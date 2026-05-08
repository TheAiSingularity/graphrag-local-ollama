# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""BasicSearch — lightweight vector-similarity RAG search.

Adapted from Microsoft GraphRAG v1.1+ for local Ollama usage (v0.1.1 architecture).

Algorithm:
  1. Embed the query with the entity text embedder.
  2. Retrieve top-k most semantically similar entities from the vector store.
  3. Gather their text units as context.
  4. Ask the LLM to answer using that context.

Unlike GlobalSearch (which uses community reports) and LocalSearch (which uses the
full entity/relationship/covariate graph), BasicSearch is fast and needs no
pre-computed community reports — it works even after a minimal index run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import tiktoken

from graphrag.model import Entity, Relationship, TextUnit
from graphrag.query.context_builder.conversation_history import ConversationHistory
from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
    map_query_to_entities,
)
from graphrag.query.context_builder.local_context import (
    build_entity_context,
    build_relationship_context,
)
from graphrag.query.context_builder.source_context import build_text_unit_context
from graphrag.query.llm.base import BaseLLM, BaseTextEmbedding
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.base import BaseSearch, SearchResult
from graphrag.vector_stores import BaseVectorStore

log = logging.getLogger(__name__)

BASIC_SEARCH_SYSTEM_PROMPT = """\
---Role---

You are a helpful assistant answering questions about documents in a knowledge base.

---Goal---

Generate a {response_type} response that directly answers the user's question, drawing on the \
most relevant entity descriptions and text excerpts provided below. Do not speculate beyond \
the provided context. If the answer is not in the context, say so.

---Context---
{context_data}
"""

DEFAULT_LLM_PARAMS: dict[str, Any] = {"max_tokens": 1500, "temperature": 0.0}


class BasicSearch(BaseSearch):
    """Lightweight vector-similarity RAG search — no community reports required."""

    def __init__(
        self,
        llm: BaseLLM,
        text_embedder: BaseTextEmbedding,
        entity_text_embeddings: BaseVectorStore,
        entities: list[Entity],
        relationships: list[Relationship] | None = None,
        text_units: list[TextUnit] | None = None,
        token_encoder: tiktoken.Encoding | None = None,
        response_type: str = "multiple paragraphs",
        top_k_entities: int = 10,
        top_k_relationships: int = 10,
        include_text_units: bool = True,
        embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
        llm_params: dict[str, Any] | None = None,
        context_builder_params: dict[str, Any] | None = None,
    ) -> None:
        # BasicSearch doesn't use a formal context builder; pass None and override search
        from graphrag.query.structured_search.base import BaseSearch
        self.llm = llm
        self.text_embedder = text_embedder
        self.entity_text_embeddings = entity_text_embeddings
        self.entities = entities
        self.relationships = relationships or []
        self.text_units = text_units or []
        self.token_encoder = token_encoder
        self.response_type = response_type
        self.top_k_entities = top_k_entities
        self.top_k_relationships = top_k_relationships
        self.include_text_units = include_text_units
        self.embedding_vectorstore_key = embedding_vectorstore_key
        self.llm_params = llm_params or DEFAULT_LLM_PARAMS
        self.context_builder_params = context_builder_params or {}
        # Required by BaseSearch ABC pattern but we don't use context_builder
        self.context_builder = None  # type: ignore[assignment]

    def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Run basic search synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.asearch(query, conversation_history, **kwargs))
            return loop.run_until_complete(self.asearch(query, conversation_history, **kwargs))
        except RuntimeError:
            return asyncio.run(self.asearch(query, conversation_history, **kwargs))

    async def asearch(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Run basic search asynchronously."""
        start_time = time.time()

        # 1. Retrieve top-k entities by embedding similarity
        top_entities = map_query_to_entities(
            query=query,
            text_embedding_vectorstore=self.entity_text_embeddings,
            text_embedder=self.text_embedder,
            all_entities=self.entities,
            embedding_vectorstore_key=self.embedding_vectorstore_key,
            k=self.top_k_entities,
        )

        # 2. Build context: entities + optionally relationships + text units
        entity_context, entity_df = build_entity_context(
            selected_entities=top_entities,
            token_encoder=self.token_encoder,
            max_tokens=2000,
            include_entity_rank=True,
        )

        relationship_context = ""
        if self.relationships:
            top_entity_ids = {e.id for e in top_entities}
            relevant_rels = [
                r for r in self.relationships
                if r.source_id in top_entity_ids or r.target_id in top_entity_ids
            ][: self.top_k_relationships]
            relationship_context, _ = build_relationship_context(
                selected_entities=top_entities,
                relationships=relevant_rels,
                token_encoder=self.token_encoder,
                max_tokens=2000,
            )

        text_unit_context = ""
        if self.include_text_units and self.text_units:
            top_entity_text_unit_ids: set[str] = set()
            for e in top_entities:
                for tid in (e.text_unit_ids or []):
                    top_entity_text_unit_ids.add(tid)
            relevant_units = [u for u in self.text_units if u.id in top_entity_text_unit_ids]
            text_unit_context, _ = build_text_unit_context(
                text_units=relevant_units,
                token_encoder=self.token_encoder,
                max_tokens=2000,
            )

        parts = [p for p in [entity_context, relationship_context, text_unit_context] if p]
        context_text = "\n\n".join(parts) if parts else "No relevant context found."

        # 3. Ask LLM
        system_prompt = BASIC_SEARCH_SYSTEM_PROMPT.format(
            context_data=context_text,
            response_type=self.response_type,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        response = await self.llm.agenerate(
            messages=messages,
            streaming=True,
            **self.llm_params,
        )

        return SearchResult(
            response=response,
            context_data={"entities": entity_df},
            context_text=context_text,
            completion_time=time.time() - start_time,
            llm_calls=1,
            prompt_tokens=num_tokens(system_prompt, self.token_encoder),
        )

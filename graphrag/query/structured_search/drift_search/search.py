# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""DRIFTSearch — Dynamic Reasoning with Iterative Feedback and Tracking.

Adapted from Microsoft GraphRAG v0.4+ for local Ollama usage (v0.1.1 architecture).

Algorithm:
  1. Primer: decompose the query into scored sub-questions using community reports.
  2. Main loop: priority-queue of sub-questions; for each, run a local-search-style
     context retrieval + LLM call that returns JSON {response, score, follow_up_queries}.
     Enqueue follow-ups and continue until the queue is drained or depth/budget limits hit.
  3. Reduce: synthesise all intermediate answers into a final response.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import logging
import time
from typing import Any

import tiktoken

from graphrag.query.structured_search.drift_search._utils import clean_up_json
from graphrag.model import CommunityReport
from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import ConversationHistory
from graphrag.query.llm.base import BaseLLM
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.base import BaseSearch, SearchResult

from .primer import DRIFTPrimer

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

DRIFT_ACTION_SYSTEM_PROMPT = """\
You are a knowledge graph analyst. Use the context data below to answer the question.
After answering, rate how well the context supports your answer and suggest 1-3 follow-up questions \
that would deepen the analysis.

Return ONLY a JSON object — no markdown, no extra text:
{{
  "response": "<your detailed answer>",
  "score": <float 0.0-1.0>,
  "follow_up_queries": ["<follow-up 1>", "<follow-up 2>"]
}}

score meaning: 0.0 = context does not support answering; 1.0 = context fully answers the question.

---Context---
{context_data}
"""

DRIFT_REDUCE_SYSTEM_PROMPT = """\
You are a senior knowledge analyst. Below are intermediate answers gathered through iterative \
knowledge-graph exploration for a user's question. Synthesise a comprehensive, coherent final \
response. Prioritise higher-scored answers. Cite supporting evidence where possible. \
Format as {response_type}.

---Intermediate answers---
{context_data}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_action_response(raw: str) -> dict[str, Any]:
    cleaned = clean_up_json(raw)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        import re
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, ValueError):
                pass
    return {"response": raw, "score": 0.3, "follow_up_queries": []}


# ──────────────────────────────────────────────────────────────────────────────
# Main search class
# ──────────────────────────────────────────────────────────────────────────────

class DRIFTSearch(BaseSearch):
    """DRIFT search: iterative graph reasoning with dynamic follow-up query generation."""

    def __init__(
        self,
        llm: BaseLLM,
        context_builder: LocalContextBuilder,
        reports: list[CommunityReport],
        token_encoder: tiktoken.Encoding | None = None,
        response_type: str = "multiple paragraphs",
        max_depth: int = 2,
        max_actions: int = 10,
        drift_k_followups: int = 2,
        primer_max_reports: int = 10,
        llm_params: dict[str, Any] | None = None,
        context_builder_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params or {"max_tokens": 1500, "temperature": 0.0},
            context_builder_params=context_builder_params or {},
        )
        self.reports = reports
        self.response_type = response_type
        self.max_depth = max_depth
        self.max_actions = max_actions
        self.drift_k_followups = drift_k_followups
        self.primer_max_reports = primer_max_reports

        self._primer = DRIFTPrimer(
            llm=llm,
            reports=reports,
            max_reports=primer_max_reports,
            llm_params=self.llm_params,
        )

    # ------------------------------------------------------------------
    # Sync entry point
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Run DRIFT search synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.asearch(query, conversation_history, **kwargs))
            return loop.run_until_complete(self.asearch(query, conversation_history, **kwargs))
        except RuntimeError:
            return asyncio.run(self.asearch(query, conversation_history, **kwargs))

    # ------------------------------------------------------------------
    # Async entry point
    # ------------------------------------------------------------------

    async def asearch(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Run DRIFT search asynchronously."""
        start_time = time.time()
        all_intermediate: list[dict[str, Any]] = []
        visited: set[str] = set()
        prompt_tokens = 0
        llm_calls = 0

        # Step 1: Primer — decompose query
        primer_items = await self._primer.adecompose(query)
        llm_calls += 1

        # Build initial priority queue: (neg_score, depth, query_text)
        # heapq is min-heap, so negate score for max-priority behaviour
        queue: list[tuple[float, int, str]] = []
        for item in primer_items:
            q = item.get("query", "")
            score = float(item.get("score", 0.5))
            if q and q not in visited:
                heapq.heappush(queue, (-score, 0, q))
                # Treat primer intermediate answers as first-depth results
                all_intermediate.append({
                    "query": q,
                    "response": item.get("intermediate_answer", ""),
                    "score": score,
                    "depth": 0,
                })
                visited.add(q)

        # Fallback: if primer produced nothing, start with the original query
        if not queue:
            heapq.heappush(queue, (-1.0, 0, query))

        # Step 2: Iterative search loop
        actions_taken = 0
        while queue and actions_taken < self.max_actions:
            neg_score, depth, sub_query = heapq.heappop(queue)
            if depth >= self.max_depth:
                continue

            # Build context using the local search context builder
            context_text, _context_records = self.context_builder.build_context(
                query=sub_query,
                conversation_history=None,
                **self.context_builder_params,
            )

            system_prompt = DRIFT_ACTION_SYSTEM_PROMPT.format(context_data=context_text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sub_query},
            ]
            prompt_tokens += num_tokens(system_prompt, self.token_encoder)

            raw = await self.llm.agenerate(messages=messages, streaming=False, **self.llm_params)
            llm_calls += 1
            parsed = _parse_action_response(raw)

            result_score = float(parsed.get("score", 0.3))
            all_intermediate.append({
                "query": sub_query,
                "response": parsed.get("response", ""),
                "score": result_score,
                "depth": depth,
            })
            actions_taken += 1

            # Enqueue follow-ups
            follow_ups: list[str] = parsed.get("follow_up_queries", [])
            for fq in follow_ups[: self.drift_k_followups]:
                if fq and fq not in visited:
                    # Follow-up score slightly lower than parent
                    fq_score = max(0.0, result_score - 0.1)
                    heapq.heappush(queue, (-fq_score, depth + 1, fq))
                    visited.add(fq)

        # Step 3: Reduce all intermediate answers
        final_response = await self._reduce(query, all_intermediate)
        llm_calls += 1

        # Build readable context summary
        context_summary = "\n\n".join(
            f"[depth={a['depth']} score={a['score']:.2f}] Q: {a['query']}\nA: {a['response']}"
            for a in sorted(all_intermediate, key=lambda x: -x["score"])
        )

        return SearchResult(
            response=final_response,
            context_data=context_summary,
            context_text=context_summary,
            completion_time=time.time() - start_time,
            llm_calls=llm_calls,
            prompt_tokens=prompt_tokens,
        )

    # ------------------------------------------------------------------
    # Reduce helper
    # ------------------------------------------------------------------

    async def _reduce(self, original_query: str, intermediate: list[dict[str, Any]]) -> str:
        if not intermediate:
            return "No relevant information found in the knowledge graph for this query."

        sorted_answers = sorted(intermediate, key=lambda x: -x["score"])
        context_parts = []
        for a in sorted_answers:
            if a["response"]:
                context_parts.append(
                    f"[score={a['score']:.2f}] Sub-question: {a['query']}\nAnswer: {a['response']}"
                )

        context_data = "\n\n---\n\n".join(context_parts[:15])
        system_prompt = DRIFT_REDUCE_SYSTEM_PROMPT.format(
            context_data=context_data,
            response_type=self.response_type,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_query},
        ]
        return await self.llm.agenerate(
            messages=messages,
            streaming=True,
            **{**self.llm_params, "max_tokens": 2000},
        )

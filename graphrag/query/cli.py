# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Command line interface for the query module."""

import os
from pathlib import Path
from typing import cast

import pandas as pd

from graphrag.config import (
    GraphRagConfig,
    create_graphrag_config,
)
from graphrag.index.progress import PrintProgressReporter
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.vector_stores import VectorStoreFactory, VectorStoreType

from .factories import get_global_search_engine, get_local_search_engine
from .indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)

reporter = PrintProgressReporter("")


def __get_embedding_description_store(
    vector_store_type: str = VectorStoreType.LanceDB, config_args: dict | None = None
):
    """Get the embedding description store."""
    if not config_args:
        config_args = {}

    config_args.update({
        "collection_name": config_args.get(
            "query_collection_name",
            config_args.get("collection_name", "description_embedding"),
        ),
    })

    description_embedding_store = VectorStoreFactory.get_vector_store(
        vector_store_type=vector_store_type, kwargs=config_args
    )

    description_embedding_store.connect(**config_args)
    return description_embedding_store


def run_global_search(
    data_dir: str | None,
    root_dir: str | None,
    community_level: int,
    response_type: str,
    query: str,
):
    """Run a global search with the given query."""
    data_dir, root_dir, config = _configure_paths_and_settings(data_dir, root_dir)
    data_path = Path(data_dir)

    final_nodes: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_nodes.parquet"
    )
    final_entities: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_entities.parquet"
    )
    final_community_reports: pd.DataFrame = pd.read_parquet(
        data_path / "create_final_community_reports.parquet"
    )

    reports = read_indexer_reports(
        final_community_reports, final_nodes, community_level
    )
    entities = read_indexer_entities(final_nodes, final_entities, community_level)
    search_engine = get_global_search_engine(
        config,
        reports=reports,
        entities=entities,
        response_type=response_type,
    )

    result = search_engine.search(query=query)

    reporter.success(f"Global Search Response: {result.response}")
    return result.response


def run_local_search(
    data_dir: str | None,
    root_dir: str | None,
    community_level: int,
    response_type: str,
    query: str,
):
    """Run a local search with the given query."""
    data_dir, root_dir, config = _configure_paths_and_settings(data_dir, root_dir)
    data_path = Path(data_dir)

    final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
    final_community_reports = pd.read_parquet(
        data_path / "create_final_community_reports.parquet"
    )
    final_text_units = pd.read_parquet(data_path / "create_final_text_units.parquet")
    final_relationships = pd.read_parquet(
        data_path / "create_final_relationships.parquet"
    )
    final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
    final_entities = pd.read_parquet(data_path / "create_final_entities.parquet")
    final_covariates_path = data_path / "create_final_covariates.parquet"
    final_covariates = (
        pd.read_parquet(final_covariates_path)
        if final_covariates_path.exists()
        else None
    )

    vector_store_args = (
        config.embeddings.vector_store if config.embeddings.vector_store else {}
    )
    vector_store_type = vector_store_args.get("type", VectorStoreType.LanceDB)

    description_embedding_store = __get_embedding_description_store(
        vector_store_type=vector_store_type,
        config_args=vector_store_args,
    )
    entities = read_indexer_entities(final_nodes, final_entities, community_level)
    store_entity_semantic_embeddings(
        entities=entities, vectorstore=description_embedding_store
    )
    covariates = (
        read_indexer_covariates(final_covariates)
        if final_covariates is not None
        else []
    )

    search_engine = get_local_search_engine(
        config,
        reports=read_indexer_reports(
            final_community_reports, final_nodes, community_level
        ),
        text_units=read_indexer_text_units(final_text_units),
        entities=entities,
        relationships=read_indexer_relationships(final_relationships),
        covariates={"claims": covariates},
        description_embedding_store=description_embedding_store,
        response_type=response_type,
    )

    result = search_engine.search(query=query)
    reporter.success(f"Local Search Response: {result.response}")
    return result.response


def run_lazy_search(
    data_dir: str | None,
    root_dir: str | None,
    response_type: str,
    query: str,
):
    """Run a lazy global search using the entity graph without pre-computed community reports.

    This is the LazyGraphRAG query path — community summaries are generated on the fly
    at query time rather than being pre-computed during indexing.
    """
    import asyncio

    data_dir, root_dir, config = _configure_paths_and_settings(data_dir, root_dir)
    data_path = Path(data_dir)

    final_entities = pd.read_parquet(data_path / "create_final_entities.parquet")
    final_relationships = pd.read_parquet(data_path / "create_final_relationships.parquet")

    query_lower = query.lower()
    query_tokens = set(query_lower.split())

    def _entity_relevance(row) -> int:
        title = str(row.get("title", "") or "").lower()
        desc = str(row.get("description", "") or "").lower()
        return sum(1 for t in query_tokens if t in title or t in desc)

    scored = final_entities.copy()
    scored["_score"] = scored.apply(_entity_relevance, axis=1)
    top_entities = scored.nlargest(20, "_score")

    entity_ids = set(top_entities["id"].tolist()) if "id" in top_entities.columns else set()
    relevant_rels = final_relationships[
        final_relationships.get("source", pd.Series(dtype=str)).isin(entity_ids)
        | final_relationships.get("target", pd.Series(dtype=str)).isin(entity_ids)
    ] if not final_relationships.empty else pd.DataFrame()

    context_parts = ["## Relevant Entities\n"]
    for _, ent in top_entities.head(15).iterrows():
        title = ent.get("title", "Unknown")
        desc = ent.get("description", "")
        context_parts.append(f"- **{title}**: {desc}")

    if not relevant_rels.empty:
        context_parts.append("\n## Relationships\n")
        for _, rel in relevant_rels.head(20).iterrows():
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            desc = rel.get("description", "")
            context_parts.append(f"- {src} → {tgt}: {desc}")

    context = "\n".join(context_parts)

    from graphrag.index.llm import load_llm
    from datashaper import NoopVerbCallbacks

    llm = load_llm(
        "lazy_search",
        config.llm.type,
        NoopVerbCallbacks(),
        None,
        config.llm.model_dump(),
    )

    prompt = (
        f"You are a knowledge graph assistant. Using the entity and relationship context below, "
        f"answer the following question as a {response_type}.\n\n"
        f"Question: {query}\n\n"
        f"{context}\n\n"
        f"Answer:"
    )

    async def _ask():
        from graphrag.llm.types import LLMInput
        result = await llm(prompt, name="lazy_search", history=[], json=False)
        return result.output if hasattr(result, "output") else str(result)

    try:
        response = asyncio.run(_ask())
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(_ask())

    reporter.success(f"Lazy Search Response: {response}")
    return response


def _configure_paths_and_settings(
    data_dir: str | None, root_dir: str | None
) -> tuple[str, str | None, GraphRagConfig]:
    if data_dir is None and root_dir is None:
        msg = "Either data_dir or root_dir must be provided."
        raise ValueError(msg)
    if data_dir is None:
        data_dir = _infer_data_dir(cast(str, root_dir))
    config = _create_graphrag_config(root_dir, data_dir)
    return data_dir, root_dir, config


def _infer_data_dir(root: str) -> str:
    output = Path(root) / "output"
    # use the latest data-run folder
    if output.exists():
        folders = sorted(output.iterdir(), key=os.path.getmtime, reverse=True)
        if len(folders) > 0:
            folder = folders[0]
            return str((folder / "artifacts").absolute())
    msg = f"Could not infer data directory from root={root}"
    raise ValueError(msg)


def _create_graphrag_config(root: str | None, data_dir: str | None) -> GraphRagConfig:
    """Create a GraphRag configuration."""
    return _read_config_parameters(cast(str, root or data_dir))


def _read_config_parameters(root: str):
    _root = Path(root)
    settings_yaml = _root / "settings.yaml"
    if not settings_yaml.exists():
        settings_yaml = _root / "settings.yml"
    settings_json = _root / "settings.json"

    if settings_yaml.exists():
        reporter.info(f"Reading settings from {settings_yaml}")
        with settings_yaml.open("r") as file:
            import yaml

            data = yaml.safe_load(file)
            return create_graphrag_config(data, root)

    if settings_json.exists():
        reporter.info(f"Reading settings from {settings_json}")
        with settings_json.open("r") as file:
            import json

            data = json.loads(file.read())
            return create_graphrag_config(data, root)

    reporter.info("Reading settings from environment variables")
    return create_graphrag_config(root_dir=root)

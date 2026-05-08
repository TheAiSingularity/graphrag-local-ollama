# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""JSON and JSONL input loader.

Supports two formats:
  - JSON array: a single file containing ``[{"text": "...", "id": "...", "title": "..."}, ...]``
  - JSONL: one JSON object per line, each with at minimum a ``"text"`` field.

Adapted from Microsoft GraphRAG v2.1+ for local Ollama usage.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from graphrag.index.config import PipelineInputConfig
from graphrag.index.progress import ProgressReporter
from graphrag.index.storage import PipelineStorage
from graphrag.index.utils import gen_md5_hash

log = logging.getLogger(__name__)

DEFAULT_FILE_PATTERN = re.compile(r".*\.(json|jsonl)$")
input_type = "json"


def _load_json_bytes(raw: bytes) -> list[dict]:
    """Parse raw bytes as a JSON array or JSONL, return list of record dicts."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return []

    # JSONL: multiple lines each being a valid JSON object
    if "\n" in text and not text.startswith("["):
        records = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("Skipping invalid JSONL line: %s…", line[:80])
        return records

    # JSON array or single object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        log.error("Failed to parse JSON file: %s", exc)
        return []

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    log.warning("JSON root is neither list nor dict; skipping")
    return []


async def load(
    config: PipelineInputConfig,
    progress: ProgressReporter | None,
    storage: PipelineStorage,
) -> pd.DataFrame:
    """Load JSON/JSONL inputs from a directory."""
    log.info("Loading JSON files from %s", config.base_dir)

    async def load_file(path: str, group: dict | None) -> pd.DataFrame:
        group = group or {}
        raw: bytes = await storage.get(path, as_bytes=True)  # type: ignore[arg-type]
        records = _load_json_bytes(raw)
        if not records:
            log.warning("No records found in %s", path)
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Ensure required columns
        if "text" not in df.columns:
            log.warning("JSON file %s has no 'text' column; skipping", path)
            return pd.DataFrame()

        if "id" not in df.columns:
            df["id"] = df.apply(lambda x: gen_md5_hash(x, x.keys()), axis=1)

        if "title" not in df.columns:
            df["title"] = Path(path).stem

        if "source" not in df.columns:
            df["source"] = path

        # Merge any group metadata
        for k, v in group.items():
            if k not in df.columns:
                df[k] = v

        return df

    file_pattern = (
        re.compile(config.file_pattern) if config.file_pattern else DEFAULT_FILE_PATTERN
    )
    files = list(
        storage.find(
            file_pattern,
            progress=progress,
            file_filter=config.file_filter,
        )
    )

    if not files:
        msg = f"No JSON/JSONL files found in {config.base_dir}"
        raise ValueError(msg)

    frames = [await load_file(path, group) for path, group in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        msg = f"All JSON files in {config.base_dir} were empty or invalid"
        raise ValueError(msg)

    result = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d JSON records from %d files", len(result), len(frames))
    return result

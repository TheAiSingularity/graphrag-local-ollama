# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""DRIFT Primer — decomposes a query into scored sub-queries using community report context."""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from graphrag.query.structured_search.drift_search._utils import clean_up_json
from graphrag.model import CommunityReport
from graphrag.query.llm.base import BaseLLM

log = logging.getLogger(__name__)

DRIFT_PRIMER_PROMPT = """\
You are an expert knowledge analyst. You are given a user question and a set of community-report \
summaries extracted from a knowledge graph.

Your task is to:
1. Decompose the original question into 3-5 focused sub-questions that together would \
   fully answer the original question.
2. For each sub-question, write a brief intermediate answer based only on the provided reports.
3. Score each sub-question 0.0-1.0 by how directly it relates to the original question.
4. Suggest 1-2 follow-up questions that would deepen the analysis.

Return ONLY a JSON object in this exact format (no markdown, no prose):
{{
  "intermediate_answers": [
    {{
      "query": "<sub-question text>",
      "intermediate_answer": "<brief answer from the context>",
      "score": 0.8,
      "follow_up_queries": ["<follow-up 1>", "<follow-up 2>"]
    }}
  ]
}}

---
Community Reports:
{context_data}
"""

_FALLBACK_SCORE = 0.5


def _build_report_context(reports: list[CommunityReport], max_reports: int = 10, max_chars: int = 4000) -> str:
    sampled = random.sample(reports, min(len(reports), max_reports))
    parts = []
    total = 0
    for r in sampled:
        text = f"[Community {r.id}] {r.title}\n{r.summary or ''}"
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def _parse_primer_response(raw: str) -> list[dict[str, Any]]:
    cleaned = clean_up_json(raw)
    try:
        data = json.loads(cleaned)
        return data.get("intermediate_answers", [])
    except (json.JSONDecodeError, ValueError):
        # Try extracting the first JSON object
        import re
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                return data.get("intermediate_answers", [])
            except (json.JSONDecodeError, ValueError):
                pass
    log.warning("DRIFT primer: failed to parse LLM response as JSON; using empty decomposition")
    return []


class DRIFTPrimer:
    """Decomposes a query into scored sub-queries using community report context."""

    def __init__(
        self,
        llm: BaseLLM,
        reports: list[CommunityReport],
        max_reports: int = 10,
        llm_params: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.reports = reports
        self.max_reports = max_reports
        self.llm_params = llm_params or {"max_tokens": 1500, "temperature": 0.0}

    def decompose(self, query: str) -> list[dict[str, Any]]:
        """Synchronous decomposition — returns list of sub-query dicts."""
        context = _build_report_context(self.reports, self.max_reports)
        system_prompt = DRIFT_PRIMER_PROMPT.format(context_data=context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        raw = self.llm.generate(messages=messages, streaming=False, **self.llm_params)
        return _parse_primer_response(raw)

    async def adecompose(self, query: str) -> list[dict[str, Any]]:
        """Async decomposition — returns list of sub-query dicts."""
        context = _build_report_context(self.reports, self.max_reports)
        system_prompt = DRIFT_PRIMER_PROMPT.format(context_data=context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        raw = await self.llm.agenerate(messages=messages, streaming=False, **self.llm_params)
        return _parse_primer_response(raw)

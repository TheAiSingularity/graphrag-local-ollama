# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Lightweight utilities for DRIFT search — avoids importing heavy graphrag.index chain."""

from __future__ import annotations


def clean_up_json(json_str: str) -> str:
    """Clean up a JSON string returned by an LLM (strip markdown fences, whitespace)."""
    json_str = (
        json_str.replace("\\n", "")
        .replace("\n", "")
        .replace("\r", "")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", "")
        .strip()
    )
    if json_str.startswith("```json"):
        json_str = json_str[len("```json"):]
    if json_str.startswith("```"):
        json_str = json_str[len("```"):]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    return json_str.strip()

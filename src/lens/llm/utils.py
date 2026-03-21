"""Shared utilities for LLM response processing."""

from __future__ import annotations


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output, extracting the JSON object.

    Looks for the outermost ``{…}`` pair inside fenced code blocks.
    Only handles JSON objects, not arrays — all LENS LLM responses are objects.

    Handles responses like::

        ```json
        {"key": "value"}
        ```
    """
    text = text.strip()
    if text.startswith("```"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
    return text

"""Core extraction logic — LLM call, JSON parsing, retry."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from lens.extract.prompts import build_extraction_prompt
from lens.llm.client import LLMClient
from lens.store.models import (
    AgenticExtraction,
    ArchitectureExtraction,
    TradeoffExtraction,
)

logger = logging.getLogger(__name__)

ExtractionTuple = tuple[
    list[dict[str, Any]],  # tradeoffs
    list[dict[str, Any]],  # architecture
    list[dict[str, Any]],  # agentic
]


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
    return text


def _validate_tradeoff(raw: dict, paper_id: str) -> dict | None:
    raw["paper_id"] = paper_id
    try:
        return TradeoffExtraction(**raw).model_dump()
    except (ValidationError, TypeError):
        logger.warning(f"Invalid tradeoff for {paper_id}")
        return None


def _validate_architecture(raw: dict, paper_id: str) -> dict | None:
    raw["paper_id"] = paper_id
    try:
        return ArchitectureExtraction(**raw).model_dump()
    except (ValidationError, TypeError):
        logger.warning(f"Invalid architecture for {paper_id}")
        return None


def _validate_agentic(raw: dict, paper_id: str) -> dict | None:
    raw["paper_id"] = paper_id
    try:
        return AgenticExtraction(**raw).model_dump()
    except (ValidationError, TypeError):
        logger.warning(f"Invalid agentic for {paper_id}")
        return None


def parse_extraction_response(
    response_text: str,
    paper_id: str,
) -> ExtractionTuple | None:
    text = _strip_code_fences(response_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    tradeoffs = [
        v for t in data.get("tradeoffs", []) if (v := _validate_tradeoff(t, paper_id)) is not None
    ]
    architecture = [
        v
        for a in data.get("architecture", [])
        if (v := _validate_architecture(a, paper_id)) is not None
    ]
    agentic = [
        v for ag in data.get("agentic", []) if (v := _validate_agentic(ag, paper_id)) is not None
    ]
    return tradeoffs, architecture, agentic


async def extract_paper(
    paper_id: str,
    title: str,
    abstract: str,
    llm_client: LLMClient,
    full_text: str | None = None,
) -> ExtractionTuple | None:
    prompt = build_extraction_prompt(title, abstract, full_text=full_text)
    messages = [{"role": "user", "content": prompt}]

    try:
        response = await llm_client.complete(messages)
    except Exception:
        logger.warning(f"LLM call failed for paper {paper_id}")
        return None

    result = parse_extraction_response(response, paper_id)
    if result is not None:
        return result

    # Retry with stricter prompt
    logger.info(f"Retrying extraction for {paper_id}")
    retry_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
        {
            "role": "user",
            "content": (
                "Your previous response was not valid JSON. "
                "Please respond with ONLY a valid JSON object. "
                "No markdown, no explanation, just the JSON."
            ),
        },
    ]

    try:
        response = await llm_client.complete(retry_messages)
    except Exception:
        logger.warning(f"LLM retry failed for paper {paper_id}")
        return None

    return parse_extraction_response(response, paper_id)

"""Core extraction logic — LLM call, JSON parsing, retry."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

import polars as pl
from pydantic import ValidationError

from lens.extract.prompts import build_extraction_prompt
from lens.llm.client import LLMClient
from lens.llm.utils import strip_code_fences
from lens.store.models import (
    AgenticExtraction,
    ArchitectureExtraction,
    TradeoffExtraction,
)
from lens.store.store import LensStore, escape_sql_string

logger = logging.getLogger(__name__)

ExtractionTuple = tuple[
    list[dict[str, Any]],  # tradeoffs
    list[dict[str, Any]],  # architecture
    list[dict[str, Any]],  # agentic
]


_strip_code_fences = strip_code_fences  # backwards-compatible alias


def _validate_tradeoff(raw: dict[str, Any], paper_id: str) -> dict[str, Any] | None:
    try:
        merged = {**raw, "paper_id": paper_id}
        return TradeoffExtraction.model_validate(merged).model_dump()
    except (ValidationError, TypeError):
        logger.warning("Invalid tradeoff for %s", paper_id)
        return None


def _validate_architecture(raw: dict[str, Any], paper_id: str) -> dict[str, Any] | None:
    try:
        merged = {**raw, "paper_id": paper_id}
        return ArchitectureExtraction.model_validate(merged).model_dump()
    except (ValidationError, TypeError):
        logger.warning("Invalid architecture for %s", paper_id)
        return None


def _validate_agentic(raw: dict[str, Any], paper_id: str) -> dict[str, Any] | None:
    try:
        merged = {**raw, "paper_id": paper_id}
        return AgenticExtraction.model_validate(merged).model_dump()
    except (ValidationError, TypeError):
        logger.warning("Invalid agentic for %s", paper_id)
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


def _delete_old_extractions(store: LensStore, paper_id: str) -> None:
    """Delete previous extractions for a paper (idempotent re-extraction)."""
    safe_id = escape_sql_string(paper_id)
    for table_name in [
        "tradeoff_extractions",
        "architecture_extractions",
        "agentic_extractions",
    ]:
        with contextlib.suppress(OSError, ValueError):
            store.get_table(table_name).delete(f"paper_id = '{safe_id}'")


def _update_paper_status(store: LensStore, paper_id: str, status: str) -> None:
    """Update a paper's extraction_status."""
    safe_id = escape_sql_string(paper_id)
    try:
        store.get_table("papers").update(
            where=f"paper_id = '{safe_id}'",
            values={"extraction_status": status},
        )
    except Exception:
        logger.warning("Failed to update status for %s", paper_id)


async def extract_papers(
    store: LensStore,
    llm_client: LLMClient,
    concurrency: int = 5,
    paper_id: str | None = None,
) -> int:
    """Extract knowledge from all pending papers in the store.

    Idempotent: re-extraction deletes old results before storing new ones.
    Updates extraction_status to 'complete' or 'incomplete'.
    """
    papers_df = store.get_table("papers").to_polars()

    if paper_id:
        papers_df = papers_df.filter(pl.col("paper_id") == paper_id)
    else:
        papers_df = papers_df.filter(pl.col("extraction_status").is_in(["pending", "incomplete"]))

    if len(papers_df) == 0:
        logger.info("No papers to extract")
        return 0

    semaphore = asyncio.Semaphore(concurrency)

    async def extract_one(row: dict) -> tuple[str, ExtractionTuple | None]:
        """Run LLM extraction concurrently; return (paper_id, result)."""
        async with semaphore:
            pid = row["paper_id"]
            result = await extract_paper(
                paper_id=pid,
                title=row["title"],
                abstract=row["abstract"],
                llm_client=llm_client,
            )
            return pid, result

    # Phase 1: Run all LLM calls concurrently
    tasks = [extract_one(row) for row in papers_df.to_dicts()]
    extraction_results = await asyncio.gather(*tasks)

    # Phase 2: Write results to LanceDB sequentially (avoids concurrent writes)
    success_count = 0
    for pid, result in extraction_results:
        _delete_old_extractions(store, pid)

        if result is None:
            _update_paper_status(store, pid, "incomplete")
            logger.warning("Extraction failed for %s", pid)
            continue

        tradeoffs, architecture, agentic = result
        if tradeoffs:
            store.add_rows("tradeoff_extractions", tradeoffs)
        if architecture:
            store.add_rows("architecture_extractions", architecture)
        if agentic:
            store.add_rows("agentic_extractions", agentic)

        _update_paper_status(store, pid, "complete")
        logger.info(
            "Extracted %s: %d tradeoffs, %d arch, %d agentic",
            pid,
            len(tradeoffs),
            len(architecture),
            len(agentic),
        )
        success_count += 1

    return success_count

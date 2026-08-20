"""Core extraction logic — LLM call, JSON parsing, retry."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from lens.extract.prompts import build_extraction_prompt
from lens.knowledge.events import log_event
from lens.llm.client import LLMClient
from lens.llm.schemas import ExtractionResponse
from lens.llm.utils import strip_code_fences
from lens.store.models import (
    AgenticExtraction,
    ArchitectureExtraction,
    TradeoffExtraction,
)
from lens.store.store import LensStore

logger = logging.getLogger(__name__)

ExtractionTuple = tuple[
    list[dict[str, Any]],  # tradeoffs
    list[dict[str, Any]],  # architecture
    list[dict[str, Any]],  # agentic
]

# Threshold below which a quote is treated as insufficient to "verify" a claim.
# ~2 words — shorter quotes ("yes", "p<.05") almost never substantiate a full
# tradeoff, so high-confidence rows with a tiny quote get demoted to "inferred".
_MIN_QUOTE_LEN = 10


def compute_verification_status(
    confidence: float,
    evidence_quote: str | None = None,
) -> str:
    """Derive a 4-label verification status from an extraction's confidence and evidence.

    For extraction types that carry a quote (tradeoffs), a substantive quote plus
    high confidence yields ``verified``. For extraction types without a quote
    (architecture, agentic), only confidence is considered. Borrowed from the
    Feynman project's verified/inferred/unverified/blocked vocabulary.
    """
    quote_ok = evidence_quote is None or len(evidence_quote.strip()) >= _MIN_QUOTE_LEN
    if confidence >= 0.8 and quote_ok:
        return "verified"
    if confidence >= 0.5:
        return "inferred"
    return "unverified"


def _validate_tradeoff(raw: dict[str, Any], paper_id: str) -> dict[str, Any] | None:
    try:
        merged = {**raw, "paper_id": paper_id}
        merged.setdefault(
            "verification_status",
            compute_verification_status(
                confidence=float(merged.get("confidence", 0.0)),
                # Coerce None → "" so an LLM-emitted `null` is treated as "no quote"
                # rather than flowing into `quote_ok=True` (the None branch).
                evidence_quote=merged.get("evidence_quote") or "",
            ),
        )
        return TradeoffExtraction.model_validate(merged).model_dump()
    except (ValidationError, TypeError) as e:
        logger.warning("Invalid tradeoff for %s: %s", paper_id, e)
        return None


def _validate_architecture(raw: dict[str, Any], paper_id: str) -> dict[str, Any] | None:
    try:
        merged = {**raw, "paper_id": paper_id}
        merged.setdefault(
            "verification_status",
            compute_verification_status(confidence=float(merged.get("confidence", 0.0))),
        )
        return ArchitectureExtraction.model_validate(merged).model_dump()
    except (ValidationError, TypeError):
        logger.warning("Invalid architecture for %s", paper_id)
        return None


def _validate_agentic(raw: dict[str, Any], paper_id: str) -> dict[str, Any] | None:
    try:
        merged = {**raw, "paper_id": paper_id}
        merged.setdefault(
            "verification_status",
            compute_verification_status(confidence=float(merged.get("confidence", 0.0))),
        )
        return AgenticExtraction.model_validate(merged).model_dump()
    except (ValidationError, TypeError):
        logger.warning("Invalid agentic for %s", paper_id)
        return None


def parse_extraction_response(
    response_text: str,
    paper_id: str,
) -> ExtractionTuple | None:
    text = strip_code_fences(response_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Attempt repair of malformed JSON (trailing commas, unquoted keys, etc.)
        from json_repair import repair_json

        try:
            data = repair_json(text, return_objects=True)
            logger.info("Repaired malformed JSON for %s", paper_id)
        except Exception:
            logger.warning("Unrepairable JSON for %s", paper_id)
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


def _item_to_raw(item: BaseModel) -> dict[str, Any]:
    """Flatten one response item onto the persistence shape.

    ``new_concepts`` arrives as typed pairs because strict structured outputs
    cannot express an open-ended map; it is rebuilt into the ``{slug: description}``
    dict the store expects.
    """
    raw = item.model_dump()
    raw["new_concepts"] = {c["name"]: c["description"] for c in raw.get("new_concepts", [])}
    return raw


def extraction_response_to_tuple(
    response: ExtractionResponse,
    paper_id: str,
) -> ExtractionTuple:
    """Convert a validated :class:`ExtractionResponse` into the store's tuple shape.

    Reuses the same per-item validators as the prompt path, so ``paper_id`` and
    ``verification_status`` are attached identically no matter which route the
    response arrived by.
    """
    tradeoffs = [
        v
        for t in response.tradeoffs
        if (v := _validate_tradeoff(_item_to_raw(t), paper_id)) is not None
    ]
    architecture = [
        v
        for a in response.architecture
        if (v := _validate_architecture(_item_to_raw(a), paper_id)) is not None
    ]
    agentic = [
        v
        for ag in response.agentic
        if (v := _validate_agentic(_item_to_raw(ag), paper_id)) is not None
    ]
    return tradeoffs, architecture, agentic


async def extract_paper(
    paper_id: str,
    title: str,
    abstract: str,
    llm_client: LLMClient,
    full_text: str | None = None,
    vocabulary: list[dict[str, str]] | None = None,
) -> ExtractionTuple | None:
    """Extract structured information from one paper.

    Uses schema-constrained decoding where the endpoint supports it, so a
    non-conforming response is impossible rather than merely repairable.
    ``complete_structured`` handles the prompt-schema fallback and the one
    corrective retry, so no separate "please emit valid JSON" round-trip is
    needed here.
    """
    prompt = build_extraction_prompt(title, abstract, full_text=full_text, vocabulary=vocabulary)
    messages = [{"role": "user", "content": prompt}]

    try:
        response = await llm_client.complete_structured(messages, ExtractionResponse)
    except Exception:
        logger.warning("LLM extraction failed for paper %s", paper_id)
        return None

    return extraction_response_to_tuple(response, paper_id)


def _delete_old_extractions(store: LensStore, paper_id: str) -> None:
    """Delete previous extractions for a paper (idempotent re-extraction)."""
    for table_name in [
        "tradeoff_extractions",
        "architecture_extractions",
        "agentic_extractions",
    ]:
        with contextlib.suppress(OSError, ValueError):
            store.delete(table_name, "paper_id = ?", (paper_id,))


def _update_paper_status(store: LensStore, paper_id: str, status: str) -> None:
    """Update a paper's extraction_status."""
    try:
        store.update("papers", "extraction_status = ?", "paper_id = ?", (status, paper_id))
    except Exception:
        logger.warning("Failed to update status for %s", paper_id)


async def extract_papers(
    store: LensStore,
    llm_client: LLMClient,
    concurrency: int = 5,
    paper_id: str | None = None,
    session_id: str | None = None,
) -> int:
    """Extract knowledge from all pending papers in the store.

    Idempotent: re-extraction deletes old results before storing new ones.
    Updates extraction_status to 'complete' or 'incomplete'.
    """
    if paper_id:
        papers = store.query("papers", "paper_id = ?", (paper_id,))
    else:
        papers = store.query("papers", "extraction_status IN ('pending', 'incomplete')")

    if not papers:
        logger.info("No papers to extract")
        return 0

    # Load vocabulary for guided extraction
    vocab_rows = store.query("vocabulary")
    vocabulary = (
        [{"name": r["name"], "kind": r["kind"]} for r in vocab_rows] if vocab_rows else None
    )

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
                vocabulary=vocabulary,
            )
            return pid, result

    # Phase 1: Run all LLM calls concurrently
    tasks = [extract_one(row) for row in papers]
    extraction_results = await asyncio.gather(*tasks)

    # Phase 2: Write results to DB sequentially (avoids concurrent writes)
    success_count = 0
    for pid, result in extraction_results:
        _delete_old_extractions(store, pid)

        if result is None:
            _update_paper_status(store, pid, "incomplete")
            logger.warning("Extraction failed for %s", pid)
            log_event(
                store,
                "extract",
                "extraction.failed",
                target_type="paper",
                target_id=pid,
                session_id=session_id,
            )
            continue

        tradeoffs, architecture, agentic = result
        if tradeoffs:
            store.add_rows("tradeoff_extractions", tradeoffs)
        if architecture:
            store.add_rows("architecture_extractions", architecture)
        if agentic:
            store.add_rows("agentic_extractions", agentic)

        _update_paper_status(store, pid, "complete")
        log_event(
            store,
            "extract",
            "extraction.completed",
            target_type="paper",
            target_id=pid,
            detail={
                "tradeoffs": len(tradeoffs),
                "architecture": len(architecture),
                "agentic": len(agentic),
            },
            session_id=session_id,
        )
        logger.info(
            "Extracted %s: %d tradeoffs, %d arch, %d agentic",
            pid,
            len(tradeoffs),
            len(architecture),
            len(agentic),
        )
        success_count += 1

    return success_count

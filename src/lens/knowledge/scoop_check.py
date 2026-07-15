"""Scoop-check: verify idea-card novelty against OpenAlex prior art."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from lens.acquire.openalex import search_openalex
from lens.llm.utils import strip_code_fences
from lens.store.store import LensStore

logger = logging.getLogger(__name__)

_VERDICTS = {"novel", "overlaps", "scooped"}

NOVELTY_SYSTEM_PROMPT = (
    "You are a research novelty auditor. Given a proposed research idea and a list "
    "of existing papers, decide whether the idea's CORE contribution is already "
    "covered. Distinguish shared keywords from the same contribution. Respond with "
    "a single JSON object and nothing else."
)


def _str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def _format_prior_art(prior_art: list[dict[str, Any]]) -> str:
    lines = []
    for i, p in enumerate(prior_art, 1):
        yr = f" ({p['year']})" if p.get("year") else ""
        lines.append(f"{i}. {p.get('title', '')}{yr}\n   {(p.get('abstract') or '')[:500]}")
    return "\n".join(lines) if lines else "(no prior art found)"


async def judge_novelty(
    card: dict[str, Any],
    prior_art: list[dict[str, Any]],
    llm_client: Any,
) -> dict[str, Any] | None:
    """Ask the LLM whether the card's core idea is already covered by prior art."""
    user = (
        "Proposed idea:\n"
        f"  Title: {card.get('title', '')}\n"
        f"  Mechanism: {card.get('mechanism', '')}\n"
        f"  Differentiation: {'; '.join(card.get('differentiation') or [])}\n\n"
        f"Existing papers:\n{_format_prior_art(prior_art)}\n\n"
        'Return JSON: {"verdict": "novel|overlaps|scooped", '
        '"colliding_papers": ["<paper title>", ...], "rationale": "<one sentence>"}\n'
        "verdict meanings: scooped = core idea already published; "
        "overlaps = substantial related work but a distinct angle; "
        "novel = no close prior art in the list."
    )
    messages = [
        {"role": "system", "content": NOVELTY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        text = await llm_client.complete(messages)
    except Exception:
        logger.warning("Novelty judge LLM call failed")
        return None

    text = strip_code_fences(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        from json_repair import repair_json

        try:
            data = repair_json(text, return_objects=True)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in _VERDICTS:
        return None
    return {
        "verdict": verdict,
        "colliding_papers": _str_list(data.get("colliding_papers")),
        "rationale": str(data.get("rationale", "")).strip(),
    }


async def run_scoop_check(
    store: LensStore,
    llm_client: Any,
    limit: int | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Novelty-check every idea card with novelty_status='unchecked'.

    Idempotent and fail-soft: a card only leaves 'unchecked' when it gets a
    real verdict; search/judge/DB failures leave it for the next run.
    """
    cards = store.query("idea_cards", "novelty_status = ?", ("unchecked",))
    cards = sorted(cards, key=lambda c: c["id"])

    counts = {"novel": 0, "overlaps": 0, "scooped": 0}
    checked = 0

    for card in cards:
        # --limit caps the number of cards actually CHECKED (given a real
        # verdict), not pre-sliced: persistently-failing cards don't consume
        # the cap and can't starve higher-id cards from ever being reached.
        if limit is not None and checked >= limit:
            break
        terms = card.get("signature_terms") or []
        query = " ".join([card.get("title", ""), *terms]).strip()
        if not query:
            logger.info("Card %d has no title/terms to search; leaving unchecked", card["id"])
            continue
        prior_art = await search_openalex(query, limit=top_k)
        if not prior_art:
            logger.info("No prior art for card %d; leaving unchecked", card["id"])
            continue

        try:
            verdict = await judge_novelty(card, prior_art, llm_client)
        except Exception:
            logger.warning("Novelty judge crashed for card %d; leaving unchecked", card["id"])
            verdict = None
        if verdict is None:
            logger.warning("Unusable novelty judgment for card %d; leaving unchecked", card["id"])
            continue

        stored_art = [
            {"title": p.get("title", ""), "url": p.get("url", ""), "year": p.get("year")}
            for p in prior_art
        ]
        note = verdict["rationale"]
        colliding = verdict.get("colliding_papers") or []
        if colliding:
            note = f"{note} (collides with: {', '.join(colliding)})".strip()
        try:
            store.update(
                "idea_cards",
                "novelty_status = ?, prior_art = ?, novelty_note = ?, novelty_checked_at = ?",
                "id = ?",
                (
                    verdict["verdict"],
                    json.dumps(stored_art),
                    note,
                    datetime.now(UTC).isoformat(),
                    card["id"],
                ),
            )
        except Exception:
            logger.warning("Failed to persist novelty verdict for card %d", card["id"])
            continue
        if verdict["verdict"] in ("scooped", "overlaps"):
            logger.info("Card %d %s — collides with %s", card["id"], verdict["verdict"], colliding)
        counts[verdict["verdict"]] += 1
        checked += 1

    logger.info("Scoop-check: %d cards checked %s", checked, counts)
    return {"checked": checked, "by_verdict": counts}

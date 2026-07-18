"""Ideation gap analysis pipeline — sparse cells, cross-pollination, LLM enrichment."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np

from lens.llm.utils import strip_code_fences
from lens.store.store import LensStore
from lens.taxonomy.vocabulary import _slugify

logger = logging.getLogger(__name__)

# Keys carried on in-memory gap dicts for downstream provenance but not
# persisted as columns on the ``ideation_gaps`` table.
_GAP_TRANSIENT_KEYS = frozenset({"source_improving_param_id", "source_worsening_param_id"})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _card_token_set(title: str, signature_terms: list[str]) -> set[str]:
    """Lowercased word-token fingerprint of a card, for near-duplicate detection.

    Combines the title and signature terms — the two fields that carry a card's
    idea-level identity — and splits into ``[a-z0-9]+`` tokens so multi-word
    phrases ("Differentiable Architecture Search") overlap at the word level.
    """
    text = title + " " + " ".join(signature_terms)
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two token sets; empty ∪ empty is 0.0 (no divide-by-zero)."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _diversified_gap_order(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin gaps across their improving parameter, best-scored first.

    All fully-empty sparse cells score an identical ``1.0``, so a plain score sort
    would surface many gaps for one parameter before touching another. Bucketing by
    ``related_params[0]`` and taking one gap per bucket per pass spreads the LLM
    budget across parameters, so distinct ideas surface early.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for gap in gaps:
        params = gap.get("related_params") or []
        key = params[0] if params else ""
        buckets[key].append(gap)
    # Highest score first within each bucket.
    for bucket in buckets.values():
        bucket.sort(key=lambda g: g.get("score", 0.0), reverse=True)
    # Stable bucket order: by the best score each bucket offers, then by key.
    ordered_buckets = sorted(
        buckets.items(),
        key=lambda kv: (-kv[1][0].get("score", 0.0), kv[0]),
    )
    queues = [list(bucket) for _, bucket in ordered_buckets]
    result: list[dict[str, Any]] = []
    while queues:
        for queue in queues:
            result.append(queue.pop(0))
        queues = [q for q in queues if q]
    return result


def find_sparse_cells(
    store: LensStore,
    min_principles: int = 2,
) -> list[dict[str, Any]]:
    """Find parameter pairs that have fewer than *min_principles* resolving them.

    Returns a list of dicts with keys:
        improving_param_id, worsening_param_id, existing_principle_ids, count
    """
    params = store.query("vocabulary", "kind = ?", ("parameter",))
    param_ids = [p["id"] for p in params]

    cells = store.query("matrix_cells")

    # Build a mapping: (improving, worsening) -> list of principle_ids
    pair_principles: dict[tuple[str, str], list[str]] = {}
    for cell in cells:
        key = (cell["improving_param_id"], cell["worsening_param_id"])
        pair_principles.setdefault(key, []).append(cell["principle_id"])

    gaps: list[dict[str, Any]] = []
    for imp_id in param_ids:
        for wors_id in param_ids:
            if imp_id == wors_id:
                continue
            key = (imp_id, wors_id)
            existing = pair_principles.get(key, [])
            if len(existing) < min_principles:
                gaps.append(
                    {
                        "improving_param_id": imp_id,
                        "worsening_param_id": wors_id,
                        "existing_principle_ids": existing,
                        "count": len(existing),
                    }
                )
    return gaps


def find_cross_pollination(
    store: LensStore,
    similarity_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """Find principles that could transfer to similar parameter pairs.

    If principle P resolves (A, B) and A' is similar to A, but P does not
    resolve (A', B), then (A', B, P) is a cross-pollination candidate.
    Checks BOTH improving and worsening roles.
    """
    params = store.query("vocabulary", "kind = ?", ("parameter",))
    if not params:
        return []

    param_ids = [p["id"] for p in params]

    # Need to get embeddings from the vec table via raw SQL
    # Since query() doesn't return embeddings (they're in the vec table), we need
    # to get them from the vocabulary_vec table
    import struct

    vec_rows = store.conn.execute(
        "SELECT vv.id, vv.embedding FROM vocabulary_vec vv "
        "INNER JOIN vocabulary v ON vv.id = v.id "
        "WHERE v.kind = 'parameter'",
    ).fetchall()

    if not vec_rows:
        return []

    # Build id -> embedding mapping
    id_to_emb: dict[str, list[float]] = {}
    from lens.store.models import EMBEDDING_DIM

    for row in vec_rows:
        pid = row[0]
        emb_bytes = row[1]
        emb = list(struct.unpack(f"{EMBEDDING_DIM}f", emb_bytes))
        id_to_emb[pid] = emb

    # Build ordered arrays matching param_ids
    valid_param_ids = [pid for pid in param_ids if pid in id_to_emb]
    if not valid_param_ids:
        return []

    embeddings = np.array([id_to_emb[pid] for pid in valid_param_ids])

    # Compute cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    similarity = normalized @ normalized.T

    cells = store.query("matrix_cells")

    # Build set of existing (improving, worsening, principle) triples
    existing_triples: set[tuple[str, str, str]] = set()
    for cell in cells:
        existing_triples.add(
            (
                cell["improving_param_id"],
                cell["worsening_param_id"],
                cell["principle_id"],
            )
        )

    # Build index from param_id -> position in valid_param_ids list
    id_to_idx = {pid: i for i, pid in enumerate(valid_param_ids)}

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for cell in cells:
        imp_id = cell["improving_param_id"]
        wors_id = cell["worsening_param_id"]
        principle_id = cell["principle_id"]

        imp_idx = id_to_idx.get(imp_id)
        wors_idx = id_to_idx.get(wors_id)
        if imp_idx is None or wors_idx is None:
            continue

        # Check improving role: find params similar to imp_id
        for j, other_id in enumerate(valid_param_ids):
            if other_id == imp_id:
                continue
            if similarity[imp_idx, j] >= similarity_threshold:
                candidate_key = (other_id, wors_id, principle_id)
                if candidate_key not in existing_triples and candidate_key not in seen:
                    seen.add(candidate_key)
                    candidates.append(
                        {
                            "improving_param_id": other_id,
                            "worsening_param_id": wors_id,
                            "principle_id": principle_id,
                            "source_improving_param_id": imp_id,
                            "source_worsening_param_id": wors_id,
                            "similarity": float(similarity[imp_idx, j]),
                            "role": "improving",
                        }
                    )

        # Check worsening role: find params similar to wors_id
        for j, other_id in enumerate(valid_param_ids):
            if other_id == wors_id:
                continue
            if similarity[wors_idx, j] >= similarity_threshold:
                candidate_key = (imp_id, other_id, principle_id)
                if candidate_key not in existing_triples and candidate_key not in seen:
                    seen.add(candidate_key)
                    candidates.append(
                        {
                            "improving_param_id": imp_id,
                            "worsening_param_id": other_id,
                            "principle_id": principle_id,
                            "source_improving_param_id": imp_id,
                            "source_worsening_param_id": wors_id,
                            "similarity": float(similarity[wors_idx, j]),
                            "role": "worsening",
                        }
                    )

    return candidates


def run_ideation(
    store: LensStore,
    min_principles: int = 2,
    similarity_threshold: float = 0.75,
) -> dict[str, Any]:
    """Run Layer 1 ideation: sparse cells + cross-pollination.

    Persists gaps and report to DB and returns a summary dict.
    """
    now = datetime.now(UTC)

    sparse = find_sparse_cells(store, min_principles=min_principles)
    cross = find_cross_pollination(store, similarity_threshold=similarity_threshold)

    # Fetch param and principle names for descriptions
    vocab = store.query("vocabulary")
    param_names = {v["id"]: v["name"] for v in vocab if v["kind"] == "parameter"}
    principle_names = {v["id"]: v["name"] for v in vocab if v["kind"] == "principle"}

    # Determine next report_id
    report_rows = store.query_sql("SELECT MAX(id) AS max_id FROM ideation_reports")
    report_id = (int(report_rows[0]["max_id"]) + 1) if report_rows[0]["max_id"] is not None else 1

    # Determine next gap_id
    gap_rows = store.query_sql("SELECT MAX(id) AS max_id FROM ideation_gaps")
    next_gap_id = (int(gap_rows[0]["max_id"]) + 1) if gap_rows[0]["max_id"] is not None else 1

    all_gaps: list[dict[str, Any]] = []

    for gap in sparse:
        imp_name = param_names.get(gap["improving_param_id"], "?")
        wors_name = param_names.get(gap["worsening_param_id"], "?")
        count = gap["count"]
        desc = (
            f"Sparse cell: {imp_name} vs {wors_name} has no principles"
            if count == 0
            else f"Sparse cell: {imp_name} vs {wors_name} has only {count} principle(s)"
        )
        gap_record = {
            "id": next_gap_id,
            "report_id": report_id,
            "gap_type": "sparse_cell",
            "description": desc,
            "related_params": [
                gap["improving_param_id"],
                gap["worsening_param_id"],
            ],
            "related_principles": gap["existing_principle_ids"],
            "related_slots": [],
            "score": 1.0 - (gap["count"] / min_principles),
            "llm_hypothesis": None,
            "created_at": now,
            "taxonomy_version": 0,
        }
        all_gaps.append(gap_record)
        next_gap_id += 1

    for cand in cross:
        imp_name = param_names.get(cand["improving_param_id"], "?")
        wors_name = param_names.get(cand["worsening_param_id"], "?")
        p_name = principle_names.get(cand["principle_id"], "?")
        desc = (
            f"Cross-pollination: {p_name} could apply to "
            f"{imp_name} vs {wors_name} "
            f"(similarity={cand['similarity']:.2f}, role={cand['role']})"
        )
        gap_record = {
            "id": next_gap_id,
            "report_id": report_id,
            "gap_type": "cross_pollination",
            "description": desc,
            "related_params": [
                cand["improving_param_id"],
                cand["worsening_param_id"],
            ],
            "related_principles": [cand["principle_id"]],
            "related_slots": [],
            "score": cand["similarity"],
            "llm_hypothesis": None,
            "created_at": now,
            "taxonomy_version": 0,
            # In-memory only (not columns on ideation_gaps): the source cell
            # that motivated the transfer, so idea-card provenance can cite the
            # exact (source_improving, source_worsening, principle) matrix cell.
            "source_improving_param_id": cand.get("source_improving_param_id"),
            "source_worsening_param_id": cand.get("source_worsening_param_id"),
        }
        all_gaps.append(gap_record)
        next_gap_id += 1

    # Persist gaps — strip in-memory-only keys that are not columns on the table.
    if all_gaps:
        persist_rows = [
            {k: v for k, v in gap.items() if k not in _GAP_TRANSIENT_KEYS} for gap in all_gaps
        ]
        store.add_rows("ideation_gaps", persist_rows)

    # Persist report
    report_record = {
        "id": report_id,
        "created_at": now,
        "taxonomy_version": 0,
        "paper_batch_size": 0,
        "gap_count": len(all_gaps),
    }
    store.add_rows("ideation_reports", [report_record])

    logger.info(
        "Ideation: %d gaps (%d sparse, %d cross-pollination)",
        len(all_gaps),
        sum(1 for g in all_gaps if g["gap_type"] == "sparse_cell"),
        sum(1 for g in all_gaps if g["gap_type"] == "cross_pollination"),
    )

    return {
        "report_id": report_id,
        "gap_count": len(all_gaps),
        "gaps": all_gaps,
    }


IDEA_CARD_SYSTEM_PROMPT = (
    "You are a research analyst. You are given a gap in an LLM-research "
    "tradeoff matrix and a catalog of ideation patterns (reusable moves for "
    "generating research ideas). Select the 1-2 patterns most applicable to "
    "the gap and apply them to produce ONE concrete, falsifiable research idea. "
    "Respond with a single JSON object and nothing else."
)


def _format_pattern_catalog(patterns: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {p['name']}: {p['description']}" for p in patterns)


def _format_grounding(
    gap: dict[str, Any],
    vocab_by_id: dict[str, dict[str, Any]],
) -> str:
    lines: list[str] = []
    for pid in gap.get("related_params", []):
        v = vocab_by_id.get(pid)
        if v:
            lines.append(f"- parameter: {v['name']} — {v['description']}")
    for pid in gap.get("related_principles", []):
        v = vocab_by_id.get(pid)
        if v:
            lines.append(f"- principle: {v['name']} — {v['description']}")
    return "\n".join(lines) if lines else "(no referenced vocabulary)"


def _build_idea_card_messages(
    gap: dict[str, Any],
    vocab_by_id: dict[str, dict[str, Any]],
    patterns: list[dict[str, Any]],
) -> list[dict[str, str]]:
    user = (
        f"Gap: {gap['description']}\n"
        f"Gap type: {gap['gap_type']}\n\n"
        f"Referenced vocabulary:\n{_format_grounding(gap, vocab_by_id)}\n\n"
        f"Ideation pattern catalog:\n{_format_pattern_catalog(patterns)}\n\n"
        "Return a JSON object with keys: "
        '"title" (<=15 words), '
        '"patterns" (list of 1-2 pattern names copied verbatim from the catalog), '
        '"hook" (1-2 sentences), '
        '"mechanism" (how the chosen pattern applies to this gap, producing a concrete artifact), '
        '"falsification" (a minimal experiment with a metric and a distinguishing prediction), '
        '"differentiation" (list of strings: how this differs from the referenced principles), '
        '"signature_terms" (list of 3-6 key terms), '
        '"confidence" (0-1 float).'
    )
    return [
        {"role": "system", "content": IDEA_CARD_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [value] if value.strip() else []
    # Ignore malformed shapes (e.g. a dict) rather than storing a Python repr.
    return []


def _parse_idea_card(text: str, valid_pattern_ids: set[str]) -> dict[str, Any] | None:
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
    title = str(data.get("title", "")).strip()
    if not title:
        return None
    patterns_raw = data.get("patterns")
    if not isinstance(patterns_raw, list):
        patterns_raw = []
    pattern_ids = [
        pid for name in patterns_raw if (pid := _slugify(str(name))) in valid_pattern_ids
    ]
    confidence_raw = data.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else 0.5
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    return {
        "title": title,
        "pattern_ids": pattern_ids,
        "hook": str(data.get("hook", "")).strip(),
        "mechanism": str(data.get("mechanism", "")).strip(),
        "falsification": str(data.get("falsification", "")).strip(),
        "differentiation": _as_str_list(data.get("differentiation")),
        "signature_terms": _as_str_list(data.get("signature_terms")),
        "confidence": confidence,
    }


def _set_gap_hypothesis(store: LensStore, gap: dict[str, Any], text: str) -> None:
    """Persist a free-text hypothesis on a gap (back-compat), tolerating DB errors."""
    gap_id = int(gap["id"])
    try:
        store.update("ideation_gaps", "llm_hypothesis = ?", "id = ?", (text, gap_id))
        gap["llm_hypothesis"] = text
    except Exception:
        logger.warning("Failed to store hypothesis for gap %d", gap_id)


def _card_paper_ids(
    gap: dict[str, Any],
    cell_papers: dict[tuple[str, str], list[str]],
    triple_papers: dict[tuple[str, str, str], list[str]],
) -> list[str]:
    """Evidence papers for a gap's idea card.

    Cross-pollination cards cite the exact source cell that motivated the
    transfer — ``(source_improving, source_worsening, principle)`` — not the
    (empty) target pair and not every cell the principle happens to resolve.
    Sparse-cell cards cite the papers already sitting at the gap's pair.
    """
    if gap["gap_type"] == "cross_pollination":
        src_imp = gap.get("source_improving_param_id")
        src_wors = gap.get("source_worsening_param_id")
        principles = gap.get("related_principles") or []
        if src_imp and src_wors and principles:
            return triple_papers.get((src_imp, src_wors, principles[0]), [])
        return []
    params = gap.get("related_params", [])
    if len(params) >= 2:
        return cell_papers.get((params[0], params[1]), [])
    return []


async def run_ideation_with_llm(
    store: LensStore,
    llm_client: Any,
    min_principles: int = 2,
    similarity_threshold: float = 0.75,
    max_cards: int = 40,
    min_gap_score: float = 0.0,
    dedup_threshold: float = 0.32,
) -> dict[str, Any]:
    """Run ideation, then generate a diverse, capped set of Idea Cards via the LLM.

    A diversity gate sits between structural gap-finding and card emission so a run
    yields ~``max_cards`` *distinct* ideas rather than one card per matrix cell:

    * ``min_gap_score`` — drop gaps scoring below this before spending any LLM call.
    * diversified order — round-robin gaps across their improving parameter.
    * ``dedup_threshold`` — skip a card whose title+signature_terms token-Jaccard
      against any already-kept card is ≥ this.
    * ``max_cards`` — stop once this many distinct cards are kept; an internal
      ``max(max_cards*3, 60)`` LLM-call budget bounds worst-case cost.
    """
    report = run_ideation(
        store,
        min_principles=min_principles,
        similarity_threshold=similarity_threshold,
    )

    vocab = store.query("vocabulary")
    vocab_by_id = {v["id"]: v for v in vocab}
    patterns = [v for v in vocab if v["kind"] == "ideation_pattern"]
    valid_pattern_ids = {p["id"] for p in patterns}

    if not patterns:
        # An existing (pre-0.11.0) or --skip-build database may not carry the
        # ideation patterns yet. Seed them on demand rather than silently
        # disabling all enrichment (load_seed_vocabulary is idempotent).
        logger.warning(
            "No ideation_pattern vocabulary present; seeding defaults before idea-card generation"
        )
        from lens.taxonomy.vocabulary import load_seed_vocabulary

        load_seed_vocabulary(store)
        vocab = store.query("vocabulary")
        vocab_by_id = {v["id"]: v for v in vocab}
        patterns = [v for v in vocab if v["kind"] == "ideation_pattern"]
        valid_pattern_ids = {p["id"] for p in patterns}

    if not patterns:
        logger.warning("No ideation_pattern vocabulary available; skipping idea-card generation")
        report["idea_cards"] = []
        return report

    # Provenance lookups from matrix cells:
    #   cell_papers   : (improving, worsening)            -> papers (sparse-cell gaps)
    #   triple_papers : (improving, worsening, principle) -> papers (cross-pollination source cell)
    cell_papers: dict[tuple[str, str], list[str]] = {}
    triple_papers: dict[tuple[str, str, str], list[str]] = {}
    for cell in store.query("matrix_cells"):
        pair = (cell["improving_param_id"], cell["worsening_param_id"])
        triple = (cell["improving_param_id"], cell["worsening_param_id"], cell["principle_id"])
        pair_bucket = cell_papers.setdefault(pair, [])
        triple_bucket = triple_papers.setdefault(triple, [])
        for pid in cell.get("paper_ids", []):
            if pid not in pair_bucket:
                pair_bucket.append(pid)
            if pid not in triple_bucket:
                triple_bucket.append(pid)

    card_rows = store.query_sql("SELECT MAX(id) AS max_id FROM idea_cards")
    next_card_id = (int(card_rows[0]["max_id"]) + 1) if card_rows[0]["max_id"] is not None else 1

    now = datetime.now(UTC)
    all_cards: list[dict[str, Any]] = []
    kept_token_sets: list[set[str]] = []
    dedup_dropped = 0

    # Diversity gate: floor by score, spread across improving params, then cap.
    eligible = [g for g in report["gaps"] if g.get("score", 0.0) >= min_gap_score]
    ordered_gaps = _diversified_gap_order(eligible)
    call_budget = max(max_cards * 3, 60)
    calls_made = 0
    gaps_reached = 0

    for gap in ordered_gaps:
        if len(all_cards) >= max_cards or calls_made >= call_budget:
            break
        gaps_reached += 1
        gap_id = int(gap["id"])
        messages = _build_idea_card_messages(gap, vocab_by_id, patterns)
        calls_made += 1
        try:
            text = await llm_client.complete(messages)
        except Exception:
            logger.warning("LLM idea-card generation failed for gap %d", gap_id)
            continue

        card = _parse_idea_card(text, valid_pattern_ids)

        # Graceful degradation: when we cannot build a properly pattern-grounded
        # card, keep the LLM's output as a free-text hypothesis (pre-0.11.0
        # behaviour) instead of dropping it entirely.
        if card is None:
            logger.warning("Unusable idea-card JSON for gap %d; keeping raw hypothesis", gap_id)
            fallback = strip_code_fences(text).strip()
            if fallback:
                _set_gap_hypothesis(store, gap, fallback)
            continue
        if not card["pattern_ids"]:
            logger.warning(
                "Idea card for gap %d referenced no known pattern; keeping hypothesis only", gap_id
            )
            if card["mechanism"]:
                _set_gap_hypothesis(store, gap, card["mechanism"])
            continue

        # Diversity gate: drop cards that restate an already-kept idea.
        token_set = _card_token_set(card["title"], card["signature_terms"])
        if any(_jaccard(token_set, kept) >= dedup_threshold for kept in kept_token_sets):
            dedup_dropped += 1
            logger.debug("Idea card for gap %d is a near-duplicate; skipping", gap_id)
            # Keep the mechanism as the gap's hypothesis so the gap isn't left bare.
            _set_gap_hypothesis(store, gap, card["mechanism"])
            continue

        card_row = {
            "id": next_card_id,
            "gap_id": gap_id,
            "report_id": int(gap["report_id"]),
            "title": card["title"],
            "pattern_ids": card["pattern_ids"],
            "hook": card["hook"],
            "mechanism": card["mechanism"],
            "falsification": card["falsification"],
            "differentiation": card["differentiation"],
            "signature_terms": card["signature_terms"],
            "paper_ids": _card_paper_ids(gap, cell_papers, triple_papers),
            "confidence": card["confidence"],
            "created_at": now,
            "taxonomy_version": gap.get("taxonomy_version", 0),
        }
        # Guard the writes so one gap's DB error is logged and skipped rather
        # than aborting the whole run and discarding earlier cards.
        try:
            store.add_rows("idea_cards", [card_row])
        except Exception:
            logger.warning("Failed to persist idea card for gap %d", gap_id)
            continue
        all_cards.append(card_row)
        kept_token_sets.append(token_set)
        next_card_id += 1

        # Back-compat: keep the free-text hypothesis column populated.
        _set_gap_hypothesis(store, gap, card["mechanism"])

    report["idea_cards"] = all_cards
    unprocessed = len(ordered_gaps) - gaps_reached
    logger.info(
        "Ideation LLM: %d distinct cards from %d gaps (%d near-dups dropped, "
        "%d LLM calls, %d gaps left unprocessed by cap/budget)",
        len(all_cards),
        gaps_reached,
        dedup_dropped,
        calls_made,
        unprocessed,
    )
    return report

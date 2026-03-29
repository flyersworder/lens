"""Contradiction matrix — aggregates tradeoff extractions via vocabulary."""

from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from typing import Any

from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def _build_vocab_name_map(store: LensStore) -> dict[str, str]:
    """Build a map from vocabulary display name to ID."""
    rows = store.query("vocabulary")
    return {r["name"]: r["id"] for r in rows}


def build_matrix(store: LensStore) -> None:
    """Build the contradiction matrix from extractions + vocabulary.

    Full rebuild — deletes all existing cells first.
    Filters extractions to confidence >= 0.5.
    """
    with contextlib.suppress(OSError, ValueError):
        store.delete("matrix_cells", "1 = 1", ())

    vocab_map = _build_vocab_name_map(store)
    if not vocab_map:
        logger.info("No vocabulary entries — skipping matrix build")
        return

    extractions = store.query("tradeoff_extractions")
    if not extractions:
        logger.info("No extractions — skipping matrix build")
        return

    extractions = [e for e in extractions if e.get("confidence", 0) >= 0.5]

    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in extractions:
        # Strip NEW: prefix if present (already accepted into vocabulary)
        improves = row["improves"]
        if improves.startswith("NEW: "):
            improves = improves[5:].strip()
        worsens = row["worsens"]
        if worsens.startswith("NEW: "):
            worsens = worsens[5:].strip()
        technique = row["technique"]
        if technique.startswith("NEW: "):
            technique = technique[5:].strip()

        imp_id = vocab_map.get(improves)
        wors_id = vocab_map.get(worsens)
        tech_id = vocab_map.get(technique)

        if imp_id is None or wors_id is None or tech_id is None:
            continue

        key = (imp_id, wors_id, tech_id)
        cells.setdefault(key, []).append(row)

    cell_rows = []
    for (imp_id, wors_id, princ_id), matches in cells.items():
        count = len(matches)
        avg_conf = sum(m["confidence"] for m in matches) / count
        paper_ids = list({m["paper_id"] for m in matches})
        cell_rows.append(
            {
                "improving_param_id": imp_id,
                "worsening_param_id": wors_id,
                "principle_id": princ_id,
                "count": count,
                "avg_confidence": round(avg_conf, 4),
                "paper_ids": paper_ids,
                "taxonomy_version": 0,
            }
        )

    if cell_rows:
        store.add_rows("matrix_cells", cell_rows)
        logger.info("Built matrix with %d cells", len(cell_rows))


def get_ranked_matrix(
    store: LensStore,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Get the contradiction matrix with top-k principles per cell pair."""
    cells = store.query("matrix_cells")
    if not cells:
        return []

    for c in cells:
        c["score"] = c["count"] * c["avg_confidence"]

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cells:
        key = (c["improving_param_id"], c["worsening_param_id"])
        groups[key].append(c)

    result: list[dict[str, Any]] = []
    for _key, group in sorted(groups.items()):
        group.sort(key=lambda x: x["score"], reverse=True)
        result.extend(group[:top_k])

    result.sort(key=lambda x: (x["improving_param_id"], x["worsening_param_id"]))
    return result

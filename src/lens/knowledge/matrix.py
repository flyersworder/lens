"""Contradiction matrix construction.

Maps raw extraction strings through the taxonomy to build matrix cells.
Each cell: (improving_param_id, worsening_param_id) -> ranked principles.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def _build_string_to_id_map(store: LensStore, table_name: str, version: int) -> dict[str, int]:
    """Build a map from raw_strings to taxonomy entry IDs."""
    rows = store.query(table_name, "taxonomy_version = ?", (version,))
    if not rows:
        return {}
    result: dict[str, int] = {}
    for row in rows:
        entry_id = row["id"]
        for s in row.get("raw_strings", []):
            result[s] = entry_id
    return result


def get_ranked_matrix(
    store: LensStore,
    taxonomy_version: int,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Get the contradiction matrix with top-k principles per cell pair.

    Returns a list of dicts ranked by score (count * avg_confidence)
    within each (improving, worsening) pair, limited to top_k.
    """
    cells = store.query("matrix_cells", "taxonomy_version = ?", (taxonomy_version,))
    if not cells:
        return []

    # Add score
    for c in cells:
        c["score"] = c["count"] * c["avg_confidence"]

    # Group by (improving, worsening) pair, rank within each group
    from collections import defaultdict

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for c in cells:
        key = (c["improving_param_id"], c["worsening_param_id"])
        groups[key].append(c)

    result: list[dict[str, Any]] = []
    for _key, group in sorted(groups.items()):
        group.sort(key=lambda x: x["score"], reverse=True)
        result.extend(group[:top_k])

    result.sort(key=lambda x: (x["improving_param_id"], x["worsening_param_id"]))
    return result


def build_matrix(
    store: LensStore,
    taxonomy_version: int,
) -> None:
    """Build the contradiction matrix from extractions + taxonomy.

    Full rebuild — deletes existing cells for this version first.
    Filters extractions to confidence >= 0.5 (spec requirement).
    """
    taxonomy_version = int(taxonomy_version)  # defense-in-depth for SQL filters
    # Delete old cells for idempotent rebuild
    with contextlib.suppress(OSError, ValueError):
        store.delete("matrix_cells", "taxonomy_version = ?", (taxonomy_version,))

    param_map = _build_string_to_id_map(store, "parameters", taxonomy_version)
    principle_map = _build_string_to_id_map(store, "principles", taxonomy_version)

    if not param_map or not principle_map:
        logger.info("No taxonomy entries — skipping matrix build")
        return

    extractions = store.query("tradeoff_extractions")
    if not extractions:
        logger.info("No extractions — skipping matrix build")
        return

    # Filter to confidence >= 0.5
    extractions = [e for e in extractions if e.get("confidence", 0) >= 0.5]

    # Map raw strings to taxonomy IDs and aggregate
    cells: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in extractions:
        imp_id = param_map.get(row["improves"])
        wors_id = param_map.get(row["worsens"])
        tech_id = principle_map.get(row["technique"])

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
                "avg_confidence": avg_conf,
                "paper_ids": paper_ids,
                "taxonomy_version": taxonomy_version,
            }
        )

    if cell_rows:
        store.add_rows("matrix_cells", cell_rows)
        logger.info("Built matrix with %d cells", len(cell_rows))

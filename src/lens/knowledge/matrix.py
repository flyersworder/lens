"""Contradiction matrix construction.

Maps raw extraction strings through the taxonomy to build matrix cells.
Each cell: (improving_param_id, worsening_param_id) -> ranked principles.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import polars as pl

from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def _build_string_to_id_map(store: LensStore, table_name: str, version: int) -> dict[str, int]:
    """Build a map from raw_strings to taxonomy entry IDs."""
    df = store.get_table(table_name).to_polars()
    if len(df) == 0:
        return {}
    df = df.filter(pl.col("taxonomy_version") == version)
    result: dict[str, int] = {}
    for row in df.to_dicts():
        entry_id = row["id"]
        for s in row.get("raw_strings", []):
            result[s] = entry_id
    return result


def get_ranked_matrix(
    store: LensStore,
    taxonomy_version: int,
    top_k: int = 4,
) -> pl.DataFrame:
    """Get the contradiction matrix with top-k principles per cell pair.

    Returns a DataFrame ranked by score (count * avg_confidence)
    within each (improving, worsening) pair, limited to top_k.
    """
    cells = store.get_table("matrix_cells").to_polars()
    if len(cells) == 0:
        return cells
    cells = cells.filter(pl.col("taxonomy_version") == taxonomy_version)
    if len(cells) == 0:
        return cells

    return (
        cells.with_columns((pl.col("count") * pl.col("avg_confidence")).alias("score"))
        .sort("score", descending=True)
        .group_by(["improving_param_id", "worsening_param_id"])
        .head(top_k)
    )


def build_matrix(
    store: LensStore,
    taxonomy_version: int,
) -> None:
    """Build the contradiction matrix from extractions + taxonomy.

    Full rebuild — deletes existing cells for this version first.
    Filters extractions to confidence >= 0.5 (spec requirement).
    """
    # Delete old cells for idempotent rebuild
    with contextlib.suppress(OSError, ValueError):
        store.get_table("matrix_cells").delete(f"taxonomy_version = {taxonomy_version}")

    param_map = _build_string_to_id_map(store, "parameters", taxonomy_version)
    principle_map = _build_string_to_id_map(store, "principles", taxonomy_version)

    if not param_map or not principle_map:
        logger.info("No taxonomy entries — skipping matrix build")
        return

    extractions = store.get_table("tradeoff_extractions").to_polars()
    if len(extractions) == 0:
        logger.info("No extractions — skipping matrix build")
        return

    # Filter to confidence >= 0.5
    extractions = extractions.filter(pl.col("confidence") >= 0.5)

    # Map raw strings to taxonomy IDs and aggregate
    cells: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in extractions.to_dicts():
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

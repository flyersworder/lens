"""LENS knowledge base linter — health checks with optional auto-fix."""

from __future__ import annotations

import logging

from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def check_orphan_vocabulary(store: LensStore) -> list[dict]:
    """Find extracted vocabulary entries with zero paper references."""
    return store.query_sql(
        "SELECT id, name, kind, description, source, paper_count "
        "FROM vocabulary WHERE paper_count = 0 AND source != 'seed'"
    )


def fix_orphans(store: LensStore) -> list[str]:
    """Delete orphan vocabulary entries. Returns list of deleted IDs."""
    orphans = check_orphan_vocabulary(store)
    deleted_ids = []
    for orphan in orphans:
        store.delete("vocabulary", "id = ?", (orphan["id"],))
        deleted_ids.append(orphan["id"])
    return deleted_ids


def check_contradictions(store: LensStore, min_count: int = 2) -> list[dict]:
    """Find parameter pairs with opposing directionality in the matrix.

    A contradiction exists when both (A improves, B worsens, principle P)
    and (B improves, A worsens, principle P) exist, each with count >= min_count.
    """
    cells = store.query("matrix_cells")
    if not cells:
        return []

    by_principle: dict[str, list[dict]] = {}
    for cell in cells:
        by_principle.setdefault(cell["principle_id"], []).append(cell)

    contradictions = []
    seen: set[tuple[str, str, str]] = set()

    for principle_id, group in by_principle.items():
        for cell in group:
            if cell["count"] < min_count:
                continue
            imp = cell["improving_param_id"]
            wors = cell["worsening_param_id"]

            for other in group:
                if other["count"] < min_count:
                    continue
                if other["improving_param_id"] == wors and other["worsening_param_id"] == imp:
                    key = (min(imp, wors), max(imp, wors), principle_id)
                    if key not in seen:
                        seen.add(key)
                        contradictions.append(
                            {
                                "params": [imp, wors],
                                "principle_id": principle_id,
                                "forward_count": cell["count"],
                                "reverse_count": other["count"],
                            }
                        )
    return contradictions

"""Browse knowledge structures — parameters, principles, matrix, papers."""

from __future__ import annotations

from typing import Any

from lens.store.store import LensStore


def list_parameters(store: LensStore, taxonomy_version: int) -> list[dict[str, Any]]:
    """List all parameters for a taxonomy version."""
    rows = store.query("parameters", "taxonomy_version = ?", (taxonomy_version,))
    # Remove embedding field from results
    for r in rows:
        r.pop("embedding", None)
    return rows


def list_principles(store: LensStore, taxonomy_version: int) -> list[dict[str, Any]]:
    """List all principles for a taxonomy version."""
    rows = store.query("principles", "taxonomy_version = ?", (taxonomy_version,))
    for r in rows:
        r.pop("embedding", None)
    return rows


def get_matrix_cell(
    store: LensStore,
    improving_param_id: int,
    worsening_param_id: int,
    taxonomy_version: int,
) -> list[dict[str, Any]]:
    """Get matrix cells for a specific parameter pair, sorted by score."""
    cells = store.query(
        "matrix_cells",
        "taxonomy_version = ? AND improving_param_id = ? AND worsening_param_id = ?",
        (taxonomy_version, improving_param_id, worsening_param_id),
    )
    if not cells:
        return []
    for c in cells:
        c["score"] = c["count"] * c["avg_confidence"]
    cells.sort(key=lambda x: x["score"], reverse=True)
    return cells


def list_matrix_overview(store: LensStore, taxonomy_version: int) -> list[dict[str, Any]]:
    """Get overview of all populated matrix cells."""
    rows = store.query_sql(
        "SELECT improving_param_id, worsening_param_id, "
        "COUNT(principle_id) AS num_principles, "
        "SUM(count) AS total_evidence "
        "FROM matrix_cells WHERE taxonomy_version = ? "
        "GROUP BY improving_param_id, worsening_param_id "
        "ORDER BY total_evidence DESC",
        (taxonomy_version,),
    )
    return rows


def list_architecture_slots(store: LensStore, taxonomy_version: int) -> list[dict[str, Any]]:
    """List all architecture slots for a version, enriched with variant_count, sorted by name."""
    rows = store.query_sql(
        "SELECT s.*, COALESCE(v.variant_count, 0) AS variant_count "
        "FROM architecture_slots s "
        "LEFT JOIN ("
        "  SELECT slot_id, COUNT(*) AS variant_count "
        "  FROM architecture_variants "
        "  WHERE taxonomy_version = ? "
        "  GROUP BY slot_id"
        ") v ON s.id = v.slot_id "
        "WHERE s.taxonomy_version = ? "
        "ORDER BY s.name",
        (taxonomy_version, taxonomy_version),
    )
    return rows


def list_architecture_variants(
    store: LensStore, slot_name: str, taxonomy_version: int
) -> list[dict[str, Any]]:
    """Find the slot by name, then list all variants with that slot_id."""
    slots = store.query(
        "architecture_slots",
        "name = ? AND taxonomy_version = ?",
        (slot_name, taxonomy_version),
    )
    if not slots:
        return []
    slot_id = slots[0]["id"]

    variants = store.query(
        "architecture_variants",
        "slot_id = ? AND taxonomy_version = ?",
        (slot_id, taxonomy_version),
    )
    for v in variants:
        v.pop("embedding", None)
    return variants


def list_agentic_patterns(
    store: LensStore, taxonomy_version: int, category: str | None = None
) -> list[dict[str, Any]]:
    """List all agentic patterns for a version, optionally filtered by category."""
    if category is not None:
        rows = store.query(
            "agentic_patterns",
            "taxonomy_version = ? AND category = ?",
            (taxonomy_version, category),
        )
    else:
        rows = store.query("agentic_patterns", "taxonomy_version = ?", (taxonomy_version,))
    for r in rows:
        r.pop("embedding", None)
    return rows


def get_architecture_timeline(
    store: LensStore, slot_name: str, taxonomy_version: int
) -> list[dict[str, Any]]:
    """List variants for a slot ordered by earliest paper date ascending."""
    slots = store.query(
        "architecture_slots",
        "name = ? AND taxonomy_version = ?",
        (slot_name, taxonomy_version),
    )
    if not slots:
        return []
    slot_id = slots[0]["id"]

    variants = store.query(
        "architecture_variants",
        "slot_id = ? AND taxonomy_version = ?",
        (slot_id, taxonomy_version),
    )
    if not variants:
        return []

    # Build a paper_id -> date map
    papers = store.query("papers")
    paper_date_map = {p["paper_id"]: p["date"] for p in papers}

    # Find earliest paper date per variant
    for v in variants:
        v.pop("embedding", None)
        paper_ids = v.get("paper_ids", [])
        dates = [paper_date_map[pid] for pid in paper_ids if pid in paper_date_map]
        v["earliest_date"] = min(dates) if dates else None

    variants.sort(key=lambda x: x.get("earliest_date") or "9999-99-99")
    return variants


def get_paper(store: LensStore, paper_id: str) -> dict[str, Any] | None:
    """Get a specific paper by ID."""
    matches = store.query("papers", "paper_id = ?", (paper_id,))
    if not matches:
        return None
    result = matches[0]
    result.pop("embedding", None)
    return result

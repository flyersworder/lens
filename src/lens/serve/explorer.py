"""Browse knowledge structures — parameters, principles, matrix, papers."""

from __future__ import annotations

import logging
from typing import Any

from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def _matches_canonical(value: str, canonical_name: str) -> bool:
    """Check if a value matches a canonical name, with or without NEW: prefix."""
    return value == canonical_name or value == f"NEW: {canonical_name}"


def list_parameters(store: LensStore) -> list[dict[str, Any]]:
    """List all vocabulary entries of kind 'parameter'."""
    rows = store.query("vocabulary", "kind = ?", ("parameter",))
    for r in rows:
        r.pop("embedding", None)
    return rows


def list_principles(store: LensStore) -> list[dict[str, Any]]:
    """List all vocabulary entries of kind 'principle'."""
    rows = store.query("vocabulary", "kind = ?", ("principle",))
    for r in rows:
        r.pop("embedding", None)
    return rows


def get_matrix_cell(
    store: LensStore,
    improving_param_id: str,
    worsening_param_id: str,
) -> list[dict[str, Any]]:
    """Get matrix cells for a specific parameter pair, sorted by score."""
    cells = store.query(
        "matrix_cells",
        "improving_param_id = ? AND worsening_param_id = ?",
        (improving_param_id, worsening_param_id),
    )
    if not cells:
        return []
    for c in cells:
        c["score"] = c["count"] * c["avg_confidence"]
    cells.sort(key=lambda x: x["score"], reverse=True)
    return cells


def list_matrix_overview(store: LensStore) -> list[dict[str, Any]]:
    """Get overview of all populated matrix cells."""
    rows = store.query_sql(
        "SELECT improving_param_id, worsening_param_id, "
        "COUNT(principle_id) AS num_principles, "
        "SUM(count) AS total_evidence "
        "FROM matrix_cells "
        "GROUP BY improving_param_id, worsening_param_id "
        "ORDER BY total_evidence DESC",
        (),
    )
    return rows


def list_architecture_slots(store: LensStore) -> list[dict[str, Any]]:
    """List all architecture slots from vocabulary."""
    rows = store.query("vocabulary", "kind = ?", ("arch_slot",))
    extractions = store.query("architecture_extractions")
    for r in rows:
        r.pop("embedding", None)
        r["variant_count"] = len(
            set(
                e["variant_name"]
                for e in extractions
                if _matches_canonical(e["component_slot"], r["name"])
            )
        )
    return rows


def list_architecture_variants(store: LensStore, slot_name: str) -> list[dict[str, Any]]:
    """List architecture variants for a given slot name."""
    extractions = store.query("architecture_extractions")
    matching = [e for e in extractions if _matches_canonical(e["component_slot"], slot_name)]
    by_name: dict[str, dict[str, Any]] = {}
    for v in matching:
        name = v["variant_name"]
        if name not in by_name:
            by_name[name] = {
                "variant_name": name,
                "slot": slot_name,
                "replaces": v.get("replaces"),
                "key_properties": v.get("key_properties", ""),
                "paper_ids": [],
                "confidence": v["confidence"],
            }
        by_name[name]["paper_ids"].append(v["paper_id"])
    return list(by_name.values())


def list_agentic_patterns(store: LensStore, category: str | None = None) -> list[dict[str, Any]]:
    """List agentic patterns from extractions, optionally filtered by category."""
    extractions = store.query("agentic_extractions")
    if category:
        extractions = [
            e for e in extractions if _matches_canonical(e.get("category", ""), category)
        ]
    by_name: dict[str, dict[str, Any]] = {}
    for e in extractions:
        name = e["pattern_name"]
        if name not in by_name:
            by_name[name] = {
                "pattern_name": name,
                "category": e.get("category", ""),
                "structure": e.get("structure", ""),
                "use_case": e.get("use_case", ""),
                "components": e.get("components", []),
                "paper_ids": [],
            }
        by_name[name]["paper_ids"].append(e["paper_id"])
    return list(by_name.values())


def get_architecture_timeline(store: LensStore, slot_name: str) -> list[dict[str, Any]]:
    """List variants for a slot ordered by earliest paper date."""
    variants = list_architecture_variants(store, slot_name)
    if not variants:
        return []
    papers = store.query("papers")
    paper_date_map = {p["paper_id"]: p["date"] for p in papers}
    for v in variants:
        dates = [paper_date_map[pid] for pid in v.get("paper_ids", []) if pid in paper_date_map]
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


def search_papers(
    store: LensStore,
    query: str | None = None,
    *,
    author: str | None = None,
    venue: str | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = 10,
    embedding_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search papers via hybrid search and/or metadata filters.

    When *query* is provided, embeds it and runs hybrid FTS5 + vector search.
    Falls back to FTS5-only if embedding fails.
    When only filters are provided, runs a direct SQL query.
    """
    filters: dict[str, str] = {}
    if author:
        filters["author"] = author
    if venue:
        filters["venue"] = venue
    if after:
        filters["after"] = after
    if before:
        filters["before"] = before

    embedding = None
    if query:
        try:
            from lens.taxonomy.embedder import embed_strings

            emb_kw = embedding_kwargs or {}
            embedding = embed_strings([query], **emb_kw)[0].tolist()
        except Exception:
            logger.warning("Embedding failed, falling back to keyword-only search")

    raw_results = store.search_papers(
        query=query,
        embedding=embedding,
        filters=filters if filters else None,
        limit=limit,
    )

    formatted = []
    for r in raw_results:
        abstract = r.get("abstract", "")
        snippet = (abstract[:150] + "...") if len(abstract) > 150 else abstract

        authors = r.get("authors", [])
        if len(authors) > 3:
            authors_display = ", ".join(authors[:3]) + ", ..."
        else:
            authors_display = ", ".join(authors)

        entry: dict[str, Any] = {
            "paper_id": r["paper_id"],
            "title": r["title"],
            "date": r.get("date", ""),
            "authors_display": authors_display,
            "abstract_snippet": snippet,
            "arxiv_id": r.get("arxiv_id", ""),
            "venue": r.get("venue"),
        }
        if "_rrf_score" in r:
            entry["score"] = r["_rrf_score"]

        formatted.append(entry)

    return formatted

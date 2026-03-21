"""Browse knowledge structures — parameters, principles, matrix, papers."""

from __future__ import annotations

from typing import Any

import polars as pl

from lens.store.store import LensStore


def list_parameters(store: LensStore, taxonomy_version: int) -> list[dict[str, Any]]:
    """List all parameters for a taxonomy version."""
    df = store.get_table("parameters").to_polars()
    df = df.filter(pl.col("taxonomy_version") == taxonomy_version)
    return df.drop("embedding").to_dicts()


def list_principles(store: LensStore, taxonomy_version: int) -> list[dict[str, Any]]:
    """List all principles for a taxonomy version."""
    df = store.get_table("principles").to_polars()
    df = df.filter(pl.col("taxonomy_version") == taxonomy_version)
    return df.drop("embedding").to_dicts()


def get_matrix_cell(
    store: LensStore,
    improving_param_id: int,
    worsening_param_id: int,
    taxonomy_version: int,
) -> list[dict[str, Any]]:
    """Get matrix cells for a specific parameter pair, sorted by score."""
    df = store.get_table("matrix_cells").to_polars()
    df = df.filter(
        (pl.col("taxonomy_version") == taxonomy_version)
        & (pl.col("improving_param_id") == improving_param_id)
        & (pl.col("worsening_param_id") == worsening_param_id)
    )
    if len(df) == 0:
        return []
    df = df.with_columns((pl.col("count") * pl.col("avg_confidence")).alias("score")).sort(
        "score", descending=True
    )
    return df.to_dicts()


def list_matrix_overview(store: LensStore, taxonomy_version: int) -> list[dict[str, Any]]:
    """Get overview of all populated matrix cells."""
    df = store.get_table("matrix_cells").to_polars()
    df = df.filter(pl.col("taxonomy_version") == taxonomy_version)
    if len(df) == 0:
        return []
    return (
        df.group_by(["improving_param_id", "worsening_param_id"])
        .agg(
            pl.col("principle_id").count().alias("num_principles"),
            pl.col("count").sum().alias("total_evidence"),
        )
        .sort("total_evidence", descending=True)
        .to_dicts()
    )


def list_architecture_slots(store: LensStore, taxonomy_version: int) -> list[dict[str, Any]]:
    """List all architecture slots for a version, enriched with variant_count, sorted by name."""
    slots_df = store.get_table("architecture_slots").to_polars()
    slots_df = slots_df.filter(pl.col("taxonomy_version") == taxonomy_version)

    variants_df = store.get_table("architecture_variants").to_polars()
    variants_df = variants_df.filter(pl.col("taxonomy_version") == taxonomy_version)
    counts_df = variants_df.group_by("slot_id").agg(pl.len().alias("variant_count"))

    slots_df = slots_df.join(counts_df, left_on="id", right_on="slot_id", how="left")
    slots_df = slots_df.with_columns(pl.col("variant_count").fill_null(0))
    return slots_df.sort("name").to_dicts()


def list_architecture_variants(
    store: LensStore, slot_name: str, taxonomy_version: int
) -> list[dict[str, Any]]:
    """Find the slot by name, then list all variants with that slot_id. Drop embedding column."""
    slots_df = store.get_table("architecture_slots").to_polars()
    slot_row = slots_df.filter(
        (pl.col("name") == slot_name) & (pl.col("taxonomy_version") == taxonomy_version)
    )
    if len(slot_row) == 0:
        return []
    slot_id = slot_row["id"][0]

    variants_df = store.get_table("architecture_variants").to_polars()
    variants_df = variants_df.filter(
        (pl.col("slot_id") == slot_id) & (pl.col("taxonomy_version") == taxonomy_version)
    )
    return variants_df.drop("embedding").to_dicts()


def list_agentic_patterns(
    store: LensStore, taxonomy_version: int, category: str | None = None
) -> list[dict[str, Any]]:
    """List all agentic patterns for a version, optionally filtered by category."""
    df = store.get_table("agentic_patterns").to_polars()
    df = df.filter(pl.col("taxonomy_version") == taxonomy_version)
    if category is not None:
        df = df.filter(pl.col("category") == category)
    return df.drop("embedding").to_dicts()


def get_architecture_timeline(
    store: LensStore, slot_name: str, taxonomy_version: int
) -> list[dict[str, Any]]:
    """List variants for a slot ordered by earliest paper date ascending."""
    slots_df = store.get_table("architecture_slots").to_polars()
    slot_row = slots_df.filter(
        (pl.col("name") == slot_name) & (pl.col("taxonomy_version") == taxonomy_version)
    )
    if len(slot_row) == 0:
        return []
    slot_id = slot_row["id"][0]

    variants_df = store.get_table("architecture_variants").to_polars()
    variants_df = variants_df.filter(
        (pl.col("slot_id") == slot_id) & (pl.col("taxonomy_version") == taxonomy_version)
    )

    papers_df = store.get_table("papers").to_polars().select(["paper_id", "date"])

    # Explode paper_ids, join to get dates, find min date per variant
    exploded = variants_df.select(["id", "paper_ids"]).explode("paper_ids")
    joined = exploded.join(papers_df, left_on="paper_ids", right_on="paper_id", how="left")
    min_dates = joined.group_by("id").agg(pl.col("date").min().alias("earliest_date"))

    variants_df = variants_df.drop("embedding")
    variants_df = variants_df.join(min_dates, on="id", how="left")
    variants_df = variants_df.sort("earliest_date")
    return variants_df.to_dicts()


def get_paper(store: LensStore, paper_id: str) -> dict[str, Any] | None:
    """Get a specific paper by ID."""
    df = store.get_table("papers").to_polars()
    matches = df.filter(pl.col("paper_id") == paper_id)
    if len(matches) == 0:
        return None
    return matches.drop("embedding").to_dicts()[0]

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


def get_paper(store: LensStore, paper_id: str) -> dict[str, Any] | None:
    """Get a specific paper by ID."""
    df = store.get_table("papers").to_polars()
    matches = df.filter(pl.col("paper_id") == paper_id)
    if len(matches) == 0:
        return None
    return matches.drop("embedding").to_dicts()[0]

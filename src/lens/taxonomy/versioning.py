"""Taxonomy version management."""

from __future__ import annotations

from datetime import UTC, datetime

from lens.store.store import LensStore


def get_latest_version(store: LensStore) -> int | None:
    """Get the latest taxonomy version number, or None if no versions exist."""
    df = store.get_table("taxonomy_versions").to_polars()
    if len(df) == 0:
        return None
    return int(df["version_id"].max())  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]


def get_next_version(store: LensStore) -> int:
    """Get the next taxonomy version number."""
    latest = get_latest_version(store)
    return (latest or 0) + 1


def record_version(
    store: LensStore,
    version_id: int,
    paper_count: int,
    param_count: int,
    principle_count: int,
    slot_count: int = 0,
    variant_count: int = 0,
    pattern_count: int = 0,
) -> None:
    """Record a taxonomy version in the store."""
    store.add_rows(
        "taxonomy_versions",
        [
            {
                "version_id": version_id,
                "created_at": datetime.now(UTC),
                "paper_count": paper_count,
                "param_count": param_count,
                "principle_count": principle_count,
                "slot_count": slot_count,
                "variant_count": variant_count,
                "pattern_count": pattern_count,
            }
        ],
    )

"""One-off recovery: rebuild a local sqlite-vec ``lens.db`` from a Turso branch.

Context (2026-07-19): a manual ``publish_to_turso`` + corpus-snapshot bootstrap
overwrote the weekly-cron-accumulated corpus (477 papers / 299 matrix cells /
1,152 extractions) with a stale 77-paper local DB, and clobbered the
corpus-snapshot release. The prod data was recovered via Turso PITR into the
``lens-prod-restore`` branch. This script pulls that branch back down into a
local sqlite-vec DB so it can be re-embedded, re-ideated, and republished to
both lens-prod and the corpus-snapshot release.

Not part of the product surface — kept in ``scripts/`` for provenance. Reads the
source branch URL + token from ``RESTORE_SRC_URL`` / ``RESTORE_SRC_TOKEN`` env.

Usage:
    RESTORE_SRC_URL=... RESTORE_SRC_TOKEN=... \
        uv run python scripts/restore_from_turso.py --target /path/to/new.db
"""

from __future__ import annotations

import argparse
import os

import libsql_client

from lens.store.store import LensStore

# Copied verbatim from the recovered branch. Order matters only in that parents
# precede children for FK-free sqlite it doesn't, but we keep the logical order.
# idea_cards is intentionally excluded — they are regenerated against the
# recovered 299-cell matrix in a later step.
TABLES_TO_COPY: list[str] = [
    "papers",
    "vocabulary",
    "tradeoff_extractions",
    "architecture_extractions",
    "agentic_extractions",
    "matrix_cells",
    "taxonomy_versions",
    "ideation_reports",
    "ideation_gaps",
    "event_log",
]


def _normalize_url(url: str) -> str:
    """libSQL client wants an https:// URL, not libsql://."""
    if url.startswith("libsql://"):
        return "https://" + url[len("libsql://") :]
    return url


def _local_columns(store: LensStore, table: str) -> set[str]:
    """Column names of the freshly-created local *table*."""
    rows = store.query_sql(f"PRAGMA table_info({table})")
    return {r["name"] for r in rows}


def copy_table(client: libsql_client.ClientSync, store: LensStore, table: str) -> int:
    """Pull every row of *table* from the remote branch into the local store.

    Target-schema-aware: only columns present in the freshly-created local table
    are copied. This drops the remote-only ``embedding`` (F32_BLOB) column
    (embeddings live in a separate sqlite-vec companion table locally and are
    regenerated afterward) and any legacy columns the current schema has since
    retired (e.g. ``new_concept_description``, superseded by the ``new_concepts``
    JSON field). JSON columns arrive as JSON strings and pass through
    ``add_rows`` unchanged (the local column stores the same string).
    """
    keep = _local_columns(store, table)
    rs = client.execute(f"SELECT * FROM {table}")
    cols = list(rs.columns)
    rows: list[dict] = []
    for r in rs.rows:
        d = {cols[i]: r[i] for i in range(len(cols)) if cols[i] in keep}
        rows.append(d)
    if not rows:
        return 0
    return store.add_rows(table, rows, ignore_conflicts=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Path for the new local lens.db")
    args = ap.parse_args()

    src_url = os.environ["RESTORE_SRC_URL"]
    src_token = os.environ["RESTORE_SRC_TOKEN"]

    client = libsql_client.create_client_sync(url=_normalize_url(src_url), auth_token=src_token)
    store = LensStore(args.target)
    store.init_tables()

    print(f"Rebuilding {args.target} from {src_url}")
    total = 0
    for table in TABLES_TO_COPY:
        n = copy_table(client, store, table)
        total += n
        print(f"  {table}: {n} rows copied")
    print(f"Done: {total} rows across {len(TABLES_TO_COPY)} tables")

    client.close()


if __name__ == "__main__":
    main()

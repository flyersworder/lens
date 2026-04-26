"""Publish a local sqlite-vec LENS database to a remote Turso libSQL DB.

Translates the local schema (sqlite-vec ``vec0`` virtual tables companion
to ``papers`` / ``vocabulary``) into the libSQL-native shape that
:class:`lens.store.turso_store.TursoStore` expects:

* ``papers`` and ``vocabulary`` gain an ``embedding F32_BLOB(EMBEDDING_DIM)``
  column on the main table, populated from the local ``_vec`` companion.
* ``papers_emb_idx`` and ``vocabulary_emb_idx`` libSQL vector indexes
  are created via ``libsql_vector_idx``.
* FTS5 virtual tables (``papers_fts``, ``vocabulary_fts``) are rebuilt
  on the remote side.
* All other tables are copied row-for-row.

Usage::

    # Set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN in environment
    # (or .env at repo root — auto-loaded via python-dotenv)
    uv run python scripts/publish_to_turso.py

    # Custom local DB and target
    uv run python scripts/publish_to_turso.py \\
        --local ~/.lens/data/lens.db \\
        --target dev   # or 'prod' (uses TURSO_DEV_* / TURSO_PROD_* env vars)

The publish is **destructive**: it drops the entire remote schema and
rebuilds it. Run it inside a try/finally in the build pipeline so a
mid-run failure leaves the remote in an unambiguously broken state
that the *next* publish run will overwrite cleanly.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import libsql_client

from lens.store.models import EMBEDDING_DIM

logger = logging.getLogger("publish_to_turso")

# Tables to copy verbatim, in dependency order (parents before children
# even though we don't enforce FK constraints on either side).
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

# Tables with vector embeddings; key column is what joins the main row
# to its companion ``_vec`` row in the local sqlite-vec schema.
EMBEDDING_TABLES: dict[str, str] = {
    "papers": "paper_id",
    "vocabulary": "id",
}

# Order matters: drop indexes/FTS first, then base tables, to avoid
# orphaned vector-index state in libSQL. `DROP TABLE` on an FTS5 virtual
# table cascades to its shadow tables (papers_fts_data, _idx, _config,
# _docsize, _content) — verified against live Turso 2026-04 — so we
# don't need to enumerate them.
DROP_STATEMENTS: list[str] = [
    "DROP INDEX IF EXISTS papers_emb_idx",
    "DROP INDEX IF EXISTS vocabulary_emb_idx",
    "DROP TABLE IF EXISTS papers_fts",
    "DROP TABLE IF EXISTS vocabulary_fts",
    *[f"DROP TABLE IF EXISTS {t}" for t in TABLES_TO_COPY],
]

FTS_DDL: list[str] = [
    "CREATE VIRTUAL TABLE papers_fts USING fts5("
    "title, abstract, content=papers, content_rowid=rowid)",
    "CREATE VIRTUAL TABLE vocabulary_fts USING fts5("
    "name, description, kind, content=vocabulary, content_rowid=rowid)",
]

INDEX_DDL: list[str] = [
    "CREATE INDEX papers_emb_idx ON papers (libsql_vector_idx(embedding))",
    "CREATE INDEX vocabulary_emb_idx ON vocabulary (libsql_vector_idx(embedding))",
]


def _normalize_url(url: str) -> str:
    """Same conversion as TursoStore — libsql:// fails on the Python client."""
    if url.startswith("libsql://"):
        return "https://" + url[len("libsql://") :]
    return url


def _resolve_target(target: str) -> tuple[str, str]:
    """Return (url, auth_token) for ``target`` ('dev', 'prod', or 'plain').

    No fallback for `prod`: production publishes must use explicit
    `TURSO_PROD_*` variables. A silent fallback to `TURSO_DATABASE_URL`
    risks publishing dev creds to prod when a developer's shell is
    misconfigured.
    """
    if target == "dev":
        url = os.environ.get("TURSO_DEV_DATABASE_URL")
        token = os.environ.get("TURSO_DEV_AUTH_TOKEN")
    elif target == "prod":
        url = os.environ.get("TURSO_PROD_DATABASE_URL")
        token = os.environ.get("TURSO_PROD_AUTH_TOKEN")
    else:  # plain
        url = os.environ.get("TURSO_DATABASE_URL")
        token = os.environ.get("TURSO_AUTH_TOKEN")

    if not url or not token:
        sys.exit(
            f"error: TURSO credentials for target '{target}' not found. "
            f"Set TURSO_{target.upper()}_DATABASE_URL and "
            f"TURSO_{target.upper()}_AUTH_TOKEN (or use --target plain "
            f"with TURSO_DATABASE_URL / TURSO_AUTH_TOKEN)."
        )
    return url, token


def open_local(db_path: str) -> sqlite3.Connection:
    """Open the local sqlite-vec DB read-only.

    We don't need the LensStore wrapper for a one-way data dump, just
    raw sqlite3 with sqlite-vec loaded so we can query the ``_vec``
    companion tables.
    """
    import sqlite_vec

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
    """Names of regular columns for a local SQLite table, in declaration order."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def _column_specs(conn: sqlite3.Connection, table: str) -> list[dict]:
    """Full PRAGMA table_info rows: each is (cid, name, type, notnull, dflt, pk)."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [
        {
            "name": row[1],
            "type": row[2] or "",
            "notnull": bool(row[3]),
            "dflt": row[4],
            "pk": int(row[5]),
        }
        for row in cur.fetchall()
    ]


def _build_remote_ddl(table: str, cols: list[dict], embedding_dim: int | None) -> str:
    """Build a libSQL ``CREATE TABLE`` statement that mirrors the local
    column list, optionally appending an ``embedding F32_BLOB(N)`` column.

    Mirrors the local schema exactly so the publish step is robust to
    column-migration drift between the running LENS version and whatever
    historical migrations the local DB has accumulated.
    """
    pieces: list[str] = []
    pk_cols = [c["name"] for c in cols if c["pk"] >= 1]

    for c in cols:
        spec = f"{c['name']} {c['type']}".strip()
        # SQLite quirk: only INTEGER PRIMARY KEY (single-column) is rowid-aliased
        # *and* implicitly NOT NULL. TEXT/REAL primary keys still need the
        # explicit NOT NULL constraint at the column level.
        is_inline_int_pk = c["pk"] == 1 and len(pk_cols) == 1 and c["type"].upper() == "INTEGER"
        if is_inline_int_pk:
            spec += " PRIMARY KEY"
        # NOT NULL: emit unless the column is an inline INTEGER PRIMARY KEY
        # (which already implies NOT NULL). Skipping the check for non-INTEGER
        # PKs would silently drop NOT NULL on `papers.paper_id` etc.
        if c["notnull"] and not is_inline_int_pk:
            spec += " NOT NULL"
        if c["dflt"] is not None:
            # PRAGMA returns DEFAULT pre-quoted for string literals (e.g. "'pending'")
            # and as numeric/identifier text otherwise. We pass it through verbatim,
            # which is correct for the LENS schema's defaults (numbers, simple
            # quoted strings, no expressions). If the schema ever adds expression
            # defaults like CURRENT_TIMESTAMP, this still works; arbitrary
            # complex defaults would need explicit re-quoting logic.
            spec += f" DEFAULT {c['dflt']}"
        pieces.append(spec)
    # Note: AUTOINCREMENT cannot be detected via PRAGMA table_info — it's only
    # visible in sqlite_schema.sql. The publish step therefore drops the
    # AUTOINCREMENT keyword for tables like `tradeoff_extractions.rowid`. This
    # is functionally fine for our read-only Turso use (rowid still increments
    # monotonically within a session); only matters if anything depends on
    # AUTOINCREMENT's "never reuse a deleted rowid" guarantee, which LENS doesn't.

    # Append the embedding column BEFORE any table-level PK constraint,
    # since SQLite requires column definitions to precede table constraints.
    if embedding_dim is not None:
        pieces.append(f"embedding F32_BLOB({embedding_dim})")

    # If the table has a non-INTEGER single-column PK or a composite PK,
    # add a separate PRIMARY KEY clause AFTER all columns.
    needs_pk_clause = pk_cols and not (
        len(pk_cols) == 1
        and any(
            c["pk"] == 1 and c["type"].upper() == "INTEGER" for c in cols if c["name"] in pk_cols
        )
    )
    if needs_pk_clause:
        pieces.append(f"PRIMARY KEY ({', '.join(pk_cols)})")

    return f"CREATE TABLE {table} (\n  " + ",\n  ".join(pieces) + "\n)"


def drop_remote_schema(client: libsql_client.ClientSync) -> None:
    """Drop everything owned by LENS on the remote DB."""
    for stmt in DROP_STATEMENTS:
        client.execute(stmt)
    logger.info("dropped existing remote schema")


def create_remote_schema(
    local: sqlite3.Connection,
    client: libsql_client.ClientSync,
    embedding_dim: int,
) -> None:
    """Create the libSQL-native schema by mirroring local table DDL.

    Tables are recreated by reading column specs from the local DB via
    ``PRAGMA table_info``; ``papers`` and ``vocabulary`` get an extra
    ``embedding F32_BLOB(N)`` column appended. This makes the publish
    step robust to local-side schema drift (e.g. historical migrations
    that newer LENS versions no longer emit by default).
    """
    for table in TABLES_TO_COPY:
        cols = _column_specs(local, table)
        if not cols:
            sys.exit(f"error: local DB has no table '{table}'. Has `lens init` been run?")
        emb_dim = embedding_dim if table in EMBEDDING_TABLES else None
        ddl = _build_remote_ddl(table, cols, emb_dim)
        logger.debug(f"DDL for {table}:\n{ddl}")
        client.execute(ddl)

    for ddl in FTS_DDL:
        client.execute(ddl)
    for ddl in INDEX_DDL:
        client.execute(ddl)
    logger.info("created remote schema (tables, FTS, indexes)")


def copy_table(
    local: sqlite3.Connection,
    client: libsql_client.ClientSync,
    table: str,
    *,
    batch_size: int = 200,
) -> int:
    """Bulk-copy rows from local ``table`` to remote, returning row count.

    Uses ``client.batch`` for efficient round-trips and respects libSQL's
    request-size cap by chunking at ``batch_size``.
    """
    cols = _column_names(local, table)
    # Local schema never has an `embedding` column — embeddings live in
    # the companion `_vec` virtual table. The remote schema does, but
    # we attach those values via UPDATE in `attach_embeddings`. Asserting
    # the invariant catches a future schema drift early.
    assert "embedding" not in cols, (
        f"unexpected `embedding` column in local table {table}; "
        f"publish flow assumes embeddings live in {table}_vec"
    )

    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)

    # Read local rows into memory. Even at a few-thousand-row scale this
    # is fine; if we need streaming we can refactor later.
    rows = local.execute(f"SELECT {col_list} FROM {table}").fetchall()
    if not rows:
        logger.info(f"  {table}: 0 rows (skipped)")
        return 0

    insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
    total = 0
    # list invariance: list[Statement] isn't assignable to list[InStatement]
    # at the type level even though Statement ∈ InStatement at runtime.
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        statements = [libsql_client.Statement(insert_sql, list(row)) for row in chunk]
        client.batch(statements)  # ty: ignore[invalid-argument-type]
        total += len(chunk)
    logger.info(f"  {table}: {total} rows copied")
    return total


def attach_embeddings(
    local: sqlite3.Connection,
    client: libsql_client.ClientSync,
    table: str,
    key_col: str,
    *,
    embedding_dim: int,
    batch_size: int = 200,
) -> int:
    """Pull embeddings from local ``{table}_vec`` and UPDATE remote rows.

    The local ``_vec`` virtual table stores ``(key_col, embedding)`` where
    ``embedding`` is already the F32 BLOB form libSQL expects — so we can
    pass the bytes through unchanged.

    Validates that the BLOB byte length matches ``embedding_dim`` so a
    dimension mismatch fails loudly here, not silently at vector-index
    query time.
    """
    rows = local.execute(f"SELECT {key_col}, embedding FROM {table}_vec").fetchall()
    if not rows:
        logger.info(f"  {table}.embedding: 0 rows (no embeddings present)")
        return 0

    expected_bytes = 4 * embedding_dim
    actual_bytes = len(rows[0]["embedding"])
    if actual_bytes != expected_bytes:
        actual_dim = actual_bytes // 4
        sys.exit(
            f"error: embedding dimension mismatch in {table}_vec: "
            f"--embedding-dim={embedding_dim} expects {expected_bytes} bytes "
            f"but row 0 has {actual_bytes} bytes (= {actual_dim}-dim). "
            f"Re-run with --embedding-dim={actual_dim} or re-embed the "
            f"local corpus to match."
        )

    # Count how many `_vec` rows have no matching key in the main table —
    # these are orphan embeddings from prior deletes that didn't cascade.
    # The UPDATE silently no-ops on those, but log a warning so users notice.
    main_keys = {r[0] for r in local.execute(f"SELECT {key_col} FROM {table}").fetchall()}
    orphan_count = sum(1 for r in rows if r[key_col] not in main_keys)
    if orphan_count:
        logger.warning(
            f"  {table}_vec has {orphan_count} orphan embedding(s) with "
            f"no matching {key_col} in {table} — silently skipped during "
            f"attach. Consider cleaning up the local DB."
        )

    update_sql = f"UPDATE {table} SET embedding = ? WHERE {key_col} = ?"
    total = 0
    # list invariance: see note in `copy_table` for why we suppress.
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        statements = [
            libsql_client.Statement(update_sql, [row["embedding"], row[key_col]]) for row in chunk
        ]
        client.batch(statements)  # ty: ignore[invalid-argument-type]
        total += len(chunk)
    logger.info(f"  {table}.embedding: {total} embeddings attached")
    return total


def rebuild_fts(client: libsql_client.ClientSync) -> None:
    """Rebuild FTS5 indexes from the canonical content tables."""
    client.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    client.execute("INSERT INTO vocabulary_fts(vocabulary_fts) VALUES('rebuild')")
    logger.info("rebuilt FTS5 indexes (papers_fts, vocabulary_fts)")


def verify_counts(local: sqlite3.Connection, client: libsql_client.ClientSync) -> bool:
    """Sanity check: row counts match between local and remote for each
    table, and FTS5 indexes have been populated to match their content tables.
    """
    ok = True
    for table in TABLES_TO_COPY:
        local_count = local.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        remote_count = client.execute(f"SELECT count(*) FROM {table}").rows[0][0]
        match = "OK" if local_count == remote_count else "FAIL"
        logger.info(f"  [{match}] {table}: local={local_count}, remote={remote_count}")
        if local_count != remote_count:
            ok = False

    # Sanity-check FTS5 indexes — after `rebuild`, fts row count equals main.
    for fts, main in [("papers_fts", "papers"), ("vocabulary_fts", "vocabulary")]:
        fts_count = client.execute(f"SELECT count(*) FROM {fts}").rows[0][0]
        main_count = client.execute(f"SELECT count(*) FROM {main}").rows[0][0]
        match = "OK" if fts_count == main_count else "FAIL"
        logger.info(f"  [{match}] {fts}: rows={fts_count}, expected={main_count}")
        if fts_count != main_count:
            ok = False
    return ok


def publish(
    local_db_path: str,
    target_url: str,
    target_auth_token: str,
    *,
    embedding_dim: int = EMBEDDING_DIM,
) -> None:
    """End-to-end publish: drop → create → copy → attach → rebuild → verify."""
    if not Path(local_db_path).exists():
        sys.exit(f"error: local DB not found: {local_db_path}")

    logger.info(f"local db: {local_db_path}")
    logger.info(f"target:   {target_url}")
    logger.info(f"embedding dim: {embedding_dim}")

    local = open_local(local_db_path)
    client = libsql_client.create_client_sync(
        url=_normalize_url(target_url), auth_token=target_auth_token
    )

    started = time.monotonic()
    try:
        logger.info("step 1/5: dropping existing remote schema")
        drop_remote_schema(client)

        logger.info("step 2/5: creating remote schema")
        create_remote_schema(local, client, embedding_dim)

        logger.info("step 3/5: copying tables")
        for table in TABLES_TO_COPY:
            copy_table(local, client, table)

        logger.info("step 4/5: attaching embeddings")
        for table, key_col in EMBEDDING_TABLES.items():
            attach_embeddings(local, client, table, key_col, embedding_dim=embedding_dim)

        logger.info("step 5/5: rebuilding FTS5 + verifying")
        rebuild_fts(client)
        if not verify_counts(local, client):
            sys.exit("error: row counts did not match between local and remote")

        elapsed = time.monotonic() - started
        logger.info(f"published successfully in {elapsed:.1f}s")
    finally:
        client.close()
        local.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a local sqlite-vec LENS database to a remote Turso libSQL DB."
    )
    parser.add_argument(
        "--local",
        default=str(Path.home() / ".lens" / "data" / "lens.db"),
        help="Path to local sqlite-vec DB (default: ~/.lens/data/lens.db)",
    )
    parser.add_argument(
        "--target",
        choices=("dev", "prod", "plain"),
        default="dev",
        help="Which Turso DB to publish to (default: dev)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=EMBEDDING_DIM,
        help=f"Embedding vector dimension (default: {EMBEDDING_DIM})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v=DEBUG (default level is INFO; the destructive nature of "
        "this script means progress should always be visible).",
    )
    args = parser.parse_args()

    # Default to INFO so each step's progress is visible; bump to DEBUG with -v.
    # WARNING-level default would silently hide the drop/create/copy stages.
    level = logging.DEBUG if args.verbose >= 1 else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    url, token = _resolve_target(args.target)
    publish(args.local, url, token, embedding_dim=args.embedding_dim)


if __name__ == "__main__":
    main()

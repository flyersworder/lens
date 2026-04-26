"""Protocol-conformance tests for the LENS storage backends.

Verifies that both :class:`LensStore` and :class:`TursoStore` satisfy
the :class:`ReadableStore` protocol, and that ``serve/*`` modules
accept either backend at runtime (not just at the type level).

The TursoStore-specific cases are skipped without ``TURSO_DEV_*``
env vars; the LensStore cases run unconditionally.
"""

from __future__ import annotations

import contextlib
import os
import struct
from pathlib import Path

import pytest

from lens.store.protocols import ReadableStore
from lens.store.store import LensStore

# Same .env.local loader as test_turso_store.py — keeps the test
# self-contained when run locally.
_ENV_LOCAL = Path(__file__).resolve().parent.parent / ".env.local"
if _ENV_LOCAL.exists():
    for line in _ENV_LOCAL.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# LensStore (always runs)
# ---------------------------------------------------------------------------


def test_lens_store_satisfies_readable_protocol(tmp_path):
    """`LensStore` is structurally compatible with `ReadableStore`."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    # @runtime_checkable enables this isinstance check.
    assert isinstance(store, ReadableStore)


def test_serve_explorer_accepts_lens_store(tmp_path):
    """serve/explorer.list_parameters works against the local backend."""
    from lens.serve.explorer import list_parameters

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    # Empty DB returns empty list — confirms type acceptance, not data.
    assert list_parameters(store) == []


# ---------------------------------------------------------------------------
# TursoStore (env-gated)
# ---------------------------------------------------------------------------


def _turso_creds_present() -> bool:
    """Re-evaluated each test invocation so env mutation between collection
    and run is respected (e.g. if a test fixture sets the var)."""
    return bool(
        os.environ.get("TURSO_DEV_DATABASE_URL") and os.environ.get("TURSO_DEV_AUTH_TOKEN")
    )


_SKIP_REASON = "TURSO_DEV_* not set; skipping integration check"


@pytest.mark.skipif(not _turso_creds_present(), reason=_SKIP_REASON)
def test_turso_store_satisfies_readable_protocol():
    from lens.store.turso_store import TursoStore

    store = TursoStore(
        url=os.environ["TURSO_DEV_DATABASE_URL"],
        auth_token=os.environ["TURSO_DEV_AUTH_TOKEN"],
    )
    try:
        assert isinstance(store, ReadableStore)
    finally:
        store.close()


@pytest.mark.skipif(not _turso_creds_present(), reason=_SKIP_REASON)
def test_serve_explorer_against_turso_store():
    """serve/explorer.list_parameters works against the libSQL backend.

    Type-and-shape integration check: it proves the Protocol typing
    isn't just a static convenience but a real contract the serve
    layer can rely on at runtime.

    Coverage scope: this test only exercises ``query`` (via
    ``list_parameters``). Broader coverage of the other 4 protocol
    methods against ``TursoStore`` lives in ``tests/test_turso_store.py``
    (vector_search, search_papers, hybrid_search, query_sql).

    Concurrency caveat: this test sets up + tears down a ``vocabulary``
    table on lens-dev. Running it concurrently with the
    ``publish-turso.yml`` GH Actions workflow against the same DB will
    race. Phase 1 prototype assumes serial execution; if we move to
    parallel CI, point this test at a dedicated ``lens-dev-test``
    database.
    """
    import libsql_client

    from lens.serve.explorer import list_parameters
    from lens.store.turso_store import TursoStore

    url = os.environ["TURSO_DEV_DATABASE_URL"]
    if url.startswith("libsql://"):
        url = "https://" + url[len("libsql://") :]
    raw = libsql_client.create_client_sync(url=url, auth_token=os.environ["TURSO_DEV_AUTH_TOKEN"])

    # Set up a minimal vocabulary table the way publish_to_turso would.
    setup_stmts = [
        "DROP INDEX IF EXISTS papers_emb_idx",
        "DROP INDEX IF EXISTS vocabulary_emb_idx",
        "DROP TABLE IF EXISTS papers_fts",
        "DROP TABLE IF EXISTS vocabulary_fts",
        "DROP TABLE IF EXISTS papers",
        "DROP TABLE IF EXISTS vocabulary",
    ]
    for s in setup_stmts:
        with contextlib.suppress(Exception):
            raw.execute(s)

    raw.execute(
        """
        CREATE TABLE vocabulary (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            description TEXT NOT NULL,
            source TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            paper_count INTEGER NOT NULL DEFAULT 0,
            avg_confidence REAL NOT NULL DEFAULT 0.0,
            embedding F32_BLOB(4)
        )
        """
    )
    raw.execute(
        "CREATE VIRTUAL TABLE vocabulary_fts USING fts5("
        "name, description, kind, content=vocabulary, content_rowid=rowid)"
    )
    raw.execute("CREATE INDEX vocabulary_emb_idx ON vocabulary (libsql_vector_idx(embedding))")

    blob = struct.pack("4f", 1.0, 0.0, 0.0, 0.0)
    raw.execute(
        "INSERT INTO vocabulary "
        "(id, name, kind, description, source, first_seen, "
        "paper_count, avg_confidence, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "throughput",
            "Throughput",
            "parameter",
            "Tokens per second.",
            "seed",
            "2026-01-01",
            1,
            0.9,
            blob,
        ],
    )

    store = TursoStore(
        url=os.environ["TURSO_DEV_DATABASE_URL"],
        auth_token=os.environ["TURSO_DEV_AUTH_TOKEN"],
    )
    try:
        params = list_parameters(store)
        assert len(params) == 1
        assert params[0]["id"] == "throughput"
        assert params[0]["name"] == "Throughput"
    finally:
        store.close()
        # Best-effort teardown so we don't leave state for the next run.
        for s in [
            "DROP INDEX IF EXISTS vocabulary_emb_idx",
            "DROP TABLE IF EXISTS vocabulary_fts",
            "DROP TABLE IF EXISTS vocabulary",
        ]:
            with contextlib.suppress(Exception):
                raw.execute(s)
        raw.close()

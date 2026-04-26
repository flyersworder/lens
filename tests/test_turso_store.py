"""Integration tests for TursoStore against a real remote libSQL DB.

Skipped by default; runs only when ``TURSO_DEV_DATABASE_URL`` and
``TURSO_DEV_AUTH_TOKEN`` are set in the environment (see ``.env.local``).

Tests build the canonical ``papers`` / ``vocabulary`` schema on the
remote DB and tear it down at the end. The dev DB is dedicated to
spike/test work — production data lives on a separate ``lens-prod`` DB.
"""

from __future__ import annotations

import contextlib
import os
import struct
from pathlib import Path

import pytest

# Load .env.local so tests pick up dev creds when running locally.
_ENV_LOCAL = Path(__file__).resolve().parent.parent / ".env.local"
if _ENV_LOCAL.exists():
    for line in _ENV_LOCAL.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


pytestmark = pytest.mark.skipif(
    not os.environ.get("TURSO_DEV_DATABASE_URL") or not os.environ.get("TURSO_DEV_AUTH_TOKEN"),
    reason="Turso dev credentials not available; set TURSO_DEV_DATABASE_URL "
    "and TURSO_DEV_AUTH_TOKEN to run these integration tests.",
)


def _drop_all(client) -> None:
    """Best-effort teardown of the canonical test schema.

    Drops indexes before tables to avoid orphaned vector-index state on
    libSQL — the DROP-on-table cascade isn't always sufficient.
    """
    for stmt in [
        "DROP INDEX IF EXISTS papers_emb_idx",
        "DROP INDEX IF EXISTS vocabulary_emb_idx",
        "DROP TABLE IF EXISTS papers_fts",
        "DROP TABLE IF EXISTS vocabulary_fts",
        "DROP TABLE IF EXISTS papers",
        "DROP TABLE IF EXISTS vocabulary",
    ]:
        with contextlib.suppress(Exception):
            client.execute(stmt)


@pytest.fixture(scope="module")
def turso_client():
    """Raw libsql client used to set up + tear down schema (module-scoped)."""
    libsql_client = pytest.importorskip("libsql_client")
    url = os.environ["TURSO_DEV_DATABASE_URL"]
    if url.startswith("libsql://"):
        url = "https://" + url[len("libsql://") :]
    c = libsql_client.create_client_sync(url=url, auth_token=os.environ["TURSO_DEV_AUTH_TOKEN"])
    yield c
    c.close()


@pytest.fixture(scope="module")
def seeded_db(turso_client):
    """Populate the dev DB with a tiny canonical-named schema."""
    c = turso_client
    _drop_all(c)

    # Canonical schema, F32_BLOB(4) for tiny test vectors.
    c.execute(
        """
        CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            authors TEXT NOT NULL,
            date TEXT NOT NULL,
            embedding F32_BLOB(4)
        )
        """
    )
    c.execute(
        """
        CREATE TABLE vocabulary (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            kind TEXT NOT NULL,
            embedding F32_BLOB(4)
        )
        """
    )
    c.execute(
        "CREATE VIRTUAL TABLE papers_fts USING fts5("
        "title, abstract, content=papers, content_rowid=rowid)"
    )
    c.execute(
        "CREATE VIRTUAL TABLE vocabulary_fts USING fts5("
        "name, description, kind, content=vocabulary, content_rowid=rowid)"
    )
    c.execute("CREATE INDEX papers_emb_idx ON papers (libsql_vector_idx(embedding))")
    c.execute("CREATE INDEX vocabulary_emb_idx ON vocabulary (libsql_vector_idx(embedding))")

    def emb(*xs: float) -> bytes:
        return struct.pack(f"{len(xs)}f", *xs)

    papers = [
        (
            "p1",
            "Attention Is All You Need",
            "transformer",
            '["A","B"]',
            "2017-06-12",
            emb(1, 0, 0, 0),
        ),
        ("p2", "BERT", "bidirectional encoder", '["C"]', "2018-10-11", emb(0, 1, 0, 0)),
        ("p3", "T5", "text-to-text transformer", '["D"]', "2019-10-23", emb(0.9, 0.1, 0, 0)),
    ]
    for row in papers:
        c.execute(
            "INSERT INTO papers (paper_id, title, abstract, authors, date, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            list(row),
        )

    vocab = [
        ("attention", "Attention", "scaled dot-product attention", "principle", emb(1, 0, 0, 0)),
        ("moe", "Mixture of Experts", "sparsely activated FFN", "principle", emb(0, 1, 0, 0)),
        ("gqa", "Grouped Query Attention", "shared KV heads", "principle", emb(0.9, 0.1, 0, 0)),
    ]
    for row in vocab:
        c.execute(
            "INSERT INTO vocabulary (id, name, description, kind, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            list(row),
        )

    c.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    c.execute("INSERT INTO vocabulary_fts(vocabulary_fts) VALUES('rebuild')")

    yield c

    _drop_all(c)


@pytest.fixture
def store(seeded_db):
    """A TursoStore connected to the seeded dev DB."""
    from lens.store.turso_store import TursoStore

    s = TursoStore(
        url=os.environ["TURSO_DEV_DATABASE_URL"],
        auth_token=os.environ["TURSO_DEV_AUTH_TOKEN"],
    )
    yield s
    s.close()


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------


def test_query_returns_all_rows(store):
    rows = store.query("papers")
    assert len(rows) == 3
    assert {r["title"] for r in rows} == {
        "Attention Is All You Need",
        "BERT",
        "T5",
    }


def test_query_with_where(store):
    rows = store.query("papers", "paper_id = ?", ("p1",))
    assert len(rows) == 1
    assert rows[0]["title"] == "Attention Is All You Need"


def test_query_deserializes_json_fields(store):
    # `authors` is a JSON-serialized list per JSON_FIELDS["papers"].
    rows = store.query("papers", "paper_id = ?", ("p1",))
    assert rows[0]["authors"] == ["A", "B"]


# ---------------------------------------------------------------------------
# query_sql()
# ---------------------------------------------------------------------------


def test_query_sql_passthrough(store):
    rows = store.query_sql("SELECT paper_id, title FROM papers ORDER BY date DESC")
    assert [r["paper_id"] for r in rows] == ["p3", "p2", "p1"]


# ---------------------------------------------------------------------------
# vector_search()
# ---------------------------------------------------------------------------


def test_vector_search_nearest_neighbor(store):
    # Query along the x-axis; nearest is p1 (1,0,0,0), then p3 (0.9,0.1,0,0).
    results = store.vector_search("papers", [1.0, 0.0, 0.0, 0.0], limit=2)
    assert [r["paper_id"] for r in results] == ["p1", "p3"]
    assert "_distance" in results[0]
    assert results[0]["_distance"] == pytest.approx(0.0, abs=1e-6)


def test_vector_search_on_vocabulary(store):
    results = store.vector_search("vocabulary", [0.0, 1.0, 0.0, 0.0], limit=1)
    assert len(results) == 1
    assert results[0]["id"] == "moe"


def test_vector_search_with_where_filter(store):
    # Filter to only `principle` kinds + nearest to x-axis. All seeded
    # vocab entries are principles, so the filter is a no-op semantically
    # but exercises the over-fetch + JOIN-WHERE branch.
    results = store.vector_search(
        "vocabulary",
        [1.0, 0.0, 0.0, 0.0],
        limit=2,
        where="t.kind = ?",
        params=("principle",),
    )
    assert [r["id"] for r in results] == ["attention", "gqa"]


def test_vector_search_with_where_filter_excludes_rows(store):
    # Filter that matches nothing — confirms the WHERE clause gates correctly.
    results = store.vector_search(
        "vocabulary",
        [1.0, 0.0, 0.0, 0.0],
        limit=5,
        where="t.kind = ?",
        params=("does_not_exist",),
    )
    assert results == []


# ---------------------------------------------------------------------------
# search_papers() — hybrid FTS5 + vector
# ---------------------------------------------------------------------------


def test_search_papers_filter_only(store):
    results = store.search_papers(filters={"after": "2018-01-01"}, limit=10)
    assert {r["paper_id"] for r in results} == {"p2", "p3"}


def test_search_papers_fts_only(store):
    results = store.search_papers(query="transformer", limit=5)
    paper_ids = {r["paper_id"] for r in results}
    # p1 (transformer) and p3 (text-to-text transformer) both match.
    assert paper_ids >= {"p1", "p3"}


def test_search_papers_hybrid(store):
    # FTS query "bidirectional" matches p2's abstract; vector points to y-axis
    # where p2 lives; either signal makes p2 the top hit.
    results = store.search_papers(
        query="bidirectional",
        embedding=[0.0, 1.0, 0.0, 0.0],
        limit=3,
    )
    assert results[0]["paper_id"] == "p2"
    assert "_rrf_score" in results[0]


def test_search_papers_hybrid_no_fts_matches(store):
    # FTS query that matches nothing in any abstract; vector signal alone
    # should still surface the nearest paper. Exercises the
    # `f.rowid IS NULL OR vm.rowid IS NOT NULL` branch of the combined CTE.
    results = store.search_papers(
        query="zzz_nonexistent_token",
        embedding=[0.0, 1.0, 0.0, 0.0],
        limit=2,
    )
    # Vector-only path should still return p2 (the y-axis paper).
    assert any(r["paper_id"] == "p2" for r in results)


def test_search_papers_filter_only_no_results(store):
    # Date filter that matches nothing returns empty list cleanly.
    results = store.search_papers(filters={"after": "2099-01-01"}, limit=5)
    assert results == []


def test_search_papers_fts_only_no_matches(store):
    # FTS query against a token absent from every paper.
    results = store.search_papers(query="zzz_nonexistent_token", limit=5)
    assert results == []


# ---------------------------------------------------------------------------
# hybrid_search() — vocabulary
# ---------------------------------------------------------------------------


def test_hybrid_search_vocabulary(store):
    # FTS "attention" matches `attention` and `gqa` (shared KV heads).
    # Vector x-axis prefers `attention` (1,0,0,0). RRF picks attention first.
    results = store.hybrid_search(
        query="attention",
        embedding=[1.0, 0.0, 0.0, 0.0],
        limit=2,
    )
    assert results[0]["id"] == "attention"
    assert "_rrf_score" in results[0]


# ---------------------------------------------------------------------------
# Pure-Python helpers (no DB needed)
# ---------------------------------------------------------------------------


def test_normalize_url():
    from lens.store.turso_store import _normalize_url

    assert _normalize_url("libsql://x.turso.io") == "https://x.turso.io"
    assert _normalize_url("https://x.turso.io") == "https://x.turso.io"
    assert _normalize_url("file:/tmp/foo.db") == "file:/tmp/foo.db"


def test_pack_embedding_round_trip():
    from lens.store.turso_store import _pack_embedding

    blob = _pack_embedding([1.0, 0.0, 0.0, 0.0])
    assert len(blob) == 16
    assert struct.unpack("4f", blob) == (1.0, 0.0, 0.0, 0.0)

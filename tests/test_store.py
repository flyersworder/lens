"""Tests for LensStore — SQLite + sqlite-vec backend."""

from datetime import datetime

from lens.store.models import EMBEDDING_DIM

# ---- 1. Init creates tables ----


def test_init_creates_regular_tables(store):
    cursor = store.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row[0] for row in cursor.fetchall()}
    expected = {
        "papers",
        "tradeoff_extractions",
        "architecture_extractions",
        "agentic_extractions",
        "vocabulary",
        "matrix_cells",
        "taxonomy_versions",
        "ideation_reports",
        "ideation_gaps",
    }
    assert expected.issubset(tables)


def test_init_creates_vec_tables(store):
    cursor = store.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row[0] for row in cursor.fetchall()}
    for vec_table in [
        "papers_vec",
        "vocabulary_vec",
    ]:
        assert vec_table in tables, f"Missing vec table: {vec_table}"


def test_init_creates_papers_fts(store):
    cursor = store.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row[0] for row in cursor.fetchall()}
    assert "papers_fts" in tables


def test_rebuild_papers_fts(store, sample_paper_data):
    store.add_papers([sample_paper_data])
    store.rebuild_papers_fts()
    cursor = store.conn.execute(
        "SELECT * FROM papers_fts WHERE papers_fts MATCH ?", ('"Attention"',)
    )
    rows = cursor.fetchall()
    assert len(rows) == 1


def test_papers_fts_populated_after_init(tmp_path):
    """papers_fts is rebuilt when upgrading a database that predates it."""
    from lens.store.store import LensStore

    # Create a database with only the papers table (simulating old schema)
    db_path = str(tmp_path / "upgrade.db")
    store = LensStore(db_path)
    store.conn.execute(
        "CREATE TABLE IF NOT EXISTS papers ("
        "paper_id TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT NOT NULL, "
        "authors TEXT NOT NULL, venue TEXT, date TEXT NOT NULL, arxiv_id TEXT NOT NULL, "
        "citations INTEGER NOT NULL DEFAULT 0, quality_score REAL NOT NULL DEFAULT 0.0, "
        "extraction_status TEXT NOT NULL DEFAULT 'pending')"
    )
    store.conn.execute(
        "INSERT INTO papers (paper_id, title, abstract, authors, venue, date, arxiv_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "p1",
            "Attention Is All You Need",
            "A new architecture...",
            '["Vaswani"]',
            None,
            "2017-06-12",
            "1706.03762",
        ),
    )
    store.conn.commit()

    # Now call init_tables — should create papers_fts AND rebuild the index
    store.init_tables()
    cursor = store.conn.execute(
        "SELECT * FROM papers_fts WHERE papers_fts MATCH ?", ('"Attention"',)
    )
    rows = cursor.fetchall()
    assert len(rows) == 1


# ---- 2. add_rows + query (basic CRUD) ----


def test_add_rows_and_query(store, sample_paper_data):
    store.add_rows("papers", [sample_paper_data])
    rows = store.query("papers")
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "2401.12345"
    assert rows[0]["title"] == "Attention Is All You Need"


# ---- 3. query with WHERE clause ----


def test_query_with_where(store, sample_paper_data):
    store.add_rows("papers", [sample_paper_data])
    other = sample_paper_data.copy()
    other["paper_id"] = "2401.99999"
    other["arxiv_id"] = "2401.99999"
    other["extraction_status"] = "complete"
    other["embedding"] = [0.2] * EMBEDDING_DIM
    store.add_rows("papers", [other])

    rows = store.query("papers", where="extraction_status = ?", params=("pending",))
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "2401.12345"


# ---- 4. update ----


def test_update(store, sample_paper_data):
    store.add_rows("papers", [sample_paper_data])
    store.update(
        "papers",
        values="extraction_status = ?",
        where="paper_id = ?",
        params=("complete", "2401.12345"),
    )
    rows = store.query("papers", where="paper_id = ?", params=("2401.12345",))
    assert rows[0]["extraction_status"] == "complete"


# ---- 5. delete ----


def test_delete(store, sample_paper_data):
    store.add_rows("papers", [sample_paper_data])
    store.delete("papers", where="paper_id = ?", params=("2401.12345",))
    rows = store.query("papers")
    assert len(rows) == 0


def test_delete_also_removes_from_vec_table(store, sample_paper_data):
    store.add_rows("papers", [sample_paper_data])
    store.delete("papers", where="paper_id = ?", params=("2401.12345",))
    vec_rows = store.conn.execute("SELECT count(*) FROM papers_vec").fetchone()[0]
    assert vec_rows == 0


# ---- 6. vector_search (nearest neighbor returns closest) ----


def test_vector_search(store):
    papers = [
        {
            "paper_id": "paper_a",
            "title": "About attention",
            "abstract": "Attention mechanisms",
            "authors": ["A"],
            "venue": None,
            "date": "2024-01-01",
            "arxiv_id": "paper_a",
            "citations": 0,
            "quality_score": 0.0,
            "extraction_status": "pending",
            "embedding": [1.0] + [0.0] * (EMBEDDING_DIM - 1),
        },
        {
            "paper_id": "paper_b",
            "title": "About distillation",
            "abstract": "Knowledge distillation",
            "authors": ["B"],
            "venue": None,
            "date": "2024-01-01",
            "arxiv_id": "paper_b",
            "citations": 0,
            "quality_score": 0.0,
            "extraction_status": "pending",
            "embedding": [0.0] + [1.0] + [0.0] * (EMBEDDING_DIM - 2),
        },
    ]
    store.add_rows("papers", papers)

    query_emb = [0.9] + [0.1] + [0.0] * (EMBEDDING_DIM - 2)
    results = store.vector_search("papers", query_emb, limit=1)
    assert len(results) == 1
    assert results[0]["paper_id"] == "paper_a"


# ---- 7. vector_search with WHERE filter ----


def test_vector_search_with_where(store):
    papers = [
        {
            "paper_id": "paper_a",
            "title": "About attention",
            "abstract": "Attention mechanisms",
            "authors": ["A"],
            "venue": None,
            "date": "2024-01-01",
            "arxiv_id": "paper_a",
            "citations": 0,
            "quality_score": 0.0,
            "extraction_status": "complete",
            "embedding": [1.0] + [0.0] * (EMBEDDING_DIM - 1),
        },
        {
            "paper_id": "paper_b",
            "title": "About distillation",
            "abstract": "Knowledge distillation",
            "authors": ["B"],
            "venue": None,
            "date": "2024-01-01",
            "arxiv_id": "paper_b",
            "citations": 0,
            "quality_score": 0.0,
            "extraction_status": "pending",
            "embedding": [0.9] + [0.1] + [0.0] * (EMBEDDING_DIM - 2),
        },
    ]
    store.add_rows("papers", papers)

    # paper_b is closer to query but filter requires "complete"
    query_emb = [0.9] + [0.1] + [0.0] * (EMBEDDING_DIM - 2)
    results = store.vector_search(
        "papers", query_emb, limit=1, where="extraction_status = ?", params=("complete",)
    )
    assert len(results) == 1
    assert results[0]["paper_id"] == "paper_a"


# ---- 8. query_sql for raw queries ----


def test_query_sql(store, sample_paper_data):
    store.add_rows("papers", [sample_paper_data])
    rows = store.query_sql(
        "SELECT paper_id, title FROM papers WHERE paper_id = ?",
        params=("2401.12345",),
    )
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "2401.12345"
    assert rows[0]["title"] == "Attention Is All You Need"


# ---- 9. init_tables is idempotent ----


def test_init_tables_idempotent(store, sample_paper_data):
    store.add_rows("papers", [sample_paper_data])
    store.init_tables()  # call again
    rows = store.query("papers")
    assert len(rows) == 1


# ---- 10. JSON list fields round-trip correctly ----


def test_json_list_fields_roundtrip(store, sample_paper_data):
    store.add_rows("papers", [sample_paper_data])
    rows = store.query("papers")
    assert rows[0]["authors"] == ["Vaswani", "Shazeer"]


def test_json_list_fields_roundtrip_ideation_gaps(store):
    row = {
        "id": 1,
        "report_id": 10,
        "gap_type": "coverage",
        "description": "Missing coverage",
        "related_params": [1, 2, 3],
        "related_principles": [4, 5],
        "related_slots": [6],
        "score": 0.8,
        "llm_hypothesis": None,
        "created_at": datetime(2024, 1, 1, 12, 0, 0),
        "taxonomy_version": 1,
    }
    store.add_rows("ideation_gaps", [row])
    rows = store.query("ideation_gaps")
    assert rows[0]["related_params"] == [1, 2, 3]
    assert rows[0]["related_principles"] == [4, 5]
    assert rows[0]["related_slots"] == [6]


# ---- 11. add_papers convenience method ----


def test_add_papers(store, sample_paper_data):
    store.add_papers([sample_paper_data])
    rows = store.query("papers")
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "2401.12345"


def test_add_papers_skips_duplicates(store, sample_paper_data):
    """add_papers should skip duplicates without crashing."""
    inserted_first = store.add_papers([sample_paper_data])
    assert inserted_first == 1

    # Insert same paper again — should be silently skipped
    inserted_second = store.add_papers([sample_paper_data])
    assert inserted_second == 0

    rows = store.query("papers")
    assert len(rows) == 1


def test_add_papers_returns_count_with_mixed_new_and_existing(store, sample_paper_data):
    """add_papers should return count of only newly inserted papers."""
    store.add_papers([sample_paper_data])

    new_paper = sample_paper_data.copy()
    new_paper["paper_id"] = "2401.99999"
    new_paper["arxiv_id"] = "2401.99999"

    inserted = store.add_papers([sample_paper_data, new_paper])
    assert inserted == 1  # only the new one

    rows = store.query("papers")
    assert len(rows) == 2


# ---- 12. search_papers ----


def test_search_papers_hybrid(store, sample_paper_data):
    """Hybrid search combines FTS5 keyword + vector similarity."""
    store.add_papers([sample_paper_data])
    query_embedding = [0.1] * EMBEDDING_DIM
    results = store.search_papers(
        query="Attention architecture",
        embedding=query_embedding,
        limit=5,
    )
    assert len(results) == 1
    assert results[0]["paper_id"] == "2401.12345"
    assert "_rrf_score" in results[0]


def test_search_papers_fts_only(store, sample_paper_data):
    """FTS-only search when no embedding is provided."""
    store.add_papers([sample_paper_data])
    results = store.search_papers(query="Attention", limit=5)
    assert len(results) == 1
    assert results[0]["paper_id"] == "2401.12345"
    assert "_rrf_score" in results[0]


def test_search_papers_filter_only(store, sample_paper_data):
    """Filter-only mode when no text query is given."""
    store.add_papers([sample_paper_data])
    results = store.search_papers(filters={"author": "Vaswani"}, limit=5)
    assert len(results) == 1
    assert results[0]["paper_id"] == "2401.12345"
    assert "_rrf_score" not in results[0]


def test_search_papers_hybrid_with_filters(store):
    """Hybrid search narrowed by date filter."""
    papers = [
        {
            "paper_id": "p1",
            "title": "Attention Mechanism Survey",
            "abstract": "A survey of attention mechanisms in deep learning.",
            "authors": ["Author A"],
            "venue": None,
            "date": "2023-01-01",
            "arxiv_id": "2301.00001",
            "citations": 10,
            "quality_score": 0.5,
            "extraction_status": "pending",
            "embedding": [0.1] * EMBEDDING_DIM,
        },
        {
            "paper_id": "p2",
            "title": "Attention in Vision Transformers",
            "abstract": "Applying attention to computer vision tasks.",
            "authors": ["Author B"],
            "venue": None,
            "date": "2024-06-01",
            "arxiv_id": "2406.00001",
            "citations": 5,
            "quality_score": 0.3,
            "extraction_status": "pending",
            "embedding": [0.2] * EMBEDDING_DIM,
        },
    ]
    store.add_papers(papers)
    results = store.search_papers(
        query="attention",
        embedding=[0.15] * EMBEDDING_DIM,
        filters={"after": "2024-01-01"},
        limit=5,
    )
    assert len(results) == 1
    assert results[0]["paper_id"] == "p2"


def test_search_papers_filter_by_venue(store):
    """Filter-only mode filters by venue substring."""
    paper = {
        "paper_id": "p1",
        "title": "A NeurIPS Paper",
        "abstract": "Some content.",
        "authors": ["Alice"],
        "venue": "NeurIPS 2024",
        "date": "2024-01-01",
        "arxiv_id": "2401.00001",
        "citations": 0,
        "quality_score": 0.5,
        "extraction_status": "pending",
    }
    store.add_papers([paper])
    results = store.search_papers(filters={"venue": "NeurIPS"}, limit=5)
    assert len(results) == 1
    assert results[0]["paper_id"] == "p1"

    results = store.search_papers(filters={"venue": "ICML"}, limit=5)
    assert results == []


def test_search_papers_filter_by_before(store):
    """Filter-only mode filters by before date."""
    papers = [
        {
            "paper_id": "p1",
            "title": "Old Paper",
            "abstract": "Content.",
            "authors": ["Alice"],
            "venue": None,
            "date": "2022-06-01",
            "arxiv_id": "2206.00001",
            "citations": 0,
            "quality_score": 0.5,
            "extraction_status": "pending",
        },
        {
            "paper_id": "p2",
            "title": "New Paper",
            "abstract": "Content.",
            "authors": ["Bob"],
            "venue": None,
            "date": "2024-06-01",
            "arxiv_id": "2406.00001",
            "citations": 0,
            "quality_score": 0.5,
            "extraction_status": "pending",
        },
    ]
    store.add_papers(papers)
    results = store.search_papers(filters={"before": "2023-01-01"}, limit=5)
    assert len(results) == 1
    assert results[0]["paper_id"] == "p1"


def test_search_papers_no_results(store):
    """Search with no matching papers returns empty list."""
    results = store.search_papers(query="nonexistent topic xyz", limit=5)
    assert results == []


def test_add_rows_without_ignore_conflicts_raises_on_duplicate(store, sample_paper_data):
    """add_rows with ignore_conflicts=False should raise on duplicate PK."""
    import sqlite3

    import pytest

    store.add_rows("papers", [sample_paper_data])
    with pytest.raises(sqlite3.IntegrityError):
        store.add_rows("papers", [sample_paper_data])


# ---- Extra: datetime serialization ----


def test_datetime_serialization(store):
    row = {
        "id": 1,
        "report_id": 10,
        "gap_type": "coverage",
        "description": "Missing coverage",
        "related_params": [],
        "related_principles": [],
        "related_slots": [],
        "score": 0.5,
        "llm_hypothesis": None,
        "created_at": datetime(2024, 6, 15, 10, 30, 0),
        "taxonomy_version": 1,
    }
    store.add_rows("ideation_gaps", [row])
    rows = store.query("ideation_gaps")
    assert rows[0]["created_at"] == "2024-06-15T10:30:00"


# ---- Extra: db_path auto-appends .db ----


def test_db_path_auto_appends_db(tmp_path):
    from lens.store.store import LensStore

    s = LensStore(str(tmp_path / "mystore"))
    assert s.db_path.endswith(".db")

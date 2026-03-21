"""Tests for LensStore — LanceDB connection and table management."""

import polars as pl

from lens.store.store import escape_sql_string


class TestEscapeSqlString:
    def test_no_quotes(self):
        assert escape_sql_string("hello") == "hello"

    def test_single_quote(self):
        assert escape_sql_string("O'Brien") == "O''Brien"

    def test_multiple_quotes(self):
        assert escape_sql_string("it's a 'test'") == "it''s a ''test''"

    def test_empty_string(self):
        assert escape_sql_string("") == ""


def test_store_init(store):
    assert store.db is not None


def test_store_init_tables(store):
    store.init_tables()
    table_names = store.db.table_names()
    assert "papers" in table_names
    assert "tradeoff_extractions" in table_names
    assert "architecture_extractions" in table_names
    assert "agentic_extractions" in table_names
    assert "parameters" in table_names
    assert "principles" in table_names
    assert "architecture_slots" in table_names
    assert "architecture_variants" in table_names
    assert "agentic_patterns" in table_names
    assert "matrix_cells" in table_names
    assert "taxonomy_versions" in table_names
    assert "ideation_reports" in table_names
    assert "ideation_gaps" in table_names


def test_store_add_and_get_paper(store, sample_paper_data):
    store.init_tables()
    store.add_papers([sample_paper_data])
    papers = store.get_table("papers").to_polars()
    assert len(papers) == 1
    assert papers["paper_id"][0] == "2401.12345"


def test_store_add_multiple_papers(store):
    store.init_tables()
    papers = [
        {
            "paper_id": f"2401.{i:05d}",
            "title": f"Paper {i}",
            "abstract": f"Abstract {i}",
            "authors": ["Author"],
            "venue": None,
            "date": "2024-01-01",
            "arxiv_id": f"2401.{i:05d}",
            "citations": 0,
            "quality_score": 0.0,
            "extraction_status": "pending",
            "embedding": [float(i) / 10] * 768,
        }
        for i in range(5)
    ]
    store.add_papers(papers)
    result = store.get_table("papers").to_polars()
    assert len(result) == 5


def test_store_get_table_as_polars(store, sample_paper_data):
    store.init_tables()
    store.add_papers([sample_paper_data])
    df = store.get_table("papers").to_polars()
    assert isinstance(df, pl.DataFrame)
    assert "paper_id" in df.columns
    assert "embedding" in df.columns


def test_store_vector_search(store):
    store.init_tables()
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
            "embedding": [1.0] + [0.0] * 767,
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
            "embedding": [0.0] + [1.0] + [0.0] * 766,
        },
    ]
    store.add_papers(papers)
    query = [0.9] + [0.1] + [0.0] * 766
    results = store.get_table("papers").search(query).limit(1).to_pandas()
    assert results.iloc[0]["paper_id"] == "paper_a"


def test_store_filtered_query(store, sample_paper_data):
    store.init_tables()
    store.add_papers([sample_paper_data])
    other = sample_paper_data.copy()
    other["paper_id"] = "2401.99999"
    other["arxiv_id"] = "2401.99999"
    other["extraction_status"] = "complete"
    store.add_papers([other])
    df = store.get_table("papers").to_polars()
    pending = df.filter(pl.col("extraction_status") == "pending")
    assert len(pending) == 1
    assert pending["paper_id"][0] == "2401.12345"


def test_store_init_tables_idempotent(store):
    store.init_tables()
    store.init_tables()
    table_names = store.db.table_names()
    assert len(table_names) == len(set(table_names))

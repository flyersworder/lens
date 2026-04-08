"""Tests for DeepXiv acquire module and schema extensions."""

from lens.store.models import EMBEDDING_DIM, Paper
from lens.store.store import LensStore


def test_paper_model_accepts_keywords_and_github_url():
    """Paper model should accept optional keywords and github_url fields."""
    p = Paper(
        paper_id="2507.01701",
        title="Test Paper",
        abstract="An abstract.",
        authors=["Alice"],
        date="2025-07-02",
        arxiv_id="2507.01701",
        keywords=["blackboard architecture", "multi-agent"],
        github_url="https://github.com/bc200/LbMAS",
    )
    assert p.keywords == ["blackboard architecture", "multi-agent"]
    assert p.github_url == "https://github.com/bc200/LbMAS"


def test_paper_model_defaults_keywords_and_github_url():
    """keywords and github_url should default to empty list and None."""
    p = Paper(
        paper_id="2507.01701",
        title="Test Paper",
        abstract="An abstract.",
        authors=["Alice"],
        date="2025-07-02",
        arxiv_id="2507.01701",
    )
    assert p.keywords == []
    assert p.github_url is None


def test_schema_migration_adds_keywords_and_github_url(tmp_path):
    """init_tables() should add keywords and github_url columns to papers."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    # Insert a paper with the new fields
    store.add_papers(
        [
            {
                "paper_id": "test-001",
                "title": "Test",
                "abstract": "Abstract",
                "authors": ["Alice"],
                "date": "2025-01-01",
                "arxiv_id": "test-001",
                "keywords": ["attention", "transformer"],
                "github_url": "https://github.com/test/repo",
                "embedding": [0.0] * EMBEDDING_DIM,
            }
        ]
    )

    rows = store.query("papers", "paper_id = ?", ("test-001",))
    assert len(rows) == 1
    assert rows[0]["keywords"] == ["attention", "transformer"]
    assert rows[0]["github_url"] == "https://github.com/test/repo"

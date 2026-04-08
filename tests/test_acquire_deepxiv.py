"""Tests for DeepXiv acquire module and schema extensions."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from lens.cli import app
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


def test_search_deepxiv_maps_results_to_paper_dicts():
    """search_deepxiv should map DeepXiv search results to LENS Paper dicts."""
    mock_reader = MagicMock()
    mock_reader.search.return_value = {
        "total": 1,
        "results": [
            {
                "arxiv_id": "2507.01701",
                "title": "Blackboard Multi-Agent Systems",
                "abstract": "We propose bMAS...",
                "authors": [
                    {"name": "Bochen Han"},
                    {"name": "Songmao Zhang"},
                ],
                "categories": ["cs.MA", "cs.AI"],
                "citation": 3,
                "score": 33.8,
                "publish_at": "2025-07-02T00:00:00",
            }
        ],
    }

    with (
        patch("lens.acquire.deepxiv.Reader", return_value=mock_reader),
        patch("lens.acquire.deepxiv.HAS_DEEPXIV", True),
    ):
        from lens.acquire.deepxiv import search_deepxiv

        papers = search_deepxiv(query="multi-agent", max_results=5)

    assert len(papers) == 1
    p = papers[0]
    assert p["paper_id"] == "2507.01701"
    assert p["arxiv_id"] == "2507.01701"
    assert p["title"] == "Blackboard Multi-Agent Systems"
    assert p["authors"] == ["Bochen Han", "Songmao Zhang"]
    assert p["citations"] == 3
    assert p["date"] == "2025-07-02"
    assert p["extraction_status"] == "pending"
    assert isinstance(p["quality_score"], float)


def test_search_deepxiv_empty_results():
    """search_deepxiv should return empty list when no results."""
    mock_reader = MagicMock()
    mock_reader.search.return_value = {"total": 0, "results": []}

    with (
        patch("lens.acquire.deepxiv.Reader", return_value=mock_reader),
        patch("lens.acquire.deepxiv.HAS_DEEPXIV", True),
    ):
        from lens.acquire.deepxiv import search_deepxiv

        papers = search_deepxiv(query="nonexistent topic xyz")

    assert papers == []


def test_fetch_deepxiv_paper_returns_rich_metadata():
    """fetch_deepxiv_paper should return keywords and github_url."""
    mock_reader = MagicMock()
    mock_reader.brief.return_value = {
        "arxiv_id": "2507.01701",
        "title": "Blackboard Multi-Agent Systems",
        "abstract": "We propose bMAS...",
        "authors": [{"name": "Bochen Han"}],
        "publish_at": "2025-07-02T00:00:00",
        "citations": 3,
        "keywords": ["blackboard architecture", "multi-agent"],
        "github_url": "https://github.com/bc200/LbMAS",
    }

    with (
        patch("lens.acquire.deepxiv.Reader", return_value=mock_reader),
        patch("lens.acquire.deepxiv.HAS_DEEPXIV", True),
    ):
        from lens.acquire.deepxiv import fetch_deepxiv_paper

        paper = fetch_deepxiv_paper("2507.01701")

    assert paper["keywords"] == ["blackboard architecture", "multi-agent"]
    assert paper["github_url"] == "https://github.com/bc200/LbMAS"
    assert paper["paper_id"] == "2507.01701"
    assert paper["authors"] == ["Bochen Han"]


def test_fetch_deepxiv_paper_handles_missing_optional_fields():
    """fetch_deepxiv_paper should handle missing keywords and github_url."""
    mock_reader = MagicMock()
    mock_reader.brief.return_value = {
        "arxiv_id": "2507.01701",
        "title": "A Paper",
        "abstract": "Abstract",
        "publish_at": "2025-07-02T00:00:00",
        "citations": 0,
    }

    with (
        patch("lens.acquire.deepxiv.Reader", return_value=mock_reader),
        patch("lens.acquire.deepxiv.HAS_DEEPXIV", True),
    ):
        from lens.acquire.deepxiv import fetch_deepxiv_paper

        paper = fetch_deepxiv_paper("2507.01701")

    assert paper["keywords"] == []
    assert paper["github_url"] is None


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

runner = CliRunner()


def test_cli_deepxiv_search(tmp_path, monkeypatch):
    """lens acquire deepxiv should search and store papers."""
    mock_reader = MagicMock()
    mock_reader.search.return_value = {
        "total": 1,
        "results": [
            {
                "arxiv_id": "2507.01701",
                "title": "Test Paper",
                "abstract": "Abstract",
                "authors": [{"name": "Alice"}],
                "publish_at": "2025-07-02T00:00:00",
                "citation": 1,
            }
        ],
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "storage:\n  data_dir: " + str(tmp_path) + "\nacquire:\n  arxiv_categories: [cs.AI]\n"
    )
    monkeypatch.setenv("LENS_CONFIG_PATH", str(config_path))

    with (
        patch("lens.acquire.deepxiv.Reader", return_value=mock_reader),
        patch("lens.acquire.deepxiv.HAS_DEEPXIV", True),
    ):
        result = runner.invoke(app, ["acquire", "deepxiv", "multi-agent"])

    assert result.exit_code == 0
    assert "1" in result.output


def test_cli_deepxiv_not_installed(tmp_path, monkeypatch):
    """lens acquire deepxiv should print helpful error when not installed."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "storage:\n  data_dir: " + str(tmp_path) + "\nacquire:\n  arxiv_categories: [cs.AI]\n"
    )
    monkeypatch.setenv("LENS_CONFIG_PATH", str(config_path))

    with patch("lens.acquire.deepxiv.HAS_DEEPXIV", False):
        result = runner.invoke(app, ["acquire", "deepxiv", "multi-agent"])

    assert result.exit_code == 1
    assert "deepxiv-sdk" in result.output.lower() or "not installed" in result.output.lower()

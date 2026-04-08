# DeepXiv Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `deepxiv-sdk` as an optional acquire source with a new `lens acquire deepxiv` CLI command, storing richer metadata (keywords, github_url) from DeepXiv's API.

**Architecture:** New `src/lens/acquire/deepxiv.py` module with `search_deepxiv()` and `fetch_deepxiv_paper()` functions that call the `deepxiv_sdk.Reader` API and return LENS Paper-shaped dicts. Schema extended with two nullable columns via existing `_COLUMN_MIGRATIONS` pattern. CLI command added to `acquire_app` in `cli.py`.

**Tech Stack:** deepxiv-sdk (optional), existing quality.py for scoring, SQLite schema migrations

**Spec:** `docs/specs/2026-04-08-deepxiv-integration-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `pyproject.toml` | Modify | Add `deepxiv` optional extra |
| `src/lens/acquire/deepxiv.py` | Create | DeepXiv Reader wrapper with `search_deepxiv()` and `fetch_deepxiv_paper()` |
| `src/lens/acquire/__init__.py` | Modify | Conditionally export DeepXiv functions |
| `src/lens/store/store.py` | Modify | Add `keywords` and `github_url` column migrations + JSON fields |
| `src/lens/store/models.py` | Modify | Add `keywords` and `github_url` fields to Paper model |
| `src/lens/cli.py` | Modify | Add `lens acquire deepxiv` subcommand |
| `tests/test_acquire_deepxiv.py` | Create | Unit tests for DeepXiv acquire module |

---

### Task 1: Schema — Add `keywords` and `github_url` to Paper

**Files:**
- Modify: `src/lens/store/models.py:23-36` (Paper model)
- Modify: `src/lens/store/store.py:26-34` (JSON_FIELDS)
- Modify: `src/lens/store/store.py:146-151` (_COLUMN_MIGRATIONS)
- Test: `tests/test_acquire_deepxiv.py`

- [ ] **Step 1: Write a failing test for the new Paper fields**

Create `tests/test_acquire_deepxiv.py`:

```python
"""Tests for DeepXiv acquire module and schema extensions."""

from lens.store.models import Paper


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_acquire_deepxiv.py::test_paper_model_accepts_keywords_and_github_url tests/test_acquire_deepxiv.py::test_paper_model_defaults_keywords_and_github_url -v`
Expected: FAIL — Paper model does not accept `keywords` or `github_url`

- [ ] **Step 3: Add fields to Paper model**

In `src/lens/store/models.py`, add two fields to the `Paper` class after `extraction_status`:

```python
class Paper(BaseModel):
    """A research paper ingested into LENS."""

    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    venue: str | None = None
    date: str
    arxiv_id: str
    citations: int = 0
    quality_score: float = 0.0
    extraction_status: str = "pending"
    keywords: list[str] = []
    github_url: str | None = None
    embedding: list[float] = []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_acquire_deepxiv.py -v`
Expected: PASS

- [ ] **Step 5: Add schema migration and JSON field registration**

In `src/lens/store/store.py`, add `keywords` to `JSON_FIELDS` for papers:

```python
JSON_FIELDS: dict[str, set[str]] = {
    "papers": {"authors", "keywords"},
    ...
}
```

Add two entries to `_COLUMN_MIGRATIONS`:

```python
_COLUMN_MIGRATIONS: list[tuple[str, str, str]] = [
    ("tradeoff_extractions", "new_concepts", "TEXT NOT NULL DEFAULT '{}'"),
    ("architecture_extractions", "new_concepts", "TEXT NOT NULL DEFAULT '{}'"),
    ("agentic_extractions", "category", "TEXT NOT NULL DEFAULT ''"),
    ("agentic_extractions", "new_concepts", "TEXT NOT NULL DEFAULT '{}'"),
    ("papers", "keywords", "TEXT NOT NULL DEFAULT '[]'"),
    ("papers", "github_url", "TEXT"),
]
```

- [ ] **Step 6: Write a test for schema migration**

Append to `tests/test_acquire_deepxiv.py`:

```python
from lens.store.store import LensStore
from lens.store.models import EMBEDDING_DIM


def test_schema_migration_adds_keywords_and_github_url(tmp_path):
    """init_tables() should add keywords and github_url columns to papers."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    # Insert a paper with the new fields
    store.add_papers([
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
    ])

    rows = store.query("papers", "paper_id = ?", ("test-001",))
    assert len(rows) == 1
    assert rows[0]["keywords"] == ["attention", "transformer"]
    assert rows[0]["github_url"] == "https://github.com/test/repo"
```

- [ ] **Step 7: Run all tests to verify**

Run: `uv run pytest tests/test_acquire_deepxiv.py -v`
Expected: All 3 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/lens/store/models.py src/lens/store/store.py tests/test_acquire_deepxiv.py
git commit -m "feat: add keywords and github_url fields to Paper schema"
```

---

### Task 2: DeepXiv acquire module

**Files:**
- Create: `src/lens/acquire/deepxiv.py`
- Modify: `src/lens/acquire/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_acquire_deepxiv.py`

- [ ] **Step 1: Add optional dependency to pyproject.toml**

In `pyproject.toml`, add the `deepxiv` extra:

```toml
[project.optional-dependencies]
litellm = ["litellm>=1.40"]
deepxiv = ["deepxiv-sdk>=0.2.0"]
```

- [ ] **Step 2: Write failing tests for `search_deepxiv`**

Append to `tests/test_acquire_deepxiv.py`:

```python
from unittest.mock import MagicMock, patch


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

    with patch("lens.acquire.deepxiv.Reader", return_value=mock_reader):
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

    with patch("lens.acquire.deepxiv.Reader", return_value=mock_reader):
        from lens.acquire.deepxiv import search_deepxiv

        papers = search_deepxiv(query="nonexistent topic xyz")

    assert papers == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_acquire_deepxiv.py::test_search_deepxiv_maps_results_to_paper_dicts -v`
Expected: FAIL — `lens.acquire.deepxiv` does not exist

- [ ] **Step 4: Implement `src/lens/acquire/deepxiv.py`**

```python
"""DeepXiv paper search and retrieval (optional dependency).

Requires: uv sync --extra deepxiv
"""

from __future__ import annotations

import logging
import re
from typing import Any

from lens.acquire.quality import quality_score

logger = logging.getLogger(__name__)

try:
    from deepxiv_sdk import Reader  # ty: ignore[unresolved-import]

    HAS_DEEPXIV = True
except ImportError:
    HAS_DEEPXIV = False

    class Reader:  # type: ignore[no-redef]
        """Placeholder when deepxiv-sdk is not installed."""


def _get_reader() -> Reader:
    """Create a Reader instance. Raises RuntimeError if deepxiv-sdk is not installed."""
    if not HAS_DEEPXIV:
        raise RuntimeError(
            "deepxiv-sdk is not installed. Run: uv sync --extra deepxiv"
        )
    return Reader()


def _extract_date(publish_at: str | None) -> str:
    """Extract YYYY-MM-DD from a datetime string."""
    if not publish_at:
        return "1970-01-01"
    match = re.match(r"(\d{4}-\d{2}-\d{2})", publish_at)
    return match.group(1) if match else publish_at[:10]


def _extract_authors(authors: list[dict | str] | None) -> list[str]:
    """Extract author name strings from DeepXiv author data."""
    if not authors:
        return []
    names = []
    for a in authors:
        if isinstance(a, dict):
            name = a.get("name", "")
        else:
            name = str(a)
        if name:
            names.append(name)
    return names


def search_deepxiv(
    query: str,
    categories: list[str] | None = None,
    since: str | None = None,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Search DeepXiv and return LENS Paper-shaped dicts.

    Parameters
    ----------
    query:
        Search query string.
    categories:
        Optional arXiv category filters (e.g. ["cs.AI", "cs.CL"]).
    since:
        Only papers after this date (YYYY-MM-DD).
    max_results:
        Maximum number of papers to return.
    """
    reader = _get_reader()
    kwargs: dict[str, Any] = {
        "query": query,
        "size": max_results,
        "search_mode": "hybrid",
    }
    if categories:
        kwargs["categories"] = categories
    if since:
        kwargs["date_from"] = since

    response = reader.search(**kwargs)
    results = response.get("results", [])

    papers = []
    for r in results:
        date = _extract_date(r.get("publish_at") or r.get("published"))
        citations = r.get("citation", 0) or 0
        venue = r.get("venue") or r.get("journal_name")
        papers.append(
            {
                "paper_id": r["arxiv_id"],
                "arxiv_id": r["arxiv_id"],
                "title": r.get("title", ""),
                "abstract": r.get("abstract", ""),
                "authors": _extract_authors(r.get("authors")),
                "date": date,
                "venue": venue,
                "citations": citations,
                "quality_score": quality_score(citations, venue, date),
                "extraction_status": "pending",
            }
        )
    return papers


def fetch_deepxiv_paper(arxiv_id: str) -> dict[str, Any]:
    """Fetch a single paper with rich metadata via DeepXiv brief().

    Returns a LENS Paper-shaped dict with keywords and github_url populated.
    """
    reader = _get_reader()
    data = reader.brief(arxiv_id)

    date = _extract_date(data.get("publish_at"))
    citations = data.get("citations", 0) or 0
    venue = data.get("venue") or data.get("journal_name")

    return {
        "paper_id": arxiv_id,
        "arxiv_id": arxiv_id,
        "title": data.get("title", ""),
        "abstract": data.get("abstract", ""),
        "authors": _extract_authors(data.get("authors")),
        "date": date,
        "venue": venue,
        "citations": citations,
        "quality_score": quality_score(citations, venue, date),
        "extraction_status": "pending",
        "keywords": data.get("keywords") or [],
        "github_url": data.get("github_url"),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_acquire_deepxiv.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Write test for `fetch_deepxiv_paper`**

Append to `tests/test_acquire_deepxiv.py`:

```python
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

    with patch("lens.acquire.deepxiv.Reader", return_value=mock_reader):
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

    with patch("lens.acquire.deepxiv.Reader", return_value=mock_reader):
        from lens.acquire.deepxiv import fetch_deepxiv_paper

        paper = fetch_deepxiv_paper("2507.01701")

    assert paper["keywords"] == []
    assert paper["github_url"] is None
```

- [ ] **Step 7: Run tests to verify**

Run: `uv run pytest tests/test_acquire_deepxiv.py -v`
Expected: All 7 tests PASS

- [ ] **Step 8: Update `__init__.py` with conditional exports**

In `src/lens/acquire/__init__.py`, add:

```python
"""LENS paper acquisition pipeline."""

from lens.acquire.arxiv import fetch_arxiv_papers
from lens.acquire.openalex import enrich_with_openalex
from lens.acquire.pdf import ingest_pdf
from lens.acquire.quality import quality_score
from lens.acquire.seed import acquire_seed, load_seed_manifest
from lens.acquire.semantic_scholar import fetch_embedding, fetch_embeddings_batch

__all__ = [
    "acquire_seed",
    "enrich_with_openalex",
    "ingest_pdf",
    "fetch_arxiv_papers",
    "fetch_embedding",
    "fetch_embeddings_batch",
    "load_seed_manifest",
    "quality_score",
]

try:
    from lens.acquire.deepxiv import (
        HAS_DEEPXIV,
        fetch_deepxiv_paper,
        search_deepxiv,
    )

    __all__ += ["HAS_DEEPXIV", "search_deepxiv", "fetch_deepxiv_paper"]
except ImportError:
    pass
```

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/lens/acquire/deepxiv.py src/lens/acquire/__init__.py tests/test_acquire_deepxiv.py
git commit -m "feat: add DeepXiv acquire module as optional dependency"
```

---

### Task 3: CLI subcommand

**Files:**
- Modify: `src/lens/cli.py:652` (after openalex command)
- Test: `tests/test_acquire_deepxiv.py`

- [ ] **Step 1: Write failing test for CLI availability guard**

Append to `tests/test_acquire_deepxiv.py`:

```python
from typer.testing import CliRunner
from lens.cli import app

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
        "data_dir: " + str(tmp_path) + "\n"
        "acquire:\n"
        "  arxiv_categories: [cs.AI]\n"
    )
    monkeypatch.setenv("LENS_CONFIG", str(config_path))

    with patch("lens.acquire.deepxiv.Reader", return_value=mock_reader):
        with patch("lens.acquire.deepxiv.HAS_DEEPXIV", True):
            result = runner.invoke(app, ["acquire", "deepxiv", "multi-agent"])

    assert result.exit_code == 0
    assert "1" in result.output


def test_cli_deepxiv_not_installed(tmp_path, monkeypatch):
    """lens acquire deepxiv should print helpful error when not installed."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data_dir: " + str(tmp_path) + "\n"
        "acquire:\n"
        "  arxiv_categories: [cs.AI]\n"
    )
    monkeypatch.setenv("LENS_CONFIG", str(config_path))

    with patch("lens.acquire.deepxiv.HAS_DEEPXIV", False):
        result = runner.invoke(app, ["acquire", "deepxiv", "multi-agent"])

    assert result.exit_code == 1
    assert "deepxiv-sdk" in result.output.lower() or "not installed" in result.output.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_acquire_deepxiv.py::test_cli_deepxiv_search -v`
Expected: FAIL — no `deepxiv` subcommand in CLI

- [ ] **Step 3: Add `deepxiv` CLI command to `cli.py`**

Add this after the `openalex` command (after line 704 in `src/lens/cli.py`):

```python
@acquire_app.command()
def deepxiv(
    query: str = typer.Argument(None, help="Search query for DeepXiv."),
    paper: str | None = typer.Option(
        None, "--paper", help="Fetch single paper by arXiv ID."
    ),
    since: str | None = typer.Option(
        None, "--since", help="Only papers after this date (YYYY-MM-DD)."
    ),
    max_results: int = typer.Option(20, "--max-results", help="Maximum papers to fetch."),
    categories: str | None = typer.Option(
        None, "--categories", help="Comma-separated arXiv categories (e.g. cs.AI,cs.CL)."
    ),
) -> None:
    """Search and fetch papers via DeepXiv (requires deepxiv-sdk)."""
    from lens.acquire.deepxiv import HAS_DEEPXIV

    if not HAS_DEEPXIV:
        rprint("[red]deepxiv-sdk not installed. Run: uv sync --extra deepxiv[/red]")
        raise typer.Exit(code=1)

    if not query and not paper:
        rprint("[red]Provide a search query or --paper ARXIV_ID[/red]")
        raise typer.Exit(code=1)

    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()
    session_id = str(uuid4())[:8]

    if paper:
        from lens.acquire.deepxiv import fetch_deepxiv_paper

        paper_data = fetch_deepxiv_paper(paper)

        existing = store.query("papers", "paper_id = ?", (paper_data["paper_id"],))
        if existing:
            rprint(f"[yellow]Paper '{paper}' already exists. Skipping.[/yellow]")
            return

        paper_data["embedding"] = [0.0] * EMBEDDING_DIM
        store.add_papers([paper_data])
        log_event(
            store,
            "ingest",
            "paper.added",
            target_type="paper",
            target_id=paper_data["paper_id"],
            detail={"title": paper_data["title"], "source": "deepxiv"},
            session_id=session_id,
        )
        rprint(f"[green]Acquired paper {paper} via DeepXiv[/green]")
    else:
        from lens.acquire.deepxiv import search_deepxiv

        cat_list = [c.strip() for c in categories.split(",")] if categories else None
        papers = search_deepxiv(
            query=query,
            categories=cat_list,
            since=since,
            max_results=max_results,
        )

        if not papers:
            rprint("[yellow]No papers found[/yellow]")
            return

        for p in papers:
            if "embedding" not in p:
                p["embedding"] = [0.0] * EMBEDDING_DIM

        store.add_papers(papers)
        for p in papers:
            log_event(
                store,
                "ingest",
                "paper.added",
                target_type="paper",
                target_id=p["paper_id"],
                detail={"title": p["title"], "source": "deepxiv"},
                session_id=session_id,
            )
        rprint(f"[green]Acquired {len(papers)} papers via DeepXiv[/green]")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_acquire_deepxiv.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `uv run pytest -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/lens/cli.py tests/test_acquire_deepxiv.py
git commit -m "feat: add 'lens acquire deepxiv' CLI subcommand"
```

---

### Task 4: Integration test

**Files:**
- Test: `tests/test_acquire_deepxiv.py`

- [ ] **Step 1: Write an integration test marked with `@pytest.mark.integration`**

Append to `tests/test_acquire_deepxiv.py`:

```python
import pytest


@pytest.mark.integration
def test_deepxiv_search_live():
    """Integration test: search DeepXiv for real papers."""
    from lens.acquire.deepxiv import HAS_DEEPXIV, search_deepxiv

    if not HAS_DEEPXIV:
        pytest.skip("deepxiv-sdk not installed")

    papers = search_deepxiv(query="transformer attention", max_results=3)
    assert len(papers) > 0
    p = papers[0]
    assert p["paper_id"]
    assert p["title"]
    assert p["date"]
    assert isinstance(p["quality_score"], float)


@pytest.mark.integration
def test_deepxiv_fetch_paper_live():
    """Integration test: fetch a known paper via DeepXiv."""
    from lens.acquire.deepxiv import HAS_DEEPXIV, fetch_deepxiv_paper

    if not HAS_DEEPXIV:
        pytest.skip("deepxiv-sdk not installed")

    paper = fetch_deepxiv_paper("2507.01701")
    assert paper["paper_id"] == "2507.01701"
    assert paper["title"]
    assert isinstance(paper["keywords"], list)
```

- [ ] **Step 2: Run the integration test to verify it passes**

Run: `uv run pytest tests/test_acquire_deepxiv.py -m integration -v`
Expected: 2 tests PASS (or SKIP if deepxiv-sdk is not installed)

- [ ] **Step 3: Run the full test suite excluding integration tests**

Run: `uv run pytest -m "not integration" -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_acquire_deepxiv.py
git commit -m "test: add integration tests for DeepXiv acquire"
```

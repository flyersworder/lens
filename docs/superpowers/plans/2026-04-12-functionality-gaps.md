# LENS Functionality Gaps — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 10 identified functionality gaps so LENS is fully operational end-to-end.

**Architecture:** All changes are additive to existing modules. The main files touched are `cli.py` (new commands + helpers), `monitor/watcher.py` (pipeline stages), and test files. No new modules are created.

**Tech Stack:** Python 3.12+, Typer CLI, SQLite + sqlite-vec, pytest, asyncio

---

### Task 1: Fix `year` Display Bug in `explore paper`

**Files:**
- Modify: `src/lens/cli.py:1196-1200`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_explore_paper_shows_date(tmp_path, sample_paper_data):
    """lens explore paper should display the paper date, not a missing 'year' field."""
    from typer.testing import CliRunner
    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.add_papers([sample_paper_data])

    result = runner.invoke(
        app,
        ["explore", "paper", "2401.12345"],
        env={"LENS_DATA_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0
    assert "2017-06-12" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_explore_paper_shows_date -v`
Expected: FAIL — the current code reads `result.get('year')` which is `None`, so no date is printed.

- [ ] **Step 3: Fix the bug**

In `src/lens/cli.py`, replace the year display block (around line 1196-1200):

```python
# Old:
    if result.get("authors"):
        rprint(f"[dim]Authors:[/dim] {result['authors']}")
    if result.get("year"):
        rprint(f"[dim]Year:[/dim] {result['year']}")

# New:
    if result.get("authors"):
        rprint(f"[dim]Authors:[/dim] {result['authors']}")
    if result.get("date"):
        rprint(f"[dim]Date:[/dim] {result['date']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py::test_explore_paper_shows_date -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/lens/cli.py tests/test_cli.py
git commit -m "fix: display paper date instead of nonexistent year field"
```

---

### Task 2: Remove `--interval` No-Op from Monitor

**Files:**
- Modify: `src/lens/cli.py:364-366`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_cli.py`:

```python
def test_monitor_no_interval_flag():
    """Monitor command should not have an --interval flag."""
    from typer.testing import CliRunner
    from lens.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["monitor", "--help"])
    assert result.exit_code == 0
    assert "--interval" not in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_monitor_no_interval_flag -v`
Expected: FAIL — `--interval` is currently present in help output.

- [ ] **Step 3: Remove the parameter**

In `src/lens/cli.py`, change the `monitor` function signature from:

```python
@app.command()
def monitor(
    interval: str = typer.Option("weekly", "--interval", help="Check interval (not yet used)."),
    trending: bool = typer.Option(False, "--trending", help="Show ideation gaps."),
) -> None:
```

to:

```python
@app.command()
def monitor(
    trending: bool = typer.Option(False, "--trending", help="Show ideation gaps."),
) -> None:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py::test_monitor_no_interval_flag -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/lens/cli.py tests/test_cli.py
git commit -m "fix: remove unused --interval flag from monitor command"
```

---

### Task 3: Add `--verbose` / `-v` Logging Control

**Files:**
- Modify: `src/lens/cli.py` (add callback near top)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_verbose_flag_accepted():
    """The -v flag should be accepted without error."""
    from typer.testing import CliRunner
    from lens.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["-v", "--help"])
    assert result.exit_code == 0


def test_verbose_flag_in_help():
    """The --verbose flag should appear in top-level help."""
    from typer.testing import CliRunner
    from lens.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.output or "-v" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py::test_verbose_flag_accepted tests/test_cli.py::test_verbose_flag_in_help -v`
Expected: FAIL — no such option exists yet.

- [ ] **Step 3: Add the verbose callback**

In `src/lens/cli.py`, add a callback to the main `app` after the app definition (around line 23):

```python
import logging

@app.callback()
def main(
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase log verbosity (-v=INFO, -vv=DEBUG)."
    ),
) -> None:
    """LENS — LLM Engineering Navigation System."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
```

Also remove the `help` string from the `app = typer.Typer(...)` call since the callback docstring replaces it:

```python
app = typer.Typer(name="lens")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py::test_verbose_flag_accepted tests/test_cli.py::test_verbose_flag_in_help -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All pass (the existing help test `test_cli_help` checks for "LENS" which is still in the callback docstring)

- [ ] **Step 6: Commit**

```bash
git add src/lens/cli.py tests/test_cli.py
git commit -m "feat: add --verbose / -v flag for logging control"
```

---

### Task 4: Add API Key Validation Helper

**Files:**
- Modify: `src/lens/cli.py` (new helper + wire into commands)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_extract_without_api_key_shows_error(tmp_path, monkeypatch):
    """lens extract should fail early with a clear message when no API key is set."""
    from typer.testing import CliRunner
    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LENS_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.conn.close()

    result = runner.invoke(app, ["extract"])
    assert result.exit_code == 1
    assert "API key not configured" in result.output


def test_analyze_without_api_key_shows_error(tmp_path, monkeypatch):
    """lens analyze should fail early with a clear message when no API key is set."""
    from typer.testing import CliRunner
    from lens.cli import app

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LENS_CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result = runner.invoke(app, ["analyze", "test query"])
    assert result.exit_code == 1
    assert "API key not configured" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py::test_extract_without_api_key_shows_error tests/test_cli.py::test_analyze_without_api_key_shows_error -v`
Expected: FAIL — commands currently try to call the LLM without checking for a key first.

- [ ] **Step 3: Add the helper and wire it in**

Add helper to `src/lens/cli.py` after `_embedding_kwargs()` (around line 148):

```python
def _require_llm_config(config: dict) -> None:
    """Exit early with a clear message if no LLM API key is configured."""
    llm_cfg = config.get("llm", {})
    api_key = llm_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        rprint(
            "[red]LLM API key not configured.[/red]\n"
            "Set it with: [bold]lens config set llm.api_key YOUR_KEY[/bold]\n"
            "Or export: [bold]export OPENROUTER_API_KEY=YOUR_KEY[/bold]"
        )
        raise typer.Exit(code=1)
```

Then add `_require_llm_config(config)` as the first call after `config = load_config(...)` in these commands:

- `extract` (around line 344, after `config = load_config(...)`)
- `analyze` (around line 176, after `config = load_config(...)`)
- `explain` (after `config = load_config(...)`)
- `monitor` (after `config = load_config(...)`, but only in the non-`--trending` branch — i.e., after the `if trending:` block returns)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py::test_extract_without_api_key_shows_error tests/test_cli.py::test_analyze_without_api_key_shows_error -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All pass. Existing tests that invoke `extract`/`analyze` either mock the LLM or set env vars — verify no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/lens/cli.py tests/test_cli.py
git commit -m "feat: validate LLM API key early in extract, analyze, explain, monitor"
```

---

### Task 5: Add `acquire semantic` CLI Command

**Files:**
- Modify: `src/lens/cli.py` (new command under `acquire_app`)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_acquire_semantic_help():
    """acquire semantic subcommand should exist."""
    from typer.testing import CliRunner
    from lens.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["acquire", "semantic", "--help"])
    assert result.exit_code == 0
    assert "SPECTER2" in result.output or "Semantic Scholar" in result.output


def test_acquire_semantic_no_papers(tmp_path, monkeypatch):
    """acquire semantic with no papers should print a message."""
    from typer.testing import CliRunner
    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.conn.close()

    result = runner.invoke(app, ["acquire", "semantic"])
    assert result.exit_code == 0
    assert "No papers" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py::test_acquire_semantic_help tests/test_cli.py::test_acquire_semantic_no_papers -v`
Expected: FAIL — command doesn't exist yet.

- [ ] **Step 3: Implement the command**

Add to `src/lens/cli.py` after the `deepxiv` command (around line 873):

```python
@acquire_app.command()
def semantic(
    paper_id: str | None = typer.Option(
        None, "--paper-id", help="Fetch SPECTER2 embedding for a specific paper."
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="Semantic Scholar API key."
    ),
) -> None:
    """Fetch SPECTER2 embeddings from Semantic Scholar."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    if paper_id:
        papers = store.query("papers", "paper_id = ?", (paper_id,))
    else:
        papers = store.query("papers")

    if not papers:
        rprint("[yellow]No papers to fetch embeddings for.[/yellow]")
        return

    # Filter to papers with zero-vector or missing embeddings
    if not paper_id:
        papers_needing_embeddings = []
        for p in papers:
            pid = p["paper_id"]
            vec_row = store.query_sql(
                "SELECT embedding FROM papers_vec WHERE paper_id = ?", (pid,)
            )
            if not vec_row:
                papers_needing_embeddings.append(p)
                continue
            # Check if embedding is all zeros by summing bytes
            import struct
            emb_bytes = vec_row[0]["embedding"]
            emb = struct.unpack(f"{EMBEDDING_DIM}f", emb_bytes)
            if all(v == 0.0 for v in emb):
                papers_needing_embeddings.append(p)
        papers = papers_needing_embeddings

    if not papers:
        rprint("[yellow]All papers already have embeddings.[/yellow]")
        return

    arxiv_ids = [p.get("arxiv_id", p["paper_id"]) for p in papers]
    pid_by_arxiv = {p.get("arxiv_id", p["paper_id"]): p["paper_id"] for p in papers}

    rprint(f"Fetching SPECTER2 embeddings for {len(arxiv_ids)} paper(s)...")

    from lens.acquire.semantic_scholar import fetch_embeddings_batch

    results = asyncio.run(fetch_embeddings_batch(arxiv_ids, api_key=api_key))

    session_id = str(uuid4())[:8]
    updated = 0
    for arxiv_id, embedding in results.items():
        if embedding is not None:
            pid = pid_by_arxiv.get(arxiv_id, arxiv_id)
            store.upsert_embedding("papers", pid, embedding)
            log_event(
                store,
                "ingest",
                "paper.embedding_updated",
                target_type="paper",
                target_id=pid,
                detail={"source": "semantic_scholar"},
                session_id=session_id,
            )
            updated += 1

    rprint(f"[green]Updated {updated} paper embeddings[/green]")
    if updated < len(arxiv_ids):
        rprint(f"[yellow]{len(arxiv_ids) - updated} papers had no SPECTER2 embedding available[/yellow]")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py::test_acquire_semantic_help tests/test_cli.py::test_acquire_semantic_no_papers -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/lens/cli.py tests/test_cli.py
git commit -m "feat: add acquire semantic command for SPECTER2 embeddings"
```

---

### Task 6: Wire Quality Score Computation

**Files:**
- Modify: `src/lens/cli.py` (add quality scoring to `acquire openalex` and `acquire seed`)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_acquire_seed_computes_quality_score(tmp_path, monkeypatch):
    """acquire seed should compute quality_score for papers with citations/venue."""
    from typer.testing import CliRunner
    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LENS_CONFIG_PATH", str(tmp_path / "config.yaml"))

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.conn.close()

    result = runner.invoke(app, ["acquire", "seed"])
    assert result.exit_code == 0

    store = LensStore(db_path)
    store.init_tables()
    papers = store.query("papers")
    # At least some seed papers should have a non-None quality_score
    scored = [p for p in papers if p.get("quality_score") is not None]
    assert len(scored) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_acquire_seed_computes_quality_score -v`
Expected: FAIL — seed papers currently don't get quality scores computed.

- [ ] **Step 3: Add quality scoring to seed and openalex commands**

In `src/lens/cli.py`, in the `seed()` command, after `count = asyncio.run(...)`, add quality scoring:

```python
    # Compute quality scores for papers that have citation/venue data
    if count > 0:
        from lens.acquire.quality import quality_score as compute_quality

        all_papers = store.query("papers")
        venue_tiers = config["acquire"].get("quality_venue_tiers")
        for p in all_papers:
            if p.get("quality_score") is not None:
                continue
            score = compute_quality(
                citations=p.get("citations", 0) or 0,
                venue=p.get("venue"),
                paper_date=p.get("date", "2020-01-01"),
                venue_tiers=venue_tiers,
            )
            store.update(
                "papers",
                "quality_score = ?",
                "paper_id = ?",
                (score, p["paper_id"]),
            )
```

In the `openalex()` command, after the enrichment loop that updates citations and venue, add quality scoring:

```python
    # Recompute quality scores now that we have citations and venue
    from lens.acquire.quality import quality_score as compute_quality

    venue_tiers = config["acquire"].get("quality_venue_tiers")
    for paper in enriched:
        pid = paper.get("paper_id", "")
        score = compute_quality(
            citations=paper.get("citations", 0) or 0,
            venue=paper.get("venue"),
            paper_date=paper.get("date", "2020-01-01"),
            venue_tiers=venue_tiers,
        )
        store.update("papers", "quality_score = ?", "paper_id = ?", (score, pid))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py::test_acquire_seed_computes_quality_score -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/lens/cli.py tests/test_cli.py
git commit -m "feat: compute quality scores after seed acquisition and OpenAlex enrichment"
```

---

### Task 7: Enhance Monitor Pipeline

**Files:**
- Modify: `src/lens/monitor/watcher.py`
- Test: `tests/test_monitor.py`

- [ ] **Step 1: Read current test file**

Read `tests/test_monitor.py` to understand existing test patterns.

- [ ] **Step 2: Write failing tests**

Add to `tests/test_monitor.py`:

```python
@pytest.mark.asyncio
async def test_monitor_cycle_with_build(tmp_path, monkeypatch):
    """Monitor cycle with run_build=True should call build_vocabulary and build_matrix."""
    from unittest.mock import AsyncMock

    from lens.monitor.watcher import run_monitor_cycle
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    mock_llm = AsyncMock()

    # Patch arxiv fetch to return nothing (no new papers)
    async def fake_fetch(**kwargs):
        return []

    monkeypatch.setattr("lens.monitor.watcher.fetch_arxiv_papers", fake_fetch)

    # Add seed vocab so build has something to work with
    from lens.taxonomy.vocabulary import load_seed_vocabulary
    load_seed_vocabulary(store)

    result = await run_monitor_cycle(
        store,
        mock_llm,
        run_build=True,
        run_ideation_flag=False,
    )
    assert "taxonomy_built" in result
    assert result["taxonomy_built"] is True


@pytest.mark.asyncio
async def test_monitor_cycle_skip_build(tmp_path, monkeypatch):
    """Monitor cycle with run_build=False should skip taxonomy and matrix build."""
    from unittest.mock import AsyncMock

    from lens.monitor.watcher import run_monitor_cycle
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    mock_llm = AsyncMock()

    async def fake_fetch(**kwargs):
        return []

    monkeypatch.setattr("lens.monitor.watcher.fetch_arxiv_papers", fake_fetch)

    result = await run_monitor_cycle(
        store,
        mock_llm,
        run_build=False,
        run_ideation_flag=False,
    )
    assert result["taxonomy_built"] is False
    assert result["matrix_built"] is False


@pytest.mark.asyncio
async def test_monitor_cycle_skip_enrich(tmp_path, monkeypatch):
    """Monitor cycle with run_enrich=False should skip OpenAlex enrichment."""
    from unittest.mock import AsyncMock

    from lens.monitor.watcher import run_monitor_cycle
    from lens.store.models import EMBEDDING_DIM
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    mock_llm = AsyncMock()

    paper = {
        "paper_id": "test-001",
        "title": "Test Paper",
        "abstract": "Test abstract.",
        "authors": ["Author"],
        "date": "2025-01-01",
        "arxiv_id": "2501.00001",
        "extraction_status": "pending",
        "embedding": [0.0] * EMBEDDING_DIM,
    }

    async def fake_fetch(**kwargs):
        return [paper]

    monkeypatch.setattr("lens.monitor.watcher.fetch_arxiv_papers", fake_fetch)

    # Mock extract_papers to avoid LLM calls
    async def fake_extract(store, client, concurrency=3, session_id=None):
        return 0

    monkeypatch.setattr("lens.monitor.watcher.extract_papers", fake_extract)

    result = await run_monitor_cycle(
        store,
        mock_llm,
        run_enrich=False,
        run_build=False,
        run_ideation_flag=False,
    )
    assert result["papers_enriched"] == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_monitor.py -v -k "build or enrich"`
Expected: FAIL — `run_monitor_cycle()` doesn't accept `run_build` or `run_enrich` yet.

- [ ] **Step 4: Implement the enhanced pipeline**

Replace the content of `src/lens/monitor/watcher.py`:

```python
"""Monitor pipeline: acquire new papers -> enrich -> extract -> build -> ideate.

Runs one monitoring cycle with configurable stages.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from lens.acquire.arxiv import fetch_arxiv_papers
from lens.extract.extractor import extract_papers
from lens.knowledge.events import log_event
from lens.llm.client import LLMClient
from lens.store.models import EMBEDDING_DIM
from lens.store.store import LensStore

logger = logging.getLogger(__name__)


async def run_monitor_cycle(
    store: LensStore,
    llm_client: LLMClient,
    query: str = "LLM",
    categories: list[str] | None = None,
    max_results: int = 50,
    run_enrich: bool = True,
    run_build: bool = True,
    run_ideation_flag: bool = True,
    ideate_with_llm: bool = False,
    openalex_mailto: str = "",
    embedding_kwargs: dict[str, Any] | None = None,
    venue_tiers: dict[str, list[str]] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run one monitoring cycle with configurable stages.

    Stages:
    1. Acquire — fetch new papers from arxiv
    2. Enrich — OpenAlex metadata (if run_enrich and openalex_mailto)
    3. Extract — LLM knowledge extraction
    4. Build — taxonomy + matrix rebuild (if run_build)
    5. Ideate — gap analysis (if run_ideation_flag)
    """
    cats = categories or ["cs.CL", "cs.LG", "cs.AI"]
    session_id = session_id or str(uuid4())[:8]

    # --- Stage 1: Acquire ---
    try:
        papers = await fetch_arxiv_papers(
            query=query,
            categories=cats,
            max_results=max_results,
        )
    except Exception:
        logger.warning("Failed to fetch papers from arxiv")
        papers = []

    existing = store.query("papers")
    existing_ids = {p["paper_id"] for p in existing}
    new_papers = [p for p in papers if p["paper_id"] not in existing_ids]

    for p in new_papers:
        if "embedding" not in p:
            p["embedding"] = [0.0] * EMBEDDING_DIM

    papers_acquired = len(new_papers)
    if new_papers:
        store.add_papers(new_papers)
        logger.info("Acquired %d new papers", papers_acquired)

    # --- Stage 2: Enrich ---
    papers_enriched = 0
    if run_enrich and openalex_mailto and papers_acquired > 0:
        try:
            from lens.acquire.openalex import enrich_with_openalex
            from lens.acquire.quality import quality_score as compute_quality

            papers_for_enrich = [
                {k: v for k, v in p.items() if k != "embedding"} for p in new_papers
            ]
            enriched = await enrich_with_openalex(papers_for_enrich, mailto=openalex_mailto)

            for paper in enriched:
                pid = paper.get("paper_id", "")
                store.update(
                    "papers",
                    "citations = ?, venue = ?",
                    "paper_id = ?",
                    (paper.get("citations", 0), paper.get("venue"), pid),
                )
                score = compute_quality(
                    citations=paper.get("citations", 0) or 0,
                    venue=paper.get("venue"),
                    paper_date=paper.get("date", "2020-01-01"),
                    venue_tiers=venue_tiers,
                )
                store.update("papers", "quality_score = ?", "paper_id = ?", (score, pid))
                papers_enriched += 1
            logger.info("Enriched %d papers via OpenAlex", papers_enriched)
        except Exception:
            logger.warning("OpenAlex enrichment failed, continuing without enrichment")

    # --- Stage 3: Extract ---
    papers_extracted = 0
    if papers_acquired > 0:
        papers_extracted = await extract_papers(
            store, llm_client, concurrency=3, session_id=session_id
        )

    # --- Stage 4: Build ---
    taxonomy_built = False
    matrix_built = False
    if run_build:
        try:
            from lens.knowledge.matrix import build_matrix
            from lens.taxonomy import get_next_version, record_version
            from lens.taxonomy.vocabulary import build_vocabulary

            emb_kw = embedding_kwargs or {}
            build_vocabulary(
                store,
                embedding_provider=emb_kw.get("provider", "local"),
                embedding_model=emb_kw.get("model"),
                embedding_api_base=emb_kw.get("api_base"),
                embedding_api_key=emb_kw.get("api_key"),
                session_id=session_id,
            )
            taxonomy_built = True

            build_matrix(store, session_id=session_id)
            matrix_built = True

            version_id = get_next_version(store)
            paper_count = len(store.query("papers"))
            vocab = store.query("vocabulary")
            record_version(
                store,
                version_id,
                paper_count=paper_count,
                param_count=len([v for v in vocab if v["kind"] == "parameter"]),
                principle_count=len([v for v in vocab if v["kind"] == "principle"]),
                slot_count=len([v for v in vocab if v["kind"] == "arch_slot"]),
                variant_count=0,
                pattern_count=len([v for v in vocab if v["kind"] == "agentic_category"]),
                session_id=session_id,
            )
            logger.info("Built taxonomy + matrix")
        except Exception:
            logger.warning("Taxonomy/matrix build failed", exc_info=True)

    # --- Stage 5: Ideate ---
    ideation_report = None
    if run_ideation_flag:
        try:
            if ideate_with_llm:
                from lens.monitor.ideation import run_ideation_with_llm

                ideation_report = await run_ideation_with_llm(store, llm_client)
            else:
                from lens.monitor.ideation import run_ideation

                ideation_report = run_ideation(store)
        except Exception:
            logger.warning("Ideation failed", exc_info=True)

    return {
        "papers_acquired": papers_acquired,
        "papers_enriched": papers_enriched,
        "papers_extracted": papers_extracted,
        "taxonomy_built": taxonomy_built,
        "matrix_built": matrix_built,
        "ideation_report": ideation_report,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: All pass (both new and existing tests — existing tests use default parameter values)

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add src/lens/monitor/watcher.py tests/test_monitor.py
git commit -m "feat: enhance monitor pipeline with enrich, build, and LLM ideation stages"
```

---

### Task 8: Wire Enhanced Monitor to CLI

**Files:**
- Modify: `src/lens/cli.py` (update `monitor` command)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_monitor_has_skip_flags():
    """Monitor should accept --skip-enrich and --skip-build flags."""
    from typer.testing import CliRunner
    from lens.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["monitor", "--help"])
    assert result.exit_code == 0
    assert "--skip-enrich" in result.output
    assert "--skip-build" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_monitor_has_skip_flags -v`
Expected: FAIL — flags don't exist yet.

- [ ] **Step 3: Update the monitor CLI command**

In `src/lens/cli.py`, replace the `monitor` command (using the signature from Task 2 with `interval` already removed):

```python
@app.command()
def monitor(
    trending: bool = typer.Option(False, "--trending", help="Show ideation gaps."),
    skip_enrich: bool = typer.Option(
        False, "--skip-enrich", help="Skip OpenAlex enrichment stage."
    ),
    skip_build: bool = typer.Option(
        False, "--skip-build", help="Skip taxonomy and matrix rebuild."
    ),
) -> None:
    """Run one monitoring cycle: acquire -> enrich -> extract -> build -> ideate."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    if trending:
        gaps = store.query("ideation_gaps")
        if not gaps:
            rprint("[yellow]No ideation gaps found.[/yellow]")
        else:
            rprint("\n[bold]Ideation Gaps:[/bold]")
            gaps.sort(key=lambda x: x.get("score", 0), reverse=True)
            for row in gaps:
                hyp = ""
                if row.get("llm_hypothesis"):
                    h = row["llm_hypothesis"][:80]
                    hyp = f" — {h}..."
                rprint(f"  [{row['gap_type']}] {row['description']}{hyp}")
        raise typer.Exit(code=0)

    _require_llm_config(config)

    from lens.llm.client import LLMClient
    from lens.monitor.watcher import run_monitor_cycle

    client = LLMClient(model=config["llm"]["extract_model"], **_llm_kwargs(config))
    cats = config["acquire"]["arxiv_categories"]
    monitor_cfg = config["monitor"]
    openalex_mailto = config["acquire"].get("openalex_mailto", "")
    session_id = str(uuid4())[:8]

    result = asyncio.run(
        run_monitor_cycle(
            store,
            client,
            categories=cats,
            run_enrich=not skip_enrich,
            run_build=not skip_build,
            run_ideation_flag=monitor_cfg["ideate"],
            ideate_with_llm=monitor_cfg.get("ideate_llm", False),
            openalex_mailto=openalex_mailto,
            embedding_kwargs=_embedding_kwargs(config),
            venue_tiers=config["acquire"].get("quality_venue_tiers"),
            session_id=session_id,
        )
    )
    rprint("[green]Monitor cycle complete:[/green]")
    rprint(f"  Papers acquired: {result['papers_acquired']}")
    if result["papers_enriched"]:
        rprint(f"  Papers enriched: {result['papers_enriched']}")
    rprint(f"  Papers extracted: {result['papers_extracted']}")
    if result["taxonomy_built"]:
        rprint("  Taxonomy + matrix: rebuilt")
    if result.get("ideation_report"):
        rprint(f"  Gaps found: {result['ideation_report']['gap_count']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py::test_monitor_has_skip_flags -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/lens/cli.py tests/test_cli.py
git commit -m "feat: wire enhanced monitor pipeline with --skip-enrich and --skip-build flags"
```

---

### Task 9: Add `lens status` Command

**Files:**
- Modify: `src/lens/cli.py` (new top-level command)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_status_empty_db(tmp_path, monkeypatch):
    """lens status on an empty DB should show zeros gracefully."""
    from typer.testing import CliRunner
    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.conn.close()

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Papers" in result.output
    assert "0" in result.output


def test_status_with_data(tmp_path, sample_paper_data, monkeypatch):
    """lens status with papers should show counts."""
    from typer.testing import CliRunner
    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.add_papers([sample_paper_data])

    from lens.taxonomy.vocabulary import load_seed_vocabulary
    load_seed_vocabulary(store)
    store.conn.close()

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Papers" in result.output
    assert "pending: 1" in result.output
    assert "Vocabulary" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py::test_status_empty_db tests/test_cli.py::test_status_with_data -v`
Expected: FAIL — `status` command doesn't exist yet.

- [ ] **Step 3: Implement the status command**

Add to `src/lens/cli.py` after the `init` command (around line 172):

```python
@app.command()
def status() -> None:
    """Show a summary of the LENS knowledge base."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    db_path = data_dir / "lens.db"
    if not db_path.exists():
        rprint("[yellow]No database found. Run 'lens init' first.[/yellow]")
        raise typer.Exit(code=1)

    store = LensStore(str(db_path))
    store.init_tables()

    # Paper counts by extraction status
    papers = store.query("papers")
    total = len(papers)
    by_status: dict[str, int] = {}
    for p in papers:
        s = p.get("extraction_status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    status_parts = ", ".join(f"{s}: {c}" for s, c in sorted(by_status.items()))

    rprint("\n[bold]LENS Knowledge Base Status[/bold]")
    rprint("=" * 40)
    rprint(f"Papers: {total} ({status_parts})" if status_parts else f"Papers: {total}")

    # Vocabulary counts by kind
    vocab = store.query("vocabulary")
    vocab_by_kind: dict[str, int] = {}
    for v in vocab:
        k = v["kind"]
        vocab_by_kind[k] = vocab_by_kind.get(k, 0) + 1
    if vocab_by_kind:
        vocab_parts = ", ".join(
            f"{c} {k}s" for k, c in sorted(vocab_by_kind.items())
        )
        rprint(f"Vocabulary: {vocab_parts}")
    else:
        rprint("Vocabulary: empty")

    # Matrix
    cells = store.query("matrix_cells")
    if cells:
        total_evidence = sum(c.get("count", 0) for c in cells)
        rprint(f"Matrix: {len(cells)} cells, {total_evidence} total evidence")
    else:
        rprint("Matrix: empty")

    # Top parameters by paper_count
    params = [v for v in vocab if v["kind"] == "parameter" and v["paper_count"] > 0]
    if params:
        params.sort(key=lambda x: x["paper_count"], reverse=True)
        top = params[:5]
        top_str = ", ".join(f"{p['name']} ({p['paper_count']})" for p in top)
        rprint(f"Top parameters: {top_str}")

    # Taxonomy version
    versions = store.query_sql(
        "SELECT * FROM taxonomy_versions ORDER BY id DESC LIMIT 1"
    )
    if versions:
        v = versions[0]
        rprint(f"Taxonomy: v{v['id']} (built: {v.get('created_at', 'unknown')})")
    else:
        rprint("Taxonomy: not built yet")

    # Last event
    events = store.query_sql(
        "SELECT * FROM event_log ORDER BY created_at DESC LIMIT 1"
    )
    if events:
        e = events[0]
        rprint(f"Last event: {e.get('created_at', '?')} ({e.get('kind', '?')})")
    else:
        rprint("Last event: none")

    # Cheap lint checks
    from lens.knowledge.linter import (
        check_missing_embeddings,
        check_orphan_vocabulary,
        check_weak_evidence,
    )

    orphans = check_orphan_vocabulary(store)
    weak = check_weak_evidence(store)
    missing_emb = check_missing_embeddings(store)
    issues = []
    if orphans:
        issues.append(f"{len(orphans)} orphans")
    if weak:
        issues.append(f"{len(weak)} weak evidence")
    if missing_emb:
        issues.append(f"{len(missing_emb)} missing embeddings")
    if issues:
        rprint(f"Issues: {', '.join(issues)}")
    else:
        rprint("Issues: none")

    rprint()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py::test_status_empty_db tests/test_cli.py::test_status_with_data -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/lens/cli.py tests/test_cli.py
git commit -m "feat: add lens status command for knowledge base overview"
```

---

### Task 10: Final Integration Check

**Files:**
- None — validation only

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass (229 original + ~12 new)

- [ ] **Step 2: Verify CLI help reflects all changes**

Run: `uv run lens --help`
Expected: `status` appears in command list.

Run: `uv run lens acquire --help`
Expected: `semantic` appears in subcommands.

Run: `uv run lens monitor --help`
Expected: `--skip-enrich`, `--skip-build` appear. No `--interval`. `-v` is documented.

- [ ] **Step 3: Verify ruff lint passes**

Run: `uv run ruff check src/lens/cli.py src/lens/monitor/watcher.py`
Expected: No errors

- [ ] **Step 4: Commit any fixups if needed**

Only if previous steps revealed issues.

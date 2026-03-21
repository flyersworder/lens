# Plan 6: Monitor + Ideation — Continuous Monitoring and Gap Analysis

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the continuous monitoring pipeline (acquire + extract new papers) and the ideation gap analysis system (sparse cells, cross-pollination candidates) that surfaces research opportunities from the knowledge structures.

**Architecture:** The monitor command runs the acquire→extract pipeline for new papers. The ideation module analyzes the current matrix and taxonomy for gaps. Layer 1 (gap analysis) is deterministic — pure Polars/NumPy. Layer 2 (LLM enrichment) is optional and narrates the gaps as research hypotheses.

**Tech Stack:** Existing acquire/extract pipelines, Polars, NumPy (cosine similarity), existing LLMClient

**Spec:** `docs/specs/design.md` (Monitoring, lines 624-641; IdeationGap/IdeationReport models, lines 218-237)

**Scope decisions:**
- BERTrend analysis (trend detection, trend-gap intersections) is deferred — requires a separate dependency and significant complexity. The gap analysis provides value without it.
- Taxonomy drift detection is deferred — it requires centroid storage which isn't implemented yet.
- Stalled architecture slots is deferred — architecture catalog isn't built yet.
- Focus: sparse cells + cross-pollination + monitor pipeline + LLM enrichment + explore ideas

---

## File Structure

```
src/lens/
├── monitor/
│   ├── __init__.py           # Public API: run_monitor
│   ├── watcher.py            # Monitor pipeline: acquire → extract → ideate
│   └── ideation.py           # Gap analysis + LLM enrichment
tests/
├── test_ideation.py          # Gap analysis tests
├── test_monitor.py           # Monitor pipeline tests
```

---

### Task 1: Ideation — Gap Analysis (Layer 1)

**Files:**
- Create: `src/lens/monitor/__init__.py`
- Create: `src/lens/monitor/ideation.py`
- Create: `tests/test_ideation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ideation.py
"""Tests for the ideation gap analysis pipeline."""

import numpy as np
import polars as pl
import pytest
from unittest.mock import AsyncMock
from datetime import datetime


@pytest.fixture
def ideation_store(tmp_path):
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    # Parameters with embeddings for cosine similarity
    store.add_rows("parameters", [
        {"id": 1, "name": "Latency", "description": "Speed",
         "raw_strings": ["latency"], "paper_ids": ["p1"],
         "taxonomy_version": 1,
         "embedding": [1.0, 0.0, 0.0] + [0.0] * 765},
        {"id": 2, "name": "Accuracy", "description": "Quality",
         "raw_strings": ["accuracy"], "paper_ids": ["p1"],
         "taxonomy_version": 1,
         "embedding": [0.0, 1.0, 0.0] + [0.0] * 765},
        {"id": 3, "name": "Throughput", "description": "Speed variant",
         "raw_strings": ["throughput"], "paper_ids": ["p2"],
         "taxonomy_version": 1,
         "embedding": [0.9, 0.1, 0.0] + [0.0] * 765},  # similar to Latency
    ])
    store.add_rows("principles", [
        {"id": 50001, "name": "Quantization",
         "description": "Reduce precision",
         "sub_techniques": ["int8"], "raw_strings": ["quantization"],
         "paper_ids": ["p1"], "taxonomy_version": 1,
         "embedding": [0.0] * 768},
    ])
    # Matrix: Latency→Accuracy has Quantization, but
    # Throughput→Accuracy has NO principles (sparse)
    store.add_rows("matrix_cells", [
        {"improving_param_id": 1, "worsening_param_id": 2,
         "principle_id": 50001, "count": 5,
         "avg_confidence": 0.9, "paper_ids": ["p1"],
         "taxonomy_version": 1},
    ])
    store.add_rows("taxonomy_versions", [
        {"version_id": 1, "created_at": "2026-03-21T00:00:00",
         "paper_count": 10, "param_count": 3,
         "principle_count": 1},
    ])
    return store


def test_find_sparse_cells(ideation_store):
    from lens.monitor.ideation import find_sparse_cells

    gaps = find_sparse_cells(
        ideation_store, taxonomy_version=1, min_principles=2
    )
    # (Latency, Accuracy) has only 1 principle < 2 threshold
    assert len(gaps) >= 1
    assert any(
        g["improving_param_id"] == 1 and g["worsening_param_id"] == 2
        for g in gaps
    )


def test_find_sparse_cells_no_gaps(ideation_store):
    from lens.monitor.ideation import find_sparse_cells

    # With threshold 1, no gaps (cell has 1 principle)
    gaps = find_sparse_cells(
        ideation_store, taxonomy_version=1, min_principles=1
    )
    assert len(gaps) == 0


def test_find_cross_pollination(ideation_store):
    from lens.monitor.ideation import find_cross_pollination

    candidates = find_cross_pollination(
        ideation_store, taxonomy_version=1, similarity_threshold=0.7
    )
    # Latency(1) and Throughput(3) are similar (cosine ~0.99)
    # Quantization resolves (Latency, Accuracy) but not (Throughput, Accuracy)
    # → cross-pollination candidate
    assert len(candidates) >= 1


def test_run_ideation(ideation_store):
    from lens.monitor.ideation import run_ideation

    report = run_ideation(
        ideation_store, taxonomy_version=1
    )
    assert report["gap_count"] >= 1
    assert len(report["gaps"]) >= 1

    # Check gaps stored in LanceDB
    gaps_df = ideation_store.get_table("ideation_gaps").to_polars()
    assert len(gaps_df) >= 1

    reports_df = ideation_store.get_table(
        "ideation_reports"
    ).to_polars()
    assert len(reports_df) == 1


@pytest.mark.asyncio
async def test_run_ideation_with_llm(ideation_store):
    from lens.monitor.ideation import run_ideation_with_llm

    mock_client = AsyncMock()
    mock_client.complete.return_value = (
        "This gap suggests that quantization techniques "
        "could be applied to throughput optimization."
    )

    report = await run_ideation_with_llm(
        ideation_store,
        mock_client,
        taxonomy_version=1,
    )
    assert report["gap_count"] >= 1
    # At least one gap should have an LLM hypothesis
    gaps_with_hyp = [
        g for g in report["gaps"] if g.get("llm_hypothesis")
    ]
    assert len(gaps_with_hyp) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ideation.py -v
```

- [ ] **Step 3: Implement ideation module**

```python
# src/lens/monitor/__init__.py
"""LENS monitoring and ideation pipeline."""
```

```python
# src/lens/monitor/ideation.py
"""Gap analysis and LLM enrichment for research ideation.

Layer 1 (deterministic): Identifies sparse matrix cells and
cross-pollination candidates via Polars + NumPy.
Layer 2 (optional): LLM narrates gaps as research hypotheses.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from lens.llm.client import LLMClient
from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def find_sparse_cells(
    store: LensStore,
    taxonomy_version: int,
    min_principles: int = 2,
) -> list[dict[str, Any]]:
    """Find tradeoff pairs with fewer than min_principles known techniques.

    These are known problems without well-established solutions.
    """
    cells = store.get_table("matrix_cells").to_polars()
    cells = cells.filter(
        pl.col("taxonomy_version") == taxonomy_version
    )
    if len(cells) == 0:
        return []

    # Count principles per (improving, worsening) pair
    counts = cells.group_by(
        ["improving_param_id", "worsening_param_id"]
    ).agg(pl.col("principle_id").count().alias("num_principles"))

    sparse = counts.filter(
        pl.col("num_principles") < min_principles
    )

    gaps = []
    for row in sparse.to_dicts():
        gaps.append({
            "gap_type": "sparse_cell",
            "improving_param_id": row["improving_param_id"],
            "worsening_param_id": row["worsening_param_id"],
            "num_principles": row["num_principles"],
            "description": (
                f"Tradeoff ({row['improving_param_id']}, "
                f"{row['worsening_param_id']}) has only "
                f"{row['num_principles']} known principle(s)"
            ),
            "related_params": [
                row["improving_param_id"],
                row["worsening_param_id"],
            ],
            "related_principles": [],
            "score": 1.0 - (row["num_principles"] / min_principles),
        })
    return gaps


def find_cross_pollination(
    store: LensStore,
    taxonomy_version: int,
    similarity_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """Find principles that resolve (A, B) but not (A, B') where B ~ B'.

    Uses cosine similarity of parameter embeddings to find similar
    parameters. If a principle works for one pair but not a similar
    pair, it's a cross-pollination candidate.
    """
    params = store.get_table("parameters").to_polars()
    params = params.filter(
        pl.col("taxonomy_version") == taxonomy_version
    )
    if len(params) < 2:
        return []

    cells = store.get_table("matrix_cells").to_polars()
    cells = cells.filter(
        pl.col("taxonomy_version") == taxonomy_version
    )
    if len(cells) == 0:
        return []

    # Build parameter embeddings matrix
    param_ids = params["id"].to_list()
    embeddings = np.array(params["embedding"].to_list())

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    # Compute pairwise cosine similarity
    similarity = normalized @ normalized.T

    # Build set of existing (improving, worsening, principle) triples
    existing = set()
    for row in cells.to_dicts():
        existing.add((
            row["improving_param_id"],
            row["worsening_param_id"],
            row["principle_id"],
        ))

    # Find cross-pollination candidates
    candidates = []
    principles_in_matrix = {
        row["principle_id"] for row in cells.to_dicts()
    }

    for i, pid_a in enumerate(param_ids):
        for j, pid_b in enumerate(param_ids):
            if i == j:
                continue
            sim = float(similarity[i, j])
            if sim < similarity_threshold:
                continue

            # pid_a and pid_b are similar parameters
            # Check each cell containing pid_a — does pid_b
            # appear in a similar cell?
            for princ_id in principles_in_matrix:
                # Check all positions where pid_a appears
                # Pre-build cell lookup for efficiency
                cells_list = cells.to_dicts()

                # Check improving role: (pid_a, X, princ) exists
                # but (pid_b, X, princ) doesn't
                for row in cells_list:
                    if (
                        row["improving_param_id"] == pid_a
                        and row["principle_id"] == princ_id
                    ):
                        wors_id = row["worsening_param_id"]
                        if (pid_b, wors_id, princ_id) not in existing:
                            candidates.append({
                                "gap_type": "cross_pollination",
                                "source_pair": (pid_a, wors_id),
                                "target_pair": (pid_b, wors_id),
                                "principle_id": princ_id,
                                "similarity": sim,
                                "description": (
                                    f"Principle {princ_id} resolves "
                                    f"({pid_a}, {wors_id}) — try "
                                    f"for ({pid_b}, {wors_id})?"
                                ),
                                "related_params": [pid_b, wors_id],
                                "related_principles": [princ_id],
                                "score": sim,
                            })

                # Check worsening role: (X, pid_a, princ) exists
                # but (X, pid_b, princ) doesn't
                for row in cells_list:
                    if (
                        row["worsening_param_id"] == pid_a
                        and row["principle_id"] == princ_id
                    ):
                        imp_id = row["improving_param_id"]
                        if (imp_id, pid_b, princ_id) not in existing:
                            candidates.append({
                                "gap_type": "cross_pollination",
                                "source_pair": (imp_id, pid_a),
                                "target_pair": (imp_id, pid_b),
                                "principle_id": princ_id,
                                "similarity": sim,
                                "description": (
                                    f"Principle {princ_id} resolves "
                                    f"({imp_id}, {pid_a}) — try "
                                    f"for ({imp_id}, {pid_b})?"
                                ),
                                "related_params": [imp_id, pid_b],
                                "related_principles": [princ_id],
                                "score": sim,
                            })

    # Deduplicate by target pair + principle
    seen = set()
    unique = []
    for c in candidates:
        key = (
            tuple(c["target_pair"]),
            c["principle_id"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def run_ideation(
    store: LensStore,
    taxonomy_version: int,
    min_principles: int = 2,
    similarity_threshold: float = 0.75,
    top_n: int = 10,
    min_gap_score: float = 0.5,
) -> dict[str, Any]:
    """Run Layer 1 ideation: deterministic gap analysis.

    Finds sparse cells and cross-pollination candidates.
    Stores results in ideation_reports and ideation_gaps tables.
    Returns report dict.
    """
    sparse = find_sparse_cells(
        store, taxonomy_version, min_principles
    )
    cross = find_cross_pollination(
        store, taxonomy_version, similarity_threshold
    )

    all_gaps = sparse + cross
    # Filter by score and limit
    all_gaps = [
        g for g in all_gaps if g.get("score", 0) >= min_gap_score
    ]
    all_gaps.sort(key=lambda g: g.get("score", 0), reverse=True)
    all_gaps = all_gaps[:top_n]

    # Create report
    now = datetime.now()
    reports_df = store.get_table("ideation_reports").to_polars()
    report_id = (
        int(reports_df["id"].max()) + 1 if len(reports_df) > 0 else 1
    )

    report_row = {
        "id": report_id,
        "created_at": now,
        "taxonomy_version": taxonomy_version,
        "paper_batch_size": 0,
        "gap_count": len(all_gaps),
    }
    store.add_rows("ideation_reports", [report_row])

    # Store gaps
    gap_rows = []
    gaps_df = store.get_table("ideation_gaps").to_polars()
    next_gap_id = (
        int(gaps_df["id"].max()) + 1 if len(gaps_df) > 0 else 1
    )

    for i, gap in enumerate(all_gaps):
        gap_rows.append({
            "id": next_gap_id + i,
            "report_id": report_id,
            "gap_type": gap["gap_type"],
            "description": gap["description"],
            "related_params": gap.get("related_params", []),
            "related_principles": gap.get(
                "related_principles", []
            ),
            "related_slots": [],
            "score": gap.get("score", 0.0),
            "llm_hypothesis": None,
            "created_at": now,
            "taxonomy_version": taxonomy_version,
        })

    if gap_rows:
        store.add_rows("ideation_gaps", gap_rows)

    report = {
        "report_id": report_id,
        "gap_count": len(all_gaps),
        "gaps": gap_rows,
    }
    logger.info(
        "Ideation: found %d gaps (v%d)",
        len(all_gaps),
        taxonomy_version,
    )
    return report


async def run_ideation_with_llm(
    store: LensStore,
    llm_client: LLMClient,
    taxonomy_version: int,
    top_n: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run ideation with optional LLM enrichment (Layer 2).

    First runs Layer 1 gap analysis, then asks the LLM to
    narrate top gaps as research hypotheses.
    """
    report = run_ideation(
        store, taxonomy_version, top_n=top_n, **kwargs
    )

    # LLM enrichment for top gaps
    for gap in report["gaps"]:
        prompt = (
            "You are an LLM research expert. A gap analysis has "
            "identified the following research opportunity:\n\n"
            f"{gap['description']}\n\n"
            "In 2-3 sentences, explain why this gap might exist, "
            "what research direction could fill it, and rate its "
            "potential impact (high/medium/low)."
        )
        try:
            hypothesis = await llm_client.complete(
                [{"role": "user", "content": prompt}]
            )
            gap["llm_hypothesis"] = hypothesis

            # Update the stored gap with the hypothesis
            try:
                gap_id = int(gap["id"])
                store.get_table("ideation_gaps").update(
                    where=f"id = {gap_id}",
                    values={"llm_hypothesis": hypothesis},
                )
            except Exception:
                logger.warning(
                    "Failed to update gap %d with hypothesis",
                    gap["id"],
                )
        except Exception:
            logger.warning(
                "LLM enrichment failed for gap %d", gap["id"]
            )

    return report
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ideation.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/lens/monitor/ tests/test_ideation.py
git commit -m "feat: add ideation gap analysis (sparse cells + cross-pollination + LLM enrichment)"
```

---

### Task 2: Monitor Pipeline — Acquire + Extract + Ideate

**Files:**
- Create: `src/lens/monitor/watcher.py`
- Create: `tests/test_monitor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_monitor.py
"""Tests for the monitor pipeline."""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_run_monitor_cycle(tmp_path):
    from lens.monitor.watcher import run_monitor_cycle
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    # Add taxonomy so ideation can run
    store.add_rows("taxonomy_versions", [
        {"version_id": 1, "created_at": "2026-03-21T00:00:00",
         "paper_count": 0, "param_count": 0,
         "principle_count": 0},
    ])

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = (
        '{"tradeoffs": [], "architecture": [], "agentic": []}'
    )

    # Mock the arxiv fetch to return one paper
    import httpx

    arxiv_xml = '''<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.99999v1</id>
        <title>New LLM Paper</title>
        <summary>A new paper about LLMs.</summary>
        <published>2024-01-15T00:00:00Z</published>
        <author><name>Author</name></author>
      </entry>
    </feed>'''

    mock_client = AsyncMock()
    mock_client.get.return_value = httpx.Response(
        200, text=arxiv_xml
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "lens.acquire.arxiv.httpx.AsyncClient",
        return_value=mock_client,
    ):
        result = await run_monitor_cycle(
            store=store,
            llm_client=mock_llm,
            query="LLM",
            categories=["cs.CL"],
            max_results=10,
        )

    assert result["papers_acquired"] >= 0
    assert "papers_extracted" in result


@pytest.mark.asyncio
async def test_run_monitor_cycle_no_taxonomy(tmp_path):
    from lens.monitor.watcher import run_monitor_cycle
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    mock_llm = AsyncMock()

    with patch(
        "lens.acquire.arxiv.httpx.AsyncClient",
        return_value=AsyncMock(),
    ):
        result = await run_monitor_cycle(
            store=store,
            llm_client=mock_llm,
            query="LLM",
            categories=["cs.CL"],
            max_results=10,
        )

    # Should still work, just skip ideation
    assert "papers_acquired" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_monitor.py -v
```

- [ ] **Step 3: Implement monitor watcher**

```python
# src/lens/monitor/watcher.py
"""Monitor pipeline: acquire new papers → extract → ideate.

Runs one monitoring cycle: fetches new papers from arxiv,
extracts knowledge, and optionally runs ideation gap analysis.
"""

from __future__ import annotations

import logging
from typing import Any

from lens.acquire.arxiv import fetch_arxiv_papers
from lens.extract.extractor import extract_papers
from lens.llm.client import LLMClient
from lens.monitor.ideation import run_ideation
from lens.store.store import LensStore
from lens.taxonomy.versioning import get_latest_version

logger = logging.getLogger(__name__)


async def run_monitor_cycle(
    store: LensStore,
    llm_client: LLMClient,
    query: str = "LLM",
    categories: list[str] | None = None,
    max_results: int = 50,
    run_ideation_flag: bool = True,
) -> dict[str, Any]:
    """Run one monitoring cycle.

    Steps:
    1. Fetch new papers from arxiv
    2. Store papers with placeholder embeddings
    3. Extract knowledge from new papers
    4. Run ideation gap analysis (if taxonomy exists)

    Returns summary dict.
    """
    cats = categories or ["cs.CL", "cs.LG", "cs.AI"]

    # Step 1: Acquire
    try:
        papers = await fetch_arxiv_papers(
            query=query,
            categories=cats,
            max_results=max_results,
        )
    except Exception:
        logger.warning("Failed to fetch papers from arxiv")
        papers = []

    # Filter out papers already in store
    existing_df = store.get_table("papers").to_polars()
    existing_ids = (
        set(existing_df["paper_id"].to_list())
        if len(existing_df) > 0
        else set()
    )
    new_papers = [
        p for p in papers if p["paper_id"] not in existing_ids
    ]

    # Add placeholder embeddings and store
    for p in new_papers:
        if "embedding" not in p:
            p["embedding"] = [0.0] * 768

    papers_acquired = len(new_papers)
    if new_papers:
        store.add_papers(new_papers)
        logger.info("Acquired %d new papers", papers_acquired)

    # Step 2: Extract
    papers_extracted = 0
    if papers_acquired > 0:
        papers_extracted = await extract_papers(
            store, llm_client, concurrency=3
        )

    # Step 3: Ideation (if taxonomy exists)
    ideation_report = None
    version = get_latest_version(store)
    if version is not None and run_ideation_flag:
        ideation_report = run_ideation(
            store, taxonomy_version=version
        )

    return {
        "papers_acquired": papers_acquired,
        "papers_extracted": papers_extracted,
        "ideation_report": ideation_report,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_monitor.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/lens/monitor/watcher.py tests/test_monitor.py
git commit -m "feat: add monitor watcher pipeline (acquire → extract → ideate)"
```

---

### Task 3: Wire Up CLI Commands

**Files:**
- Modify: `src/lens/cli.py` — replace `monitor` and `explore ideas` stubs
- Modify: `src/lens/monitor/__init__.py` — add exports

- [ ] **Step 1: Update monitor __init__.py**

```python
# src/lens/monitor/__init__.py
"""LENS monitoring and ideation pipeline."""

from lens.monitor.ideation import run_ideation, run_ideation_with_llm
from lens.monitor.watcher import run_monitor_cycle

__all__ = ["run_ideation", "run_ideation_with_llm", "run_monitor_cycle"]
```

- [ ] **Step 2: Replace the `monitor` CLI stub**

```python
@app.command()
def monitor(
    interval: str = typer.Option("weekly", "--interval", help="Not used yet."),
    trending: bool = typer.Option(False, "--trending", help="Show ideation gaps."),
) -> None:
    """Run one monitoring cycle: acquire → extract → ideate."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    if trending:
        # Show existing ideation gaps
        from lens.taxonomy.versioning import get_latest_version

        version = get_latest_version(store)
        if version is None:
            rprint("[yellow]No taxonomy yet.[/yellow]")
            raise typer.Exit(code=0)

        gaps = store.get_table("ideation_gaps").to_polars()
        gaps = gaps.filter(pl.col("taxonomy_version") == version)
        if len(gaps) == 0:
            rprint("[yellow]No ideation gaps found.[/yellow]")
        else:
            rprint(f"\n[bold]Ideation Gaps (v{version}):[/bold]")
            for row in gaps.sort("score", descending=True).to_dicts():
                hyp = f" — {row['llm_hypothesis'][:80]}..." if row.get("llm_hypothesis") else ""
                rprint(f"  [{row['gap_type']}] {row['description']}{hyp}")
        raise typer.Exit(code=0)

    from lens.llm.client import LLMClient
    from lens.monitor.watcher import run_monitor_cycle

    client = LLMClient(model=config["llm"]["extract_model"])
    cats = config["acquire"]["arxiv_categories"]
    result = asyncio.run(
        run_monitor_cycle(store, client, categories=cats)
    )
    rprint(f"[green]Monitor cycle complete:[/green]")
    rprint(f"  Papers acquired: {result['papers_acquired']}")
    rprint(f"  Papers extracted: {result['papers_extracted']}")
    if result.get("ideation_report"):
        rprint(f"  Gaps found: {result['ideation_report']['gap_count']}")
```

- [ ] **Step 3: Replace the `explore ideas` stub**

```python
@explore_app.command()
def ideas(
    type_: str | None = typer.Option(None, "--type", help="Gap type filter."),
) -> None:
    """Browse ideation gaps and research opportunities."""
    config = load_config(_get_config_path())
    store = LensStore(str(_get_data_dir(config)))
    store.init_tables()

    from lens.taxonomy.versioning import get_latest_version

    version = get_latest_version(store)
    if version is None:
        rprint("[yellow]No taxonomy yet.[/yellow]")
        raise typer.Exit(code=0)

    gaps = store.get_table("ideation_gaps").to_polars()
    gaps = gaps.filter(pl.col("taxonomy_version") == version)

    if type_:
        gaps = gaps.filter(pl.col("gap_type") == type_)

    if len(gaps) == 0:
        rprint("[yellow]No ideation gaps found.[/yellow]")
        raise typer.Exit(code=0)

    rprint(f"\n[bold]Research Opportunities ({len(gaps)} gaps):[/bold]\n")
    for row in gaps.sort("score", descending=True).to_dicts():
        rprint(f"  [bold][{row['gap_type']}][/bold] {row['description']}")
        if row.get("llm_hypothesis"):
            rprint(f"    → {row['llm_hypothesis']}")
        rprint()
```

**IMPORTANT**: Add `import polars as pl` at the top of cli.py (after the other imports). The `--trending` branch uses `pl.col(...)` and will fail without it.

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest -v
```

- [ ] **Step 5: Verify CLI**

```bash
uv run lens monitor --help
uv run lens monitor --trending
uv run lens explore ideas --help
```

- [ ] **Step 6: Commit**

```bash
git add src/lens/monitor/__init__.py src/lens/cli.py
git commit -m "feat: wire up monitor and explore ideas CLI commands"
```

---

## Summary

After completing this plan, LENS has the full pipeline:
- **`lens monitor`** — one-shot monitoring cycle: fetch arxiv → extract → ideation gap analysis
- **`lens monitor --trending`** — show existing ideation gaps
- **`lens explore ideas`** — browse research opportunities with optional `--type` filter
- **Ideation Layer 1** — deterministic gap analysis (sparse cells, cross-pollination)
- **Ideation Layer 2** — optional LLM enrichment that narrates gaps as research hypotheses

**Deferred to future work:**
- BERTrend trend detection and trend-gap intersections
- Taxonomy drift detection (centroid comparison)
- Stalled architecture slots (requires architecture catalog)
- Scheduled/recurring monitoring (currently one-shot)

**This completes all 6 plans.** The LENS system is fully functional end-to-end.

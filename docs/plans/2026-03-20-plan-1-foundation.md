# Plan 1: Foundation — Project Setup, Models, Store, Config, CLI Skeleton

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the LENS project scaffold with Pydantic/LanceModel schemas, LanceDB store, YAML config, and Typer CLI skeleton — so all subsequent plans have a solid foundation to build on.

**Architecture:** Single LanceDB embedded database with Pydantic `LanceModel` schemas defining all tables. Polars for in-memory analytics via zero-copy Arrow interchange. Typer CLI as thin wrapper over library functions. Config via YAML at `~/.lens/config.yaml`.

**Tech Stack:** Python 3.12+, uv (package manager), LanceDB, Polars, Pydantic, Typer, PyYAML

**Spec:** `docs/specs/design.md`

---

## File Structure

```
lens/
├── pyproject.toml                    # Project metadata, dependencies, entry points
├── CLAUDE.md                         # Development guidance for AI agents
├── src/
│   └── lens/
│       ├── __init__.py               # Public API exports
│       ├── cli.py                    # Typer CLI app with command groups
│       ├── config.py                 # Config loading/saving/defaults
│       ├── store/
│       │   ├── __init__.py           # Re-exports LensStore
│       │   ├── store.py              # LensStore class — LanceDB connection + table accessors
│       │   └── models.py             # All Pydantic LanceModel schemas (Layer 0-3)
│       └── acquire/                  # Empty __init__.py — placeholder for Plan 2
│           └── __init__.py
├── tests/
│   ├── conftest.py                   # Shared fixtures (tmp_path store, sample data)
│   ├── test_models.py                # Pydantic model validation tests
│   ├── test_store.py                 # LensStore CRUD tests
│   └── test_config.py                # Config load/save/defaults tests
└── docs/
    └── specs/
        └── design.md                 # Design spec (already exists)
```

---

### Task 1: Project Setup with uv

**Files:**
- Create: `pyproject.toml`
- Create: `CLAUDE.md`
- Create: `src/lens/__init__.py`

- [ ] **Step 1: Initialize uv project**

```bash
cd /Users/qingye/Documents/lens
uv init --lib --python 3.12
```

This will generate a `pyproject.toml` and `src/lens/__init__.py`. If files already exist, uv will skip them.

- [ ] **Step 2: Edit pyproject.toml**

Replace the generated `pyproject.toml` with:

```toml
[project]
name = "lens"
version = "0.1.0"
description = "LLM Engineering Navigation System — discovers recurring solution patterns from LLM research papers"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "lancedb>=0.17",
    "polars>=1.0",
    "pydantic>=2.0",
    "typer>=0.12",
    "pyyaml>=6.0",
    "litellm>=1.40",
    "rich>=13.0",
]

[project.scripts]
lens = "lens.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: marks tests that hit live APIs (deselect with '-m \"not integration\"')",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
```

- [ ] **Step 3: Install dependencies**

```bash
uv sync
```

Expected: all dependencies install successfully, `.venv` created.

- [ ] **Step 4: Create CLAUDE.md**

```markdown
# LENS Development Guide

## Quick Reference
- **Package manager**: `uv` — use `uv sync` to install, `uv add <pkg>` to add dependencies
- **Run CLI**: `uv run lens <command>`
- **Run tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/test_file.py::test_name -v`

## Architecture
- Single LanceDB database at `~/.lens/data/lens.lance`
- All models in `src/lens/store/models.py` as Pydantic `LanceModel` classes
- Analytics via Polars (zero-copy from Arrow)
- CLI via Typer in `src/lens/cli.py`
- Config at `~/.lens/config.yaml`

## Conventions
- Public API is synchronous; async internals wrapped with `asyncio.run()`
- All LanceDB tables use Pydantic LanceModel schemas
- Tests use tmp_path fixtures for isolated LanceDB instances
- No mocking of LanceDB — use real embedded instances in tests
```

- [ ] **Step 5: Create empty src/lens/__init__.py**

```python
"""LENS — LLM Engineering Navigation System."""
```

- [ ] **Step 6: Verify project structure**

```bash
uv run python -c "import lens; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml CLAUDE.md src/lens/__init__.py uv.lock .python-version
git commit -m "feat: initialize project with uv, dependencies, and CLAUDE.md"
```

---

### Task 2: Pydantic LanceModel Schemas

**Files:**
- Create: `src/lens/store/__init__.py`
- Create: `src/lens/store/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Create store package init**

Note: `LensStore` doesn't exist yet — the import will be added in Task 3, Step 5.

```python
# src/lens/store/__init__.py
"""LENS storage layer."""
```

- [ ] **Step 2: Write failing tests for models**

```python
# tests/test_models.py
"""Tests for Pydantic LanceModel schemas."""

import pytest
from datetime import datetime


def test_paper_model_valid():
    from lens.store.models import Paper

    paper = Paper(
        paper_id="2401.12345",
        title="Attention Is All You Need",
        abstract="We propose a new architecture...",
        authors=["Vaswani", "Shazeer"],
        venue="NeurIPS",
        date="2017-06-12",
        arxiv_id="1706.03762",
        citations=100000,
        quality_score=0.95,
        extraction_status="pending",
        embedding=[0.1] * 768,
    )
    assert paper.paper_id == "2401.12345"
    assert paper.extraction_status == "pending"
    assert len(paper.embedding) == 768


def test_paper_model_nullable_venue():
    from lens.store.models import Paper

    paper = Paper(
        paper_id="2401.99999",
        title="Some Preprint",
        abstract="Abstract text",
        authors=["Author"],
        venue=None,
        date="2024-01-01",
        arxiv_id="2401.99999",
        citations=0,
        quality_score=0.1,
        extraction_status="pending",
        embedding=[0.0] * 768,
    )
    assert paper.venue is None


def test_tradeoff_extraction_model():
    from lens.store.models import TradeoffExtraction

    ext = TradeoffExtraction(
        paper_id="2401.12345",
        improves="mathematical reasoning accuracy",
        worsens="inference time per token",
        technique="chain-of-thought with self-consistency",
        context="on GSM8K benchmark",
        confidence=0.85,
        evidence_quote="We observe a 15% improvement...",
    )
    assert ext.confidence == 0.85


def test_architecture_extraction_model():
    from lens.store.models import ArchitectureExtraction

    ext = ArchitectureExtraction(
        paper_id="2401.12345",
        component_slot="attention mechanism",
        variant_name="grouped-query attention",
        replaces="multi-head attention",
        key_properties="reduces KV cache by sharing keys/values",
        confidence=0.9,
    )
    assert ext.replaces == "multi-head attention"


def test_architecture_extraction_replaces_nullable():
    from lens.store.models import ArchitectureExtraction

    ext = ArchitectureExtraction(
        paper_id="2401.12345",
        component_slot="activation function",
        variant_name="SwiGLU",
        replaces=None,
        key_properties="smooth gated activation",
        confidence=0.7,
    )
    assert ext.replaces is None


def test_agentic_extraction_model():
    from lens.store.models import AgenticExtraction

    ext = AgenticExtraction(
        paper_id="2401.12345",
        pattern_name="reflexion",
        structure="single agent with self-critique loop and memory",
        use_case="code generation with iterative debugging",
        components=["actor", "evaluator", "memory"],
        confidence=0.8,
    )
    assert ext.components == ["actor", "evaluator", "memory"]


def test_parameter_model():
    from lens.store.models import Parameter

    param = Parameter(
        id=1,
        name="Inference Latency",
        description="Time to generate output tokens",
        raw_strings=["inference time", "latency"],
        paper_ids=["2401.12345"],
        taxonomy_version=1,
        embedding=[0.1] * 768,
    )
    assert param.name == "Inference Latency"
    assert len(param.embedding) == 768


def test_principle_model():
    from lens.store.models import Principle

    principle = Principle(
        id=1,
        name="Knowledge Distillation",
        description="Transfer knowledge from large to small model",
        sub_techniques=["response distillation", "feature distillation"],
        raw_strings=["distillation", "model compression"],
        paper_ids=["2401.12345"],
        taxonomy_version=1,
        embedding=[0.1] * 768,
    )
    assert principle.sub_techniques == ["response distillation", "feature distillation"]


def test_architecture_slot_model():
    from lens.store.models import ArchitectureSlot

    slot = ArchitectureSlot(
        id=1,
        name="Attention Mechanism",
        description="Core attention computation in transformer blocks",
        taxonomy_version=1,
    )
    assert slot.name == "Attention Mechanism"


def test_architecture_variant_model():
    from lens.store.models import ArchitectureVariant

    variant = ArchitectureVariant(
        id=1,
        slot_id=1,
        name="Grouped-Query Attention",
        replaces=[2, 3],
        properties="reduces KV cache by sharing keys/values across query groups",
        paper_ids=["2305.13245"],
        taxonomy_version=1,
        embedding=[0.1] * 768,
    )
    assert variant.replaces == [2, 3]


def test_agentic_pattern_model():
    from lens.store.models import AgenticPattern

    pattern = AgenticPattern(
        id=1,
        name="Reflexion",
        category="single-agent",
        description="Single agent with self-critique loop and memory",
        components=["actor", "evaluator", "memory"],
        use_cases=["code generation", "reasoning tasks"],
        paper_ids=["2303.11366"],
        taxonomy_version=1,
        embedding=[0.1] * 768,
    )
    assert pattern.category == "single-agent"


def test_matrix_cell_model():
    from lens.store.models import MatrixCell

    cell = MatrixCell(
        improving_param_id=1,
        worsening_param_id=2,
        principle_id=3,
        count=5,
        avg_confidence=0.82,
        paper_ids=["2401.12345", "2402.67890"],
        taxonomy_version=1,
    )
    assert cell.count == 5
    assert cell.avg_confidence == 0.82


def test_taxonomy_version_model():
    from lens.store.models import TaxonomyVersion

    tv = TaxonomyVersion(
        version_id=1,
        created_at=datetime(2026, 3, 20, 12, 0, 0),
        paper_count=200,
        param_count=25,
        principle_count=35,
    )
    assert tv.version_id == 1


def test_ideation_gap_model():
    from lens.store.models import IdeationGap

    gap = IdeationGap(
        id=1,
        report_id=1,
        gap_type="sparse_cell",
        description="Tradeoff (accuracy, latency) has only 1 known principle",
        related_params=[1, 2],
        related_principles=[3],
        related_slots=[],
        score=0.8,
        llm_hypothesis=None,
        created_at=datetime(2026, 3, 20),
        taxonomy_version=1,
    )
    assert gap.gap_type == "sparse_cell"
    assert gap.llm_hypothesis is None


def test_ideation_report_model():
    from lens.store.models import IdeationReport

    report = IdeationReport(
        id=1,
        created_at=datetime(2026, 3, 20),
        taxonomy_version=1,
        paper_batch_size=50,
        gap_count=10,
    )
    assert report.gap_count == 10
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lens.store.models'`

- [ ] **Step 4: Write all LanceModel schemas**

```python
# src/lens/store/models.py
"""Pydantic LanceModel schemas for all LENS data types.

Each model maps to a LanceDB table. Models with `embedding: Vector(768)` fields
support vector similarity search. All models use Pydantic v2 validation.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel
from lancedb.pydantic import LanceModel, Vector


# --- Layer 0: Papers ---


class Paper(LanceModel):
    """A research paper with metadata and SPECTER2 embedding."""

    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    venue: Optional[str] = None
    date: str
    arxiv_id: str
    citations: int = 0
    quality_score: float = 0.0
    extraction_status: str = "pending"
    embedding: Vector(768)  # type: ignore[valid-type]


# --- Layer 1: Raw Extractions ---


class TradeoffExtraction(LanceModel):
    """A tradeoff extracted from a paper: improving X worsens Y, resolved by technique."""

    paper_id: str
    improves: str
    worsens: str
    technique: str
    context: str
    confidence: float
    evidence_quote: str


class ArchitectureExtraction(LanceModel):
    """An architecture contribution extracted from a paper."""

    paper_id: str
    component_slot: str
    variant_name: str
    replaces: Optional[str] = None
    key_properties: str
    confidence: float


class AgenticExtraction(LanceModel):
    """An agentic pattern extracted from a paper."""

    paper_id: str
    pattern_name: str
    structure: str
    use_case: str
    components: list[str]
    confidence: float


# --- Layer 2: Taxonomy ---


class Parameter(LanceModel):
    """An emergent LLM design parameter (e.g., 'Inference Latency')."""

    id: int
    name: str
    description: str
    raw_strings: list[str]
    paper_ids: list[str]
    taxonomy_version: int
    embedding: Vector(768)  # type: ignore[valid-type]


class Principle(LanceModel):
    """An emergent solution principle (e.g., 'Knowledge Distillation')."""

    id: int
    name: str
    description: str
    sub_techniques: list[str]
    raw_strings: list[str]
    paper_ids: list[str]
    taxonomy_version: int
    embedding: Vector(768)  # type: ignore[valid-type]


class ArchitectureSlot(LanceModel):
    """A replaceable component position in an LLM architecture."""

    id: int
    name: str
    description: str
    taxonomy_version: int


class ArchitectureVariant(LanceModel):
    """A concrete implementation within an architecture slot."""

    id: int
    slot_id: int
    name: str
    replaces: list[int]
    properties: str
    paper_ids: list[str]
    taxonomy_version: int
    embedding: Vector(768)  # type: ignore[valid-type]


class AgenticPattern(LanceModel):
    """A recurring pattern for building LLM-based agents."""

    id: int
    name: str
    category: str
    description: str
    components: list[str]
    use_cases: list[str]
    paper_ids: list[str]
    taxonomy_version: int
    embedding: Vector(768)  # type: ignore[valid-type]


# --- Query Response Models (not stored in LanceDB) ---


class ExplanationResult(BaseModel):
    """Result of a `lens explain` query. Computed, not stored."""

    resolved_type: str
    resolved_id: int
    resolved_name: str
    narrative: str
    evolution: list[str]
    tradeoffs: list[dict]
    connections: list[str]
    paper_refs: list[str]
    alternatives: list[dict]


# --- Layer 3: Knowledge Structures (stored) ---


class MatrixCell(LanceModel):
    """One cell in the contradiction matrix: (improving, worsening) -> principle."""

    improving_param_id: int
    worsening_param_id: int
    principle_id: int
    count: int
    avg_confidence: float
    paper_ids: list[str]
    taxonomy_version: int


class TaxonomyVersion(LanceModel):
    """Metadata for a taxonomy version."""

    version_id: int
    created_at: datetime
    paper_count: int
    param_count: int
    principle_count: int


class IdeationGap(LanceModel):
    """A gap identified by the ideation pipeline."""

    id: int
    report_id: int
    gap_type: str
    description: str
    related_params: list[int]
    related_principles: list[int]
    related_slots: list[int]
    score: float
    llm_hypothesis: Optional[str] = None
    created_at: datetime
    taxonomy_version: int


class IdeationReport(LanceModel):
    """Summary of one ideation pipeline run."""

    id: int
    created_at: datetime
    taxonomy_version: int
    paper_batch_size: int
    gap_count: int
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_models.py -v
```

Expected: all 16 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lens/store/ tests/test_models.py
git commit -m "feat: add Pydantic LanceModel schemas for all data types"
```

---

### Task 3: LensStore — LanceDB Connection and Table Management

**Files:**
- Create: `src/lens/store/store.py`
- Create: `tests/conftest.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write shared test fixtures**

```python
# tests/conftest.py
"""Shared test fixtures."""

import pytest

from lens.store.store import LensStore


@pytest.fixture
def store(tmp_path):
    """Create a LensStore backed by a temporary directory."""
    return LensStore(str(tmp_path / "test.lance"))


@pytest.fixture
def sample_paper_data():
    """Sample paper data as a dict (for table.add())."""
    return {
        "paper_id": "2401.12345",
        "title": "Attention Is All You Need",
        "abstract": "We propose a new architecture...",
        "authors": ["Vaswani", "Shazeer"],
        "venue": "NeurIPS",
        "date": "2017-06-12",
        "arxiv_id": "1706.03762",
        "citations": 100000,
        "quality_score": 0.95,
        "extraction_status": "pending",
        "embedding": [0.1] * 768,
    }
```

- [ ] **Step 2: Write failing store tests**

```python
# tests/test_store.py
"""Tests for LensStore — LanceDB connection and table management."""

import polars as pl
from lens.store.models import Paper, TradeoffExtraction, Parameter


def test_store_init(store):
    """LensStore connects to LanceDB and has a db attribute."""
    assert store.db is not None


def test_store_init_tables(store):
    """init_tables() creates all expected tables."""
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
    """Can add a paper and retrieve it."""
    store.init_tables()
    store.add_papers([sample_paper_data])
    papers = store.get_table("papers").to_polars()
    assert len(papers) == 1
    assert papers["paper_id"][0] == "2401.12345"


def test_store_add_multiple_papers(store):
    """Can add multiple papers at once."""
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
    """get_table() returns a LanceDB table that converts to Polars."""
    store.init_tables()
    store.add_papers([sample_paper_data])
    df = store.get_table("papers").to_polars()
    assert isinstance(df, pl.DataFrame)
    assert "paper_id" in df.columns
    assert "embedding" in df.columns


def test_store_vector_search(store):
    """Can perform vector similarity search on papers."""
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
    # Search for something close to paper_a's embedding
    query = [0.9] + [0.1] + [0.0] * 766
    results = (
        store.get_table("papers")
        .search(query)
        .limit(1)
        .to_pandas()
    )
    assert results.iloc[0]["paper_id"] == "paper_a"


def test_store_filtered_query(store, sample_paper_data):
    """Can filter table rows with where()."""
    store.init_tables()
    store.add_papers([sample_paper_data])
    # Add another paper with different status
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
    """Calling init_tables() twice does not error or duplicate tables."""
    store.init_tables()
    store.init_tables()
    table_names = store.db.table_names()
    # No duplicates
    assert len(table_names) == len(set(table_names))
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_store.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lens.store.store'`

- [ ] **Step 4: Implement LensStore**

```python
# src/lens/store/store.py
"""LensStore — LanceDB connection and table management."""

from __future__ import annotations

from typing import Any

import lancedb

from lens.store.models import (
    AgenticExtraction,
    AgenticPattern,
    ArchitectureExtraction,
    ArchitectureSlot,
    ArchitectureVariant,
    IdeationGap,
    IdeationReport,
    MatrixCell,
    Paper,
    Parameter,
    Principle,
    TaxonomyVersion,
    TradeoffExtraction,
)

# Maps table name -> LanceModel schema
TABLE_SCHEMAS: dict[str, type] = {
    "papers": Paper,
    "tradeoff_extractions": TradeoffExtraction,
    "architecture_extractions": ArchitectureExtraction,
    "agentic_extractions": AgenticExtraction,
    "parameters": Parameter,
    "principles": Principle,
    "architecture_slots": ArchitectureSlot,
    "architecture_variants": ArchitectureVariant,
    "agentic_patterns": AgenticPattern,
    "matrix_cells": MatrixCell,
    "taxonomy_versions": TaxonomyVersion,
    "ideation_reports": IdeationReport,
    "ideation_gaps": IdeationGap,
}


class LensStore:
    """Single LanceDB connection managing all LENS tables.

    Args:
        data_dir: Parent directory for data storage. LanceDB will be at {data_dir}/lens.lance.
                  Pass a full .lance path directly if preferred (for testing).

    Usage:
        store = LensStore("/path/to/data")
        store.init_tables()
        store.add_papers([{...}])
        df = store.get_table("papers").to_polars()
    """

    def __init__(self, data_dir: str) -> None:
        lance_path = data_dir if data_dir.endswith(".lance") else f"{data_dir}/lens.lance"
        self.db = lancedb.connect(lance_path)

    def init_tables(self) -> None:
        """Create all tables if they don't already exist."""
        existing = set(self.db.table_names())
        for name, schema in TABLE_SCHEMAS.items():
            if name not in existing:
                self.db.create_table(name, schema=schema)

    def get_table(self, name: str) -> lancedb.table.Table:
        """Get a LanceDB table by name."""
        return self.db.open_table(name)

    def add_papers(self, data: list[dict[str, Any]]) -> None:
        """Add papers to the papers table."""
        self.get_table("papers").add(data)

    def add_rows(self, table_name: str, data: list[dict[str, Any]]) -> None:
        """Add rows to any table by name."""
        self.get_table(table_name).add(data)
```

- [ ] **Step 5: Update store __init__.py**

```python
# src/lens/store/__init__.py
"""LENS storage layer."""

from lens.store.store import LensStore

__all__ = ["LensStore"]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_store.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/lens/store/ tests/conftest.py tests/test_store.py
git commit -m "feat: add LensStore with LanceDB table management and CRUD"
```

---

### Task 4: Configuration Management

**Files:**
- Create: `src/lens/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

```python
# tests/test_config.py
"""Tests for LENS configuration management."""

import yaml


def test_default_config():
    from lens.config import default_config

    cfg = default_config()
    assert cfg["llm"]["default_model"] == "openrouter/anthropic/claude-sonnet-4-6"
    assert cfg["llm"]["extract_model"] == "openrouter/google/gemini-2.5-flash"
    assert cfg["storage"]["data_dir"] == "~/.lens/data"
    assert cfg["taxonomy"]["target_parameters"] == 25
    assert cfg["monitor"]["ideate"] is True


def test_load_config_returns_defaults_when_no_file(tmp_path):
    from lens.config import load_config

    cfg = load_config(config_path=tmp_path / "nonexistent.yaml")
    assert cfg["llm"]["default_model"] == "openrouter/anthropic/claude-sonnet-4-6"


def test_load_config_merges_with_file(tmp_path):
    from lens.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.dump({"llm": {"default_model": "custom/model"}})
    )
    cfg = load_config(config_path=config_file)
    # Overridden value
    assert cfg["llm"]["default_model"] == "custom/model"
    # Default value still present
    assert cfg["llm"]["extract_model"] == "openrouter/google/gemini-2.5-flash"


def test_save_config(tmp_path):
    from lens.config import save_config, load_config

    config_file = tmp_path / "config.yaml"
    save_config({"llm": {"default_model": "test/model"}}, config_path=config_file)
    assert config_file.exists()
    cfg = load_config(config_path=config_file)
    assert cfg["llm"]["default_model"] == "test/model"


def test_config_set_nested_key(tmp_path):
    from lens.config import load_config, set_config_value, save_config

    config_file = tmp_path / "config.yaml"
    cfg = load_config(config_path=config_file)
    set_config_value(cfg, "llm.default_model", "new/model")
    assert cfg["llm"]["default_model"] == "new/model"


def test_resolved_data_dir():
    from lens.config import resolve_data_dir

    cfg = {"storage": {"data_dir": "~/.lens/data"}}
    resolved = resolve_data_dir(cfg)
    assert "~" not in resolved
    assert resolved.endswith(".lens/data")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lens.config'`

- [ ] **Step 3: Implement config module**

```python
# src/lens/config.py
"""LENS configuration management.

Config is stored as YAML at ~/.lens/config.yaml by default.
Missing keys fall back to defaults. Nested keys are accessed with dot notation
(e.g., 'llm.default_model').
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "default_model": "openrouter/anthropic/claude-sonnet-4-6",
        "extract_model": "openrouter/google/gemini-2.5-flash",
        "label_model": "openrouter/anthropic/claude-sonnet-4-6",
    },
    "acquire": {
        "arxiv_categories": ["cs.CL", "cs.LG", "cs.AI"],
        "quality_min_citations": 0,
        "quality_venue_tiers": {
            "tier1": ["ICML", "NeurIPS", "ICLR", "ACL", "EMNLP", "COLM"],
            "tier2": ["AAAI", "NAACL", "EACL", "COLING"],
        },
    },
    "taxonomy": {
        "target_parameters": 25,
        "target_principles": 35,
        "min_cluster_size": 3,
        "embedding_model": "specter2",
    },
    "monitor": {
        "ideate": True,
        "ideate_llm": False,
        "ideate_top_n": 10,
        "ideate_min_gap_score": 0.5,
    },
    "storage": {
        "data_dir": "~/.lens/data",
    },
}

DEFAULT_CONFIG_PATH = Path("~/.lens/config.yaml").expanduser()


def default_config() -> dict[str, Any]:
    """Return a deep copy of the default configuration."""
    return copy.deepcopy(DEFAULT_CONFIG)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load config from YAML file, merged with defaults."""
    path = config_path or DEFAULT_CONFIG_PATH
    cfg = default_config()
    if Path(path).exists():
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    return cfg


def save_config(config: dict[str, Any], config_path: Path | None = None) -> None:
    """Save config to YAML file."""
    path = config_path or DEFAULT_CONFIG_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def set_config_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested config value using dot notation (e.g., 'llm.default_model')."""
    keys = dotted_key.split(".")
    d = config
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def resolve_data_dir(config: dict[str, Any]) -> str:
    """Resolve the storage data directory, expanding ~ to home."""
    return str(Path(config["storage"]["data_dir"]).expanduser())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_config.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lens/config.py tests/test_config.py
git commit -m "feat: add YAML config management with defaults and deep merge"
```

---

### Task 5: CLI Skeleton with Typer

**Files:**
- Create: `src/lens/cli.py`
- Create: `src/lens/acquire/__init__.py`

- [ ] **Step 1: Write CLI smoke test**

We test the CLI via Typer's testing utilities:

```python
# tests/test_cli.py
"""Tests for the LENS CLI skeleton."""

from typer.testing import CliRunner

from lens.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LENS" in result.output or "lens" in result.output


def test_cli_init(tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "Initialized" in result.output


def test_cli_init_force(tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "data"))
    # Init twice, second with --force
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["init", "--force"])
    assert result.exit_code == 0


def test_cli_config_show(tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_CONFIG_PATH", str(tmp_path / "config.yaml"))
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "default_model" in result.output


def test_cli_config_set(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("LENS_CONFIG_PATH", str(config_path))
    result = runner.invoke(app, ["config", "set", "llm.default_model", "test/model"])
    assert result.exit_code == 0
    assert config_path.exists()


def test_cli_acquire_group_exists():
    result = runner.invoke(app, ["acquire", "--help"])
    assert result.exit_code == 0


def test_cli_extract_group_exists():
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0


def test_cli_build_group_exists():
    result = runner.invoke(app, ["build", "--help"])
    assert result.exit_code == 0


def test_cli_analyze_exists():
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0


def test_cli_explain_exists():
    result = runner.invoke(app, ["explain", "--help"])
    assert result.exit_code == 0


def test_cli_explore_group_exists():
    result = runner.invoke(app, ["explore", "--help"])
    assert result.exit_code == 0


def test_cli_monitor_exists():
    result = runner.invoke(app, ["monitor", "--help"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lens.cli'`

- [ ] **Step 3: Implement CLI skeleton**

```python
# src/lens/cli.py
"""LENS CLI — Typer application with all command groups.

Commands are stubs that will be implemented in subsequent plans.
Only `init`, `config show`, and `config set` are functional in Plan 1.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print as rprint

from lens.config import (
    load_config,
    resolve_data_dir,
    save_config,
    set_config_value,
)
from lens.store.store import LensStore

app = typer.Typer(name="lens", help="LENS — LLM Engineering Navigation System")

# --- Subcommand groups ---
acquire_app = typer.Typer(help="Acquire papers from various sources")
build_app = typer.Typer(help="Build taxonomy and knowledge structures")
explore_app = typer.Typer(help="Browse knowledge structures")
config_app = typer.Typer(help="Manage configuration")

app.add_typer(acquire_app, name="acquire")
app.add_typer(build_app, name="build")
app.add_typer(explore_app, name="explore")
app.add_typer(config_app, name="config")


def _get_config_path() -> Path:
    return Path(os.environ.get("LENS_CONFIG_PATH", "~/.lens/config.yaml")).expanduser()


def _get_data_dir() -> str:
    env_dir = os.environ.get("LENS_DATA_DIR")
    if env_dir:
        return env_dir
    cfg = load_config(config_path=_get_config_path())
    return resolve_data_dir(cfg)


# --- Core commands ---


@app.command()
def init(force: bool = typer.Option(False, help="Reset and reinitialize everything")) -> None:
    """Initialize LENS databases."""
    data_dir = _get_data_dir()
    lance_path = os.path.join(data_dir, "lens.lance")
    if force and os.path.exists(lance_path):
        shutil.rmtree(lance_path)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    store = LensStore(data_dir)
    store.init_tables()
    rprint(f"[green]Initialized LENS at {data_dir}[/green]")


@app.command()
def analyze(
    query: str = typer.Argument(help="Problem description"),
    type: Optional[str] = typer.Option(None, help="Query type: architecture, agentic"),
) -> None:
    """Analyze a problem and suggest solutions."""
    rprint(f"[yellow]analyze not yet implemented (query: {query})[/yellow]")
    raise typer.Exit(code=0)


@app.command(name="explain")
def explain_cmd(
    concept: str = typer.Argument(help="Concept to explain"),
    related: bool = typer.Option(False, help="Emphasize connected concepts"),
    evolution: bool = typer.Option(False, help="Focus on evolution tree"),
    tradeoffs: bool = typer.Option(False, help="Focus on tradeoffs"),
) -> None:
    """Explain an LLM concept with adaptive depth."""
    rprint(f"[yellow]explain not yet implemented (concept: {concept})[/yellow]")
    raise typer.Exit(code=0)


@app.command()
def extract(
    paper_id: Optional[str] = typer.Option(None, help="Re-extract specific paper"),
    model: Optional[str] = typer.Option(None, help="LLM model to use"),
    concurrency: int = typer.Option(5, help="Concurrent LLM calls"),
) -> None:
    """Extract knowledge from papers."""
    rprint("[yellow]extract not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@app.command()
def monitor(
    interval: str = typer.Option("weekly", help="Check interval: daily, weekly"),
    trending: bool = typer.Option(False, help="Show trending topics and ideation gaps"),
) -> None:
    """Monitor arxiv for new papers."""
    rprint("[yellow]monitor not yet implemented[/yellow]")
    raise typer.Exit(code=0)


# --- Acquire subcommands ---


@acquire_app.command()
def seed() -> None:
    """Ingest curated seed papers."""
    rprint("[yellow]acquire seed not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@acquire_app.command()
def arxiv(
    query: str = typer.Option("LLM", help="Search query"),
    since: Optional[str] = typer.Option(None, help="Fetch papers since date"),
) -> None:
    """Fetch papers from arxiv."""
    rprint("[yellow]acquire arxiv not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@acquire_app.command()
def file(path: str = typer.Argument(help="Path to PDF file")) -> None:
    """Ingest a single paper from PDF."""
    rprint("[yellow]acquire file not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@acquire_app.command()
def openalex(enrich: bool = typer.Option(False, help="Enrich existing papers")) -> None:
    """Fetch/enrich papers from OpenAlex."""
    rprint("[yellow]acquire openalex not yet implemented[/yellow]")
    raise typer.Exit(code=0)


# --- Build subcommands ---


@build_app.command()
def taxonomy() -> None:
    """Run clustering to build taxonomy."""
    rprint("[yellow]build taxonomy not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@build_app.command()
def matrix() -> None:
    """Populate contradiction matrix from taxonomy."""
    rprint("[yellow]build matrix not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@build_app.command(name="all")
def build_all() -> None:
    """Full rebuild: taxonomy + matrix + catalogs."""
    rprint("[yellow]build all not yet implemented[/yellow]")
    raise typer.Exit(code=0)


# --- Explore subcommands ---


@explore_app.command()
def parameters() -> None:
    """Browse discovered parameters."""
    rprint("[yellow]explore parameters not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def principles() -> None:
    """Browse discovered principles."""
    rprint("[yellow]explore principles not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command(name="matrix")
def explore_matrix(
    param_a: Optional[int] = typer.Argument(None, help="Improving parameter ID"),
    param_b: Optional[int] = typer.Argument(None, help="Worsening parameter ID"),
) -> None:
    """Browse contradiction matrix."""
    rprint("[yellow]explore matrix not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def architecture(slot: Optional[str] = typer.Argument(None, help="Slot name")) -> None:
    """Browse architecture catalog."""
    rprint("[yellow]explore architecture not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def agents(category: Optional[str] = typer.Argument(None, help="Pattern category")) -> None:
    """Browse agentic patterns."""
    rprint("[yellow]explore agents not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command(name="evolution")
def explore_evolution(slot: str = typer.Argument(help="Slot name")) -> None:
    """Browse architecture evolution trajectories."""
    rprint("[yellow]explore evolution not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def paper(paper_id: str = typer.Argument(help="Paper ID")) -> None:
    """Browse a specific paper's extractions."""
    rprint("[yellow]explore paper not yet implemented[/yellow]")
    raise typer.Exit(code=0)


@explore_app.command()
def ideas(
    type: Optional[str] = typer.Option(None, help="Gap type filter"),
) -> None:
    """Browse ideation reports and gaps."""
    rprint("[yellow]explore ideas not yet implemented[/yellow]")
    raise typer.Exit(code=0)


# --- Config subcommands ---


@config_app.command()
def show() -> None:
    """Show current configuration."""
    cfg = load_config(config_path=_get_config_path())
    rprint(yaml.dump(cfg, default_flow_style=False, sort_keys=False))


@config_app.command(name="set")
def config_set(
    key: str = typer.Argument(help="Config key in dot notation (e.g., llm.default_model)"),
    value: str = typer.Argument(help="Value to set"),
) -> None:
    """Set a configuration value."""
    config_path = _get_config_path()
    cfg = load_config(config_path=config_path)
    set_config_value(cfg, key, value)
    save_config(cfg, config_path=config_path)
    rprint(f"[green]Set {key} = {value}[/green]")
```

- [ ] **Step 4: Create acquire package placeholder**

```python
# src/lens/acquire/__init__.py
"""LENS paper acquisition — implemented in Plan 2."""
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests across test_models.py, test_store.py, test_config.py, and test_cli.py PASS.

- [ ] **Step 7: Verify CLI entry point works**

```bash
uv run lens --help
uv run lens init --help
uv run lens config show
```

Expected: help text and config output displayed correctly.

- [ ] **Step 8: Commit**

```bash
git add src/lens/cli.py src/lens/acquire/__init__.py tests/test_cli.py
git commit -m "feat: add Typer CLI skeleton with all command groups and stubs"
```

---

### Task 6: Wire Up Public API and Final Integration

**Files:**
- Modify: `src/lens/__init__.py`

- [ ] **Step 1: Update public API exports**

```python
# src/lens/__init__.py
"""LENS — LLM Engineering Navigation System.

Public API for programmatic use. CLI is a thin wrapper over these functions.
Pipeline functions (acquire, extract, build) will be implemented in Plans 2-4.
Query functions (analyze, explain, explore) will be implemented in Plan 5.
"""

from lens.config import load_config, resolve_data_dir
from lens.store.models import ExplanationResult
from lens.store.store import LensStore

__all__ = [
    "ExplanationResult",
    "LensStore",
    "load_config",
    "resolve_data_dir",
]
```

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "from lens import LensStore, load_config; print('Public API OK')"
```

Expected: prints `Public API OK`.

- [ ] **Step 3: Run full test suite one final time**

```bash
uv run pytest -v
```

Expected: ALL tests pass (models: 16, store: 7, config: 6, cli: 12 = ~41 tests).

- [ ] **Step 4: Commit**

```bash
git add src/lens/__init__.py
git commit -m "feat: wire up public API exports for LensStore and config"
```

---

## Summary

After completing this plan, LENS has:
- **Project scaffold** with uv, all dependencies, and entry points
- **13 Pydantic LanceModel schemas** covering all 4 data layers
- **LensStore** class with LanceDB connection, table init, add, get, and vector search
- **Config system** with YAML persistence, defaults, deep merge, and dot-notation set
- **CLI skeleton** with all command groups and stubs (init + config functional)
- **~41 tests** covering models, store, config, and CLI
- **CLAUDE.md** with development conventions

**Next:** Plan 2 (Acquire) will implement paper acquisition from arxiv, OpenAlex, Semantic Scholar, and the seed loader.

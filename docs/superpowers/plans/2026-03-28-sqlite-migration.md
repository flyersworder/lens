# SQLite Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LanceDB + Polars with SQLite + sqlite-vec + plain Python, removing ~3 heavy dependencies while preserving all functionality.

**Architecture:** New `LensStore` backed by SQLite with sqlite-vec for vector search. All Polars operations become SQL queries or plain Python. Models become Pydantic BaseModel (validation only). Config restructured with dedicated `embeddings` section.

**Tech Stack:** Python 3.12+, SQLite 3.47+, sqlite-vec 0.1.7, Pydantic 2.x

**Spec:** `docs/superpowers/specs/2026-03-28-sqlite-migration-design.md`

---

### Task 1: New store layer (SQLite + sqlite-vec)

Build the new `LensStore` from scratch. This is the foundation everything else depends on.

**Files:**
- Rewrite: `src/lens/store/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write store tests**

Rewrite `tests/test_store.py` to test the new API. These tests define the contract:

```python
"""Tests for LensStore — SQLite + sqlite-vec."""

import pytest

from lens.store.store import LensStore


@pytest.fixture
def store(tmp_path):
    s = LensStore(str(tmp_path / "test.db"))
    s.init_tables()
    return s


def test_store_init(store):
    assert store.conn is not None


def test_store_init_tables(store):
    rows = store.query_sql("SELECT name FROM sqlite_master WHERE type='table'")
    names = {r["name"] for r in rows}
    assert "papers" in names
    assert "parameters" in names
    assert "matrix_cells" in names
    assert "taxonomy_versions" in names


def test_store_add_and_query(store):
    store.add_rows("papers", [{
        "paper_id": "2401.12345",
        "title": "Test Paper",
        "abstract": "Abstract",
        "authors": ["Author A", "Author B"],
        "venue": "NeurIPS",
        "date": "2024-01-15",
        "arxiv_id": "2401.12345",
        "citations": 100,
        "quality_score": 0.9,
        "extraction_status": "pending",
        "embedding": [0.1] * 768,
    }])
    rows = store.query("papers")
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "2401.12345"
    assert rows[0]["authors"] == ["Author A", "Author B"]  # JSON deserialized


def test_store_query_with_where(store):
    store.add_rows("papers", [
        {"paper_id": "a", "title": "A", "abstract": "", "authors": [],
         "date": "", "arxiv_id": "a", "extraction_status": "pending",
         "embedding": [0.0] * 768},
        {"paper_id": "b", "title": "B", "abstract": "", "authors": [],
         "date": "", "arxiv_id": "b", "extraction_status": "complete",
         "embedding": [0.0] * 768},
    ])
    rows = store.query("papers", where="extraction_status = ?", params=["pending"])
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "a"


def test_store_update(store):
    store.add_rows("papers", [{
        "paper_id": "x", "title": "X", "abstract": "", "authors": [],
        "date": "", "arxiv_id": "x", "extraction_status": "pending",
        "embedding": [0.0] * 768,
    }])
    store.update("papers", {"extraction_status": "complete"}, where="paper_id = ?", params=["x"])
    rows = store.query("papers", where="paper_id = ?", params=["x"])
    assert rows[0]["extraction_status"] == "complete"


def test_store_delete(store):
    store.add_rows("papers", [{
        "paper_id": "x", "title": "X", "abstract": "", "authors": [],
        "date": "", "arxiv_id": "x", "extraction_status": "pending",
        "embedding": [0.0] * 768,
    }])
    store.delete("papers", where="paper_id = ?", params=["x"])
    rows = store.query("papers")
    assert len(rows) == 0


def test_store_vector_search(store):
    store.add_rows("papers", [
        {"paper_id": "near", "title": "Near", "abstract": "", "authors": [],
         "date": "", "arxiv_id": "near", "extraction_status": "pending",
         "embedding": [1.0, 0.0] + [0.0] * 766},
        {"paper_id": "far", "title": "Far", "abstract": "", "authors": [],
         "date": "", "arxiv_id": "far", "extraction_status": "pending",
         "embedding": [0.0, 1.0] + [0.0] * 766},
    ])
    results = store.vector_search("papers", embedding=[1.0, 0.0] + [0.0] * 766, limit=1)
    assert len(results) == 1
    assert results[0]["paper_id"] == "near"
    assert "_distance" in results[0]


def test_store_vector_search_with_filter(store):
    store.add_rows("parameters", [
        {"id": 1, "name": "A", "description": "d", "raw_strings": ["a"],
         "paper_ids": ["p1"], "taxonomy_version": 1,
         "embedding": [1.0, 0.0] + [0.0] * 766},
        {"id": 2, "name": "B", "description": "d", "raw_strings": ["b"],
         "paper_ids": ["p2"], "taxonomy_version": 2,
         "embedding": [0.9, 0.1] + [0.0] * 766},
    ])
    results = store.vector_search(
        "parameters", embedding=[1.0, 0.0] + [0.0] * 766,
        limit=5, where="taxonomy_version = ?", params=[1]
    )
    assert len(results) == 1
    assert results[0]["name"] == "A"


def test_store_query_sql(store):
    store.add_rows("parameters", [
        {"id": 1, "name": "X", "description": "d", "raw_strings": ["x"],
         "paper_ids": ["p1"], "taxonomy_version": 1,
         "embedding": [0.0] * 768},
        {"id": 2, "name": "Y", "description": "d", "raw_strings": ["y"],
         "paper_ids": ["p1"], "taxonomy_version": 1,
         "embedding": [0.0] * 768},
    ])
    rows = store.query_sql(
        "SELECT COUNT(*) AS cnt FROM parameters WHERE taxonomy_version = ?",
        params=[1]
    )
    assert rows[0]["cnt"] == 2


def test_store_init_tables_idempotent(store):
    store.init_tables()  # call again
    rows = store.query("papers")
    assert isinstance(rows, list)


def test_store_json_list_fields(store):
    store.add_rows("agentic_extractions", [{
        "paper_id": "p1",
        "pattern_name": "ReAct",
        "structure": "loop",
        "use_case": "tool use",
        "components": ["LLM", "tools", "memory"],
        "confidence": 0.9,
    }])
    rows = store.query("agentic_extractions")
    assert rows[0]["components"] == ["LLM", "tools", "memory"]
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL — new API doesn't exist yet.

- [ ] **Step 3: Implement new LensStore**

Rewrite `src/lens/store/store.py`:

```python
"""LensStore — SQLite + sqlite-vec database layer."""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any

import sqlite_vec

from lens.store.models import EMBEDDING_DIM

# Tables that have companion _vec tables for vector search
VECTOR_TABLES = {
    "papers": "paper_id",
    "parameters": "id",
    "principles": "id",
    "architecture_variants": "id",
    "agentic_patterns": "id",
}

# Fields that are stored as JSON TEXT in SQLite
JSON_FIELDS: dict[str, set[str]] = {
    "papers": {"authors"},
    "agentic_extractions": {"components"},
    "parameters": {"raw_strings", "paper_ids"},
    "principles": {"sub_techniques", "raw_strings", "paper_ids"},
    "architecture_variants": {"replaces", "paper_ids"},
    "agentic_patterns": {"components", "use_cases", "paper_ids"},
    "matrix_cells": {"paper_ids"},
    "ideation_gaps": {"related_params", "related_principles", "related_slots"},
}

_SCHEMA_SQL = """
-- Layer 0: Papers
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL DEFAULT '',
    authors TEXT NOT NULL DEFAULT '[]',
    venue TEXT,
    date TEXT DEFAULT '',
    arxiv_id TEXT NOT NULL DEFAULT '',
    citations INTEGER DEFAULT 0,
    quality_score REAL DEFAULT 0.0,
    extraction_status TEXT DEFAULT 'pending'
);

-- Layer 1: Extractions
CREATE TABLE IF NOT EXISTS tradeoff_extractions (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    improves TEXT NOT NULL,
    worsens TEXT NOT NULL,
    technique TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL,
    evidence_quote TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS architecture_extractions (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    component_slot TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    replaces TEXT,
    key_properties TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agentic_extractions (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    pattern_name TEXT NOT NULL,
    structure TEXT NOT NULL DEFAULT '',
    use_case TEXT NOT NULL DEFAULT '',
    components TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL
);

-- Layer 2: Taxonomy
CREATE TABLE IF NOT EXISTS parameters (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    raw_strings TEXT NOT NULL DEFAULT '[]',
    paper_ids TEXT NOT NULL DEFAULT '[]',
    taxonomy_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS principles (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    sub_techniques TEXT NOT NULL DEFAULT '[]',
    raw_strings TEXT NOT NULL DEFAULT '[]',
    paper_ids TEXT NOT NULL DEFAULT '[]',
    taxonomy_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS architecture_slots (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    taxonomy_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS architecture_variants (
    id INTEGER PRIMARY KEY,
    slot_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    replaces TEXT NOT NULL DEFAULT '[]',
    properties TEXT NOT NULL DEFAULT '',
    paper_ids TEXT NOT NULL DEFAULT '[]',
    taxonomy_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS agentic_patterns (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    components TEXT NOT NULL DEFAULT '[]',
    use_cases TEXT NOT NULL DEFAULT '[]',
    paper_ids TEXT NOT NULL DEFAULT '[]',
    taxonomy_version INTEGER NOT NULL
);

-- Layer 3: Matrix
CREATE TABLE IF NOT EXISTS matrix_cells (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    improving_param_id INTEGER NOT NULL,
    worsening_param_id INTEGER NOT NULL,
    principle_id INTEGER NOT NULL,
    count INTEGER NOT NULL,
    avg_confidence REAL NOT NULL,
    paper_ids TEXT NOT NULL DEFAULT '[]',
    taxonomy_version INTEGER NOT NULL
);

-- Versioning
CREATE TABLE IF NOT EXISTS taxonomy_versions (
    version_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    paper_count INTEGER NOT NULL,
    param_count INTEGER NOT NULL,
    principle_count INTEGER NOT NULL,
    slot_count INTEGER DEFAULT 0,
    variant_count INTEGER DEFAULT 0,
    pattern_count INTEGER DEFAULT 0
);

-- Ideation
CREATE TABLE IF NOT EXISTS ideation_reports (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    taxonomy_version INTEGER NOT NULL,
    paper_batch_size INTEGER NOT NULL,
    gap_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ideation_gaps (
    id INTEGER PRIMARY KEY,
    report_id INTEGER NOT NULL,
    gap_type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    related_params TEXT NOT NULL DEFAULT '[]',
    related_principles TEXT NOT NULL DEFAULT '[]',
    related_slots TEXT NOT NULL DEFAULT '[]',
    score REAL NOT NULL,
    llm_hypothesis TEXT,
    created_at TEXT NOT NULL,
    taxonomy_version INTEGER NOT NULL
);
"""


class LensStore:
    """SQLite + sqlite-vec database for LENS."""

    def __init__(self, db_path: str) -> None:
        if not db_path.endswith(".db"):
            db_path = db_path.rstrip("/") + "/lens.db"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        # Load sqlite-vec extension
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def init_tables(self) -> None:
        """Create all tables if they don't exist."""
        self.conn.executescript(_SCHEMA_SQL)
        # Create vec0 virtual tables for vector search
        for table, id_col in VECTOR_TABLES.items():
            id_type = "TEXT" if id_col == "paper_id" else "INTEGER"
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {table}_vec USING vec0(
                    {id_col} {id_type} PRIMARY KEY,
                    embedding FLOAT[{EMBEDDING_DIM}] distance_metric=cosine
                )
            """)
        self.conn.commit()

    def add_rows(self, table: str, rows: list[dict]) -> None:
        """Insert rows into a table. Auto-serializes JSON fields and embeddings."""
        if not rows:
            return
        json_cols = JSON_FIELDS.get(table, set())
        has_vec = table in VECTOR_TABLES

        for row in rows:
            row = dict(row)  # copy to avoid mutating caller's data
            embedding = row.pop("embedding", None) if has_vec else None

            # Serialize JSON fields
            for col in json_cols:
                if col in row and not isinstance(row[col], str):
                    row[col] = json.dumps(row[col])

            # Serialize datetime objects to ISO strings
            for key, val in row.items():
                if hasattr(val, "isoformat"):
                    row[key] = val.isoformat()

            # Insert main row
            cols = list(row.keys())
            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join(cols)
            self.conn.execute(
                f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})",
                [row[c] for c in cols],
            )

            # Insert embedding into vec table
            if has_vec and embedding is not None:
                id_col = VECTOR_TABLES[table]
                row_id = row[id_col]
                emb_bytes = struct.pack(f"{len(embedding)}f", *embedding)
                self.conn.execute(
                    f"INSERT INTO {table}_vec ({id_col}, embedding) VALUES (?, ?)",
                    (row_id, emb_bytes),
                )

        self.conn.commit()

    def query(
        self, table: str, where: str = "", params: list | None = None
    ) -> list[dict]:
        """Query a table, returning list[dict] with JSON fields deserialized."""
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        cursor = self.conn.execute(sql, params or [])
        rows = [dict(r) for r in cursor.fetchall()]
        # Deserialize JSON fields
        json_cols = JSON_FIELDS.get(table, set())
        for row in rows:
            for col in json_cols:
                if col in row and isinstance(row[col], str):
                    row[col] = json.loads(row[col])
        return rows

    def query_sql(self, sql: str, params: list | None = None) -> list[dict]:
        """Execute raw SQL, return list[dict]. JSON fields NOT auto-deserialized."""
        cursor = self.conn.execute(sql, params or [])
        return [dict(r) for r in cursor.fetchall()]

    def update(
        self, table: str, values: dict, where: str, params: list | None = None
    ) -> None:
        """Update rows in a table."""
        set_clause = ", ".join(f"{k} = ?" for k in values)
        set_values = list(values.values())
        self.conn.execute(
            f"UPDATE {table} SET {set_clause} WHERE {where}",
            set_values + (params or []),
        )
        self.conn.commit()

    def delete(self, table: str, where: str, params: list | None = None) -> None:
        """Delete rows from a table. Also deletes from companion vec table."""
        self.conn.execute(f"DELETE FROM {table} WHERE {where}", params or [])
        if table in VECTOR_TABLES:
            self.conn.execute(
                f"DELETE FROM {table}_vec WHERE {where}", params or []
            )
        self.conn.commit()

    def vector_search(
        self,
        table: str,
        embedding: list[float],
        limit: int = 5,
        where: str = "",
        params: list | None = None,
    ) -> list[dict]:
        """k-NN vector search via sqlite-vec, returning full rows with _distance."""
        id_col = VECTOR_TABLES[table]
        query_bytes = struct.pack(f"{len(embedding)}f", *embedding)
        json_cols = JSON_FIELDS.get(table, set())

        if where:
            sql = f"""
                SELECT t.*, v.distance AS _distance
                FROM {table}_vec v
                JOIN {table} t ON t.{id_col} = v.{id_col}
                WHERE v.embedding MATCH ? AND v.k = ?
                AND {where}
                ORDER BY v.distance
                LIMIT ?
            """
            cursor = self.conn.execute(
                sql, [query_bytes, limit * 3, *(params or []), limit]
            )
        else:
            sql = f"""
                SELECT t.*, v.distance AS _distance
                FROM {table}_vec v
                JOIN {table} t ON t.{id_col} = v.{id_col}
                WHERE v.embedding MATCH ? AND v.k = ?
                ORDER BY v.distance
            """
            cursor = self.conn.execute(sql, [query_bytes, limit])

        rows = [dict(r) for r in cursor.fetchall()]
        for row in rows:
            for col in json_cols:
                if col in row and isinstance(row[col], str):
                    row[col] = json.loads(row[col])
        return rows

    # Convenience aliases
    def add_papers(self, data: list[dict]) -> None:
        """Append papers to the papers table."""
        self.add_rows("papers", data)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_store.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/lens/store/store.py tests/test_store.py
git commit -m "feat: rewrite LensStore with SQLite + sqlite-vec backend"
```

---

### Task 2: Update models and config

**Files:**
- Modify: `src/lens/store/models.py`
- Modify: `src/lens/config.py`
- Modify: `pyproject.toml`
- Test: `tests/test_models.py`, `tests/test_config.py`

- [ ] **Step 1: Update models.py — LanceModel → BaseModel**

Remove `lancedb` imports. Models become plain `BaseModel`. Remove `Vector()` annotations. Keep `EMBEDDING_DIM`, validators, `ExplanationResult`.

```python
"""Pydantic model schemas for all LENS data types."""

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

EMBEDDING_DIM = 768
"""Default embedding vector dimension."""

VALID_EXTRACTION_STATUSES = {"pending", "complete", "incomplete", "failed"}


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    authors: list[str] = []
    venue: str | None = None
    date: str = ""
    arxiv_id: str = ""
    citations: int = 0
    quality_score: float = 0.0
    extraction_status: str = "pending"
    embedding: list[float] = []

    @field_validator("date")
    @classmethod
    def _check_date_format(cls, v: str) -> str:
        if v and not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"date must be YYYY-MM-DD format, got '{v}'")
        return v

    @field_validator("extraction_status")
    @classmethod
    def _check_extraction_status(cls, v: str) -> str:
        if v not in VALID_EXTRACTION_STATUSES:
            raise ValueError(
                f"extraction_status must be one of {VALID_EXTRACTION_STATUSES}, got '{v}'"
            )
        return v


class TradeoffExtraction(BaseModel):
    paper_id: str
    improves: str
    worsens: str
    technique: str
    context: str = ""
    confidence: float
    evidence_quote: str = ""


class ArchitectureExtraction(BaseModel):
    paper_id: str
    component_slot: str
    variant_name: str
    replaces: str | None = None
    key_properties: str = ""
    confidence: float


class AgenticExtraction(BaseModel):
    paper_id: str
    pattern_name: str
    structure: str = ""
    use_case: str = ""
    components: list[str] = []
    confidence: float


class Parameter(BaseModel):
    id: int
    name: str
    description: str = ""
    raw_strings: list[str] = []
    paper_ids: list[str] = []
    taxonomy_version: int
    embedding: list[float] = []


class Principle(BaseModel):
    id: int
    name: str
    description: str = ""
    sub_techniques: list[str] = []
    raw_strings: list[str] = []
    paper_ids: list[str] = []
    taxonomy_version: int
    embedding: list[float] = []


class ArchitectureSlot(BaseModel):
    id: int
    name: str
    description: str = ""
    taxonomy_version: int


class ArchitectureVariant(BaseModel):
    id: int
    slot_id: int
    name: str
    replaces: list[int] = []
    properties: str = ""
    paper_ids: list[str] = []
    taxonomy_version: int
    embedding: list[float] = []


class AgenticPattern(BaseModel):
    id: int
    name: str
    category: str = ""
    description: str = ""
    components: list[str] = []
    use_cases: list[str] = []
    paper_ids: list[str] = []
    taxonomy_version: int
    embedding: list[float] = []


class MatrixCell(BaseModel):
    improving_param_id: int
    worsening_param_id: int
    principle_id: int
    count: int
    avg_confidence: float
    paper_ids: list[str] = []
    taxonomy_version: int


class TaxonomyVersion(BaseModel):
    version_id: int
    created_at: datetime
    paper_count: int
    param_count: int
    principle_count: int
    slot_count: int = 0
    variant_count: int = 0
    pattern_count: int = 0


class IdeationGap(BaseModel):
    id: int
    report_id: int
    gap_type: str
    description: str = ""
    related_params: list[int] = []
    related_principles: list[int] = []
    related_slots: list[int] = []
    score: float
    llm_hypothesis: str | None = None
    created_at: datetime
    taxonomy_version: int


class IdeationReport(BaseModel):
    id: int
    created_at: datetime
    taxonomy_version: int
    paper_batch_size: int
    gap_count: int


class ExplanationResult(BaseModel):
    resolved_type: str
    resolved_id: int
    resolved_name: str
    narrative: str
    evolution: list[str]
    tradeoffs: list[dict]
    connections: list[str]
    paper_refs: list[str]
    alternatives: list[dict]
```

- [ ] **Step 2: Restructure config.py**

Move embedding config from `taxonomy` to new `embeddings` section:

```python
DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "default_model": "openrouter/anthropic/claude-sonnet-4-6",
        "extract_model": "openrouter/google/gemini-2.5-flash",
        "label_model": "openrouter/anthropic/claude-sonnet-4-6",
        "api_base": "",
        "api_key": "",
    },
    "embeddings": {
        "provider": "local",
        "model": "specter2",
        "dimensions": 768,
        "api_base": "",
        "api_key": "",
    },
    "acquire": {
        "arxiv_categories": ["cs.CL", "cs.LG", "cs.AI"],
        "openalex_mailto": "",
        "quality_min_citations": 0,
        "quality_venue_tiers": {
            "tier1": ["ICML", "NeurIPS", "ICLR", "ACL", "EMNLP", "COLM"],
            "tier2": ["AAAI", "NAACL", "EACL", "COLING"],
        },
    },
    "taxonomy": {
        "target_parameters": 25,
        "target_principles": 35,
        "target_arch_variants": 20,
        "target_agentic_patterns": 15,
        "min_cluster_size": 3,
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
```

- [ ] **Step 3: Update pyproject.toml**

Remove `lancedb`, `polars`, `pandas`. Keep `sqlite-vec`. Update version description if needed.

```toml
dependencies = [
    "pydantic>=2.0",
    "typer>=0.12",
    "pyyaml>=6.0",
    "openai>=1.0",
    "rich>=13.0",
    "httpx>=0.28.1",
    "tenacity>=9.1.4",
    "sentence-transformers>=5.3.0",
    "hdbscan>=0.8.41",
    "sqlite-vec>=0.1.7",
]
```

- [ ] **Step 4: Update test_models.py**

Remove LanceDB-specific model tests. Update to use plain BaseModel:

```python
from datetime import datetime
from lens.store.models import Paper, TaxonomyVersion, EMBEDDING_DIM

def test_paper_model():
    p = Paper(paper_id="123", title="T", abstract="A", authors=["X"],
              date="2024-01-01", arxiv_id="123", embedding=[0.0] * EMBEDDING_DIM)
    assert p.paper_id == "123"
    assert p.authors == ["X"]

def test_paper_date_validation():
    import pytest
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        Paper(paper_id="x", title="T", date="bad", arxiv_id="x")

def test_paper_extraction_status_validation():
    import pytest
    with pytest.raises(ValueError, match="extraction_status"):
        Paper(paper_id="x", title="T", arxiv_id="x", extraction_status="invalid")

def test_taxonomy_version_with_catalog_counts():
    tv = TaxonomyVersion(
        version_id=1, created_at=datetime.now(), paper_count=10,
        param_count=5, principle_count=10,
        slot_count=3, variant_count=12, pattern_count=8,
    )
    assert tv.slot_count == 3

def test_taxonomy_version_defaults_backward_compat():
    tv = TaxonomyVersion(
        version_id=1, created_at=datetime.now(), paper_count=10,
        param_count=5, principle_count=10,
    )
    assert tv.slot_count == 0
    assert tv.variant_count == 0
    assert tv.pattern_count == 0
```

- [ ] **Step 5: Run uv sync and tests**

Run: `uv sync && uv run pytest tests/test_store.py tests/test_models.py tests/test_config.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/lens/store/models.py src/lens/config.py pyproject.toml tests/test_models.py uv.lock
git commit -m "feat: models to BaseModel, config restructured, remove lancedb/polars/pandas deps"
```

---

### Task 3: Migrate taxonomy pipeline (versioning, embedder, __init__)

**Files:**
- Modify: `src/lens/taxonomy/versioning.py`
- Modify: `src/lens/taxonomy/embedder.py`
- Modify: `src/lens/taxonomy/__init__.py`
- Test: `tests/test_taxonomy.py`

- [ ] **Step 1: Migrate versioning.py**

Replace Polars with `store.query_sql`:

```python
"""Taxonomy version management."""
from __future__ import annotations
from datetime import UTC, datetime
from lens.store.store import LensStore


def get_latest_version(store: LensStore) -> int | None:
    rows = store.query_sql("SELECT MAX(version_id) AS max_id FROM taxonomy_versions")
    max_id = rows[0]["max_id"] if rows else None
    return max_id


def get_next_version(store: LensStore) -> int:
    latest = get_latest_version(store)
    return (latest or 0) + 1


def record_version(
    store: LensStore, version_id: int, paper_count: int,
    param_count: int, principle_count: int,
    slot_count: int = 0, variant_count: int = 0, pattern_count: int = 0,
) -> None:
    store.add_rows("taxonomy_versions", [{
        "version_id": version_id,
        "created_at": datetime.now(UTC),
        "paper_count": paper_count,
        "param_count": param_count,
        "principle_count": principle_count,
        "slot_count": slot_count,
        "variant_count": variant_count,
        "pattern_count": pattern_count,
    }])
```

- [ ] **Step 2: Update embedder.py**

Read embedding config from the new `embeddings` section. Remove `EMBEDDING_DIM` import (the embedder doesn't need it — the store handles dimension via vec table creation). The `embed_strings` function and its local/cloud paths stay the same.

The only change: update the import in `__init__.py` to pass config from the new `embeddings` section instead of `taxonomy`.

- [ ] **Step 3: Migrate taxonomy/__init__.py**

This is the largest single file change. Replace all Polars operations with SQL/Python:

Key changes:
- `_collect_strings_from_table`: use `store.query()` instead of `store.get_table().to_polars()`
- `_build_paper_id_map`: same pattern
- `_build_taxonomy_entries`: stays mostly the same (operates on dicts, not DataFrames)
- `_next_id`: use `store.query_sql("SELECT MAX(id) ...")`
- `build_taxonomy`: replace all `store.get_table().to_polars()` calls with `store.query()`
- Architecture stage: replace Polars `.filter()` with list comprehension or SQL
- Agentic stage: same pattern
- Pass embedding config from new `embeddings` section

The implementer should read the full current file and systematically replace every `pl.col()`, `.filter()`, `.to_polars()`, `.to_dicts()` call.

- [ ] **Step 4: Update tests/test_taxonomy.py**

Remove all `import polars` and `EMBEDDING_DIM` references from Polars usage. Update store fixture to use new `LensStore`. The test logic stays the same — just the API calls change.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_taxonomy.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: migrate taxonomy pipeline from Polars to SQL/Python"
```

---

### Task 4: Migrate extract, knowledge, serve, and monitor layers

**Files:**
- Modify: `src/lens/extract/extractor.py`
- Modify: `src/lens/knowledge/matrix.py`
- Modify: `src/lens/serve/analyzer.py`
- Modify: `src/lens/serve/explainer.py`
- Modify: `src/lens/serve/explorer.py`
- Modify: `src/lens/monitor/watcher.py`
- Modify: `src/lens/monitor/ideation.py`
- Test: all corresponding test files

- [ ] **Step 1: Migrate extractor.py**

Replace:
- `store.get_table("papers").to_polars()` → `store.query("papers", where=...)`
- `store.get_table(t).delete(f"paper_id = '{safe_id}'")` → `store.delete(t, "paper_id = ?", [pid])`
- `store.get_table("papers").update(where=..., values=...)` → `store.update("papers", values, "paper_id = ?", [pid])`
- Remove `escape_sql_string` import (parameterized queries handle this)

- [ ] **Step 2: Migrate matrix.py**

Replace Polars groupby/agg with SQL:
```python
# Instead of Polars-based aggregation, use:
rows = store.query("tradeoff_extractions", where="confidence >= ?", params=[0.5])
# Then aggregate in Python or use SQL GROUP BY via query_sql()
```

The `_build_string_to_id_map` function uses `store.query()` to get parameters/principles and builds a Python dict.

`build_matrix` uses `store.delete()` for cleanup and `store.add_rows()` for writing cells.

- [ ] **Step 3: Migrate analyzer.py**

Replace LanceDB vector search with `store.vector_search()`:
```python
# Before:
lance_table = cast(Any, table._table)
raw_results = lance_table.search(query_embedding).where(...).limit(5).to_list()

# After:
results = store.vector_search("architecture_variants", embedding=query_embedding,
                               limit=5, where="taxonomy_version = ?", params=[version])
```

Remove `cast(Any, ...)` hack — no longer needed.

- [ ] **Step 4: Migrate explainer.py**

Replace vector search in `resolve_concept`:
```python
# Before:
results = store.get_table(table_name).search(query_embedding).where(...).limit(top_k).to_list()

# After:
results = store.vector_search(table_name, embedding=query_embedding,
                               limit=top_k, where="taxonomy_version = ?", params=[version])
```

Replace Polars operations in `graph_walk` with `store.query()`.

- [ ] **Step 5: Migrate explorer.py**

Replace all Polars operations with SQL:

```python
# list_architecture_slots: JOIN with variant counts
def list_architecture_slots(store, taxonomy_version):
    return store.query_sql("""
        SELECT s.*, COALESCE(v.cnt, 0) AS variant_count
        FROM architecture_slots s
        LEFT JOIN (
            SELECT slot_id, COUNT(*) AS cnt FROM architecture_variants
            WHERE taxonomy_version = ? GROUP BY slot_id
        ) v ON s.id = v.slot_id
        WHERE s.taxonomy_version = ?
        ORDER BY s.name
    """, params=[taxonomy_version, taxonomy_version])

# get_architecture_timeline: JOIN with papers for dates
def get_architecture_timeline(store, slot_name, taxonomy_version):
    # Get slot_id first
    slots = store.query("architecture_slots",
                        where="name = ? AND taxonomy_version = ?",
                        params=[slot_name, taxonomy_version])
    if not slots:
        return []
    slot_id = slots[0]["id"]
    # Get variants with min paper date via JSON + papers join
    variants = store.query("architecture_variants",
                           where="slot_id = ? AND taxonomy_version = ?",
                           params=[slot_id, taxonomy_version])
    papers = {r["paper_id"]: r["date"] for r in store.query("papers")}
    for v in variants:
        if isinstance(v["paper_ids"], str):
            v["paper_ids"] = json.loads(v["paper_ids"])
        dates = [papers.get(pid, "") for pid in v["paper_ids"] if papers.get(pid)]
        v["earliest_date"] = min(dates) if dates else ""
    variants.sort(key=lambda v: v.get("earliest_date", ""))
    return variants
```

- [ ] **Step 6: Migrate watcher.py and ideation.py**

Replace Polars with `store.query()`. For `ideation.py`, the `find_sparse_cells` and `find_cross_pollination` functions load data from `store.query()` and work with Python dicts/lists — same logic, different data loading.

- [ ] **Step 7: Update all test files**

Remove all `import polars`, `from lens.store.models import EMBEDDING_DIM` (where only used for `[0.0] * EMBEDDING_DIM` — replace with `[0.0] * 768` or import EMBEDDING_DIM from models which still exists).

Update conftest.py fixture:
```python
@pytest.fixture
def store(tmp_path):
    from lens.store.store import LensStore
    s = LensStore(str(tmp_path / "test.db"))
    s.init_tables()
    return s
```

- [ ] **Step 8: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All tests pass.

- [ ] **Step 9: Commit**

```bash
git commit -m "feat: migrate all callers from Polars to SQL/Python"
```

---

### Task 5: Migrate CLI and clean up imports

**Files:**
- Modify: `src/lens/cli.py`
- Modify: `src/lens/acquire/seed.py`
- Modify: `src/lens/acquire/pdf.py`
- Modify: `src/lens/__init__.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Update cli.py**

Remove `import polars as pl`, `from lens.store.models import EMBEDDING_DIM` (where only used for placeholder embeddings — keep the import from models).

Replace:
- `store.get_table("papers").to_polars()` → `store.query("papers")`
- `store.get_table("ideation_gaps").to_polars()` → `store.query("ideation_gaps", where=...)`
- `papers_table.update(where=..., values=...)` → `store.update("papers", values, where, params)`
- Polars `pl.col()` filtering → SQL WHERE clauses
- Remove `escape_sql_string` import

Pass embedding config from `config["embeddings"]` instead of `config["taxonomy"]` to `build_taxonomy`.

- [ ] **Step 2: Update acquire/seed.py and acquire/pdf.py**

Replace `store.get_table("papers").to_polars()` → `store.query("papers")`.
Keep `EMBEDDING_DIM` import for placeholder embeddings.

- [ ] **Step 3: Update __init__.py**

Remove LanceDB imports from the public API. Export `LensStore` from the new module.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: migrate CLI and acquire layer, remove all Polars imports"
```

---

### Task 6: Update documentation and final cleanup

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/specs/design.md`

- [ ] **Step 1: Update README.md**

Update Architecture section: remove Polars, replace LanceDB with SQLite + sqlite-vec. Update config examples with new `embeddings` section.

- [ ] **Step 2: Update CLAUDE.md**

Remove references to Polars, LanceDB, `EMBEDDING_DIM` convention (still exists but simpler now). Add SQLite conventions.

- [ ] **Step 3: Update design.md**

Update tech stack table, storage section, config section.

- [ ] **Step 4: Verify no lancedb/polars imports remain**

Run: `grep -rn "import polars\|import lancedb\|from polars\|from lancedb" src/ tests/`
Expected: No matches.

- [ ] **Step 5: Run full test suite one final time**

Run: `uv run pytest -x -q`
Expected: All tests pass.

- [ ] **Step 6: Verify CLI**

Run: `uv run lens --help`
Run: `uv run lens config show`
Expected: Both work correctly.

- [ ] **Step 7: Commit**

```bash
git commit -m "docs: update all documentation for SQLite migration"
```

- [ ] **Step 8: Push**

```bash
git push
```

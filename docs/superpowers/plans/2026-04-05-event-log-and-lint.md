# Event Log & Lint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unified event log table and a 6-check lint command with optional auto-fix to the Lens knowledge base.

**Architecture:** A single `event_log` table records all mutations (ingest, extract, build, lint, fix). A `log_event()` helper is called explicitly at each instrumentation site. Six lint checks query the data tables and event log to find issues. Lint findings and fixes are themselves logged as events.

**Tech Stack:** SQLite (via existing `LensStore`), Pydantic models, Typer CLI, pytest with `tmp_path` fixtures.

**Spec:** `docs/superpowers/specs/2026-04-05-event-log-and-lint-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/lens/store/models.py` | Add `EventLog` and `LintReport` Pydantic models |
| `src/lens/store/store.py` | Add `event_log` table DDL and migration |
| `src/lens/knowledge/events.py` | **New** — `log_event()` helper + `query_events()` for reading |
| `src/lens/knowledge/linter.py` | **New** — 6 lint checks + `lint()` orchestrator |
| `src/lens/cli.py` | Add `lens lint` and `lens log` commands |
| `src/lens/extract/extractor.py` | Emit extract events in Phase 2 |
| `src/lens/taxonomy/vocabulary.py` | Emit vocabulary events in `process_new_concepts()` and `build_vocabulary()` |
| `src/lens/knowledge/matrix.py` | Emit build event at end of `build_matrix()` |
| `src/lens/taxonomy/versioning.py` | Emit event in `record_version()` |
| `tests/test_event_log.py` | **New** — event log unit tests |
| `tests/test_linter.py` | **New** — lint check unit tests |
| `tests/test_lint_integration.py` | **New** — integration tests |

---

### Task 1: EventLog Model + Table Schema

**Files:**
- Modify: `src/lens/store/models.py:203` (after last model)
- Modify: `src/lens/store/store.py:36-131` (_TABLE_DDL list)
- Modify: `src/lens/store/store.py:26-33` (JSON_FIELDS dict)
- Test: `tests/test_event_log.py`

- [ ] **Step 1: Write failing test for event_log table creation**

```python
# tests/test_event_log.py
"""Tests for the event log system."""

from lens.store.store import LensStore


def test_event_log_table_exists(tmp_path):
    """The event_log table should be created by init_tables()."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    rows = store.query_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='event_log'")
    assert len(rows) == 1
    assert rows[0]["name"] == "event_log"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_log.py::test_event_log_table_exists -v`
Expected: FAIL — `event_log` table does not exist.

- [ ] **Step 3: Add EventLog model to models.py**

Add after `ExplanationResult` (around line 220):

```python
# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


class EventLog(BaseModel):
    """A single event in the LENS audit log."""

    id: int | None = None
    timestamp: str
    kind: str  # ingest | extract | build | lint | fix
    action: str  # e.g. paper.added, orphan.found
    target_type: str | None = None  # paper | vocabulary | extraction | matrix
    target_id: str | None = None
    detail: dict | None = None
    session_id: str | None = None
```

- [ ] **Step 4: Add event_log DDL to _TABLE_DDL in store.py**

Append to the `_TABLE_DDL` list (after the `ideation_gaps` entry, around line 131):

```python
    """CREATE TABLE IF NOT EXISTS event_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        kind TEXT NOT NULL,
        action TEXT NOT NULL,
        target_type TEXT,
        target_id TEXT,
        detail TEXT,
        session_id TEXT
    )""",
```

- [ ] **Step 5: Add event_log to JSON_FIELDS in store.py**

Add to the `JSON_FIELDS` dict (around line 26):

```python
    "event_log": {"detail"},
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_event_log.py::test_event_log_table_exists -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/lens/store/models.py src/lens/store/store.py tests/test_event_log.py
git commit -m "feat: add EventLog model and event_log table schema"
```

---

### Task 2: log_event() Helper + query_events()

**Files:**
- Create: `src/lens/knowledge/events.py`
- Test: `tests/test_event_log.py`

- [ ] **Step 1: Write failing tests for log_event and query_events**

Append to `tests/test_event_log.py`:

```python
from lens.knowledge.events import log_event, query_events


def test_log_event_writes(tmp_path):
    """log_event() should insert a row into event_log."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    log_event(store, kind="ingest", action="paper.added",
              target_type="paper", target_id="test-paper-1",
              detail={"title": "Test Paper", "source": "arxiv"},
              session_id="sess-001")

    rows = store.query("event_log")
    assert len(rows) == 1
    assert rows[0]["kind"] == "ingest"
    assert rows[0]["action"] == "paper.added"
    assert rows[0]["target_type"] == "paper"
    assert rows[0]["target_id"] == "test-paper-1"
    assert rows[0]["detail"] == {"title": "Test Paper", "source": "arxiv"}
    assert rows[0]["session_id"] == "sess-001"
    assert rows[0]["timestamp"]  # non-empty


def test_log_event_session_grouping(tmp_path):
    """Events with the same session_id should be queryable together."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    log_event(store, kind="ingest", action="paper.added",
              target_id="p1", session_id="sess-A")
    log_event(store, kind="extract", action="extraction.completed",
              target_id="p1", session_id="sess-A")
    log_event(store, kind="ingest", action="paper.added",
              target_id="p2", session_id="sess-B")

    events = query_events(store, session_id="sess-A")
    assert len(events) == 2
    assert all(e["session_id"] == "sess-A" for e in events)


def test_query_events_filters(tmp_path):
    """query_events() should support kind, since, and limit filters."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    log_event(store, kind="ingest", action="paper.added", target_id="p1")
    log_event(store, kind="extract", action="extraction.completed", target_id="p1")
    log_event(store, kind="ingest", action="paper.added", target_id="p2")

    # Filter by kind
    ingest_events = query_events(store, kind="ingest")
    assert len(ingest_events) == 2

    # Limit
    limited = query_events(store, limit=1)
    assert len(limited) == 1

    # Since (all events are from today, so a past date returns all)
    all_events = query_events(store, since="2020-01-01")
    assert len(all_events) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_event_log.py -v -k "not table_exists"`
Expected: FAIL — `ModuleNotFoundError: No module named 'lens.knowledge.events'`

- [ ] **Step 3: Implement events.py**

```python
# src/lens/knowledge/events.py
"""Unified event log for LENS — records all mutations for audit and lint."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from lens.store.store import LensStore


def log_event(
    store: LensStore,
    kind: str,
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    detail: dict | None = None,
    session_id: str | None = None,
) -> None:
    """Append one event to the event_log table."""
    store.add_rows(
        "event_log",
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "action": action,
                "target_type": target_type,
                "target_id": target_id,
                "detail": detail or {},
                "session_id": session_id,
            }
        ],
    )


def query_events(
    store: LensStore,
    kind: str | None = None,
    since: str | None = None,
    limit: int = 20,
    session_id: str | None = None,
) -> list[dict]:
    """Query event_log with optional filters. Returns newest-first."""
    clauses: list[str] = []
    params: list[str | int] = []

    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if since:
        clauses.append("timestamp >= ?")
        params.append(since)
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)

    where = " AND ".join(clauses) if clauses else "1 = 1"
    sql = f"SELECT * FROM event_log WHERE {where} ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = store.query_sql(sql, tuple(params))
    for row in rows:
        if isinstance(row.get("detail"), str):
            row["detail"] = json.loads(row["detail"])
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_event_log.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/lens/knowledge/events.py tests/test_event_log.py
git commit -m "feat: add log_event() helper and query_events() for event log"
```

---

### Task 3: Lint Check — Orphan Vocabulary

**Files:**
- Create: `src/lens/knowledge/linter.py`
- Test: `tests/test_linter.py`

- [ ] **Step 1: Write failing tests for orphan detection and fix**

```python
# tests/test_linter.py
"""Tests for the LENS knowledge base linter."""

from lens.knowledge.linter import check_orphan_vocabulary, fix_orphans
from lens.store.store import LensStore
from lens.taxonomy.vocabulary import load_seed_vocabulary


def test_lint_orphan_vocabulary(tmp_path):
    """Extracted entries with paper_count=0 should be flagged as orphans."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    # Add an extracted entry with no papers
    store.add_rows("vocabulary", [{
        "id": "orphan-concept",
        "name": "Orphan Concept",
        "kind": "parameter",
        "description": "A concept with no evidence",
        "source": "extracted",
        "first_seen": "2026-04-01",
        "paper_count": 0,
        "avg_confidence": 0.0,
    }])

    orphans = check_orphan_vocabulary(store)
    assert len(orphans) == 1
    assert orphans[0]["id"] == "orphan-concept"


def test_lint_orphan_ignores_seed(tmp_path):
    """Seed vocabulary with paper_count=0 should NOT be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    orphans = check_orphan_vocabulary(store)
    assert len(orphans) == 0


def test_lint_fix_orphans(tmp_path):
    """fix_orphans() should delete orphan entries."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows("vocabulary", [{
        "id": "orphan-concept",
        "name": "Orphan Concept",
        "kind": "parameter",
        "description": "A concept with no evidence",
        "source": "extracted",
        "first_seen": "2026-04-01",
        "paper_count": 0,
        "avg_confidence": 0.0,
    }])

    deleted = fix_orphans(store)
    assert len(deleted) == 1
    assert deleted[0] == "orphan-concept"

    remaining = store.query("vocabulary", "id = ?", ("orphan-concept",))
    assert len(remaining) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_linter.py -v -k orphan`
Expected: FAIL — `ModuleNotFoundError: No module named 'lens.knowledge.linter'`

- [ ] **Step 3: Implement orphan check in linter.py**

```python
# src/lens/knowledge/linter.py
"""LENS knowledge base linter — health checks with optional auto-fix."""

from __future__ import annotations

import logging

from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def check_orphan_vocabulary(store: LensStore) -> list[dict]:
    """Find extracted vocabulary entries with zero paper references."""
    return store.query_sql(
        "SELECT id, name, kind, description, source, paper_count "
        "FROM vocabulary WHERE paper_count = 0 AND source != 'seed'"
    )


def fix_orphans(store: LensStore) -> list[str]:
    """Delete orphan vocabulary entries. Returns list of deleted IDs."""
    orphans = check_orphan_vocabulary(store)
    deleted_ids = []
    for orphan in orphans:
        store.delete("vocabulary", "id = ?", (orphan["id"],))
        deleted_ids.append(orphan["id"])
    return deleted_ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_linter.py -v -k orphan`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/lens/knowledge/linter.py tests/test_linter.py
git commit -m "feat: add orphan vocabulary lint check with fix"
```

---

### Task 4: Lint Check — Contradictions

**Files:**
- Modify: `src/lens/knowledge/linter.py`
- Modify: `tests/test_linter.py`

- [ ] **Step 1: Write failing test for contradiction detection**

Append to `tests/test_linter.py`:

```python
from lens.knowledge.linter import check_contradictions


def test_lint_contradictions(tmp_path):
    """Opposing matrix cells (A->B and B->A) should be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    # A improves latency, worsens accuracy (2 papers)
    store.add_rows("matrix_cells", [
        {
            "improving_param_id": "inference-latency",
            "worsening_param_id": "model-accuracy",
            "principle_id": "quantization",
            "count": 2,
            "avg_confidence": 0.8,
            "paper_ids": ["p1", "p2"],
            "taxonomy_version": 1,
        },
        # B improves accuracy, worsens latency (2 papers) — contradiction!
        {
            "improving_param_id": "model-accuracy",
            "worsening_param_id": "inference-latency",
            "principle_id": "quantization",
            "count": 2,
            "avg_confidence": 0.7,
            "paper_ids": ["p3", "p4"],
            "taxonomy_version": 1,
        },
    ])

    contradictions = check_contradictions(store)
    assert len(contradictions) == 1
    pair = contradictions[0]
    assert set(pair["params"]) == {"inference-latency", "model-accuracy"}
    assert pair["principle_id"] == "quantization"


def test_lint_contradictions_ignores_weak(tmp_path):
    """Single-paper contradictions (count < 2) should not be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    store.add_rows("matrix_cells", [
        {
            "improving_param_id": "inference-latency",
            "worsening_param_id": "model-accuracy",
            "principle_id": "quantization",
            "count": 2,
            "avg_confidence": 0.8,
            "paper_ids": ["p1", "p2"],
            "taxonomy_version": 1,
        },
        {
            "improving_param_id": "model-accuracy",
            "worsening_param_id": "inference-latency",
            "principle_id": "quantization",
            "count": 1,  # only 1 paper — not strong enough
            "avg_confidence": 0.7,
            "paper_ids": ["p3"],
            "taxonomy_version": 1,
        },
    ])

    contradictions = check_contradictions(store)
    assert len(contradictions) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_linter.py -v -k contradiction`
Expected: FAIL — `ImportError: cannot import name 'check_contradictions'`

- [ ] **Step 3: Implement contradiction check**

Add to `src/lens/knowledge/linter.py`:

```python
def check_contradictions(store: LensStore, min_count: int = 2) -> list[dict]:
    """Find parameter pairs with opposing directionality in the matrix.

    A contradiction exists when both (A improves, B worsens, principle P)
    and (B improves, A worsens, principle P) exist, each with count >= min_count.
    """
    cells = store.query("matrix_cells")
    if not cells:
        return []

    # Index cells by (param_set, principle) for fast lookup
    by_principle: dict[str, list[dict]] = {}
    for cell in cells:
        by_principle.setdefault(cell["principle_id"], []).append(cell)

    contradictions = []
    seen: set[tuple[str, str, str]] = set()

    for principle_id, group in by_principle.items():
        for cell in group:
            if cell["count"] < min_count:
                continue
            imp = cell["improving_param_id"]
            wors = cell["worsening_param_id"]

            # Look for the reverse
            for other in group:
                if other["count"] < min_count:
                    continue
                if other["improving_param_id"] == wors and other["worsening_param_id"] == imp:
                    key = (min(imp, wors), max(imp, wors), principle_id)
                    if key not in seen:
                        seen.add(key)
                        contradictions.append({
                            "params": [imp, wors],
                            "principle_id": principle_id,
                            "forward_count": cell["count"],
                            "reverse_count": other["count"],
                        })
    return contradictions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_linter.py -v -k contradiction`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/lens/knowledge/linter.py tests/test_linter.py
git commit -m "feat: add contradiction lint check"
```

---

### Task 5: Lint Check — Weak Evidence

**Files:**
- Modify: `src/lens/knowledge/linter.py`
- Modify: `tests/test_linter.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_linter.py`:

```python
from lens.knowledge.linter import check_weak_evidence


def test_lint_weak_evidence(tmp_path):
    """Entries with paper_count=1 or low confidence should be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows("vocabulary", [
        {
            "id": "strong-concept",
            "name": "Strong Concept",
            "kind": "parameter",
            "description": "Well-supported",
            "source": "extracted",
            "first_seen": "2026-04-01",
            "paper_count": 5,
            "avg_confidence": 0.8,
        },
        {
            "id": "weak-one-paper",
            "name": "Weak One Paper",
            "kind": "parameter",
            "description": "Only one paper",
            "source": "extracted",
            "first_seen": "2026-04-01",
            "paper_count": 1,
            "avg_confidence": 0.9,
        },
        {
            "id": "weak-low-conf",
            "name": "Weak Low Conf",
            "kind": "principle",
            "description": "Low confidence",
            "source": "extracted",
            "first_seen": "2026-04-01",
            "paper_count": 3,
            "avg_confidence": 0.3,
        },
    ])

    findings = check_weak_evidence(store, confidence_threshold=0.5)
    ids = {f["id"] for f in findings}
    assert "weak-one-paper" in ids
    assert "weak-low-conf" in ids
    assert "strong-concept" not in ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_linter.py::test_lint_weak_evidence -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement weak evidence check**

Add to `src/lens/knowledge/linter.py`:

```python
def check_weak_evidence(
    store: LensStore, confidence_threshold: float = 0.5
) -> list[dict]:
    """Find vocabulary entries with thin evidence (1 paper or low confidence)."""
    return store.query_sql(
        "SELECT id, name, kind, paper_count, avg_confidence "
        "FROM vocabulary "
        "WHERE paper_count = 1 OR avg_confidence < ?",
        (confidence_threshold,),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_linter.py::test_lint_weak_evidence -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lens/knowledge/linter.py tests/test_linter.py
git commit -m "feat: add weak evidence lint check"
```

---

### Task 6: Lint Check — Missing Embeddings

**Files:**
- Modify: `src/lens/knowledge/linter.py`
- Modify: `tests/test_linter.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_linter.py`:

```python
from lens.knowledge.linter import check_missing_embeddings, fix_missing_embeddings


def test_lint_missing_embeddings(tmp_path):
    """Vocabulary entries without a vec row should be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    # Add vocab entry WITHOUT embedding (no companion vec row)
    store.add_rows("vocabulary", [{
        "id": "no-vec",
        "name": "No Vec Entry",
        "kind": "parameter",
        "description": "Missing embedding",
        "source": "extracted",
        "first_seen": "2026-04-01",
        "paper_count": 2,
        "avg_confidence": 0.8,
    }])

    findings = check_missing_embeddings(store)
    assert len(findings) == 1
    assert findings[0]["id"] == "no-vec"


def test_lint_missing_embeddings_none_when_present(tmp_path):
    """Vocabulary entries WITH a vec row should not be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows("vocabulary", [{
        "id": "has-vec",
        "name": "Has Vec Entry",
        "kind": "parameter",
        "description": "Has embedding",
        "source": "extracted",
        "first_seen": "2026-04-01",
        "paper_count": 2,
        "avg_confidence": 0.8,
        "embedding": [0.1] * 768,
    }])

    findings = check_missing_embeddings(store)
    assert len(findings) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_linter.py -v -k missing_embedding`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement missing embeddings check**

Add to `src/lens/knowledge/linter.py`:

```python
def check_missing_embeddings(store: LensStore) -> list[dict]:
    """Find vocabulary entries with no corresponding row in vocabulary_vec."""
    return store.query_sql(
        "SELECT v.id, v.name, v.kind "
        "FROM vocabulary v "
        "LEFT JOIN vocabulary_vec vv ON v.id = vv.id "
        "WHERE vv.id IS NULL"
    )


def fix_missing_embeddings(
    store: LensStore,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
) -> list[str]:
    """Generate and store embeddings for entries missing them. Returns fixed IDs."""
    from lens.taxonomy.embedder import embed_strings

    missing = check_missing_embeddings(store)
    if not missing:
        return []

    texts = [f"{r['name']}: {r.get('kind', '')}" for r in missing]
    embeddings = embed_strings(
        texts,
        provider=embedding_provider,
        model_name=embedding_model,
        api_base=embedding_api_base,
        api_key=embedding_api_key,
    )

    fixed_ids = []
    for row, emb in zip(missing, embeddings, strict=True):
        store.upsert_embedding("vocabulary", row["id"], emb.tolist())
        fixed_ids.append(row["id"])
    return fixed_ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_linter.py -v -k missing_embedding`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/lens/knowledge/linter.py tests/test_linter.py
git commit -m "feat: add missing embeddings lint check with fix"
```

---

### Task 7: Lint Check — Stale Extractions

**Files:**
- Modify: `src/lens/knowledge/linter.py`
- Modify: `tests/test_linter.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_linter.py`:

```python
from lens.knowledge.linter import check_stale_extractions, fix_stale_extractions
from lens.store.models import EMBEDDING_DIM


def test_lint_stale_extractions(tmp_path):
    """Papers with non-complete status should be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows("papers", [
        {
            "paper_id": "complete-paper",
            "title": "Good Paper",
            "abstract": "All done",
            "authors": ["A"],
            "date": "2026-01-01",
            "arxiv_id": "2601.00001",
            "extraction_status": "complete",
            "embedding": [0.0] * EMBEDDING_DIM,
        },
        {
            "paper_id": "failed-paper",
            "title": "Bad Paper",
            "abstract": "Failed",
            "authors": ["B"],
            "date": "2026-01-01",
            "arxiv_id": "2601.00002",
            "extraction_status": "failed",
            "embedding": [0.0] * EMBEDDING_DIM,
        },
        {
            "paper_id": "pending-paper",
            "title": "Waiting Paper",
            "abstract": "Pending",
            "authors": ["C"],
            "date": "2026-01-01",
            "arxiv_id": "2601.00003",
            "extraction_status": "pending",
            "embedding": [0.0] * EMBEDDING_DIM,
        },
    ])

    findings = check_stale_extractions(store)
    ids = {f["paper_id"] for f in findings}
    assert "failed-paper" in ids
    assert "pending-paper" in ids
    assert "complete-paper" not in ids


def test_lint_fix_stale_requeues(tmp_path):
    """fix_stale_extractions() should reset status to pending."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows("papers", [{
        "paper_id": "failed-paper",
        "title": "Bad Paper",
        "abstract": "Failed",
        "authors": ["B"],
        "date": "2026-01-01",
        "arxiv_id": "2601.00002",
        "extraction_status": "failed",
        "embedding": [0.0] * EMBEDDING_DIM,
    }])

    requeued = fix_stale_extractions(store)
    assert requeued == ["failed-paper"]

    paper = store.query("papers", "paper_id = ?", ("failed-paper",))
    assert paper[0]["extraction_status"] == "pending"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_linter.py -v -k stale`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement stale extractions check**

Add to `src/lens/knowledge/linter.py`:

```python
def check_stale_extractions(store: LensStore) -> list[dict]:
    """Find papers with non-complete extraction status."""
    return store.query_sql(
        "SELECT paper_id, title, extraction_status "
        "FROM papers "
        "WHERE extraction_status IN ('pending', 'incomplete', 'failed')"
    )


def fix_stale_extractions(store: LensStore) -> list[str]:
    """Reset stale papers to 'pending' for re-extraction. Returns fixed paper_ids."""
    stale = check_stale_extractions(store)
    fixed_ids = []
    for paper in stale:
        store.update(
            "papers",
            "extraction_status = ?",
            "paper_id = ?",
            ("pending", paper["paper_id"]),
        )
        fixed_ids.append(paper["paper_id"])
    return fixed_ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_linter.py -v -k stale`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/lens/knowledge/linter.py tests/test_linter.py
git commit -m "feat: add stale extractions lint check with fix"
```

---

### Task 8: Lint Check — Near-Duplicates

**Files:**
- Modify: `src/lens/knowledge/linter.py`
- Modify: `tests/test_linter.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_linter.py`:

```python
import numpy as np

from lens.knowledge.linter import check_near_duplicates


def test_lint_near_duplicates(tmp_path):
    """Vocabulary entries with very similar embeddings should be paired."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    # Create two nearly identical embeddings
    base_emb = np.random.RandomState(42).randn(EMBEDDING_DIM).astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb)
    similar_emb = base_emb + np.random.RandomState(43).randn(EMBEDDING_DIM).astype(np.float32) * 0.01
    similar_emb = similar_emb / np.linalg.norm(similar_emb)
    different_emb = np.random.RandomState(99).randn(EMBEDDING_DIM).astype(np.float32)
    different_emb = different_emb / np.linalg.norm(different_emb)

    store.add_rows("vocabulary", [
        {
            "id": "concept-a",
            "name": "Concept A",
            "kind": "parameter",
            "description": "First concept",
            "source": "extracted",
            "first_seen": "2026-04-01",
            "paper_count": 3,
            "avg_confidence": 0.8,
            "embedding": base_emb.tolist(),
        },
        {
            "id": "concept-a-variant",
            "name": "Concept A Variant",
            "kind": "parameter",
            "description": "Nearly identical to first",
            "source": "extracted",
            "first_seen": "2026-04-01",
            "paper_count": 1,
            "avg_confidence": 0.7,
            "embedding": similar_emb.tolist(),
        },
        {
            "id": "concept-b",
            "name": "Concept B",
            "kind": "parameter",
            "description": "Completely different",
            "source": "extracted",
            "first_seen": "2026-04-01",
            "paper_count": 2,
            "avg_confidence": 0.9,
            "embedding": different_emb.tolist(),
        },
    ])

    pairs = check_near_duplicates(store, similarity_threshold=0.92)
    assert len(pairs) == 1
    pair_ids = {pairs[0]["id_a"], pairs[0]["id_b"]}
    assert pair_ids == {"concept-a", "concept-a-variant"}


def test_lint_near_duplicates_different_kinds(tmp_path):
    """Near-duplicates across different kinds should NOT be paired."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    base_emb = np.random.RandomState(42).randn(EMBEDDING_DIM).astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb)

    store.add_rows("vocabulary", [
        {
            "id": "param-x",
            "name": "Param X",
            "kind": "parameter",
            "description": "A parameter",
            "source": "extracted",
            "first_seen": "2026-04-01",
            "paper_count": 2,
            "avg_confidence": 0.8,
            "embedding": base_emb.tolist(),
        },
        {
            "id": "principle-x",
            "name": "Principle X",
            "kind": "principle",
            "description": "A principle",
            "source": "extracted",
            "first_seen": "2026-04-01",
            "paper_count": 2,
            "avg_confidence": 0.8,
            "embedding": base_emb.tolist(),
        },
    ])

    pairs = check_near_duplicates(store, similarity_threshold=0.92)
    assert len(pairs) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_linter.py -v -k near_duplicate`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement near-duplicates check**

Add to `src/lens/knowledge/linter.py`. Uses sqlite-vec's `MATCH` subquery to retrieve each entry's embedding directly from the vec table and find nearest neighbors — avoids needing to deserialize binary embeddings in Python:

```python
def check_near_duplicates(
    store: LensStore, similarity_threshold: float = 0.92
) -> list[dict]:
    """Find vocabulary entries with cosine similarity above threshold within the same kind.

    Uses sqlite-vec's MATCH subquery to compare embeddings directly in SQL.
    Returns deduplicated pairs (A,B not B,A).
    """
    vocab = store.query("vocabulary")
    if not vocab:
        return []

    by_kind: dict[str, list[dict]] = {}
    for entry in vocab:
        by_kind.setdefault(entry["kind"], []).append(entry)

    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    max_distance = 1.0 - similarity_threshold

    for kind, entries in by_kind.items():
        if len(entries) < 2:
            continue

        entry_ids = {e["id"] for e in entries}

        for entry in entries:
            try:
                neighbors = store.query_sql(
                    "SELECT id, distance "
                    "FROM vocabulary_vec "
                    "WHERE embedding MATCH (SELECT embedding FROM vocabulary_vec WHERE id = ?) "
                    "AND k = ? AND id != ?",
                    (entry["id"], len(entries), entry["id"]),
                )
            except Exception:
                continue

            for neighbor in neighbors:
                if neighbor["distance"] > max_distance:
                    continue
                if neighbor["id"] not in entry_ids:
                    continue

                key = (min(entry["id"], neighbor["id"]), max(entry["id"], neighbor["id"]))
                if key in seen:
                    continue
                seen.add(key)

                neighbor_entry = next(e for e in entries if e["id"] == neighbor["id"])
                pairs.append({
                    "id_a": entry["id"],
                    "name_a": entry["name"],
                    "id_b": neighbor["id"],
                    "name_b": neighbor_entry["name"],
                    "kind": kind,
                    "distance": neighbor["distance"],
                })
    return pairs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_linter.py -v -k near_duplicate`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/lens/knowledge/linter.py tests/test_linter.py
git commit -m "feat: add near-duplicates lint check"
```

---

### Task 9: Lint Orchestrator + LintReport

**Files:**
- Modify: `src/lens/store/models.py` (add `LintReport`)
- Modify: `src/lens/knowledge/linter.py` (add `lint()` orchestrator)
- Modify: `tests/test_linter.py`

- [ ] **Step 1: Add LintReport model**

Add to `src/lens/store/models.py` after `EventLog`:

```python
class LintReport(BaseModel):
    """Summary of a lint run."""

    orphans: list[dict] = []
    contradictions: list[dict] = []
    weak_evidence: list[dict] = []
    missing_embeddings: list[dict] = []
    stale_extractions: list[dict] = []
    near_duplicates: list[dict] = []
    fixes_applied: list[dict] = []
```

- [ ] **Step 2: Write failing test for lint orchestrator**

Append to `tests/test_linter.py`:

```python
from lens.knowledge.linter import lint


def test_lint_orchestrator_runs_all_checks(tmp_path):
    """lint() should run all checks and return a LintReport."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    # Add one orphan to make the report non-empty
    store.add_rows("vocabulary", [{
        "id": "orphan-test",
        "name": "Orphan Test",
        "kind": "parameter",
        "description": "No papers",
        "source": "extracted",
        "first_seen": "2026-04-01",
        "paper_count": 0,
        "avg_confidence": 0.0,
    }])

    report = lint(store)
    assert len(report.orphans) == 1
    assert report.orphans[0]["id"] == "orphan-test"
    assert isinstance(report.contradictions, list)
    assert isinstance(report.weak_evidence, list)
    assert isinstance(report.missing_embeddings, list)
    assert isinstance(report.stale_extractions, list)
    assert isinstance(report.near_duplicates, list)
    assert report.fixes_applied == []


def test_lint_with_fix(tmp_path):
    """lint(fix=True) should apply fixes and record them."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows("vocabulary", [{
        "id": "orphan-fix-test",
        "name": "Orphan Fix Test",
        "kind": "parameter",
        "description": "Will be deleted",
        "source": "extracted",
        "first_seen": "2026-04-01",
        "paper_count": 0,
        "avg_confidence": 0.0,
    }])

    report = lint(store, fix=True)
    assert len(report.fixes_applied) >= 1
    assert any(f["action"] == "orphan.deleted" for f in report.fixes_applied)

    # Verify the orphan was actually deleted
    remaining = store.query("vocabulary", "id = ?", ("orphan-fix-test",))
    assert len(remaining) == 0


def test_lint_check_filter(tmp_path):
    """lint(checks=['orphans']) should only run the orphan check."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows("vocabulary", [{
        "id": "orphan-filter",
        "name": "Orphan Filter",
        "kind": "parameter",
        "description": "No papers",
        "source": "extracted",
        "first_seen": "2026-04-01",
        "paper_count": 0,
        "avg_confidence": 0.0,
    }])

    # Add a stale paper too — but we only run orphan check
    store.add_rows("papers", [{
        "paper_id": "stale-p",
        "title": "Stale",
        "abstract": "Stale",
        "authors": ["A"],
        "date": "2026-01-01",
        "arxiv_id": "2601.00099",
        "extraction_status": "failed",
        "embedding": [0.0] * EMBEDDING_DIM,
    }])

    report = lint(store, checks=["orphans"])
    assert len(report.orphans) == 1
    assert report.stale_extractions == []  # not run
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_linter.py -v -k "orchestrator or with_fix or check_filter"`
Expected: FAIL — `ImportError: cannot import name 'lint'`

- [ ] **Step 4: Implement lint orchestrator**

Add to `src/lens/knowledge/linter.py`:

```python
from lens.knowledge.events import log_event
from lens.store.models import LintReport


# Map of check name -> (check_function, fix_function_or_None)
_ALL_CHECKS = {
    "orphans": "check_orphan_vocabulary",
    "contradictions": "check_contradictions",
    "weak_evidence": "check_weak_evidence",
    "missing_embeddings": "check_missing_embeddings",
    "stale": "check_stale_extractions",
    "near_duplicates": "check_near_duplicates",
}


def lint(
    store: LensStore,
    fix: bool = False,
    session_id: str | None = None,
    checks: list[str] | None = None,
    confidence_threshold: float = 0.5,
    similarity_threshold: float = 0.92,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
) -> LintReport:
    """Run lint checks and optionally apply fixes. Returns a LintReport."""
    active_checks = set(checks) if checks else set(_ALL_CHECKS.keys())

    report = LintReport()

    # 1. Orphans
    if "orphans" in active_checks:
        report.orphans = check_orphan_vocabulary(store)
        for finding in report.orphans:
            log_event(store, "lint", "orphan.found",
                      target_type="vocabulary", target_id=finding["id"],
                      detail={"name": finding["name"], "kind": finding["kind"]},
                      session_id=session_id)

    # 2. Contradictions
    if "contradictions" in active_checks:
        report.contradictions = check_contradictions(store)
        for finding in report.contradictions:
            log_event(store, "lint", "contradiction.found",
                      target_type="matrix",
                      detail={"params": finding["params"],
                              "principle_id": finding["principle_id"]},
                      session_id=session_id)

    # 3. Weak evidence
    if "weak_evidence" in active_checks:
        report.weak_evidence = check_weak_evidence(store, confidence_threshold)
        for finding in report.weak_evidence:
            log_event(store, "lint", "weak_evidence.found",
                      target_type="vocabulary", target_id=finding["id"],
                      detail={"paper_count": finding["paper_count"],
                              "avg_confidence": finding["avg_confidence"]},
                      session_id=session_id)

    # 4. Missing embeddings
    if "missing_embeddings" in active_checks:
        report.missing_embeddings = check_missing_embeddings(store)
        for finding in report.missing_embeddings:
            log_event(store, "lint", "missing_embedding.found",
                      target_type="vocabulary", target_id=finding["id"],
                      detail={"name": finding["name"]},
                      session_id=session_id)

    # 5. Stale extractions
    if "stale" in active_checks:
        report.stale_extractions = check_stale_extractions(store)
        for finding in report.stale_extractions:
            log_event(store, "lint", "stale_extraction.found",
                      target_type="paper", target_id=finding["paper_id"],
                      detail={"status": finding["extraction_status"]},
                      session_id=session_id)

    # 6. Near-duplicates
    if "near_duplicates" in active_checks:
        report.near_duplicates = check_near_duplicates(store, similarity_threshold)
        for finding in report.near_duplicates:
            log_event(store, "lint", "near_duplicate.found",
                      target_type="vocabulary",
                      detail={"id_a": finding["id_a"], "id_b": finding["id_b"],
                              "kind": finding["kind"]},
                      session_id=session_id)

    # Apply fixes if requested
    if fix:
        if "orphans" in active_checks and report.orphans:
            deleted = fix_orphans(store)
            for oid in deleted:
                report.fixes_applied.append({"action": "orphan.deleted", "target_id": oid})
                log_event(store, "fix", "orphan.deleted",
                          target_type="vocabulary", target_id=oid,
                          session_id=session_id)

        if "missing_embeddings" in active_checks and report.missing_embeddings:
            fixed = fix_missing_embeddings(
                store, embedding_provider, embedding_model,
                embedding_api_base, embedding_api_key,
            )
            for fid in fixed:
                report.fixes_applied.append({"action": "embedding.repaired", "target_id": fid})
                log_event(store, "fix", "embedding.repaired",
                          target_type="vocabulary", target_id=fid,
                          session_id=session_id)

        if "stale" in active_checks and report.stale_extractions:
            requeued = fix_stale_extractions(store)
            for pid in requeued:
                report.fixes_applied.append({"action": "extraction.requeued", "target_id": pid})
                log_event(store, "fix", "extraction.requeued",
                          target_type="paper", target_id=pid,
                          session_id=session_id)

    return report
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_linter.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/lens/store/models.py src/lens/knowledge/linter.py tests/test_linter.py
git commit -m "feat: add lint orchestrator with LintReport and event logging"
```

---

### Task 10: CLI — `lens lint` Command

**Files:**
- Modify: `src/lens/cli.py`

- [ ] **Step 1: Add lint command to cli.py**

Add a new Typer command after the existing `vocab_app` commands (around line 840). Add `uuid4` import at the top of cli.py:

At the top of `cli.py`, add to imports:
```python
from uuid import uuid4
```

Then add the command:

```python
@app.command()
def lint(
    fix: bool = typer.Option(False, "--fix", help="Apply safe auto-fixes after reporting."),
    check: str | None = typer.Option(None, "--check", help="Comma-separated checks to run (orphans,contradictions,weak_evidence,missing_embeddings,stale,near_duplicates)."),
    threshold_confidence: float = typer.Option(0.5, "--threshold-confidence", help="Weak evidence confidence cutoff."),
    threshold_similarity: float = typer.Option(0.92, "--threshold-similarity", help="Near-duplicate cosine similarity threshold."),
) -> None:
    """Health-check the knowledge base for issues."""
    from lens.knowledge.linter import lint as run_lint

    store = _get_store()
    config = load_config()
    session_id = str(uuid4())[:8]

    checks = [c.strip() for c in check.split(",")] if check else None

    emb_cfg = config.get("embeddings", {})
    report = run_lint(
        store,
        fix=fix,
        session_id=session_id,
        checks=checks,
        confidence_threshold=threshold_confidence,
        similarity_threshold=threshold_similarity,
        embedding_provider=emb_cfg.get("provider", "local"),
        embedding_model=emb_cfg.get("model"),
        embedding_api_base=emb_cfg.get("api_base"),
        embedding_api_key=emb_cfg.get("api_key"),
    )

    typer.echo(f"\nLint Report (session {session_id})")
    typer.echo("─" * 36)
    typer.echo(f"  Orphan vocabulary:     {len(report.orphans)} found")
    typer.echo(f"  Contradictions:        {len(report.contradictions)} found")
    typer.echo(f"  Weak evidence:         {len(report.weak_evidence)} found")
    typer.echo(f"  Missing embeddings:    {len(report.missing_embeddings)} found")
    typer.echo(f"  Stale extractions:     {len(report.stale_extractions)} found")
    typer.echo(f"  Near-duplicates:       {len(report.near_duplicates)} pairs found")
    typer.echo("─" * 36)

    total = (len(report.orphans) + len(report.contradictions) + len(report.weak_evidence)
             + len(report.missing_embeddings) + len(report.stale_extractions)
             + len(report.near_duplicates))
    typer.echo(f"  Total issues:          {total}\n")

    if fix and report.fixes_applied:
        orphan_fixes = sum(1 for f in report.fixes_applied if f["action"] == "orphan.deleted")
        emb_fixes = sum(1 for f in report.fixes_applied if f["action"] == "embedding.repaired")
        requeue_fixes = sum(1 for f in report.fixes_applied if f["action"] == "extraction.requeued")
        if orphan_fixes:
            typer.echo(f"  Fixed: deleted {orphan_fixes} orphan vocabulary entries")
        if emb_fixes:
            typer.echo(f"  Fixed: embedded {emb_fixes} missing vocabulary entries")
        if requeue_fixes:
            typer.echo(f"  Fixed: requeued {requeue_fixes} stale extractions")
        typer.echo()
    elif not fix and total > 0:
        typer.echo("Use --fix to apply safe auto-fixes.\n")
```

- [ ] **Step 2: Run CLI smoke test**

Run: `uv run lens lint --help`
Expected: Shows help text with `--fix`, `--check`, `--threshold-confidence`, `--threshold-similarity` options.

- [ ] **Step 3: Commit**

```bash
git add src/lens/cli.py
git commit -m "feat: add lens lint CLI command"
```

---

### Task 11: CLI — `lens log` Command

**Files:**
- Modify: `src/lens/cli.py`

- [ ] **Step 1: Add log command to cli.py**

Add after the `lint` command:

```python
@app.command(name="log")
def show_log(
    kind: str | None = typer.Option(None, "--kind", help="Filter by event kind (ingest, extract, build, lint, fix)."),
    since: str | None = typer.Option(None, "--since", help="Show events after this date (YYYY-MM-DD)."),
    limit: int = typer.Option(20, "--limit", help="Max events to show."),
    session: str | None = typer.Option(None, "--session", help="Show events from a specific session."),
) -> None:
    """Show the event log."""
    from lens.knowledge.events import query_events

    store = _get_store()
    events = query_events(store, kind=kind, since=since, limit=limit, session_id=session)

    if not events:
        typer.echo("No events found.")
        return

    for event in events:
        ts = event["timestamp"][:16].replace("T", " ")  # "2026-04-05 14:23"
        k = event["kind"]
        action = event["action"]
        target = ""
        if event.get("target_type") and event.get("target_id"):
            target = f"{event['target_type']}:{event['target_id']}"

        # Format detail as short parenthetical
        detail_str = ""
        detail = event.get("detail")
        if detail:
            if isinstance(detail, str):
                import json
                detail = json.loads(detail)
            if isinstance(detail, dict):
                parts = [f"{v}" for v in detail.values()]
                if parts:
                    detail_str = f"  ({', '.join(parts[:3])})"

        typer.echo(f"{ts}  {k:<8} {action:<28} {target}{detail_str}")
```

- [ ] **Step 2: Run CLI smoke test**

Run: `uv run lens log --help`
Expected: Shows help text with `--kind`, `--since`, `--limit`, `--session` options.

- [ ] **Step 3: Commit**

```bash
git add src/lens/cli.py
git commit -m "feat: add lens log CLI command"
```

---

### Task 12: Instrument Extraction Pipeline

**Files:**
- Modify: `src/lens/extract/extractor.py:202-230` (Phase 2 write loop)

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_lint_integration.py
"""Integration tests for lint + event log."""

from lens.knowledge.events import query_events
from lens.store.models import EMBEDDING_DIM
from lens.store.store import LensStore
from lens.taxonomy.vocabulary import load_seed_vocabulary


def test_extract_events_logged(tmp_path):
    """After extraction, events should appear in the log."""
    # We can't easily run the full LLM extraction, so we test that
    # the event logging infrastructure works by simulating what
    # extractor.py does post-extraction.
    from lens.knowledge.events import log_event

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    # Simulate: extraction completed for a paper
    log_event(store, "extract", "extraction.completed",
              target_type="paper", target_id="test-paper-1",
              detail={"tradeoffs": 2, "architecture": 1, "agentic": 0},
              session_id="test-session")

    events = query_events(store, kind="extract")
    assert len(events) == 1
    assert events[0]["action"] == "extraction.completed"
    assert events[0]["target_id"] == "test-paper-1"
```

- [ ] **Step 2: Instrument extractor.py**

Add import at top of `src/lens/extract/extractor.py`:

```python
from lens.knowledge.events import log_event
```

In the Phase 2 loop (around lines 202-230), add event logging after each paper:

Replace the section from `if result is None:` through the end of the loop body with:

```python
        if result is None:
            _update_paper_status(store, pid, "incomplete")
            logger.warning("Extraction failed for %s", pid)
            log_event(store, "extract", "extraction.failed",
                      target_type="paper", target_id=pid,
                      session_id=session_id)
            continue

        tradeoffs, architecture, agentic = result
        if tradeoffs:
            store.add_rows("tradeoff_extractions", tradeoffs)
        if architecture:
            store.add_rows("architecture_extractions", architecture)
        if agentic:
            store.add_rows("agentic_extractions", agentic)

        _update_paper_status(store, pid, "complete")
        log_event(store, "extract", "extraction.completed",
                  target_type="paper", target_id=pid,
                  detail={"tradeoffs": len(tradeoffs),
                          "architecture": len(architecture),
                          "agentic": len(agentic)},
                  session_id=session_id)
        logger.info(
            "Extracted %s: %d tradeoffs, %d arch, %d agentic",
            pid,
            len(tradeoffs),
            len(architecture),
            len(agentic),
        )
        success_count += 1
```

Also add `session_id` parameter to `extract_papers`:

```python
async def extract_papers(
    store: LensStore,
    llm_client: LLMClient,
    concurrency: int = 5,
    paper_id: str | None = None,
    session_id: str | None = None,
) -> int:
```

- [ ] **Step 3: Run integration test**

Run: `uv run pytest tests/test_lint_integration.py::test_extract_events_logged -v`
Expected: PASS

- [ ] **Step 4: Run existing extraction tests to confirm no regression**

Run: `uv run pytest tests/test_extract.py tests/test_extraction.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/lens/extract/extractor.py tests/test_lint_integration.py
git commit -m "feat: instrument extraction pipeline with event logging"
```

---

### Task 13: Instrument Vocabulary + Build Pipelines

**Files:**
- Modify: `src/lens/taxonomy/vocabulary.py:254-361` (process_new_concepts + build_vocabulary)
- Modify: `src/lens/knowledge/matrix.py:21-84` (build_matrix)
- Modify: `src/lens/taxonomy/versioning.py:25-50` (record_version)

- [ ] **Step 1: Instrument process_new_concepts in vocabulary.py**

Add import at top of `src/lens/taxonomy/vocabulary.py`:

```python
from lens.knowledge.events import log_event
```

Add `session_id` parameter to `process_new_concepts`:

```python
def process_new_concepts(store: LensStore, session_id: str | None = None) -> dict[str, int]:
```

After `store.add_rows("vocabulary", new_rows)` (around line 303), add:

```python
    if new_rows:
        store.add_rows("vocabulary", new_rows)
        logger.info("Accepted %d new vocabulary entries", len(new_rows))
        for row in new_rows:
            log_event(store, "extract", "vocabulary.created",
                      target_type="vocabulary", target_id=row["id"],
                      detail={"name": row["name"], "kind": row["kind"]},
                      session_id=session_id)
```

After the update loop (around line 321), for updated entries:

```python
        store.update(
            "vocabulary",
            "paper_count = ?, avg_confidence = ?",
            "id = ?",
            (len(unique_papers), round(avg_conf, 4), entry_id),
        )
        log_event(store, "extract", "vocabulary.updated",
                  target_type="vocabulary", target_id=entry_id,
                  detail={"paper_count": len(unique_papers),
                          "avg_confidence": round(avg_conf, 4)},
                  session_id=session_id)
        updated += 1
```

- [ ] **Step 2: Instrument build_vocabulary**

Add `session_id` parameter to `build_vocabulary`:

```python
def build_vocabulary(
    store: LensStore,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
    session_id: str | None = None,
) -> dict[str, int]:
```

Pass `session_id` through to `process_new_concepts`:

```python
    stats = process_new_concepts(store, session_id=session_id)
```

After the FTS rebuild (around line 353), add:

```python
    log_event(store, "build", "taxonomy.built",
              detail={"new_entries": stats["new_entries"],
                      "updated_entries": stats["updated_entries"],
                      "embedded": len(to_embed)},
              session_id=session_id)
```

- [ ] **Step 3: Instrument build_matrix in matrix.py**

Add import at top of `src/lens/knowledge/matrix.py`:

```python
from lens.knowledge.events import log_event
```

Add `session_id` parameter to `build_matrix`:

```python
def build_matrix(store: LensStore, session_id: str | None = None) -> None:
```

After `store.add_rows("matrix_cells", cell_rows)` (around line 83), add:

```python
    if cell_rows:
        store.add_rows("matrix_cells", cell_rows)
        logger.info("Built matrix with %d cells", len(cell_rows))
        log_event(store, "build", "matrix.built",
                  detail={"cells": len(cell_rows)},
                  session_id=session_id)
```

- [ ] **Step 4: Instrument record_version in versioning.py**

Add import at top of `src/lens/taxonomy/versioning.py`:

```python
from lens.knowledge.events import log_event
```

Add `session_id` parameter to `record_version`:

```python
def record_version(
    store: LensStore,
    version_id: int,
    paper_count: int,
    param_count: int,
    principle_count: int,
    slot_count: int = 0,
    variant_count: int = 0,
    pattern_count: int = 0,
    session_id: str | None = None,
) -> None:
```

After the `store.add_rows("taxonomy_versions", [...])` call, add:

```python
    log_event(store, "build", "version.recorded",
              target_type="taxonomy_version",
              target_id=str(version_id),
              detail={"paper_count": paper_count,
                      "param_count": param_count,
                      "principle_count": principle_count},
              session_id=session_id)
```

- [ ] **Step 5: Run existing tests to confirm no regression**

Run: `uv run pytest tests/test_vocabulary.py tests/test_matrix.py tests/test_taxonomy.py -v`
Expected: ALL PASS (session_id defaults to None, so existing callers are unaffected)

- [ ] **Step 6: Commit**

```bash
git add src/lens/taxonomy/vocabulary.py src/lens/knowledge/matrix.py src/lens/taxonomy/versioning.py
git commit -m "feat: instrument vocabulary, matrix, and versioning with event logging"
```

---

### Task 14: Instrument Acquire Pipelines in CLI

**Files:**
- Modify: `src/lens/cli.py` (seed, arxiv, file, openalex commands)

- [ ] **Step 1: Add event logging to acquire commands**

In `cli.py`, add import:
```python
from lens.knowledge.events import log_event
```

**In the `seed()` command** (line 301): The seed command calls `_acquire_seed_async(store)` which returns a count. Since we don't have individual paper objects here, log a single summary event after the call:

```python
def seed() -> None:
    """Ingest curated seed papers from the manifest."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()
    session_id = str(uuid4())[:8]
    count = asyncio.run(_acquire_seed_async(store))
    if count > 0:
        log_event(store, "ingest", "paper.added",
                  detail={"source": "seed", "count": count},
                  session_id=session_id)
    rprint(f"[green]Acquired {count} seed papers[/green]")
```

**In the `arxiv()` command** (line 318): The `papers` list is available after `_fetch_arxiv_async`. Log after `store.add_papers(papers)` (line 342):

```python
    store.add_papers(papers)
    session_id = str(uuid4())[:8]
    for p in papers:
        log_event(store, "ingest", "paper.added",
                  target_type="paper", target_id=p["paper_id"],
                  detail={"title": p["title"], "source": "arxiv"},
                  session_id=session_id)
    rprint(f"[green]Acquired {len(papers)} papers from arxiv[/green]")
```

**In the `file()` command** (line 355): The `paper` dict is returned by `ingest_pdf(path)` (line 373). Log after `store.add_papers([paper])` (line 381):

```python
    store.add_papers([paper])
    session_id = str(uuid4())[:8]
    log_event(store, "ingest", "paper.added",
              target_type="paper", target_id=paper["paper_id"],
              detail={"title": paper["title"], "source": "file"},
              session_id=session_id)
    rprint(f"[green]Ingested {path.name} as paper '{paper['paper_id']}'[/green]")
```

**In the `openalex()` command** (line 386): The enrichment loop iterates `enriched` papers (line 413). Log inside the existing loop after each `store.update`:

```python
    session_id = str(uuid4())[:8]
    updated_count = 0
    for paper in enriched:
        pid = paper.get("paper_id", "")
        store.update(
            "papers",
            "citations = ?, venue = ?",
            "paper_id = ?",
            (paper.get("citations", 0), paper.get("venue"), pid),
        )
        log_event(store, "ingest", "paper.enriched",
                  target_type="paper", target_id=pid,
                  session_id=session_id)
        updated_count += 1
```

**Thread `session_id` into build commands.** In `taxonomy()` (line 437), `build_matrix_cmd()` (line 478), and `build_all()` (line 497):

```python
# In taxonomy():
    session_id = str(uuid4())[:8]
    stats = build_vocabulary(store, ..., session_id=session_id)
    record_version(store, version_id, ..., session_id=session_id)

# In build_matrix_cmd():
    session_id = str(uuid4())[:8]
    build_matrix(store, session_id=session_id)

# In extract():
    session_id = str(uuid4())[:8]
    asyncio.run(extract_papers(store, llm_client, ..., session_id=session_id))
```

- [ ] **Step 2: Run CLI smoke tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add src/lens/cli.py
git commit -m "feat: instrument acquire and build CLI commands with event logging"
```

---

### Task 15: Integration Tests — Lint + Event Log

**Files:**
- Modify: `tests/test_lint_integration.py`

- [ ] **Step 1: Add integration tests**

Append to `tests/test_lint_integration.py`:

```python
from lens.knowledge.linter import lint


def test_lint_log_events_recorded(tmp_path):
    """Running lint should record lint.* events in the event log."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    # Add an orphan to trigger a finding
    store.add_rows("vocabulary", [{
        "id": "integ-orphan",
        "name": "Integration Orphan",
        "kind": "parameter",
        "description": "Test",
        "source": "extracted",
        "first_seen": "2026-04-01",
        "paper_count": 0,
        "avg_confidence": 0.0,
    }])

    lint(store, session_id="integ-sess")

    events = query_events(store, kind="lint", session_id="integ-sess")
    assert len(events) >= 1
    assert any(e["action"] == "orphan.found" for e in events)


def test_fix_events_recorded(tmp_path):
    """Running lint --fix should record fix.* events in the event log."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows("vocabulary", [{
        "id": "fix-orphan",
        "name": "Fix Orphan",
        "kind": "parameter",
        "description": "Will be fixed",
        "source": "extracted",
        "first_seen": "2026-04-01",
        "paper_count": 0,
        "avg_confidence": 0.0,
    }])

    lint(store, fix=True, session_id="fix-sess")

    fix_events = query_events(store, kind="fix", session_id="fix-sess")
    assert len(fix_events) >= 1
    assert any(e["action"] == "orphan.deleted" for e in fix_events)


def test_lint_no_false_positives_on_fresh_data(tmp_path):
    """Lint on a clean DB with seed vocabulary should report no orphans."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    report = lint(store, checks=["orphans"])
    assert len(report.orphans) == 0
```

- [ ] **Step 2: Run all integration tests**

Run: `uv run pytest tests/test_lint_integration.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_lint_integration.py
git commit -m "test: add lint + event log integration tests"
```

---

### Task 16: Run Full Test Suite + Final Verification

**Files:** None — verification only.

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS. No regressions in existing tests.

- [ ] **Step 2: Run lint on a fresh init**

Run: `uv run lens init && uv run lens vocab init && uv run lens lint`
Expected: Shows lint report with 0 orphans (seed vocab is exempt), possibly some missing embeddings if vocab init doesn't embed.

- [ ] **Step 3: Check log**

Run: `uv run lens log`
Expected: Shows recent events (at minimum the lint events from step 2).

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git add -u
git commit -m "chore: final cleanup for event log and lint feature"
```

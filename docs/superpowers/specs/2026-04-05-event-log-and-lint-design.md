# Event Log & Lint — Design Spec

**Date:** 2026-04-05
**Status:** Approved
**Inspired by:** [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

## Overview

Two integrated features for Lens:

1. **Event Log** — a unified `event_log` table recording all mutations (ingest, extract, build, lint, fix) with a `lens log` CLI command for querying.
2. **Lint** — a `lens lint` command that health-checks the knowledge base across six categories, with optional `--fix` for safe auto-repairs.

Both features share the same table: lint findings and fixes are logged as events, enabling history tracking and staleness detection.

## Scope

**In scope:** Event log table + model, `log_event()` helper, instrumentation of existing code paths, six lint checks, report + fix mode, `lens lint` and `lens log` CLI commands, tests.

**Out of scope:** Query promotion (explain results becoming vocabulary entries) — deferred to a follow-up.

---

## 1. Event Log Table

### Schema

```sql
CREATE TABLE IF NOT EXISTS event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    kind TEXT NOT NULL,
    action TEXT NOT NULL,
    target_type TEXT,
    target_id TEXT,
    detail TEXT,
    session_id TEXT
);
```

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK | Auto-increment row ID |
| `timestamp` | TEXT | ISO 8601 datetime |
| `kind` | TEXT | Event category: `ingest`, `extract`, `build`, `lint`, `fix` |
| `action` | TEXT | Specific action: `paper.added`, `orphan.found`, etc. |
| `target_type` | TEXT (nullable) | Affected entity type: `paper`, `vocabulary`, `extraction`, `matrix` |
| `target_id` | TEXT (nullable) | Affected entity's ID |
| `detail` | TEXT (nullable) | JSON blob with action-specific context |
| `session_id` | TEXT (nullable) | UUID grouping events from one CLI invocation |

### Model

```python
class EventLog(BaseModel):
    id: int | None = None
    timestamp: str
    kind: str
    action: str
    target_type: str | None = None
    target_id: str | None = None
    detail: dict | None = None
    session_id: str | None = None
```

### Event Taxonomy

| kind | action | Emitted when |
|------|--------|--------------|
| `ingest` | `paper.added` | acquire seed/arxiv/file adds a paper |
| `ingest` | `paper.enriched` | acquire openalex enriches a paper |
| `extract` | `extraction.completed` | extract finishes a paper successfully |
| `extract` | `extraction.failed` | extract fails on a paper |
| `extract` | `vocabulary.created` | process_new_concepts adds a NEW: entry |
| `extract` | `vocabulary.updated` | process_new_concepts updates paper_count/confidence |
| `build` | `taxonomy.built` | build taxonomy completes |
| `build` | `matrix.built` | build matrix completes |
| `build` | `version.recorded` | taxonomy version recorded |
| `lint` | `orphan.found` | orphan vocabulary detected |
| `lint` | `contradiction.found` | opposing matrix cells detected |
| `lint` | `weak_evidence.found` | thin evidence detected |
| `lint` | `missing_embedding.found` | vocabulary entry missing vector |
| `lint` | `stale_extraction.found` | stuck paper detected |
| `lint` | `near_duplicate.found` | high-similarity pair detected |
| `fix` | `orphan.deleted` | orphan entry removed |
| `fix` | `embedding.repaired` | missing embedding generated |
| `fix` | `extraction.requeued` | stale paper reset to pending |
| `fix` | `duplicate.merged` | near-duplicate merged |

### Helper

```python
def log_event(store, kind, action, target_type=None, target_id=None,
              detail=None, session_id=None):
    """Append one event to the event_log table."""
```

Called explicitly at each instrumentation site. No decorator magic.

### `detail` Examples

- `paper.added`: `{"title": "...", "source": "arxiv"}`
- `extraction.completed`: `{"tradeoffs": 3, "architecture": 2, "agentic": 1}`
- `vocabulary.created`: `{"name": "...", "kind": "parameter", "from_paper": "..."}`
- `orphan.found`: `{"name": "...", "kind": "principle", "paper_count": 0}`
- `extraction.requeued`: `{"previous_status": "failed"}`

---

## 2. Event Log Instrumentation

Events are emitted inline in existing code paths:

| Code path | File | Events |
|-----------|------|--------|
| `acquire seed/arxiv/file` | `cli.py` + acquire modules | `ingest.paper.added` per paper |
| `acquire openalex --enrich` | `cli.py` + openalex module | `ingest.paper.enriched` per paper |
| `extract` | `extractor.py` | `extract.extraction.completed` or `extract.extraction.failed` per paper |
| `process_new_concepts` | `vocabulary.py` | `extract.vocabulary.created` / `extract.vocabulary.updated` per entry |
| `build taxonomy` | `vocabulary.py` | `build.taxonomy.built` with stats |
| `build matrix` | `matrix.py` | `build.matrix.built` with stats |
| `record_version` | `versioning.py` | `build.version.recorded` with version_id |
| `lint` checks | `linter.py` (new) | `lint.*` per finding |
| `lint --fix` | `linter.py` (new) | `fix.*` per repair |

The `session_id` is generated once per CLI invocation in `cli.py` and threaded through to all called functions.

---

## 3. Lint Checks

New module: `src/lens/knowledge/linter.py`

Six checks, each a standalone function returning a list of findings. All are pure DB queries — no LLM calls.

### 3.1 Orphan Vocabulary

Find vocabulary entries where `paper_count == 0 AND source != 'seed'`. Seed entries are exempt.

**Fix:** Delete the orphan entry and its embedding.

### 3.2 Contradictions

Find parameter pairs where both `(A improves, B worsens)` and `(B improves, A worsens)` exist in `matrix_cells`, each with `count >= 2`.

**Fix:** Report only. Contradictions need human judgment.

### 3.3 Weak Evidence

Find vocabulary entries with `paper_count == 1 OR avg_confidence < threshold` (default threshold: 0.5).

**Fix:** Report only. User decides whether to seek more papers or prune.

### 3.4 Missing Embeddings

Find vocabulary entries with no corresponding row in `vocabulary_vec`.

**Fix:** Generate and store embeddings for the missing entries.

### 3.5 Stale Extractions

Find papers with `extraction_status IN ('pending', 'incomplete', 'failed')`. Cross-reference with `event_log` to report how long they've been stuck.

**Fix:** Reset status to `pending` so next `lens extract` retries them.

### 3.6 Near-Duplicates

For each vocabulary entry, find others in the same `kind` where cosine similarity > threshold (default: 0.92). Group pairs to avoid reporting A-B and B-A.

**Fix:** Merge the lower-`paper_count` entry into the higher one:
1. In `tradeoff_extractions`: rewrite `improves` and `worsens` columns where they match the duplicate's name
2. In `tradeoff_extractions`: rewrite `technique` column similarly
3. In `architecture_extractions`: rewrite `component_slot` column
4. In `agentic_extractions`: rewrite `category` column
5. Sum `paper_count`, recalculate `avg_confidence`
6. Delete the duplicate vocabulary entry + its embedding
7. Log the merge as `fix.duplicate.merged`

### Orchestration

```python
def lint(store, fix=False, session_id=None,
         checks=None, confidence_threshold=0.5,
         similarity_threshold=0.92,
         embedding_provider=None, embedding_model=None,
         embedding_api_base=None, embedding_api_key=None) -> LintReport:
```

Runs all (or filtered) checks, logs findings as `lint.*` events, optionally applies fixes logged as `fix.*` events.

```python
class LintReport(BaseModel):
    orphans: list[dict]
    contradictions: list[dict]
    weak_evidence: list[dict]
    missing_embeddings: list[dict]
    stale_extractions: list[dict]
    near_duplicates: list[dict]
    fixes_applied: list[dict]
```

---

## 4. CLI Commands

### `lens lint`

```
lens lint [--fix] [--check CHECKS] [--threshold-confidence FLOAT] [--threshold-similarity FLOAT]
```

- Default: runs all 6 checks, prints summary table
- `--fix`: applies safe auto-fixes after reporting
- `--check`: comma-separated filter (e.g. `--check orphans,stale`)
- `--threshold-confidence 0.5`: weak evidence cutoff
- `--threshold-similarity 0.92`: near-duplicate cosine threshold

**Output:**
```
Lint Report (session abc123)
────────────────────────────
  Orphan vocabulary:     3 found
  Contradictions:        1 found
  Weak evidence:         7 found
  Missing embeddings:    0 found
  Stale extractions:     2 found
  Near-duplicates:       2 pairs found
────────────────────────────
  Total issues:          15

Use --fix to apply safe auto-fixes.
```

With `--fix`:
```
  Fixed: embedded 0 missing vocabulary entries
  Fixed: requeued 2 stale extractions
  Fixed: deleted 3 orphan vocabulary entries
```

### `lens log`

```
lens log [--kind KIND] [--since DATE] [--limit N] [--session SESSION_ID]
```

- Default: last 20 events, newest first
- `--kind ingest`: filter by event kind
- `--since 2026-04-01`: events after date
- `--limit 50`: control count
- `--session ID`: show one session's events

**Output:**
```
2026-04-05 14:23  ingest   paper.added         paper:attention-is-all-you-need
2026-04-05 14:24  extract  extraction.completed paper:attention-is-all-you-need  (3 tradeoffs, 2 arch)
2026-04-05 14:24  extract  vocabulary.created   vocab:flash-attention            (parameter, from paper)
2026-04-05 14:30  lint     orphan.found         vocab:dead-concept               (0 papers)
```

---

## 5. Testing

All tests use real SQLite via `tmp_path` — no mocking.

### Unit: `tests/test_linter.py`

- `test_lint_orphan_vocabulary` — entry with paper_count=0 flagged
- `test_lint_contradictions` — opposing matrix cells detected
- `test_lint_weak_evidence` — entry with paper_count=1 flagged
- `test_lint_missing_embeddings` — entry without vec row flagged
- `test_lint_stale_extractions` — paper with status "failed" flagged
- `test_lint_near_duplicates` — near-identical embeddings paired
- `test_lint_fix_orphans` — orphan deleted when fix=True
- `test_lint_fix_requeue` — status reset to "pending"
- `test_lint_fix_merge_duplicates` — lower-count entry merged

### Unit: `tests/test_event_log.py`

- `test_log_event_writes` — event appears in table
- `test_log_event_session_grouping` — session_id groups events
- `test_log_query_filters` — kind/since/limit filtering works

### Integration: `tests/test_lint_integration.py`

- `test_lint_after_extract` — no false positives on fresh data
- `test_lint_log_events_recorded` — lint.* events in event_log
- `test_fix_events_recorded` — fix.* events logged

---

## 6. Files Changed

| File | Change |
|------|--------|
| `src/lens/store/models.py` | Add `EventLog`, `LintReport` models |
| `src/lens/store/store.py` | Add `event_log` table schema, migration |
| `src/lens/knowledge/linter.py` | **New** — six lint checks + orchestrator |
| `src/lens/knowledge/events.py` | **New** — `log_event()` helper + query functions |
| `src/lens/cli.py` | Add `lens lint` and `lens log` commands, generate session_id |
| `src/lens/extract/extractor.py` | Emit extract events |
| `src/lens/taxonomy/vocabulary.py` | Emit vocabulary events |
| `src/lens/knowledge/matrix.py` | Emit build events |
| `src/lens/taxonomy/versioning.py` | Emit version events |
| `src/lens/cli.py` (acquire callbacks) | Emit ingest events after each acquire subcommand |
| `tests/test_linter.py` | **New** — lint unit tests |
| `tests/test_event_log.py` | **New** — event log unit tests |
| `tests/test_lint_integration.py` | **New** — integration tests |

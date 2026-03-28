# SQLite Migration — Design Spec

**Date**: 2026-03-28
**Status**: Draft

## Problem

LENS uses LanceDB as its embedded database with Polars for analytics. This stack introduces:
- Heavy dependencies (~hundreds of MB for lancedb + polars + pandas + pyarrow)
- Type checker friction (LanceDB's `Vector()` in type annotations, Polars `.max()` return types)
- API instability (`_TableWrapper`, `_DatabaseWrapper` hacks for LanceDB version differences)
- No parameterized queries (manual `escape_sql_string` for SQL injection prevention)
- Concurrent write safety issues (required restructuring `extract_papers`)

At LENS's scale (~5000 papers, ~35 taxonomy entries), none of LanceDB's advanced features (multimodal blobs, Lance columnar format, MVCC versioning) are used. The primary multimodal use case (paper figures) will use file-based storage with text descriptions, not database blobs.

## Goal

Replace LanceDB + Polars with SQLite + sqlite-vec + plain Python. Remove three heavy dependencies, simplify the store layer, and gain parameterized queries, WAL-mode concurrency, and battle-tested stability.

## Design Decisions

### Single SQLite database file
One file at `~/.lens/data/lens.db` contains both regular tables and sqlite-vec virtual tables. Simple to backup, simple to reason about.

### JSON serialization for list fields
Fields like `authors`, `paper_ids`, `components`, `replaces` are stored as JSON TEXT columns. All list-content filtering happens in Python after loading rows, never via SQL. No junction tables.

### Plain Python instead of Polars
All callers switch from `store.get_table(t).to_polars().filter(...)` to `store.query(t, where=..., params=[...])` returning `list[dict]`. Polars operations (filter, groupby, join, sort) become SQL or Python equivalents. At LENS's scale, this is simpler with no performance penalty.

### Pydantic models for validation only
Models become plain `BaseModel` classes used for input validation. Table schemas are defined as SQL `CREATE TABLE` statements in the store module. The `Vector()` type annotation disappears entirely.

## New Store API

```python
class LensStore:
    def __init__(self, db_path: str)
        # Opens/creates SQLite DB (auto-appends .db if not present)
        # Loads sqlite-vec extension
        # Sets PRAGMA journal_mode=WAL, foreign_keys=ON

    def init_tables(self) -> None
        # Creates all 13 regular tables + 5 vec0 virtual tables
        # Idempotent (IF NOT EXISTS)

    def add_rows(self, table: str, rows: list[dict]) -> None
        # INSERT rows into table
        # Auto-serializes list/dict fields to JSON
        # For vector tables: extracts "embedding" field, inserts into companion _vec table

    def query(self, table: str, where: str = "", params: list | None = None) -> list[dict]
        # SELECT * FROM table [WHERE ...]
        # Returns list[dict] with JSON fields auto-deserialized to lists
        # Parameterized queries (? placeholders)

    def update(self, table: str, values: dict, where: str, params: list | None = None) -> None
        # UPDATE table SET ... WHERE ...
        # Parameterized

    def delete(self, table: str, where: str, params: list | None = None) -> None
        # DELETE FROM table WHERE ...
        # Parameterized

    def vector_search(self, table: str, embedding: list[float],
                      limit: int = 5, where: str = "", params: list | None = None) -> list[dict]
        # k-NN search via sqlite-vec on {table}_vec
        # JOINs vec results with main table for full row data
        # Returns rows with _distance field
        # Optional WHERE filter applied after vec search (over-fetches then truncates)

    def query_sql(self, sql: str, params: list | None = None) -> list[dict]
        # Execute raw SQL, return list[dict]
        # For complex queries (GROUP BY, JOIN, subqueries) that don't fit query()
        # JSON fields are NOT auto-deserialized (caller handles if needed)
```

### Removed from current API
- `get_table()` — no more table wrapper objects
- `escape_sql_string()` — replaced by parameterized queries everywhere
- `_TableWrapper`, `_DatabaseWrapper` — no longer needed

## Schema

### Naming convention
- Regular tables: `papers`, `parameters`, etc.
- Vector tables: `papers_vec`, `parameters_vec`, etc. (companion tables)

### ID columns
- `papers` uses `paper_id TEXT PRIMARY KEY` (arxiv ID is the natural key)
- All other tables with IDs use `id INTEGER PRIMARY KEY`
- Extraction tables use `rowid INTEGER PRIMARY KEY AUTOINCREMENT`

### Datetime storage
All datetime fields (`created_at`) are stored as ISO 8601 TEXT strings (e.g., `"2026-03-28T12:00:00+00:00"`). This matches the current behavior — Pydantic serializes datetime to string when creating dicts.

### Vector tables
5 tables have vector companions: `papers`, `parameters`, `principles`, `architecture_variants`, `agentic_patterns`.

```sql
-- Example: parameters + parameters_vec
CREATE TABLE parameters (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    raw_strings TEXT NOT NULL,       -- JSON list[str]
    paper_ids TEXT NOT NULL,         -- JSON list[str]
    taxonomy_version INTEGER NOT NULL
);

CREATE VIRTUAL TABLE parameters_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding FLOAT[768] distance_metric=cosine  -- EMBEDDING_DIM, cosine for embeddings
);
```

### JSON fields (complete list)
| Table | Field | Python type |
|-------|-------|-------------|
| papers | authors | list[str] |
| agentic_extractions | components | list[str] |
| parameters | raw_strings, paper_ids | list[str] |
| principles | sub_techniques, raw_strings, paper_ids | list[str] |
| architecture_variants | replaces, paper_ids | list[int], list[str] |
| agentic_patterns | components, use_cases, paper_ids | list[str] |
| matrix_cells | paper_ids | list[str] |
| ideation_gaps | related_params, related_principles, related_slots | list[int] |

### Full table definitions

**Layer 0 — Papers:**
```sql
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT NOT NULL,
    venue TEXT,
    date TEXT,
    arxiv_id TEXT NOT NULL,
    citations INTEGER DEFAULT 0,
    quality_score REAL DEFAULT 0.0,
    extraction_status TEXT DEFAULT 'pending'
);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_vec USING vec0(
    paper_id TEXT PRIMARY KEY,
    embedding FLOAT[{EMBEDDING_DIM}] distance_metric=cosine
);
```

**Layer 1 — Extractions:**
```sql
CREATE TABLE IF NOT EXISTS tradeoff_extractions (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    improves TEXT NOT NULL,
    worsens TEXT NOT NULL,
    technique TEXT NOT NULL,
    context TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_quote TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS architecture_extractions (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    component_slot TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    replaces TEXT,
    key_properties TEXT NOT NULL,
    confidence REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agentic_extractions (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    pattern_name TEXT NOT NULL,
    structure TEXT NOT NULL,
    use_case TEXT NOT NULL,
    components TEXT NOT NULL,
    confidence REAL NOT NULL
);
```

**Layer 2 — Taxonomy:**
```sql
CREATE TABLE IF NOT EXISTS parameters (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    raw_strings TEXT NOT NULL,
    paper_ids TEXT NOT NULL,
    taxonomy_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS principles (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    sub_techniques TEXT NOT NULL,
    raw_strings TEXT NOT NULL,
    paper_ids TEXT NOT NULL,
    taxonomy_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS architecture_slots (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    taxonomy_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS architecture_variants (
    id INTEGER PRIMARY KEY,
    slot_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    replaces TEXT NOT NULL,
    properties TEXT NOT NULL,
    paper_ids TEXT NOT NULL,
    taxonomy_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS agentic_patterns (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    components TEXT NOT NULL,
    use_cases TEXT NOT NULL,
    paper_ids TEXT NOT NULL,
    taxonomy_version INTEGER NOT NULL
);
```

Vector companions for parameters, principles, architecture_variants, agentic_patterns follow the same pattern as papers_vec.

**Layer 3 — Matrix:**
```sql
CREATE TABLE IF NOT EXISTS matrix_cells (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    improving_param_id INTEGER NOT NULL,
    worsening_param_id INTEGER NOT NULL,
    principle_id INTEGER NOT NULL,
    count INTEGER NOT NULL,
    avg_confidence REAL NOT NULL,
    paper_ids TEXT NOT NULL,
    taxonomy_version INTEGER NOT NULL
);
```

**Versioning:**
```sql
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
```

**Ideation:**
```sql
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
    description TEXT NOT NULL,
    related_params TEXT NOT NULL,
    related_principles TEXT NOT NULL,
    related_slots TEXT NOT NULL,
    score REAL NOT NULL,
    llm_hypothesis TEXT,
    created_at TEXT NOT NULL,
    taxonomy_version INTEGER NOT NULL
);
```

## Vector search implementation

Uses sqlite-vec's `vec0` virtual tables with binary-packed embeddings.

**Insert** (inside `add_rows` for vector tables):
```python
embedding = row.pop("embedding")
# Insert main row normally
# Then insert embedding:
embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
conn.execute(
    f"INSERT INTO {table}_vec ({id_col}, embedding) VALUES (?, ?)",
    (row_id, embedding_bytes)
)
```

**Search** (in `vector_search` method):
```python
query_bytes = struct.pack(f"{len(embedding)}f", *embedding)

if where:
    # Over-fetch from vec, then filter via JOIN with main table
    sql = f"""
        SELECT t.*, v.distance AS _distance
        FROM {table}_vec v
        JOIN {table} t ON t.{id_col} = v.{id_col}
        WHERE v.embedding MATCH ? AND v.k = ?
        AND {where}
        ORDER BY v.distance
        LIMIT ?
    """
    rows = conn.execute(sql, [query_bytes, limit * 3, *params, limit]).fetchall()
else:
    sql = f"""
        SELECT t.*, v.distance AS _distance
        FROM {table}_vec v
        JOIN {table} t ON t.{id_col} = v.{id_col}
        WHERE v.embedding MATCH ? AND v.k = ?
        ORDER BY v.distance
    """
    rows = conn.execute(sql, [query_bytes, limit]).fetchall()
```

Over-fetching by 3x when filtering ensures we get enough results after the WHERE clause eliminates non-matching rows.

## Caller migration patterns

### Pattern 1: Simple read + filter (most common)
```python
# Before:
df = store.get_table("papers").to_polars()
df = df.filter(pl.col("extraction_status").is_in(["pending", "incomplete"]))
results = df.to_dicts()

# After:
results = store.query(
    "papers",
    where="extraction_status IN (?, ?)",
    params=["pending", "incomplete"]
)
```

### Pattern 2: Read all rows
```python
# Before:
df = store.get_table("parameters").to_polars()

# After:
rows = store.query("parameters")
```

### Pattern 3: Group by + aggregate
```python
# Before (explorer.py list_matrix_overview):
df.group_by(["improving_param_id", "worsening_param_id"]).agg(
    pl.col("principle_id").count().alias("num_principles"),
    pl.col("count").sum().alias("total_evidence"),
).sort("total_evidence", descending=True)

# After:
store.query(
    "matrix_cells",
    where="taxonomy_version = ?",
    params=[version],
    # Actually: use a raw SQL query for this
)
# Or add a query_sql() method for complex queries:
rows = store.query_sql("""
    SELECT improving_param_id, worsening_param_id,
           COUNT(*) AS num_principles, SUM(count) AS total_evidence
    FROM matrix_cells WHERE taxonomy_version = ?
    GROUP BY improving_param_id, worsening_param_id
    ORDER BY total_evidence DESC
""", params=[version])
```

For complex analytics (group by, join, aggregate), a `query_sql()` escape hatch lets callers write raw SQL. This is cleaner than building a query builder abstraction.

### Pattern 4: Join
```python
# Before (explorer.py list_architecture_slots with variant counts):
slots_df.join(counts_df, left_on="id", right_on="slot_id", how="left")

# After:
rows = store.query_sql("""
    SELECT s.*, COALESCE(cnt, 0) AS variant_count FROM architecture_slots s
    LEFT JOIN (
        SELECT slot_id, COUNT(*) AS cnt FROM architecture_variants
        WHERE taxonomy_version = ? GROUP BY slot_id
    ) v ON s.id = v.slot_id
    WHERE s.taxonomy_version = ?
    ORDER BY s.name
""", params=[version, version])
```

### Pattern 5: Vector search
```python
# Before (explainer.py):
results = store.get_table("parameters").search(query_embedding) \
    .where(f"taxonomy_version = {taxonomy_version}").limit(top_k).to_list()

# After:
results = store.vector_search(
    "parameters", embedding=query_embedding,
    limit=top_k, where="taxonomy_version = ?", params=[taxonomy_version]
)
```

### Pattern 6: Max ID
```python
# Before:
max_ver: int = df.select(pl.col("version_id").max()).item() or 0

# After:
rows = store.query_sql("SELECT MAX(version_id) AS max_id FROM taxonomy_versions")
max_ver = rows[0]["max_id"] if rows and rows[0]["max_id"] is not None else 0
```

## Dependencies

**Removed:**
- `lancedb>=0.17`
- `polars>=1.0`
- `pandas>=3.0.1` (was only needed for LanceDB)

**Added:**
- `sqlite-vec>=0.1`

**Unchanged:**
- `pydantic>=2.0` (still used for validation, response models)
- All other dependencies

## models.py transformation

Models change from `LanceModel` to `BaseModel`:

```python
# Before:
from lancedb.pydantic import LanceModel, Vector

class Paper(LanceModel):
    paper_id: str
    ...
    embedding: Vector(EMBEDDING_DIM)

# After:
from pydantic import BaseModel

class Paper(BaseModel):
    paper_id: str
    ...
    embedding: list[float] = []  # not stored in main table, goes to _vec
```

`EMBEDDING_DIM` stays as a constant, used by the store layer for vec table creation.

Validators (`_check_date_format`, `_check_extraction_status`) stay on `Paper`.

The `ExplanationResult` model is already a `BaseModel` — no change needed.

## Files to modify

| File | Change |
|------|--------|
| `store/store.py` | Full rewrite: SQLite + sqlite-vec |
| `store/models.py` | LanceModel → BaseModel, remove Vector() |
| `pyproject.toml` | Remove lancedb/polars/pandas, add sqlite-vec |
| `taxonomy/__init__.py` | Replace Polars with SQL/Python |
| `taxonomy/versioning.py` | Replace Polars with SQL |
| `taxonomy/embedder.py` | Remove EMBEDDING_DIM import (stays in models.py) |
| `extract/extractor.py` | Replace get_table().delete/update with store.delete/update |
| `knowledge/matrix.py` | Replace Polars groupby/join with SQL |
| `serve/analyzer.py` | Replace vector search API |
| `serve/explainer.py` | Replace vector search API |
| `serve/explorer.py` | Replace Polars filter/join/sort with SQL |
| `monitor/watcher.py` | Replace Polars filter with SQL |
| `monitor/ideation.py` | Replace Polars with SQL/Python |
| `cli.py` | Remove Polars imports, update store calls |
| `acquire/seed.py` | Update store calls |
| `acquire/pdf.py` | Remove EMBEDDING_DIM import if unused directly |
| `config.py` | No change (storage.data_dir stays) |
| `tests/*` | Update all fixtures and assertions |
| `README.md`, `CLAUDE.md`, `design.md` | Update tech stack |

## Testing strategy

- All existing test logic preserved — same behaviors tested, different API calls
- Test fixtures create SQLite DBs via `tmp_path` (same pattern as LanceDB)
- Vector search test uses sqlite-vec (real extension, no mocking)
- `conftest.py` fixture: `LensStore(str(tmp_path / "test.db"))`

## Migration risks

**Mitigated:**
- Volume of changes (every file) — each change is mechanical, not creative
- JSON serialization correctness — test fixtures exercise all JSON fields
- Vector search: use `distance_metric=cosine` on all vec0 tables (better for embedding similarity than L2)

**Low risk:**
- datetime handling — SQLite stores as TEXT (ISO format), same as current behavior
- Concurrent writes — SQLite WAL mode is safer than LanceDB embedded mode

**Gotchas discovered during testing:**
- `INSERT OR REPLACE` does not work on vec0 virtual tables — must `DELETE` then `INSERT` to update embeddings
- sqlite-vec is pre-v1 (currently v0.1.7) — pin version to avoid breaking changes
- All vec0 tables use `distance_metric=cosine` for consistency with embedding model expectations

## Out of scope

- Data migration tool (no production data exists — `lens init --force` is fine)
- Query optimization / indexing (can add later if needed at scale)
- Alternative vector backends (sqlite-vec is sufficient)

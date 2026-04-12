# Paper Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-level `lens search` command that finds papers via hybrid search (FTS5 keyword + sqlite-vec semantic) with optional metadata filters.

**Architecture:** New `papers_fts` FTS5 virtual table indexed on title + abstract. New `search_papers()` method on `LensStore` mirrors the existing `hybrid_search()` pattern but targets papers. A thin `search_papers()` function in `explorer.py` handles embedding the query and applying filters. The CLI wires it up as a top-level `search` command.

**Tech Stack:** SQLite FTS5, sqlite-vec, sentence-transformers (or cloud embeddings via openai SDK), Typer CLI, Rich for output formatting.

**Spec:** `docs/specs/2026-04-12-paper-search-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/lens/store/store.py` | Modify | Add `papers_fts` table, `rebuild_papers_fts()`, `search_papers()` |
| `src/lens/serve/explorer.py` | Modify | Add `search_papers()` serve function |
| `src/lens/cli.py` | Modify | Add top-level `search` command |
| `tests/test_store.py` | Modify | Tests for FTS5 table, `rebuild_papers_fts()`, `search_papers()` |
| `tests/test_explorer.py` | Modify | Tests for `search_papers()` serve function |
| `tests/test_cli.py` | Modify | Test for `search` CLI command |

---

### Task 1: FTS5 Table + Rebuild Method

**Files:**
- Modify: `src/lens/store/store.py:205-211` (init_tables, after vocabulary_fts)
- Modify: `src/lens/store/store.py:371-374` (after rebuild_vocabulary_fts)
- Test: `tests/test_store.py`

- [ ] **Step 1: Write failing test for papers_fts table creation**

In `tests/test_store.py`, add after the existing `test_init_creates_vec_tables` test (line ~34):

```python
def test_init_creates_papers_fts(store):
    cursor = store.conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row[0] for row in cursor.fetchall()}
    assert "papers_fts" in tables
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py::test_init_creates_papers_fts -v`
Expected: FAIL with `AssertionError`

- [ ] **Step 3: Add papers_fts creation in init_tables**

In `src/lens/store/store.py`, after the vocabulary_fts creation block (line 209), add:

```python
        # FTS5 table for paper hybrid search (keyword + vector)
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts "
            "USING fts5(title, abstract, content=papers, content_rowid=rowid)"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py::test_init_creates_papers_fts -v`
Expected: PASS

- [ ] **Step 5: Write failing test for rebuild_papers_fts**

In `tests/test_store.py`, add:

```python
def test_rebuild_papers_fts(store, sample_paper_data):
    store.add_papers([sample_paper_data])
    store.rebuild_papers_fts()
    cursor = store.conn.execute(
        "SELECT * FROM papers_fts WHERE papers_fts MATCH ?", ('"Attention"',)
    )
    rows = cursor.fetchall()
    assert len(rows) == 1
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py::test_rebuild_papers_fts -v`
Expected: FAIL with `AttributeError: 'LensStore' object has no attribute 'rebuild_papers_fts'`

- [ ] **Step 7: Implement rebuild_papers_fts**

In `src/lens/store/store.py`, after `rebuild_vocabulary_fts()` (line 374), add:

```python
    def rebuild_papers_fts(self) -> None:
        """Rebuild the papers FTS5 index from current papers data."""
        self.conn.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
        self.conn.commit()
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py::test_rebuild_papers_fts -v`
Expected: PASS

- [ ] **Step 9: Wire rebuild into add_papers**

In `src/lens/store/store.py`, modify `add_papers()` (line 284-289) to rebuild FTS after insertion:

```python
    def add_papers(self, data: list[dict]) -> int:
        """Insert papers, skipping duplicates by primary key.

        Returns the number of new papers actually inserted.
        """
        count = self.add_rows("papers", data, ignore_conflicts=True)
        if count > 0:
            self.rebuild_papers_fts()
        return count
```

- [ ] **Step 10: Run all store tests**

Run: `uv run pytest tests/test_store.py -v`
Expected: All pass

- [ ] **Step 11: Commit**

```bash
git add src/lens/store/store.py tests/test_store.py
git commit -m "feat(store): add papers_fts table and rebuild method"
```

---

### Task 2: search_papers Store Method

**Files:**
- Modify: `src/lens/store/store.py:374` (after rebuild_papers_fts)
- Test: `tests/test_store.py`

- [ ] **Step 1: Write failing test for hybrid search on papers**

In `tests/test_store.py`, add:

```python
def test_search_papers_hybrid(store, sample_paper_data):
    """Hybrid search combines FTS5 keyword + vector similarity."""
    store.add_papers([sample_paper_data])
    query_embedding = [0.1] * EMBEDDING_DIM
    results = store.search_papers(
        query="Attention architecture",
        embedding=query_embedding,
        limit=5,
    )
    assert len(results) == 1
    assert results[0]["paper_id"] == "2401.12345"
    assert "_rrf_score" in results[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py::test_search_papers_hybrid -v`
Expected: FAIL with `AttributeError: 'LensStore' object has no attribute 'search_papers'`

- [ ] **Step 3: Implement search_papers**

In `src/lens/store/store.py`, after `rebuild_papers_fts()`, add:

```python
    def search_papers(
        self,
        query: str | None = None,
        embedding: list[float] | None = None,
        filters: dict[str, str] | None = None,
        limit: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Search papers using hybrid FTS5 + vector search with optional filters.

        When *query* and *embedding* are provided, runs hybrid search with RRF.
        When only *query* is provided, runs FTS5-only keyword search.
        When neither is provided, runs a filtered query (requires *filters*).

        Supported filter keys: author, venue, after, before.
        """
        json_cols = JSON_FIELDS.get("papers", set())

        # Build filter WHERE clause fragments
        filter_clauses: list[str] = []
        filter_params: list[str] = []
        if filters:
            if filters.get("author"):
                filter_clauses.append("t.authors LIKE ?")
                filter_params.append(f"%{filters['author']}%")
            if filters.get("venue"):
                filter_clauses.append("t.venue LIKE ?")
                filter_params.append(f"%{filters['venue']}%")
            if filters.get("after"):
                filter_clauses.append("t.date >= ?")
                filter_params.append(filters["after"])
            if filters.get("before"):
                filter_clauses.append("t.date <= ?")
                filter_params.append(filters["before"])

        filter_where = " AND ".join(filter_clauses) if filter_clauses else ""

        # Filter-only mode: no text query
        if not query:
            sql = "SELECT * FROM papers t"
            if filter_where:
                sql += f" WHERE {filter_where}"
            sql += " ORDER BY t.date DESC LIMIT ?"
            params_tuple = tuple(filter_params) + (limit,)
            cursor = self.conn.execute(sql, params_tuple)
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in cursor.fetchall():
                d: dict[str, Any] = {}
                for i, col in enumerate(columns):
                    val = row[i]
                    if col in json_cols and isinstance(val, str):
                        d[col] = json.loads(val)
                    else:
                        d[col] = val
                results.append(d)
            return results

        # Hybrid or FTS-only mode
        fts_query = " OR ".join(f'"{token}"' for token in query.strip().split())
        fetch_limit = limit * 3

        if embedding is not None:
            emb_bytes = _pack_embedding(embedding)
            sql = """
            WITH fts_matches AS (
                SELECT
                    rowid,
                    row_number() OVER (ORDER BY rank) AS rank_number
                FROM papers_fts
                WHERE papers_fts MATCH ?
                LIMIT ?
            ),
            vec_matches AS (
                SELECT
                    paper_id,
                    row_number() OVER (ORDER BY distance) AS rank_number,
                    distance
                FROM papers_vec
                WHERE embedding MATCH ? AND k = ?
            ),
            combined AS (
                SELECT
                    t.paper_id,
                    coalesce(1.0 / (? + f.rank_number), 0.0) AS fts_score,
                    coalesce(1.0 / (? + vm.rank_number), 0.0) AS vec_score,
                    vm.distance AS vec_distance
                FROM papers t
                LEFT JOIN fts_matches f ON t.rowid = f.rowid
                LEFT JOIN vec_matches vm ON t.paper_id = vm.paper_id
                WHERE f.rowid IS NOT NULL OR vm.paper_id IS NOT NULL
            )
            SELECT
                t.*,
                c.fts_score + c.vec_score AS _rrf_score,
                c.vec_distance AS _distance
            FROM combined c
            JOIN papers t ON t.paper_id = c.paper_id
            """
            base_params: list[Any] = [
                fts_query, fetch_limit,
                emb_bytes, fetch_limit,
                rrf_k, rrf_k,
            ]
        else:
            # FTS-only mode (no embedding available)
            sql = """
            WITH fts_matches AS (
                SELECT
                    rowid,
                    row_number() OVER (ORDER BY rank) AS rank_number
                FROM papers_fts
                WHERE papers_fts MATCH ?
                LIMIT ?
            )
            SELECT
                t.*,
                1.0 / (? + f.rank_number) AS _rrf_score,
                NULL AS _distance
            FROM fts_matches f
            JOIN papers t ON t.rowid = f.rowid
            """
            base_params = [fts_query, fetch_limit, rrf_k]

        if filter_where:
            sql += f" WHERE {filter_where}"
            base_params.extend(filter_params)

        sql += " ORDER BY _rrf_score DESC LIMIT ?"
        base_params.append(limit)

        cursor = self.conn.execute(sql, tuple(base_params))
        columns = [desc[0] for desc in cursor.description]

        results = []
        for row in cursor.fetchall():
            d: dict[str, Any] = {}
            for i, col in enumerate(columns):
                val = row[i]
                if col in json_cols and isinstance(val, str):
                    d[col] = json.loads(val)
                else:
                    d[col] = val
            results.append(d)
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py::test_search_papers_hybrid -v`
Expected: PASS

- [ ] **Step 5: Write failing test for FTS-only mode**

In `tests/test_store.py`, add:

```python
def test_search_papers_fts_only(store, sample_paper_data):
    """FTS-only search when no embedding is provided."""
    store.add_papers([sample_paper_data])
    results = store.search_papers(query="Attention", limit=5)
    assert len(results) == 1
    assert results[0]["paper_id"] == "2401.12345"
    assert "_rrf_score" in results[0]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py::test_search_papers_fts_only -v`
Expected: PASS (implementation already handles this case)

- [ ] **Step 7: Write failing test for filter-only mode**

In `tests/test_store.py`, add:

```python
def test_search_papers_filter_only(store, sample_paper_data):
    """Filter-only mode when no text query is given."""
    store.add_papers([sample_paper_data])
    results = store.search_papers(filters={"author": "Vaswani"}, limit=5)
    assert len(results) == 1
    assert results[0]["paper_id"] == "2401.12345"
    assert "_rrf_score" not in results[0]
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py::test_search_papers_filter_only -v`
Expected: PASS

- [ ] **Step 9: Write test for combined hybrid + filter**

In `tests/test_store.py`, add:

```python
def test_search_papers_hybrid_with_filters(store):
    """Hybrid search narrowed by date filter."""
    papers = [
        {
            "paper_id": "p1",
            "title": "Attention Mechanism Survey",
            "abstract": "A survey of attention mechanisms in deep learning.",
            "authors": ["Author A"],
            "venue": None,
            "date": "2023-01-01",
            "arxiv_id": "2301.00001",
            "citations": 10,
            "quality_score": 0.5,
            "extraction_status": "pending",
            "embedding": [0.1] * EMBEDDING_DIM,
        },
        {
            "paper_id": "p2",
            "title": "Attention in Vision Transformers",
            "abstract": "Applying attention to computer vision tasks.",
            "authors": ["Author B"],
            "venue": None,
            "date": "2024-06-01",
            "arxiv_id": "2406.00001",
            "citations": 5,
            "quality_score": 0.3,
            "extraction_status": "pending",
            "embedding": [0.2] * EMBEDDING_DIM,
        },
    ]
    store.add_papers(papers)
    results = store.search_papers(
        query="attention",
        embedding=[0.15] * EMBEDDING_DIM,
        filters={"after": "2024-01-01"},
        limit=5,
    )
    assert len(results) == 1
    assert results[0]["paper_id"] == "p2"
```

- [ ] **Step 10: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py::test_search_papers_hybrid_with_filters -v`
Expected: PASS

- [ ] **Step 11: Write test for empty results**

In `tests/test_store.py`, add:

```python
def test_search_papers_no_results(store):
    """Search with no matching papers returns empty list."""
    results = store.search_papers(query="nonexistent topic xyz", limit=5)
    assert results == []
```

- [ ] **Step 12: Run all store tests**

Run: `uv run pytest tests/test_store.py -v`
Expected: All pass

- [ ] **Step 13: Commit**

```bash
git add src/lens/store/store.py tests/test_store.py
git commit -m "feat(store): add search_papers with hybrid search and filters"
```

---

### Task 3: search_papers Serve Function

**Files:**
- Modify: `src/lens/serve/explorer.py:145` (end of file)
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write failing test for search_papers serve function**

In `tests/test_explorer.py`, add:

```python
def test_search_papers_with_query(store, sample_paper_data):
    """search_papers returns formatted results with scores."""
    from lens.serve.explorer import search_papers

    store.add_papers([sample_paper_data])
    results = search_papers(store, query="Attention")
    assert len(results) == 1
    assert results[0]["paper_id"] == "2401.12345"
    assert results[0]["title"] == "Attention Is All You Need"
    # Abstract should be truncated
    assert len(results[0]["abstract_snippet"]) <= 153  # 150 + "..."


def test_search_papers_authors_truncated(store):
    """Authors list is truncated to 3 with ellipsis."""
    from lens.serve.explorer import search_papers

    paper = {
        "paper_id": "p1",
        "title": "Multi Author Paper",
        "abstract": "A paper with many authors about transformers.",
        "authors": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "venue": None,
        "date": "2024-01-01",
        "arxiv_id": "2401.00001",
        "citations": 0,
        "quality_score": 0.5,
        "extraction_status": "pending",
    }
    store.add_papers([paper])
    results = search_papers(store, query="transformers")
    assert results[0]["authors_display"] == "Alice, Bob, Charlie, ..."


def test_search_papers_filter_only(store, sample_paper_data):
    """Filter-only mode works without a text query."""
    from lens.serve.explorer import search_papers

    store.add_papers([sample_paper_data])
    results = search_papers(store, author="Vaswani")
    assert len(results) == 1
    assert results[0]["paper_id"] == "2401.12345"
    assert results[0].get("score") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_explorer.py::test_search_papers_with_query -v`
Expected: FAIL with `ImportError: cannot import name 'search_papers'`

- [ ] **Step 3: Implement search_papers in explorer.py**

At the end of `src/lens/serve/explorer.py`, add:

```python
def search_papers(
    store: LensStore,
    query: str | None = None,
    *,
    author: str | None = None,
    venue: str | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = 10,
    embedding_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search papers via hybrid search and/or metadata filters.

    When *query* is provided, embeds it and runs hybrid FTS5 + vector search.
    Falls back to FTS5-only if embedding fails.
    When only filters are provided, runs a direct SQL query.
    """
    filters: dict[str, str] = {}
    if author:
        filters["author"] = author
    if venue:
        filters["venue"] = venue
    if after:
        filters["after"] = after
    if before:
        filters["before"] = before

    embedding = None
    if query:
        try:
            from lens.taxonomy.embedder import embed_strings

            emb_kw = embedding_kwargs or {}
            embedding = embed_strings([query], **emb_kw)[0].tolist()
        except Exception:
            pass  # Fall back to FTS-only

    raw_results = store.search_papers(
        query=query,
        embedding=embedding,
        filters=filters if filters else None,
        limit=limit,
    )

    formatted = []
    for r in raw_results:
        abstract = r.get("abstract", "")
        snippet = (abstract[:150] + "...") if len(abstract) > 150 else abstract

        authors = r.get("authors", [])
        if isinstance(authors, str):
            import json as _json
            try:
                authors = _json.loads(authors)
            except (ValueError, TypeError):
                authors = [authors]
        if len(authors) > 3:
            authors_display = ", ".join(authors[:3]) + ", ..."
        else:
            authors_display = ", ".join(authors)

        entry: dict[str, Any] = {
            "paper_id": r["paper_id"],
            "title": r["title"],
            "date": r.get("date", ""),
            "authors_display": authors_display,
            "abstract_snippet": snippet,
            "arxiv_id": r.get("arxiv_id", ""),
            "venue": r.get("venue"),
        }
        if "_rrf_score" in r:
            entry["score"] = r["_rrf_score"]

        formatted.append(entry)

    return formatted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_explorer.py -k search_papers -v`
Expected: All 3 pass

- [ ] **Step 5: Run all explorer tests**

Run: `uv run pytest tests/test_explorer.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/lens/serve/explorer.py tests/test_explorer.py
git commit -m "feat(explorer): add search_papers serve function"
```

---

### Task 4: CLI search Command

**Files:**
- Modify: `src/lens/cli.py` (after `explain` command, around line 281)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing test for CLI search command**

In `tests/test_cli.py`, add:

```python
def test_search_by_query(tmp_path, sample_paper_data):
    """lens search with a text query returns matching papers."""
    from typer.testing import CliRunner

    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.add_papers([sample_paper_data])

    result = runner.invoke(app, ["search", "Attention"], env={
        "LENS_DATA_DIR": str(tmp_path),
    })
    assert result.exit_code == 0
    assert "Attention Is All You Need" in result.output


def test_search_by_author(tmp_path, sample_paper_data):
    """lens search --author filters by author name."""
    from typer.testing import CliRunner

    from lens.cli import app
    from lens.store.store import LensStore

    runner = CliRunner()

    db_path = str(tmp_path / "lens.db")
    store = LensStore(db_path)
    store.init_tables()
    store.add_papers([sample_paper_data])

    result = runner.invoke(app, ["search", "--author", "Vaswani"], env={
        "LENS_DATA_DIR": str(tmp_path),
    })
    assert result.exit_code == 0
    assert "Attention Is All You Need" in result.output


def test_search_no_args(tmp_path):
    """lens search with no query and no filters shows an error."""
    from typer.testing import CliRunner

    from lens.cli import app

    runner = CliRunner()

    result = runner.invoke(app, ["search"], env={
        "LENS_DATA_DIR": str(tmp_path),
    })
    assert result.exit_code == 1
    assert "Provide a search query" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_search_by_query -v`
Expected: FAIL (no `search` command registered)

- [ ] **Step 3: Implement the search CLI command**

In `src/lens/cli.py`, after the `explain` command (around line 281), add:

```python
@app.command()
def search(
    query: str | None = typer.Argument(None, help="Text to search for."),
    author: str | None = typer.Option(None, "--author", help="Filter by author name."),
    venue: str | None = typer.Option(None, "--venue", help="Filter by venue."),
    after: str | None = typer.Option(None, "--after", help="Papers on or after date (YYYY-MM-DD)."),
    before: str | None = typer.Option(None, "--before", help="Papers on or before date (YYYY-MM-DD)."),
    limit: int = typer.Option(10, "--limit", help="Max results."),
) -> None:
    """Search papers by keyword, semantic similarity, or metadata filters."""
    if not query and not author and not venue and not after and not before:
        rprint("[red]Provide a search query or at least one filter (--author, --venue, --after, --before).[/red]")
        raise typer.Exit(code=1)

    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.serve.explorer import search_papers

    emb_kw = _embedding_kwargs(config)
    results = search_papers(
        store,
        query=query,
        author=author,
        venue=venue,
        after=after,
        before=before,
        limit=limit,
        embedding_kwargs=emb_kw,
    )

    if not results:
        rprint("[yellow]No papers found.[/yellow]")
        raise typer.Exit(code=0)

    rprint(f"\n[bold]Found {len(results)} paper{'s' if len(results) != 1 else ''}:[/bold]\n")
    for i, r in enumerate(results, 1):
        score_str = f"[{r['score']:.2f}] " if r.get("score") is not None else ""
        venue_str = f" · {r['venue']}" if r.get("venue") else ""
        rprint(f"  {i}. {score_str}{r['title']} ({r['date']})")
        rprint(f"     arxiv:{r['arxiv_id']}{venue_str} · {r['authors_display']}")
        rprint(f"     {r['abstract_snippet']}")
        rprint()
```

- [ ] **Step 4: Run CLI tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -k search -v`
Expected: All 3 pass

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All pass (214+ tests, 2 skipped)

- [ ] **Step 6: Commit**

```bash
git add src/lens/cli.py tests/test_cli.py
git commit -m "feat(cli): add top-level search command with hybrid search and filters"
```

---

### Task 5: Existing Data Migration

**Files:**
- Modify: `src/lens/store/store.py:186` (init_tables)

Existing databases won't have `papers_fts` populated. The FTS5 table is created by `init_tables()`, but it will be empty for databases that already have papers. We need to rebuild the FTS index when the table is first created.

- [ ] **Step 1: Write failing test for FTS rebuild on init**

In `tests/test_store.py`, add:

```python
def test_papers_fts_populated_after_init(store, sample_paper_data):
    """papers_fts is populated for papers added before init_tables is called."""
    # Simulate existing database: add paper, then re-init (as if upgrading)
    store.add_papers([sample_paper_data])
    # Re-init to simulate opening an existing database
    store.init_tables()
    cursor = store.conn.execute(
        "SELECT * FROM papers_fts WHERE papers_fts MATCH ?", ('"Attention"',)
    )
    rows = cursor.fetchall()
    assert len(rows) == 1
```

- [ ] **Step 2: Run test to verify it passes or fails**

Run: `uv run pytest tests/test_store.py::test_papers_fts_populated_after_init -v`

Note: This test may already pass because `add_papers` calls `rebuild_papers_fts()` (added in Task 1). If it passes, the migration is already handled. If it fails, proceed to step 3.

- [ ] **Step 3: Add rebuild to init_tables if needed**

If step 2 fails, add after the papers_fts creation in `init_tables()`:

```python
        # Populate papers FTS for existing data (idempotent rebuild)
        self.rebuild_papers_fts()
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/lens/store/store.py tests/test_store.py
git commit -m "feat(store): ensure papers_fts is populated for existing databases"
```

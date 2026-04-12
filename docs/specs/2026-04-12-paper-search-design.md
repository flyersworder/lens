# Paper Search — Design Spec

## Overview

Add a top-level `lens search` command that finds papers using hybrid search
(FTS5 keyword + sqlite-vec semantic) with optional metadata filters.

## CLI Interface

```
lens search "attention mechanisms"           # hybrid search
lens search --author "Vaswani"               # filter-only
lens search "efficiency" --after 2024-01-01  # hybrid + filter
lens search --venue "NeurIPS" --limit 5      # filter + limit
```

### Arguments & Flags

| Param     | Type                | Default | Description                          |
|-----------|---------------------|---------|--------------------------------------|
| `query`   | positional, optional| None    | Text to search (hybrid mode)         |
| `--author`| string              | None    | Substring match on authors JSON      |
| `--venue` | string              | None    | Substring match on venue             |
| `--after` | string              | None    | Papers dated on or after (YYYY-MM-DD)|
| `--before`| string              | None    | Papers on or before (YYYY-MM-DD)     |
| `--limit` | int                 | 10      | Max results                          |

When no text query and no filters are provided, show an error.

### Output Format

```
Found 3 papers:

  1. [0.87] Attention Is All You Need (2017-06-12)
     arxiv:1706.03762 · Vaswani, Shazeer, Parmar, ...
     A new simple network architecture based on attention mechanisms...

  2. [0.64] FlashAttention: Fast and Memory-Efficient... (2022-05-27)
     arxiv:2205.14135 · Dao, Fu, Ermon, ...
     We propose FlashAttention, an IO-aware exact attention...
```

- Score shown only in hybrid mode (not filter-only).
- Authors truncated to first 3, with "..." if more.
- Abstract truncated to ~150 characters.

## Two Modes

### Hybrid Search Mode (text query provided)

1. Embed the query string using the configured embedding provider.
2. Run FTS5 keyword search on `papers_fts` (title + abstract).
3. Run vector cosine similarity on `papers_vec`.
4. Combine results via Reciprocal Rank Fusion (RRF), same as `hybrid_search()`.
5. Apply metadata filters (author, venue, date) to narrow results.
6. Return top-N by RRF score.

### Filter-Only Mode (no text query, filters provided)

1. Skip FTS5 and vector search entirely.
2. Run a plain `SELECT` with WHERE clauses for the provided filters.
3. Order by date descending.
4. Return top-N.

## Store Layer Changes

### New FTS5 table (`papers_fts`)

Added in `init_tables()`:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts
USING fts5(title, abstract, content=papers, content_rowid=rowid)
```

This is a content-synced FTS5 table, same pattern as `vocabulary_fts`.

### New methods on `LensStore`

- `rebuild_papers_fts()` — rebuild the FTS5 index from current papers data.
  Same pattern as `rebuild_vocabulary_fts()`.

- `search_papers(query, embedding, filters, limit, rrf_k)` — hybrid search
  on papers. Mirrors the existing `hybrid_search()` method but operates on
  `papers_fts` + `papers_vec` and returns paper dicts with RRF scores.
  When no query/embedding is provided, falls back to filtered SQL query.

### FTS5 sync

The FTS5 content table needs to stay in sync with the `papers` table.
Trigger `rebuild_papers_fts()` at the end of `add_papers()` when new rows
are inserted (i.e., when the return count > 0). This keeps the index
current without requiring callers to remember a separate step.

## Serve Layer

A `search_papers()` function in `explorer.py` that:

- Embeds the query string (when text query is provided).
- Calls `store.search_papers()` with embedding + filters.
- Truncates abstracts and trims author lists for display.
- Returns results with scores.

## Embedding Fallback

If no embedding provider is configured or embedding fails, fall back to
FTS5-only keyword search (skip the vector component). This keeps the
command usable without embeddings set up.

## Testing

- FTS5 keyword matching on title and abstract.
- Vector similarity search on papers.
- RRF fusion scoring (combined keyword + vector).
- Each filter independently (author, venue, after, before).
- Combined hybrid search + filters.
- Filter-only mode (no text query).
- Empty results.
- FTS5 fallback when embeddings unavailable.

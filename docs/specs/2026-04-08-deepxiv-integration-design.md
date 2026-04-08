# DeepXiv Integration Design

## Overview

Integrate [deepxiv-sdk](https://github.com/DeepXiv/deepxiv_sdk) as an optional acquire source for LENS. DeepXiv provides agent-optimized paper search with hybrid retrieval (BM25 + vector), citation counts, keywords, GitHub URLs, and progressive reading — complementing LENS's existing arXiv and OpenAlex sources.

## Goals

1. **New `lens acquire deepxiv` subcommand** for paper search and single-paper fetch
2. **Optional dependency** — when `deepxiv-sdk` is installed, the command is available; otherwise LENS works as before
3. **Richer metadata** — store `keywords` and `github_url` from DeepXiv in the papers table
4. **No changes to extract pipeline** — progressive reading optimization is deferred

## Optional Dependency

Add `deepxiv` as an optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
litellm = ["litellm>=1.40"]
deepxiv = ["deepxiv-sdk>=0.2.0"]
```

Install with `uv sync --extra deepxiv`.

## Module: `src/lens/acquire/deepxiv.py`

Availability guard at module top, matching the `litellm` pattern in `llm/client.py`:

```python
try:
    from deepxiv_sdk import Reader
    HAS_DEEPXIV = True
except ImportError:
    HAS_DEEPXIV = False
```

### `search_deepxiv(query, categories, since, max_results) -> list[dict]`

- Calls `Reader.search()` with hybrid mode
- Maps results to LENS Paper schema: `paper_id`, `arxiv_id`, `title`, `abstract`, `authors`, `date`, `citations`, `venue`, `quality_score`, `extraction_status`
- Computes `quality_score` via existing `quality.py`
- DeepXiv provides citations directly — no separate OpenAlex enrichment needed
- Synchronous (DeepXiv API is sync; no `asyncio.run()` needed)

### `fetch_deepxiv_paper(arxiv_id) -> dict`

- Calls `Reader.brief()` to get TLDR, keywords, GitHub URL, citation count
- Returns a Paper-shaped dict with `keywords` and `github_url` populated
- Used for single-paper fetch mode

## CLI: `lens acquire deepxiv`

### Search mode (default)

```
lens acquire deepxiv "transformer architecture" --max-results 20 --since 2025-01-01 --categories cs.AI,cs.CL
```

- Calls `search_deepxiv()`, stores via `store.add_papers()`
- Logs events with `source: "deepxiv"`

### Single paper mode

```
lens acquire deepxiv --paper 2507.01701
```

- Calls `fetch_deepxiv_paper()` for richer metadata
- Gets keywords, GitHub URL, citation count in one call

### Error handling

- If `deepxiv-sdk` not installed: print `"deepxiv-sdk not installed. Run: uv sync --extra deepxiv"` and exit
- DeepXiv API errors (`RateLimitError`, `NotFoundError`, etc.) caught and logged gracefully

### Flags

Mirrors existing `lens acquire arxiv` where they overlap:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-results` | 20 | Number of papers to fetch |
| `--since` | None | Only papers after this date (YYYY-MM-DD) |
| `--categories` | None | Comma-separated arXiv categories |
| `--paper` | None | Single arXiv ID to fetch (switches to single-paper mode) |

## Schema Changes

Add two nullable columns to the `papers` table via `_COLUMN_MIGRATIONS` in `store.py`:

| Column | Type | Source | Purpose |
|--------|------|--------|---------|
| `keywords` | JSON list | DeepXiv `brief()` | Feeds into vocabulary, aids search |
| `github_url` | TEXT | DeepXiv `brief()` | Reproducibility, code availability |

Existing databases upgrade seamlessly via `ALTER TABLE` in `init_tables()`. Both columns are nullable — non-DeepXiv papers simply have `NULL` values.

`keywords` is a JSON list field, added to the JSON serialization list in `store.py`.

## Testing

### Unit tests (`tests/test_acquire_deepxiv.py`)

- Mock `Reader` class (external API boundary — mocking is appropriate here)
- Test DeepXiv dict -> LENS Paper dict mapping
- Test quality score computation on DeepXiv results
- Test handling of missing/optional fields (no github_url, no keywords)
- Test availability guard: `HAS_DEEPXIV = False` -> CLI prints helpful error

### Integration test

- One `@pytest.mark.integration` test hitting real DeepXiv API
- Skipped in CI by default

## Out of Scope

- **Extract pipeline changes** — progressive reading (brief -> head -> section for triage) is a future optimization
- **Provider abstraction** — no base class, no refactoring of `arxiv.py`
- **Trending/social impact commands** — can be added later
- **Transparent replacement of `lens acquire arxiv`** — commands stay separate

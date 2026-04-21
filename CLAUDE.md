# LENS Development Guide

## Quick Reference
- **Package manager**: `uv` — use `uv sync` to install, `uv add <pkg>` to add dependencies
- **Pre-commit hooks**: `prek` (Rust-based pre-commit replacement) — install with `prek install`
- **Run CLI**: `uv run lens <command>`
- **Run tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/test_file.py::test_name -v`

## Architecture
- Single SQLite database at `~/.lens/data/lens.db` with sqlite-vec for vector search
- All models in `src/lens/store/models.py` as Pydantic `BaseModel` classes (validation only)
- Table schemas defined as SQL in `src/lens/store/store.py`
- Embedding dimension controlled by `EMBEDDING_DIM` constant in `models.py`
- LLM: openai SDK (core) + litellm (optional, `uv sync --extra litellm`). Supports gateway mode via `llm.api_base`
- Paper search: deepxiv-sdk (optional, `uv sync --extra deepxiv`). Hybrid search + progressive reading via `lens acquire deepxiv`
- Embeddings: local (sentence-transformers) or cloud (openai/litellm), via `embeddings.provider` config
- **Vocabulary** — canonical `vocabulary` table stores parameters, principles, arch slots, and agentic categories with text IDs (slugs). Seed vocabulary in `taxonomy/vocabulary.py`. Extraction prompt injects vocabulary for guided extraction; `NEW:` prefix for novel concepts.
- **Taxonomy** — single `build_vocabulary()` processes all extraction types. No clustering.
- **Hybrid search** — FTS5 keyword + sqlite-vec vector search combined via Reciprocal Rank Fusion (RRF). Used by `explain` for concept resolution.
- **Schema migrations** — `_COLUMN_MIGRATIONS` in `store.py` handles upgrades from older schemas via `ALTER TABLE` in `init_tables()`.
- **Event log** — `event_log` table records all mutations (ingest, extract, build, lint, fix). `log_event()` helper in `knowledge/events.py`. Each CLI invocation gets a `session_id` for grouping.
- **Linter** — `knowledge/linter.py` has 7 checks (orphans, contradictions, weak evidence, missing embeddings, stale extractions, near-duplicates, unverified extractions) with optional auto-fix. Lint findings and fixes are themselves logged as events.
- **Extraction trust labels** — each row in `tradeoff_extractions` / `architecture_extractions` / `agentic_extractions` carries a `verification_status` from `{verified, inferred, unverified, blocked}`, auto-computed by `extract.extractor.compute_verification_status` from confidence + evidence-quote length. `blocked` is reserved for a future verify-quotes pass. Breakdown shown by `lens status`.
- **Provenance sidecars** — `lens analyze --provenance PATH` and `lens explain --provenance PATH` write a YAML sidecar (`serve/provenance.py`) tying each claim to the backing `paper_ids` + `vocab_ids`. Built at the CLI boundary; service layer stays format-agnostic.
- **Monitor pipeline** — 5 configurable stages: acquire (arxiv) → enrich (OpenAlex + quality scores) → extract (LLM) → build (taxonomy + matrix) → ideate (gap analysis, optional LLM). Flags: `--skip-enrich`, `--skip-build`. Config: `monitor.ideate_llm` for LLM-enriched ideation.
- **Quality scoring** — `acquire/quality.py` computes 0-1 scores (citations + venue tier + recency). Auto-computed after seed acquisition and OpenAlex enrichment.
- **SPECTER2 embeddings** — `lens acquire semantic` fetches paper embeddings from Semantic Scholar API. Filters to papers with zero-vector or missing embeddings.
- **Status command** — `lens status` shows paper counts by extraction status, vocabulary by kind, matrix density, top parameters, taxonomy version, last event, and cheap lint checks.
- **Logging** — `--verbose / -v` flag on all commands. `-v` = INFO, `-vv` = DEBUG.
- **API key validation** — `_require_llm_config()` checks for LLM API key before commands that need it (extract, analyze, explain, monitor). Clear error message with setup instructions.
- CLI via Typer in `src/lens/cli.py`
- Config at `~/.lens/config.yaml`

## Conventions
- Public API is synchronous; async internals wrapped with `asyncio.run()`
- Use `store.query(table, where, params)` for reads, `store.query_sql()` for complex SQL
- Use parameterized queries (`?` placeholders) — never string-interpolate values into SQL
- Use `EMBEDDING_DIM` from `lens.store.models` instead of hardcoding vector dimensions
- JSON list fields (authors, paper_ids, etc.) are auto-serialized/deserialized by the store
- Tests use tmp_path fixtures for isolated SQLite instances
- No mocking of SQLite — use real embedded instances in tests

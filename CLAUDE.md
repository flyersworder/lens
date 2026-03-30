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
- Embeddings: local (sentence-transformers) or cloud (openai/litellm), via `embeddings.provider` config
- **Vocabulary** — canonical `vocabulary` table stores parameters, principles, arch slots, and agentic categories with text IDs (slugs). Seed vocabulary in `taxonomy/vocabulary.py`. Extraction prompt injects vocabulary for guided extraction; `NEW:` prefix for novel concepts.
- **Taxonomy** — single `build_vocabulary()` processes all extraction types. No clustering.
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

# LENS Development Guide

## Quick Reference
- **Package manager**: `uv` — use `uv sync` to install, `uv add <pkg>` to add dependencies
- **Run CLI**: `uv run lens <command>`
- **Run tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/test_file.py::test_name -v`

## Architecture
- Single LanceDB database at `~/.lens/data/lens.lance`
- All models in `src/lens/store/models.py` as Pydantic `LanceModel` classes
- Embedding dimension controlled by `EMBEDDING_DIM` constant in `models.py`
- Embeddings: local (sentence-transformers) or cloud (litellm), via `taxonomy.embedding_provider` config
- Analytics via Polars (zero-copy from Arrow)
- CLI via Typer in `src/lens/cli.py`
- Config at `~/.lens/config.yaml`

## Conventions
- Public API is synchronous; async internals wrapped with `asyncio.run()`
- All LanceDB tables use Pydantic LanceModel schemas
- Use `EMBEDDING_DIM` from `lens.store.models` instead of hardcoding vector dimensions
- Tests use tmp_path fixtures for isolated LanceDB instances
- No mocking of LanceDB — use real embedded instances in tests

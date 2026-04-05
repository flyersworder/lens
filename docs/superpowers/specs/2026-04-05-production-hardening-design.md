# Production Hardening — Design Spec

**Date:** 2026-04-05
**Status:** Approved

## Overview

Three independent additions to make Lens production-ready:

1. **Config validation** — warn on invalid `config.yaml` values at load time
2. **Backup/export** — `lens export` and `lens import` commands for DB backup/restore
3. **Deployment guide** — `docs/deployment.md` covering API keys, gateway mode, data directory, backup, and a pre-deployment checklist

## Scope

**In scope:** Config validation function, export/import CLI commands, deployment documentation.

**Out of scope:** Config schema migration, incremental backups, JSON/CSV export, REST API.

---

## 1. Config Validation

### Approach

Add `validate_config(config)` to `src/lens/config.py`. Called at the end of `load_config()`. Prints warnings via `logging.warning()` for invalid values. Never raises — config still loads with defaults merged in.

### Validation Rules

| Field | Constraint | Warning message |
|-------|-----------|-----------------|
| `llm.default_model` | Non-empty string | "llm.default_model is empty — LLM calls will fail" |
| `llm.extract_model` | Non-empty string | "llm.extract_model is empty — extraction will fail" |
| `embeddings.provider` | One of `"local"`, `"cloud"` | "embeddings.provider '{value}' is not recognized (expected 'local' or 'cloud')" |
| `embeddings.dimensions` | Positive integer | "embeddings.dimensions must be a positive integer" |
| `storage.data_dir` | Expandable path (no exception on `Path.expanduser()`) | "storage.data_dir '{value}' is not a valid path" |
| `acquire.arxiv_categories` | Non-empty list of strings | "acquire.arxiv_categories must be a non-empty list" |
| `monitor.ideate_top_n` | Positive integer | "monitor.ideate_top_n must be a positive integer" |
| `monitor.ideate_min_gap_score` | Float in 0.0–1.0 | "monitor.ideate_min_gap_score must be between 0.0 and 1.0" |

### Implementation

Plain function with `isinstance` checks — no Pydantic model. Returns a list of warning strings (for testability) in addition to logging them.

```python
def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate config and return list of warnings. Also logs each warning."""
```

### Testing

- `test_validate_config_valid` — default config produces no warnings
- `test_validate_config_empty_model` — empty model string produces warning
- `test_validate_config_invalid_provider` — unknown provider produces warning
- `test_validate_config_bad_dimensions` — non-positive dimensions produces warning
- `test_validate_config_invalid_categories` — empty list produces warning
- `test_validate_config_gap_score_range` — out-of-range score produces warning

---

## 2. Backup/Export

### Commands

**`lens export [--output PATH]`**

- Copies `~/.lens/data/lens.db` to a timestamped backup file
- Default output: `./lens-backup-YYYY-MM-DDTHHMM.db` (current directory)
- `--output` overrides destination path
- Prints output path and file size on success
- Errors if source DB doesn't exist
- Uses `shutil.copy2` (preserves metadata)

**`lens import <path> [--force]`**

- Copies backup file to `~/.lens/data/lens.db`
- Refuses to overwrite existing DB unless `--force` is passed
- Runs `init_tables()` after import to apply any pending schema migrations
- Errors if backup file doesn't exist

### Implementation

Both commands go in `src/lens/cli.py`. The actual copy logic is trivial (`shutil.copy2`). No new module needed.

### Testing

- `test_export_creates_backup` — export creates file at expected path
- `test_export_custom_output` — `--output` overrides destination
- `test_export_missing_db` — errors when source DB missing
- `test_import_restores_db` — import copies file to data dir
- `test_import_refuses_overwrite` — refuses without `--force`
- `test_import_with_force` — overwrites with `--force`
- `test_import_runs_migrations` — tables exist after import

---

## 3. Deployment Guide

### File

`docs/deployment.md`

### Sections

1. **API Key Management**
   - Precedence order: env vars > `.env` file > `config.yaml`
   - Environment variables: `OPENROUTER_API_KEY` (or provider-specific)
   - `.env` file in project root (loaded automatically by python-dotenv)
   - `config.yaml` fields: `llm.api_key`, `embeddings.api_key`
   - Recommendation: env vars in production, gateway mode preferred

2. **Gateway Mode Setup**
   - Example config for litellm proxy, vLLM, Ollama
   - Benefits: API keys stay on gateway, not in application config
   - Minimal config needed: `llm.api_base` + optional `llm.api_key`

3. **Data Directory**
   - Default: `~/.lens/data`
   - Override: `LENS_DATA_DIR` env var or `storage.data_dir` in config
   - Persistent volume mounting for containers
   - Permissions: data dir should be writable by the Lens process

4. **Backup & Restore**
   - `lens export` usage with examples
   - `lens import` usage with `--force` flag
   - Recommended backup frequency: before major extractions

5. **Config Validation**
   - What warnings mean
   - How to fix common issues
   - Running `lens config show` to verify

6. **Pre-Deployment Checklist**
   - API key configured (env var or config)
   - Data directory on persistent storage
   - Seed papers loaded (`lens acquire seed`)
   - Vocabulary initialized (`lens vocab init`)
   - Config validated (no warnings on `lens config show`)
   - Tests passing (`uv run pytest`)

### README Change

Add to the "LLM Backend" section: `For production deployment, see [docs/deployment.md](docs/deployment.md).`

---

## 4. Files Changed

| File | Change |
|------|--------|
| `src/lens/config.py` | Add `validate_config()`, call from `load_config()` |
| `src/lens/cli.py` | Add `lens export` and `lens import` commands |
| `docs/deployment.md` | **New** — deployment guide |
| `README.md` | Add link to deployment guide |
| `tests/test_config.py` | Add validation tests |
| `tests/test_cli.py` | Add export/import tests |

# Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add config validation, DB backup/restore commands, and a deployment guide to make Lens production-ready.

**Architecture:** Three independent additions: (1) `validate_config()` in config.py warns on invalid fields at load time, (2) `lens export` / `lens import` CLI commands for SQLite file backup/restore, (3) `docs/deployment.md` covering API keys, gateway mode, and a deployment checklist.

**Tech Stack:** Python (existing config.py, cli.py), shutil for file copy, Typer CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-04-05-production-hardening-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/lens/config.py` | Add `validate_config()`, call from `load_config()` |
| `src/lens/cli.py` | Add `lens export` and `lens import` commands |
| `docs/deployment.md` | **New** — deployment guide |
| `README.md` | Add link to deployment guide |
| `tests/test_config.py` | Add validation tests |
| `tests/test_cli.py` | Add export/import tests |

---

### Task 1: Config Validation — Tests

**Files:**
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for validate_config**

Append to `tests/test_config.py`:

```python
from lens.config import validate_config


def test_validate_config_valid():
    """Default config should produce no warnings."""
    from lens.config import default_config

    warnings = validate_config(default_config())
    assert warnings == []


def test_validate_config_empty_model():
    """Empty model strings should produce warnings."""
    from lens.config import default_config

    cfg = default_config()
    cfg["llm"]["default_model"] = ""
    cfg["llm"]["extract_model"] = ""
    warnings = validate_config(cfg)
    assert any("llm.default_model" in w for w in warnings)
    assert any("llm.extract_model" in w for w in warnings)


def test_validate_config_invalid_provider():
    """Unknown embedding provider should produce a warning."""
    from lens.config import default_config

    cfg = default_config()
    cfg["embeddings"]["provider"] = "invalid"
    warnings = validate_config(cfg)
    assert any("embeddings.provider" in w for w in warnings)


def test_validate_config_bad_dimensions():
    """Non-positive dimensions should produce a warning."""
    from lens.config import default_config

    cfg = default_config()
    cfg["embeddings"]["dimensions"] = -1
    warnings = validate_config(cfg)
    assert any("embeddings.dimensions" in w for w in warnings)


def test_validate_config_invalid_categories():
    """Empty arxiv_categories should produce a warning."""
    from lens.config import default_config

    cfg = default_config()
    cfg["acquire"]["arxiv_categories"] = []
    warnings = validate_config(cfg)
    assert any("arxiv_categories" in w for w in warnings)


def test_validate_config_gap_score_range():
    """Out-of-range gap score should produce a warning."""
    from lens.config import default_config

    cfg = default_config()
    cfg["monitor"]["ideate_min_gap_score"] = 1.5
    warnings = validate_config(cfg)
    assert any("ideate_min_gap_score" in w for w in warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v -k validate`
Expected: FAIL — `ImportError: cannot import name 'validate_config'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_config.py
git commit -m "test: add config validation tests (red)"
```

---

### Task 2: Config Validation — Implementation

**Files:**
- Modify: `src/lens/config.py:72-79` (add validate_config and call from load_config)

- [ ] **Step 1: Add validate_config to config.py**

Add `import logging` at the top of `src/lens/config.py` (after `from pathlib import Path`):

```python
import logging
```

Add the `validate_config` function before `load_config` (around line 62, after `_deep_merge`):

```python
logger = logging.getLogger(__name__)

_VALID_EMBEDDING_PROVIDERS = {"local", "cloud"}


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate config values and return list of warnings. Also logs each warning."""
    warnings: list[str] = []

    def _warn(msg: str) -> None:
        warnings.append(msg)
        logger.warning(msg)

    # LLM model names
    llm = config.get("llm", {})
    if not llm.get("default_model"):
        _warn("llm.default_model is empty — LLM calls will fail")
    if not llm.get("extract_model"):
        _warn("llm.extract_model is empty — extraction will fail")

    # Embedding provider
    emb = config.get("embeddings", {})
    provider = emb.get("provider", "")
    if provider not in _VALID_EMBEDDING_PROVIDERS:
        _warn(
            f"embeddings.provider '{provider}' is not recognized"
            f" (expected one of: {', '.join(sorted(_VALID_EMBEDDING_PROVIDERS))})"
        )

    # Embedding dimensions
    dims = emb.get("dimensions", 0)
    if not isinstance(dims, int) or dims <= 0:
        _warn("embeddings.dimensions must be a positive integer")

    # Storage data dir
    storage = config.get("storage", {})
    data_dir = storage.get("data_dir", "")
    if data_dir:
        try:
            Path(data_dir).expanduser()
        except RuntimeError:
            _warn(f"storage.data_dir '{data_dir}' is not a valid path")
    else:
        _warn("storage.data_dir is empty")

    # Arxiv categories
    categories = config.get("acquire", {}).get("arxiv_categories", [])
    if not isinstance(categories, list) or len(categories) == 0:
        _warn("acquire.arxiv_categories must be a non-empty list")

    # Monitor settings
    mon = config.get("monitor", {})
    top_n = mon.get("ideate_top_n", 0)
    if not isinstance(top_n, int) or top_n <= 0:
        _warn("monitor.ideate_top_n must be a positive integer")

    gap_score = mon.get("ideate_min_gap_score", 0.0)
    if not isinstance(gap_score, (int, float)) or gap_score < 0.0 or gap_score > 1.0:
        _warn("monitor.ideate_min_gap_score must be between 0.0 and 1.0")

    return warnings
```

- [ ] **Step 2: Call validate_config from load_config**

In `load_config()` (around line 79), add `validate_config(cfg)` before the return:

Change:
```python
def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    cfg = default_config()
    if Path(path).exists():
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    return cfg
```

To:
```python
def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    cfg = default_config()
    if Path(path).exists():
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, user_cfg)
    validate_config(cfg)
    return cfg
```

- [ ] **Step 3: Run validation tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: ALL PASS (both new validation tests and existing config tests)

- [ ] **Step 4: Run full test suite to check for regressions**

Run: `uv run pytest --tb=short`
Expected: ALL PASS. The `validate_config` call in `load_config` should not break any existing tests since the default config is valid and produces no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/lens/config.py
git commit -m "feat: add config validation with warnings on invalid fields"
```

---

### Task 3: Backup Export Command — Tests

**Files:**
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for lens export**

Append to `tests/test_cli.py`. First check what imports and fixtures exist at the top of the file, then add:

```python
from lens.store.store import LensStore


def test_export_creates_backup(tmp_path):
    """lens export should create a timestamped backup file."""
    # Create a source database
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "lens.db"
    store = LensStore(str(db_path))
    store.init_tables()
    store.conn.close()

    output_path = tmp_path / "backup.db"

    from lens.cli import _export_db

    _export_db(source=db_path, destination=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_export_missing_db(tmp_path):
    """lens export should raise if source DB doesn't exist."""
    import pytest

    from lens.cli import _export_db

    with pytest.raises(FileNotFoundError):
        _export_db(source=tmp_path / "nonexistent.db", destination=tmp_path / "out.db")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v -k export`
Expected: FAIL — `ImportError: cannot import name '_export_db'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_cli.py
git commit -m "test: add export command tests (red)"
```

---

### Task 4: Backup Export Command — Implementation

**Files:**
- Modify: `src/lens/cli.py`

- [ ] **Step 1: Add shutil import and _export_db helper**

Add `import shutil` to the imports at the top of `src/lens/cli.py` (near the existing `import asyncio`).

Add the helper function before the CLI commands (after `_llm_kwargs` or similar helper functions):

```python
def _export_db(source: Path, destination: Path) -> None:
    """Copy the SQLite database to the destination path."""
    if not source.exists():
        raise FileNotFoundError(f"Database not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
```

- [ ] **Step 2: Add the lens export command**

Add after the existing `show_log` command (around line 420):

```python
@app.command()
def export(
    output: str | None = typer.Option(
        None, "--output", help="Destination path for backup file."
    ),
) -> None:
    """Back up the LENS database to a file."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    db_path = data_dir / "lens.db"

    if not db_path.exists():
        rprint("[red]No database found. Run 'lens init' first.[/red]")
        raise typer.Exit(code=1)

    if output:
        dest = Path(output)
    else:
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
        dest = Path(f"lens-backup-{ts}.db")

    _export_db(source=db_path, destination=dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    rprint(f"[green]Exported database to {dest} ({size_mb:.1f} MB)[/green]")
```

- [ ] **Step 3: Run export tests**

Run: `uv run pytest tests/test_cli.py -v -k export`
Expected: ALL PASS

- [ ] **Step 4: Smoke test**

Run: `uv run lens export --help`
Expected: Shows help with `--output` option.

- [ ] **Step 5: Commit**

```bash
git add src/lens/cli.py
git commit -m "feat: add lens export command for database backup"
```

---

### Task 5: Import/Restore Command — Tests

**Files:**
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for lens import**

Append to `tests/test_cli.py`:

```python
def test_import_restores_db(tmp_path):
    """lens import should copy backup to data dir."""
    # Create a backup file (a real DB)
    backup_path = tmp_path / "backup.db"
    store = LensStore(str(backup_path))
    store.init_tables()
    store.conn.close()

    # Target data dir
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target_db = data_dir / "lens.db"

    from lens.cli import _import_db

    _import_db(source=backup_path, destination=target_db, force=False)

    assert target_db.exists()
    assert target_db.stat().st_size > 0


def test_import_refuses_overwrite(tmp_path):
    """lens import should refuse to overwrite without --force."""
    import pytest

    backup_path = tmp_path / "backup.db"
    store = LensStore(str(backup_path))
    store.init_tables()
    store.conn.close()

    target_db = tmp_path / "data" / "lens.db"
    target_db.parent.mkdir()
    target_db.write_text("existing")

    from lens.cli import _import_db

    with pytest.raises(FileExistsError):
        _import_db(source=backup_path, destination=target_db, force=False)


def test_import_with_force(tmp_path):
    """lens import --force should overwrite existing DB."""
    backup_path = tmp_path / "backup.db"
    store = LensStore(str(backup_path))
    store.init_tables()
    store.conn.close()

    target_db = tmp_path / "data" / "lens.db"
    target_db.parent.mkdir()
    target_db.write_text("existing")

    from lens.cli import _import_db

    _import_db(source=backup_path, destination=target_db, force=True)

    assert target_db.stat().st_size > len("existing")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v -k import`
Expected: FAIL — `ImportError: cannot import name '_import_db'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_cli.py
git commit -m "test: add import command tests (red)"
```

---

### Task 6: Import/Restore Command — Implementation

**Files:**
- Modify: `src/lens/cli.py`

- [ ] **Step 1: Add _import_db helper**

Add after `_export_db`:

```python
def _import_db(source: Path, destination: Path, force: bool = False) -> None:
    """Copy a backup database to the destination path."""
    if not source.exists():
        raise FileNotFoundError(f"Backup file not found: {source}")
    if destination.exists() and not force:
        raise FileExistsError(
            f"Database already exists at {destination}. Use --force to overwrite."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
```

- [ ] **Step 2: Add the lens import command**

Add after the `export` command:

```python
@app.command(name="import")
def import_db(
    path: Path = typer.Argument(..., help="Path to backup database file."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing database."),
) -> None:
    """Restore the LENS database from a backup file."""
    if not path.exists():
        rprint(f"[red]Backup file not found: {path}[/red]")
        raise typer.Exit(code=1)

    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    target_db = data_dir / "lens.db"

    if target_db.exists() and not force:
        rprint(
            f"[red]Database already exists at {target_db}. "
            f"Use --force to overwrite.[/red]"
        )
        raise typer.Exit(code=1)

    data_dir.mkdir(parents=True, exist_ok=True)
    _import_db(source=path, destination=target_db, force=force)

    # Run migrations on imported database
    store = LensStore(str(target_db))
    store.init_tables()

    rprint(f"[green]Restored database from {path} to {target_db}[/green]")
```

- [ ] **Step 3: Run import tests**

Run: `uv run pytest tests/test_cli.py -v -k import`
Expected: ALL PASS

- [ ] **Step 4: Smoke test**

Run: `uv run lens import --help`
Expected: Shows help with `PATH` argument and `--force` option.

- [ ] **Step 5: Commit**

```bash
git add src/lens/cli.py
git commit -m "feat: add lens import command for database restore"
```

---

### Task 7: Deployment Guide

**Files:**
- Create: `docs/deployment.md`
- Modify: `README.md`

- [ ] **Step 1: Create docs/deployment.md**

```markdown
# LENS Deployment Guide

Production deployment guide for LENS — covers API key management, gateway mode, data persistence, backup, and a pre-deployment checklist.

## API Key Management

LENS needs API keys for LLM calls and (optionally) cloud embeddings. Keys are resolved in this order (first match wins):

1. **Environment variables** (recommended for production)
   - `OPENROUTER_API_KEY` — used by the default OpenRouter models
   - Or any provider-specific variable supported by your LLM backend

2. **`.env` file** in the project root
   - Loaded automatically by python-dotenv at startup
   - Good for local development

3. **`config.yaml`** fields
   - `llm.api_key` and `embeddings.api_key`
   - Stored in `~/.lens/config.yaml`
   - Least secure — keys are in plaintext on disk

**Recommendation:** Use environment variables in production. For maximum security, use gateway mode (below) so API keys never reach the application.

## Gateway Mode

Gateway mode points LENS at an OpenAI-compatible proxy (litellm, vLLM, Ollama). The proxy handles authentication — LENS only needs the gateway URL.

```yaml
# ~/.lens/config.yaml
llm:
  api_base: "http://litellm-gateway:4000/v1"
  api_key: "gateway-internal-key"  # optional, depends on gateway config
  default_model: "gpt-4"
  extract_model: "gpt-4"
```

**Benefits:**
- API keys stay on the gateway, not in the application
- Switch providers without changing application config
- Rate limiting and cost tracking at the gateway level
- No `litellm` dependency needed in the application

**Without gateway** (direct mode): install litellm (`uv sync --extra litellm`) and set API keys via environment variables.

## Data Directory

LENS stores its SQLite database at `~/.lens/data/lens.db` by default.

**Override with:**
- Environment variable: `LENS_DATA_DIR=/path/to/data`
- Config: `storage.data_dir: "/path/to/data"` in `config.yaml`

**For containers:** Mount the data directory as a persistent volume:

```bash
docker run -v /host/lens-data:/root/.lens/data lens:latest
```

**Permissions:** The data directory must be writable by the LENS process. SQLite also creates `-wal` and `-shm` files alongside the database.

## Backup & Restore

**Export** (backup):
```bash
# Default: creates lens-backup-YYYY-MM-DDTHHMM.db in current directory
uv run lens export

# Custom destination
uv run lens export --output /backups/lens-2026-04-05.db
```

**Import** (restore):
```bash
# Restore from backup (refuses if DB exists)
uv run lens import /backups/lens-2026-04-05.db

# Overwrite existing DB
uv run lens import /backups/lens-2026-04-05.db --force
```

Import automatically runs schema migrations on the restored database.

**When to back up:**
- Before major extractions (`lens extract` on new papers)
- Before `lens lint --fix` (especially duplicate merges)
- Before `lens init --force` (which destroys the database)

## Config Validation

LENS validates `config.yaml` on every command and warns about invalid values. Warnings do not prevent execution — commands still run with defaults.

Common warnings and fixes:

| Warning | Fix |
|---------|-----|
| `llm.default_model is empty` | Set a model: `lens config set llm.default_model openrouter/anthropic/claude-sonnet-4-6` |
| `embeddings.provider 'X' is not recognized` | Use `local` or `cloud`: `lens config set embeddings.provider local` |
| `embeddings.dimensions must be a positive integer` | Set dimensions: `lens config set embeddings.dimensions 768` |

Run `lens config show` to verify your configuration.

## Pre-Deployment Checklist

- [ ] **API key configured** — env var, `.env`, or `config.yaml` (verify with `lens config show`)
- [ ] **Data directory on persistent storage** — not ephemeral container filesystem
- [ ] **Database initialized** — `lens init`
- [ ] **Seed papers loaded** — `lens acquire seed`
- [ ] **Vocabulary initialized** — `lens vocab init`
- [ ] **Config validated** — no warnings from `lens config show`
- [ ] **Backup strategy** — periodic `lens export` to external storage
- [ ] **Tests passing** — `uv run pytest` (185+ tests)
```

- [ ] **Step 2: Add link to README.md**

In `README.md`, after the "LLM Backend" heading section (around line 107, after the first paragraph), add:

```markdown
For production deployment, see [docs/deployment.md](docs/deployment.md).
```

- [ ] **Step 3: Commit**

```bash
git add docs/deployment.md README.md
git commit -m "docs: add deployment guide with API key management, backup, and checklist"
```

---

### Task 8: Full Test Suite + Smoke Test

**Files:** None — verification only.

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS. No regressions.

- [ ] **Step 2: Smoke test export/import**

```bash
uv run lens init
uv run lens export --output /tmp/lens-test-backup.db
uv run lens import /tmp/lens-test-backup.db --force
```
Expected: Both commands succeed with green output.

- [ ] **Step 3: Smoke test config validation**

```bash
uv run lens config show
```
Expected: No warnings printed (default config is valid).

- [ ] **Step 4: Verify deployment guide renders**

Check `docs/deployment.md` exists and is well-formatted.

- [ ] **Step 5: Final commit if any cleanup needed**

```bash
git add -u
git commit -m "chore: final cleanup for production hardening"
```

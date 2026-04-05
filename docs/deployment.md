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
# Default: creates lens-backup-YYYY-MM-DDTHHMMSS.db in current directory
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
- [ ] **Tests passing** — `uv run pytest` (190+ tests)

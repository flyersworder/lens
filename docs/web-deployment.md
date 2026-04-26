# LENS — Public Web Deployment Architecture

Architecture for shipping LENS as a public read-only website on a free tier, using Turso for the database, GitHub Actions for the build pipeline, and Vercel for the web frontend + API.

**Status:** Spike validated 2026-04-26; implementation starting
**Date:** 2026-04-26
**Companion doc:** [`deployment.md`](deployment.md) covers self-hosted / local deployment; this doc covers public web deployment.

---

## Locked decisions (ADRs)

These are the architectural commitments confirmed 2026-04-26 after spike validation. Changes from here require a documented rationale.

| # | Decision | Choice | Rationale |
|---|---|---|---|
| **D1** | Embedding model (build + runtime) | `openai/text-embedding-3-small` (1536-dim) via OpenRouter | Battle-tested, ecosystem-friendly, ~$0 at LENS scale, compatible with future pgvector if we migrate to Neon. |
| **D2** | Request-time LLM | `deepseek/deepseek-v4-flash` ($0.14/M in, $0.28/M out) | New April 2026 release, 5.4× cheaper output than Gemini 3.1 Flash Lite Preview, 1M context, MoE 284B/13B-active, ~$2.10/mo at projected volume — well under $5/mo cap. |
| **D3** | Response cache | 24-hour TTL, key = `endpoint + query.lower().strip()` | Coarse but predictable; bounded staleness; lets caching actually pay. |
| **D4** | Repository structure | Monorepo, single deploy: Next.js at root + `api/*.py` auto-detected by Vercel | Atomic versioning, one deploy, what Vercel docs assume. |
| **D5** | Local-vs-Turso `Store` mode | Dual-backend: existing `Store` (sqlite-vec) for CLI/build/tests; new `TursoStore` (libSQL native) for Vercel runtime | Spike confirmed `file:` URLs lack libSQL vector functions; dual-backend isolates Turso-specific code and keeps tests offline. |

---

## Goals and constraints

1. **Public read-only website** — anyone can run `analyze`, `explain`, and `search` queries against the LENS knowledge base via HTTP.
2. **Near-zero-dollar prototype** — fits inside the free tiers of all platforms; the only paid line item is a one-time OpenRouter credit purchase to unlock cheap-but-good LLM models for request-time disambiguation and narrative generation.
3. **Live LLM at request time, but cheap and rate-limited** — `analyze` and `explain` retain their LLM-driven UX (concept disambiguation, narrative). Calls go to a cheap OpenRouter model with per-IP throttling, response caching, and a hard spending cap.
4. **Single Python source of truth** — the same `serve/analyzer.py` / `serve/explainer.py` / `serve/explorer.py` modules used by the CLI also power the web API, deployed as Python Vercel Functions. No TypeScript port, no two-implementation drift.
5. **Migration-friendly** — every component can be swapped out independently as the project grows.

---

## High-level architecture

```
┌──────────────────────────────────────────────────────────────┐
│  GitHub Actions (free for public repos, 2000 min/mo)         │
│                                                              │
│  Weekly cron: run `lens monitor` end-to-end                  │
│    1. acquire  — pull new arXiv papers                       │
│    2. enrich   — OpenAlex citation counts + quality scores   │
│    3. extract  — LLM extraction (OpenRouter free models)     │
│    4. build    — vocabulary, matrix, embeddings              │
│    5. ideate   — gap analysis (optional)                     │
│  Push the resulting SQLite file to Turso via libSQL sync.    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  Turso (libSQL, 5 GB free, 500 M reads/mo)                   │
│                                                              │
│  Holds:                                                      │
│    • papers, vocabulary, extractions                         │
│    • contradiction_matrix, architecture variants, patterns   │
│    • FTS5 indices, libSQL native vector indices              │
│    • event_log                                               │
│                                                              │
│  Edge replicas in 30+ regions; embedded replica syncs into   │
│  each Vercel function instance for sub-ms reads.             │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  Vercel (free Hobby tier — 1M invocations, 4hr CPU/mo)       │
│                                                              │
│  Next.js (App Router) — static-rendered shell:               │
│    • / — landing + search box                                │
│    • /analyze — tradeoff resolution UI                       │
│    • /explain/:concept — concept explanation page            │
│    • /search — paper search results                          │
│                                                              │
│  Python FastAPI (Vercel Functions, Python runtime):          │
│    • POST /api/analyze    — calls serve/analyzer.py directly │
│    • POST /api/explain    — calls serve/explainer.py         │
│    • GET  /api/search     — calls serve/explorer.py          │
│                                                              │
│  Each function opens a Turso embedded replica on cold start  │
│  (128 KiB partial sync, cached in /tmp). Live LLM calls go   │
│  to OpenRouter with per-IP rate limit + response cache.      │
└──────────────────────────────────────────────────────────────┘
```

---

## Component responsibilities

### Turso (the database)

| Property | Value |
|---|---|
| Engine | libSQL (SQLite-compatible fork) |
| Free tier | 5 GB storage, 500 M row reads/mo, 10 M row writes/mo, 100 DBs |
| Vector search | Native libSQL vector type + LM-DiskANN index. Index DDL: `CREATE INDEX … USING libsql_vector_idx(column)`; queries use `vector_top_k(idx_name, query_vec, k)` |
| FTS | SQLite FTS5 (unchanged from current LENS schema) |
| Replication | **Partial-sync embedded replicas** — fetches 128 KiB pages on demand into `/tmp`, persists across warm invocations. First page-miss request pays a small download; subsequent reads hit the local SQLite file. |
| Migration cost from current LENS | Low — schema is already SQLite-shaped |
| Maturity caveat | The Turso/Vercel integration package is **in BETA** as of April 2026. Production-readiness has improved but is worth re-checking before scaling. |

**Why Turso instead of Neon for the prototype:**
- LENS already uses sqlite-vec → libSQL is a near drop-in replacement
- Embedded replicas eliminate cold-start latency (Neon compute wakes on demand, ~300 ms first request)
- 5 GB free tier vs Neon's 0.5 GB — comfortable headroom for embeddings
- No SQL-dialect rewrite, no FTS5→tsvector migration

**Why we'd migrate to Neon later (deferred):**
- Postgres ecosystem (LangChain, LlamaIndex, pgvector tooling)
- Database branching for the build pipeline
- More predictable pricing at scale (linear $/GB-month vs. tier jumps)

See **[Migration path: Turso → Neon](#migration-path-turso--neon)** below.

#### Verified conventions (from spike)

These are gotchas confirmed by the live spike against `lens-dev`. Codify them in `TursoStore`:

1. **URL scheme conversion is required.** The Python `libsql-client` interprets `libsql://…` as a WebSocket transport that fails with HTTP 505 against current Turso edge servers. Always rewrite the URL to `https://…` before passing to `create_client_sync`. The `turso db show` command emits `libsql://` by convention; convert at the edge of `TursoStore`.
2. **Vector functions only work on remote Turso, not on `file:` URLs.** This is why `Store` (sqlite-vec) and `TursoStore` (libSQL native) coexist as separate backends — local file mode of `libsql-client` is plain SQLite without extensions. Tests against `TursoStore` therefore require a real remote DB (use `lens-dev`).
3. **Vector index DDL:** `CREATE INDEX <idx_name> ON <tbl> (libsql_vector_idx(<col>))` — column must be `F32_BLOB(<dim>)`.
4. **Insertion accepts both literal and parameterized vectors:**
   - Literal: `INSERT … VALUES (?, vector('[0.1, 0.2, …]'))` — convenient for one-offs
   - Parameterized blob: `INSERT … VALUES (?, ?)` with `struct.pack(f"{dim}f", *floats)` as the second argument — preferred for bulk inserts (no string-formatting overhead)
5. **`vector_top_k` aliasing pattern (mandatory):**
   ```sql
   SELECT v.* FROM vector_top_k('<idx>', vector32(?), <k>) AS k
   JOIN <tbl> v ON v.rowid = k.id
   ```
   Without `AS k` aliasing the `id` column is ambiguous and the server rejects the query. The Python client surfaces this as a confusing `KeyError: 'result'` rather than the underlying SQL error — always run `turso db shell` to see real server-side errors when the Python client raises a parse error.
6. **FTS5 works identically on remote Turso** as on local SQLite — no syntax changes, `content_rowid=rowid`, `MATCH ?`, and `INSERT INTO <name>(<name>) VALUES('rebuild')` all behave the same.

### GitHub Actions (the build pipeline)

| Property | Value |
|---|---|
| Free tier | **Unlimited minutes for public repos** on standard runners (the 2000 min/mo limit applies only to private repos) |
| Schedule | Weekly cron — sufficient for arXiv velocity |
| Secrets | `OPENROUTER_API_KEY`, `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN` |
| LLM | Cheap-but-good OpenRouter models (e.g. `deepseek/deepseek-chat`, `google/gemini-flash`, `openai/gpt-4o-mini`, or a free-tier model if a $10+ credit balance is maintained for the 1000 req/day cap) via the `litellm` extra |
| Output | SQLite file synced to Turso primary |
| Atomicity | Build runs `pull → modify → push` against a working copy; wrap in `try/finally` so a mid-run failure doesn't push a half-modified DB. |

**Workflow shape** (`.github/workflows/monitor.yml`):

```yaml
name: Weekly LENS Monitor
on:
  schedule:
    - cron: "0 6 * * 1"   # Mondays 06:00 UTC
  workflow_dispatch:        # manual trigger for backfills

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd  # v6
      - uses: astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b  # v8.1.0
      - run: uv python install 3.12
      - run: uv sync --all-extras --dev --frozen

      # Pull current DB from Turso into local lens.db
      - run: uv run python scripts/pull_from_turso.py
        env:
          TURSO_DATABASE_URL: ${{ secrets.TURSO_DATABASE_URL }}
          TURSO_AUTH_TOKEN:   ${{ secrets.TURSO_AUTH_TOKEN }}

      # Run the full monitor cycle
      - run: uv run lens monitor
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}

      # Push updated DB back to Turso
      - run: uv run python scripts/push_to_turso.py
        env:
          TURSO_DATABASE_URL: ${{ secrets.TURSO_DATABASE_URL }}
          TURSO_AUTH_TOKEN:   ${{ secrets.TURSO_AUTH_TOKEN }}
```

**Sync strategy:** simplest viable approach — `pull → run → push` against a single Turso DB. Once the DB grows large enough that pulling is slow, switch to incremental sync via the libSQL Python client (planned but not blocking).

### Vercel (the web tier)

| Property | Value |
|---|---|
| Free Hobby tier | 1 M function invocations/mo, **4 hours Active CPU/mo**, 360 GB-hours Provisioned Memory/mo, 1 M edge requests/mo, 100 GB bandwidth/mo |
| Function timeout | 60 s on Hobby (Pro extends to 13 min via Fluid Compute) |
| Function bundle limit | 500 MB unzipped (raised from 250 MB in 2025) |
| ToS caveat | Hobby is "personal, non-commercial use only." A research demo qualifies; if LENS later carries ads, sponsorships, or paid features, upgrade to Pro ($20/mo). |
| Frontend | Next.js 15 App Router, statically rendered shell + dynamic API routes |
| API | **Python FastAPI** as Vercel Functions (Python runtime), calling the existing `src/lens/serve/*` modules directly |
| Cold start | Vercel's Fluid Compute does bytecode caching + automatic pre-warming for FastAPI; cold-start prevention is now a platform feature, not a plan-tier feature |
| Persistent FS | None — but Turso embedded replicas live in `/tmp` per instance |

**Why Python on Vercel (and not a TypeScript port):**

- **Zero porting work.** `serve/analyzer.py` (233 lines), `serve/explainer.py` (443 lines), and `serve/explorer.py` (219 lines) get imported as-is into Python Vercel Functions. The same code that powers the CLI powers the web API. No drift, no double maintenance.
- **First-class platform support since 2025.** Vercel's Python runtime now ships zero-config FastAPI detection, full ASGI/WSGI support, bytecode caching, and Fluid Compute cold-start prevention.
- **Bundle fits comfortably.** The runtime needs `fastapi`, `libsql-client`, `openai`/`litellm`, `pydantic`, `tenacity`, `json-repair`, `pyyaml` — well under the 500 MB cap. **Critically: `sentence-transformers` and `torch` are dropped from both runtime and build paths** by setting `embeddings.provider: cloud` in `~/.lens/config.yaml` — embeddings come from an OpenRouter-hosted embedding model (free or near-free), so torch is never installed. This also speeds up GitHub Actions runs significantly (no multi-GB PyTorch download).
- **libsql-client Python is production-ready** for HTTP-mode access; the embedded-replica package is in BETA but our footprint (3 read-only endpoints) is exactly its sweet spot.

**API shape** (`api/` directory at repo root, Vercel auto-detects):

```python
# api/analyze.py
from fastapi import FastAPI, Request
from libsql_client import create_client
from lens.serve.analyzer import analyze_tradeoff
from lens.llm.client import LLMClient
import os

app = FastAPI()
db = create_client(
    url=os.environ["TURSO_DATABASE_URL"],
    auth_token=os.environ["TURSO_AUTH_TOKEN"],
    sync_url=os.environ["TURSO_DATABASE_URL"],  # embedded replica
)
llm_client = LLMClient(...)  # OpenRouter, cheap model

@app.post("/api/analyze")
async def analyze(req: Request):
    body = await req.json()
    # rate-limit check, cache lookup, then:
    return await analyze_tradeoff(body["query"], db, llm_client)
```

**Adapter layer needed** (small): `serve/analyzer.py` currently expects the LENS `Store` object. We add a thin libSQL-backed `Store` shim (or extend `Store` with a libSQL connection mode) so the existing code paths work unchanged. Estimated effort: ~0.5 day.

---

### Embedding and LLM strategy

The site makes two kinds of model calls at request time: **one embedding call** (to vectorize the user's query for hybrid search) and **one or more LLM calls** (concept disambiguation in `explain`, narrative generation in `analyze`). Both go through OpenRouter.

| Call type | Where | Model recommendation | Why |
|---|---|---|---|
| Embedding (build) | GH Actions, weekly | `nvidia/llama-nemotron-embed-vl-1b-v2:free` *or* `openai/text-embedding-3-small` (paid, $0.02/M tokens) | Free model is fine for batch; paid is ~free at LENS scale and removes rate-limit concerns. |
| Embedding (runtime) | Vercel Functions | Same model as build (must match `EMBEDDING_DIM`) | The query vector must live in the same space as stored vectors. |
| LLM disambiguation + narrative | Vercel Functions | `deepseek/deepseek-chat` or `google/gemini-flash` (paid, both <$1/M tokens) | Cheap, capable, stable. Free-tier models risk hitting the 1000/day cap during traffic spikes. |
| LLM extraction | GH Actions, weekly | Same model — or stronger model for higher-quality extractions if budget allows | Extraction quality compounds; spending a few cents extra per run pays back. |

**Critical schema constraint:** the embedding model used at *build* time and the embedding model used at *runtime* must match (same model → same `EMBEDDING_DIM` → same vector space). If you swap embedding models later, the entire DB must be re-embedded. The current LENS schema migration system (`_COLUMN_MIGRATIONS` in `store.py`) handles dimension changes via `ALTER TABLE`, but a full re-embed of all rows is required. Pick once, commit.

### Request-time LLM safety: rate limit, cache, spending cap

Live LLM at the edge is cheap *only if* it's protected. The Vercel Python Functions need three guardrails:

1. **Per-IP rate limit** — e.g., 10 requests/min per IP, enforced via Vercel KV (free 30K req/mo). Prevents bot loops from draining credits.
2. **Response cache** — every `analyze`/`explain` request keyed by query string + type, cached for 24 h in Vercel KV or as a stored row in Turso. Most public queries repeat (`"reduce hallucination"`, `"MoE explained"`, etc.); caching ~10× the underlying API cost.
3. **OpenRouter spending cap** — set a hard monthly limit in OpenRouter dashboard (e.g., $5/mo). If exceeded, the API returns `402` and the Function falls back to a cached or static response.

**Estimated cost per active month** (with 10K visitors, average 3 queries each, 80% cache hit rate):
- Cache misses: ~6,000 LLM calls
- Average 1.5K tokens in + 500 tokens out per call ≈ 12 M tokens total
- At DeepSeek Chat pricing (~$0.14/M input, $0.28/M output): **~$2/month**
- Plus embedding calls: ~30K embeddings × 100 tokens × $0.02/M = **<$0.10/month**

Budget cap of **$5/month** gives ~2× safety margin and triple-digit-thousand visitor headroom.

## Data flow

### Build path (weekly, offline)

1. GitHub Actions runner clones LENS repo, installs dependencies via `uv sync`.
2. Runner pulls current `lens.db` from Turso into local filesystem.
3. `lens monitor` runs the 5-stage pipeline against the local DB, calling OpenRouter free models for extraction.
4. Updated `lens.db` is pushed back to Turso primary.
5. Turso replicates to edge regions automatically.

### Read path (per request, online)

1. User hits `vercel.app/explain/grouped-query-attention`.
2. Next.js server component calls `/api/explain`.
3. Vercel Function instance (warm or cold) opens a Turso embedded replica.
4. Function runs hybrid search (FTS5 + vector) + matrix lookup against the *local* SQLite file.
5. Results returned to the page, rendered, sent to user.
6. In the background, the embedded replica syncs any new writes from the primary.

---

## Cost projection

### Free-tier capacity for LENS

| Resource | Free limit | LENS estimated usage at 6 months | Headroom |
|---|---|---|---|
| Turso storage | 5 GB | ~200 MB (≈ 2K papers × 100 KB) | 25× |
| Turso row reads | 500 M / mo | ~3 M (10K visitors × 30 queries × 10 rows) | 150× |
| Turso row writes | 10 M / mo | ~50K (weekly monitor cycles) | 200× |
| GitHub Actions (public repo) | **unlimited** | ~80 min/mo (4 weekly runs × 20 min, much faster without torch) | n/a |
| Vercel bandwidth | 100 GB / mo | ~5 GB (10K visitors × 500 KB) | 20× |
| Vercel invocations | 1 M / mo | ~30K | 33× |
| Vercel Active CPU | **4 hours / mo** | ~1 hour (avg ~100 ms × 30K invocations) | 4× |
| Vercel Edge Requests | 1 M / mo | ~50K | 20× |
| OpenRouter (paid credits ≥$10) | 1000 req/day on free models, or pay-per-use on cheap models | well within with caching | comfortable |
| OpenRouter spend (cheap models) | hard cap $5/mo (configurable) | ~$2/mo at 10K visitors | 2.5× |

**Verdict:** The whole stack is essentially free except for ~$2–5/month in OpenRouter usage if traffic grows. Vercel Active CPU is the tightest constraint and the first thing to watch — at higher traffic, increase response-cache hit rate to keep CPU time per request low.

### When you'd start paying — and how much

The growth path to watch:

| Threshold crossed | Action | Cost |
|---|---|---|
| **Turso 5 GB → 9 GB** (≈ 50K papers with embeddings) | Upgrade to Turso Developer | **$5/mo** |
| **Turso 9 GB → 24 GB** | Upgrade to Turso Scaler | **$25/mo** |
| **Vercel 100 GB bandwidth/mo exceeded** | Upgrade to Vercel Pro | **$20/mo** |
| **GitHub Actions 2000 min exceeded** | Buy minutes ($0.008/min) | typically <$2/mo unless monitor runs daily |
| **OpenRouter free models hit rate limit** | Pay-as-you-go for paid models | ~$1–5/run depending on model |

**Realistic 12-month total at modest growth:** $0/mo (months 1–6), $5/mo (months 6–12) for Turso Developer if the corpus grows past 5 GB. Everything else stays free.

**Decision point for migrating to Neon:**
- Hit Turso Scaler ($25/mo for 24 GB), *and*
- Want pgvector / Postgres ecosystem, *or*
- Need branching for a multi-environment pipeline

At that point Neon Launch is $5/mo + $0.30/GB-month storage, which can be cheaper if storage is the dominant cost.

---

## Migration cost from current LENS

The repo is currently SQLite + sqlite-vec with default local embeddings. To ship the public version:

| Task | Estimate | Notes |
|---|---|---|
| Build new `TursoStore` class (read API parity with `Store`) | 1 day | libSQL-native vectors via `vector_top_k` + RRF; `Store` itself untouched, sqlite-vec stays for local/build/test |
| Write `scripts/publish_to_turso.py` (translates sqlite-vec schema → libSQL native, copies data, rebuilds FTS5) | 1 day | Reads local `lens.db`, recreates schema with `F32_BLOB` columns + `libsql_vector_idx`, batch-inserts |
| Wire `TursoStore` into `serve/analyzer.py`, `explainer.py`, `explorer.py` (accept either backend) | 0.5 day | Both expose same query API; small adapter shim |
| Add tests for `TursoStore` against `lens-dev` (gated on `TURSO_DEV_*` env vars) | 0.5 day | Skip when env vars absent so default test runs stay offline |
| Switch `embeddings.provider` from `local` to `cloud` (OpenRouter), re-embed corpus once | 0.5 day | `embedder.py` cloud path already wired; config + rebuild run |
| Build FastAPI app under `api/` with 3 endpoints + per-IP rate limit + KV cache | 1 day | Direct imports of `serve/*.py` modules with `TursoStore` backend |
| Build minimal Next.js frontend (3 pages, calls `/api/*`) | 1 day | Tailwind + shadcn/ui for speed |
| Wire GitHub Actions monitor workflow + `publish_to_turso.py` post-step | 0.5 day | Secret config, uv setup, try/finally around publish |
| Deploy to Vercel, set env vars (Turso URL/token, OpenRouter key), spending cap | 0.5 day | |
| **Total** | **~6.5 days** | for a single developer |

The dual-backend approach is more work than the original "swap `Store` internals" plan but trades risk for predictability: tests and local dev never need internet, and the Turso-specific code is contained to `TursoStore` + one publish script.

The TypeScript port is gone, replaced by a thin FastAPI layer that imports the existing Python modules — net savings versus the prior plan, with stronger correctness guarantees (no two implementations to drift).

---

## Open questions / decisions deferred

1. **Frontend framework choice.** Next.js is the obvious default; Astro or SvelteKit would also work and are sometimes leaner. Defaulting to Next.js because it's Vercel-native.
2. **Authentication.** None for the prototype. If we add user accounts later, Clerk free tier (10K MAU) or Vercel's own auth integration.
3. **Analytics.** Vercel Analytics free tier is enough for the prototype.
4. **Provenance sidecars.** `lens analyze --provenance` and `lens explain --provenance` currently write YAML files. On Vercel Functions there's no persistent FS — return the YAML in the HTTP response body when `?provenance=true` is set, or omit the feature from the public site. Default: return-in-response.
5. **DB versioning.** Each weekly monitor run replaces the live DB. If we want history (or rollback), use Turso's branching (paid) or push timestamped snapshots to a GitHub release as backup.
6. **Cache backend.** Vercel KV (free 30K req/mo) for response cache and rate-limit counters. If KV usage outgrows the free tier, fall back to a small `cache` table in Turso.

---

## Migration path: Turso → Neon

When the prototype graduates to a product, migrating to Neon is a contained project:

1. **Schema translation** (1 day) — SQLite → Postgres DDL. Most types map 1:1; the gotchas are `JSON` columns (use `jsonb`) and `INTEGER PRIMARY KEY` (use `BIGSERIAL`).
2. **FTS rewrite** (1.5 days) — SQLite FTS5 → Postgres `tsvector` + GIN index. Postgres FTS doesn't ship BM25; install `pg_search` (or accept tsvector ranking, which differs from FTS5's BM25 in retrieval quality). Tune the RRF weights afterward.
3. **Vector index swap** (0.5 day) — libSQL vectors → `pgvector` with HNSW. Embedding columns are unchanged if the embedding model stays constant.
4. **Store layer rewrite** (1.5 days) — `store.py` swaps `sqlite3` for `asyncpg` or `psycopg`; `?` placeholders become `$1, $2`. Query helpers stay structurally similar.
5. **API layer** — *no rewrite needed.* The Python FastAPI `api/` modules just point at the new Neon-backed `Store`. This is the major win versus the prior TypeScript-based plan.
6. **Build pipeline** — point the GitHub Actions cron at Neon instead of Turso. Use a `staging` branch for builds, promote to `main` on success — *this is the feature we're trading away today and gaining at migration time*.

**Total estimated migration cost: 6–8 days.** Larger than the original Turso build, primarily because of the FTS5 → Postgres FTS rewrite. Worth it once branching, pgvector ecosystem maturity, or per-GB pricing becomes the bottleneck.

---

## Implementation checklist

When ready to ship, work through these in order:

**Database & embedding setup**
- [x] ~~Create Turso account, create `lens-prod` and `lens-dev` databases, capture URL + auth tokens~~ — done 2026-04-26; credentials in `.env.local`
- [x] ~~Validate libSQL native vectors + FTS5 against remote Turso (Spike 2)~~ — passed 2026-04-26
- [ ] Top up OpenRouter with $10 credit (one-time) to unlock 1000 req/day on free models *and* enable cheap paid models
- [ ] Add `libsql-client` to `pyproject.toml` as a `[project.optional-dependencies] turso` extra
- [ ] Build new `TursoStore` class with read API matching `Store` (vector queries via `vector_top_k`, FTS5 unchanged)
- [ ] Write `scripts/publish_to_turso.py` to translate sqlite-vec schema → libSQL native and copy data
- [ ] Wire `TursoStore` into `serve/analyzer.py`, `serve/explainer.py`, `serve/explorer.py` via a small adapter
- [ ] Add `TursoStore` integration tests gated on `TURSO_DEV_*` env vars (skip when offline)
- [ ] Switch local config to `embeddings.provider: cloud` with OpenRouter base URL; verify `embedder.py` cloud path works end-to-end
- [ ] Re-embed the existing corpus once with the chosen embedding model (locks `EMBEDDING_DIM`)

**Build pipeline**
- [ ] Write `scripts/pull_from_turso.py` and `scripts/push_to_turso.py` (try/finally around the modify step)
- [ ] Add `.github/workflows/monitor.yml` with weekly cron + `workflow_dispatch`
- [ ] Configure GitHub Secrets: `OPENROUTER_API_KEY`, `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`
- [ ] Run the workflow once via `workflow_dispatch` to seed the prod DB

**Web tier**
- [ ] Create `api/` directory at repo root with `analyze.py`, `explain.py`, `search.py` FastAPI handlers
- [ ] Wire per-IP rate limit + response cache via Vercel KV
- [ ] Set OpenRouter monthly spending cap in OpenRouter dashboard
- [ ] Scaffold Next.js project in `web/` (or separate repo) with 3 pages
- [ ] Deploy to Vercel; set env vars: `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`, `OPENROUTER_API_KEY`, `KV_REST_API_URL`, `KV_REST_API_TOKEN`
- [ ] Smoke-test all three endpoints against prod Turso (warm and cold start)
- [ ] Verify `/tmp` libSQL replica behaviour under sustained traffic

**Polish**
- [ ] (Optional) Configure custom domain
- [ ] Add Vercel Analytics
- [ ] Update README with the public URL and a note that the site is non-commercial under Vercel Hobby ToS

---

## References

- [Turso pricing](https://turso.tech/pricing) — current free tier and paid tiers
- [Turso AI & Embeddings](https://docs.turso.tech/features/ai-and-embeddings) — `libsql_vector_idx`, `vector_top_k`, LM-DiskANN
- [Bringing SQLite to Vercel Functions with Turso](https://turso.tech/blog/serverless) — partial sync, `/tmp` cache, BETA status
- [Vercel Hobby Plan](https://vercel.com/docs/plans/hobby) and [Vercel Functions Limits](https://vercel.com/docs/functions/limitations) — invocations, Active CPU, bundle size
- [Deploy a FastAPI app on Vercel](https://vercel.com/docs/frameworks/backend/fastapi) — Python runtime, Fluid Compute
- [GitHub Actions billing](https://docs.github.com/en/billing/managing-billing-for-your-products/managing-billing-for-github-actions/about-billing-for-github-actions) — public-repo unlimited minutes
- [OpenRouter Embeddings](https://openrouter.ai/docs/api/reference/embeddings) and [Free Models](https://openrouter.ai/collections/free-models)
- [OpenRouter Rate Limits](https://openrouter.ai/docs/api/reference/limits) — 50/day no-credit, 1000/day with $10+ credits
- [`docs/deployment.md`](deployment.md) — self-hosted deployment (the existing guide)
- [`docs/architecture.md`](architecture.md) — system design

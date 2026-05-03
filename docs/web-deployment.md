# LENS — Public Web Deployment Architecture

Architecture for shipping LENS as a public read-only website on a free tier, using Turso for the database, GitHub Actions for the build pipeline, and Vercel for the web frontend + API.

**Status:** **Live in production** — `https://lens-fawn.vercel.app` (Vercel Hobby) backed by **`lens-prod`** Turso DB with the real 77-paper corpus. All 4 routes return 200; full stack (cloud embeddings + DeepSeek V4 Flash + libSQL hybrid search + FastAPI) verified end-to-end 2026-04-26 against `lens-prod`.
**Date:** 2026-04-26
**Companion doc:** [`deployment.md`](deployment.md) covers self-hosted / local deployment; this doc covers public web deployment.

---

## Deployment topology (2026-05-03)

ADR D4 ("single deploy") is **superseded**. Combining Next.js + Python in one Vercel project blew the 500 MB unzipped function bundle cap; multiple workarounds were attempted and none held. The deploy is now split into two Vercel projects in the same `flyersworders-projects` team:

| Project | Vercel ID | Config file | Deploys what |
|---|---|---|---|
| `lens` | `prj_ga9XFH4y8Q5jheH5zxlw4SkB9h4e` | `vercel.json` | Next.js frontend; `/api/*` rewrites to the `lens-api` host |
| `lens-api` | `prj_1C1cnBnJZpy7U8qoNfZAqkeWYGYM` | `vercel.api.json` | Python FastAPI in `api/index.py` |

Both projects link from the same repo root. The Vercel CLI selects one or the other via `VERCEL_PROJECT_ID` + `--local-config` — there is no longer a `.vercel/` swap dance.

**CI deploys via `.github/workflows/deploy-vercel.yml`** on every push to `main`:

1. Deploy `lens-api` → `https://lens-api-chi.vercel.app`
2. Smoke-test `GET /api/health` (6 retries, 5 s apart, to ride out cold-start)
3. Deploy `lens` → `https://lens-fawn.vercel.app`

Backend-first ordering is mandatory: the web project's `vercel.json` pins `/api/*` to `https://lens-api-chi.vercel.app`, so a contract-changing web deploy would talk to a stale API if the order were reversed. A failed health check halts the web deploy, leaving the user-facing alias on the previous good build.

Required GitHub Secrets: `VERCEL_TOKEN`. Org/project IDs are inlined in the workflow (not secret).

To deploy manually from a laptop, link the right project before running `vercel deploy`:

```sh
# API
VERCEL_PROJECT_ID=prj_1C1cnBnJZpy7U8qoNfZAqkeWYGYM \
VERCEL_ORG_ID=team_aWx4s8YoVMOWVPeo3S8FO2RD \
  vercel deploy --prod --local-config=vercel.api.json

# Web
VERCEL_PROJECT_ID=prj_ga9XFH4y8Q5jheH5zxlw4SkB9h4e \
VERCEL_ORG_ID=team_aWx4s8YoVMOWVPeo3S8FO2RD \
  vercel deploy --prod
```

---

## Live state snapshot (2026-04-26)

| Component | Where it lives | Notes |
|---|---|---|
| **Public URL (web)** | `https://lens-fawn.vercel.app` | Production alias on Vercel Hobby (`flyersworders-projects/lens`). Rewrites `/api/*` to the API project. |
| **Public URL (API)** | `https://lens-api-chi.vercel.app` | Separate Vercel project (`flyersworders-projects/lens-api`). See [Deployment topology](#deployment-topology-2026-05-03). |
| **Endpoints** | `/api/health`, `/api/search`, `/api/analyze`, `/api/explain`, `/api/stats`, `/api/track` | Defined in `api/index.py`, served by the `lens-api` project. `stats` aggregates corpus counts (5-min in-process cache); `track` writes a row into `usage_events` on Turso (lazy DDL, no-op when `TURSO_*` is unset). |
| **Frontend** | Next.js 16 App Router at repo root (`app/`, `lib/`) | Three pages — `/` (search + landing), `/analyze`, `/explain/[concept]` — plus a global `StatsBar`. Uses Tailwind; no shadcn registry. Tracks views + submissions via `lib/api.ts#track()` (sendBeacon). |
| **Database (prod)** | Turso `lens-prod` (`libsql://lens-prod-flyersworder.aws-eu-west-1.turso.io`) | Live API source. 77 papers, 82 vocab, 65 matrix cells, 80 tradeoff extractions; embeddings now 1536-dim `text-embedding-3-small` (matches runtime query space — no truncation). |
| **Database (dev)** | Turso `lens-dev` (`libsql://lens-dev-flyersworder.aws-eu-west-1.turso.io`) | Used by `tests/test_turso_store.py` and the Phase 1 publish workflow; safe to wipe. Same corpus snapshot, same 1536-dim embeddings. |
| **Storage adapter** | `TursoStore` in `src/lens/store/turso_store.py` | libSQL native vectors via `vector_top_k` + `vector_distance_cos` |
| **LLM** | OpenRouter `deepseek/deepseek-v4-flash` | Per ADR D2; ~$20 credit balance, **weekly cap $5** (set 2026-04-28 — ~$20/mo effective ceiling) |
| **Embeddings (runtime)** | OpenRouter `openai/text-embedding-3-small` (1536-dim) | Per ADR D1; cloud provider, runs on every search/explain |
| **Build/publish path** | `scripts/build_seed_fixture.py` + `scripts/publish_to_turso.py` + `tests/test_turso_store.py` | Drop-recreate-copy-verify; ~100s for 728 rows + 174 embeddings |
| **CI Phase 1** | `.github/workflows/publish-turso.yml` (manual trigger) | Validates publish chain on every workflow_dispatch; no LLM cost |
| **CI Phase 2** | `.github/workflows/monitor.yml` (Mondays 06:00 UTC + manual) | Pulls `lens.db` from `corpus-snapshot` release → `lens monitor` → publish to lens-prod → re-uploads `lens.db` to release. ~$0.50–1/run at current corpus size. |
| **State preservation** | GitHub release `corpus-snapshot`, asset `lens.db` | Single-asset rolling snapshot. Atomically replaced (`gh release upload --clobber`) at the end of every successful monitor run. |
| **Local-only path** | `LensStore` (sqlite-vec); `--extra local` install | Used by CLI, build pipeline, and 296-test suite |
| **Vercel env vars set** | `TURSO_DATABASE_URL` (→ lens-prod), `TURSO_AUTH_TOKEN`, `OPENROUTER_API_KEY` | Production environment only; preview/dev not configured |
| **GitHub Secrets** | `TURSO_DEV_*` (Phase 1 publish) and `TURSO_PROD_*` (Phase 2 monitor cron, queued) | `OPENROUTER_API_KEY` will be added when Phase 2 lands |
| **Test count** | 296 (256 offline + 17 TursoStore + 19 API/protocol + 4 protocol-specific) | All green |

### Verified working (last checked 2026-04-26)

```
GET /api/search?q=transformer&limit=5
  → 5 real papers (Attention Is All You Need, RoFormer, GQA, Llama 3, Qwen2.5-VL)

POST /api/analyze {"query": "reduce inference latency without adding computational overhead"}
  → improving=Inference Latency, worsening=Model Accuracy
  → 3 principles ranked: Multi-Query Attention (0.95), Quantization (0.95), Anchor-based self-attention (0.95)

POST /api/explain {"query": "grouped query attention"}
  → resolved_id=grouped-query-attention, 5026-char narrative, 1 tradeoff connection
```

### Costs incurred so far
- **Turso**: free tier (5 GB / 500 M reads / 10 M writes)
- **Vercel**: free Hobby tier (within 1 M invocations / 4 hr CPU)
- **GitHub Actions**: unlimited for public repo
- **OpenRouter**: ~$0.05 used during deploy validation; ~$2–5/mo projected; hard cap **$5/week** (configured)

---

## Picking this up from a fresh context

If you (or future-you) are returning to this work after a break, this is the order of next steps in priority order. Each item is independently completable in ~1 day or less.

> **OpenRouter spend cap: $5/week** (configured 2026-04-28; tighter than the per-month default since the prototype has no rate limiting or response cache yet — deferred to step 5 below). At ~$0.0005 per LLM call (DeepSeek V4 Flash, 1.5K-in / 0.5K-out), $5 covers ~10K cache-miss calls per week, well above the projected legitimate traffic and well below what a determined bot loop could burn through. The cap is the standing safeguard until `api/_middleware.py` lands. Re-evaluate once the rate limit + cache ship — at that point you can raise the cap or convert it to a higher monthly ceiling.

### 1. ~~Provision `lens-prod` so test runs don't wipe production data~~ — **done 2026-04-26**

`lens-prod` is live, seeded from the local 77-paper corpus, and Vercel production points at it. `TURSO_PROD_*` is in `.env` (auto-loaded by `python-dotenv`) and GitHub Secrets so `python scripts/publish_to_turso.py --target prod` works locally and in CI. The dev DB (`lens-dev`) is now disposable: `pytest` and the Phase 1 publish workflow can wipe it without affecting the live site.

Reference: ADR D5, `_resolve_target` in `scripts/publish_to_turso.py:114`.

### 2. ~~Re-embed the corpus with `text-embedding-3-small`~~ — **done 2026-04-26**

`EMBEDDING_DIM` is now 1536, matching `text-embedding-3-small` natively. The `~/.lens/config.yaml` `embeddings.*` block already pointed at the cloud provider; only the dim mismatch (stored 768, query 1536-truncated-to-768) was wrong. `scripts/reembed_corpus.py` drops the `_vec` virtual tables, recreates them at the new dim, and re-embeds every paper (`title + abstract`) and vocab row (`name: description`) in ~2 s. Republished to `lens-prod` and `lens-dev` (343.9 s each — 2× the prior time, since vectors are 2× wider). Sanity check: `/api/search?q=transformer` ranks Attention Is All You Need, RoFormer, and FLatten Transformer in the top 5.

The earlier 15 vocabulary orphan embeddings (referenced in step 1) are gone naturally — the rebuild only embeds rows that exist in `vocabulary`, so stale entries from an old taxonomy version don't survive.

### 3. ~~Phase 2 GH Actions monitor cron~~ — **done 2026-04-26**

`.github/workflows/monitor.yml` runs Mondays 06:00 UTC (and on `workflow_dispatch`). State preservation uses a single-asset GitHub release: `corpus-snapshot/lens.db` is downloaded at the start of the run, mutated by `lens monitor`, published to `lens-prod`, and atomically re-uploaded (`gh release upload --clobber`). Concurrency group `monitor` prevents a manual dispatch from racing the weekly cron mid-publish.

The release was bootstrapped once from the local `~/.lens/data/lens.db`. From here the workflow is the single source of truth for `lens-prod`'s contents — manual `publish_to_turso.py --target prod` runs from a developer's machine will be silently overwritten by the next cron unless that change has also been pushed to `corpus-snapshot`.

**First real run not yet triggered** — wait for Monday's cron, or run via `gh workflow run monitor.yml` for an immediate paid backfill (~$0.50–1 OpenRouter spend; runs ~10–20 min).

Why a release asset instead of `pull_from_turso.py`: writing the inverse of the publish script (translating libSQL-native vectors back to sqlite-vec companion tables) doubles the surface area; `turso db dump`/`restore` adds a Turso-region dependency to the runner and is slower cross-region.

### 4. ~~Next.js frontend~~ — **done 2026-04-28**

Next.js 16 (App Router) lives at the repo root (not `web/` — Vercel's monorepo+Python+Next.js pattern wants the framework at root, and ADR D4's intent is "single deploy"). Tailwind, no shadcn (the three pages don't earn a registry init). Pages:

- `/` — hero search → live results from `/api/search`, with prompt chips and the global `StatsBar` in the footer
- `/analyze` — toggle for `tradeoff` / `architecture` / `agentic`; renders `improving`/`worsening` cards + ranked principles, with deep-links into `/explain/<id>`
- `/explain/[concept]` — narrative + evolution + connected concepts + paper refs, with sibling links

Two API additions backing the UI:
- `/api/stats` — aggregates `papers`, `vocabulary` by kind, `matrix_cells`, `tradeoffs`, `taxonomy_version`. Cached for 5 minutes per warm instance via an `lru_cache` keyed on `int(time.time() // 300)`.
- `/api/track` — POST `{event, query?}` writing a row into `usage_events` on Turso (lazy `CREATE TABLE IF NOT EXISTS` on first call). Queries are SHA-256-hashed (16 hex) before persistence — analytics without storing raw user input. No-ops locally when `TURSO_*` is unset.

`vercel.json` now combines installs: `npm install --no-audit --no-fund && uv sync --frozen --extra web --no-dev --no-editable`, plus `buildCommand: "next build"`.

### 4a. Local development

`scripts/dev.sh` boots both halves of the stack — uvicorn on `:8000`, Next.js on `:3000` — under a single shell with a cleanup trap. Ctrl+C tears down both. **Default mode is `preview`** (production build + `next start`), which keeps the runtime footprint to ~100 MB instead of the ~1.5–2 GB Turbopack `next dev` uses. On a memory-constrained laptop, `next dev` cascading into swap-thrash → kernel panic is a documented failure mode in this repo (sessions on 2026-04-28 saw two MBA restarts before we switched the default).

```sh
./scripts/dev.sh                          # preview mode (default, low memory)
./scripts/dev.sh --stop                   # tear down a running session
LENS_DEV_MODE=hot ./scripts/dev.sh        # next dev + Turbopack (HMR, heavy)
LENS_DEV_MODE=webpack ./scripts/dev.sh    # next dev --no-turbopack (HMR, lighter)
./scripts/dev.sh --turso                  # use TURSO_PROD_* instead of local DB
LENS_API_PORT=8001 ./scripts/dev.sh       # alternate backend port
```

`Ctrl+C` from your terminal works in the normal foreground case. If you launched dev.sh as a background job (`./scripts/dev.sh &`), bash inherits a SIG_IGN mask for INT/TERM that the in-script trap can't reliably override on macOS bash 3.2 — use `./scripts/dev.sh --stop` instead, which reads `/tmp/lens-dev.pids` and recursively tears down the full process tree.

**Why preview, not dev, by default**: `next dev` (with Turbopack or webpack) holds the source tree + node_modules + a TS type-checker in memory and re-checks on every keystroke. `next build` is a one-time spike that exits cleanly; `next start` afterwards is a small Node HTTP server with no compiler in the loop. For *inspecting* the UI (the common case after first scaffold), preview is strictly better. Switch to `hot` only when actively iterating on the frontend.

**Why `--extra local-store` instead of `--extra local`**: the latter pulls `sentence-transformers` + `torch` (~400 MB resident, dragged in via litellm's plugin discovery). The runtime uses cloud embeddings (per ADR D1), so torch is dead weight in dev. The `[local-store]` extra carries only `sqlite-vec` for `LensStore`'s vector index. If you actually need offline embeddings (rare in web-tier work), pass `--extra local-embed` or the back-compat `--extra local`.

**Why a script and not a one-liner**: detached background processes (the original launch path) survived shell exit and piled up memory until macOS gave up. `dev.sh` keeps both children attached to the parent shell's TTY so a single Ctrl+C cleanly terminates both via a `trap` on `INT/TERM/EXIT`.

**`next.config.mjs` rewrite contract**: the rewrite condition (`if (process.env.VERCEL) return []`) is evaluated at **build time**, then baked into `.next/routes-manifest.json`. Locally `next build` runs without `VERCEL` set, so the proxy `/api/* → 127.0.0.1:$LENS_API_PORT` is included in the manifest and `next start` honours it. On Vercel the platform sets `VERCEL=1` for both build and runtime, so the rewrite resolves to `[]` and the platform's own `api/*.py` auto-detection wins. If you ever change the condition, **rebuild** — the preview-mode `dev.sh` does this for you on every launch.

### 4b. Inspecting `usage_events`

Phase-2 monitor and the `/api/track` endpoint both write into a single Turso table. There's no read API yet — query it directly via the Turso CLI:

```sh
turso db shell lens-prod \
  "SELECT event, COUNT(*) AS n
     FROM usage_events
    WHERE ts >= strftime('%s','now') - 7*86400
    GROUP BY event
    ORDER BY n DESC;"
```

Useful follow-ups:

- **Top distinct queries (last 7 days):** `SELECT query_hash, COUNT(*) FROM usage_events WHERE query_hash IS NOT NULL AND ts >= strftime('%s','now') - 7*86400 GROUP BY query_hash ORDER BY 2 DESC LIMIT 20;` — `query_hash` is a 16-char SHA-256 prefix, so duplicate queries collide deterministically without exposing user input.
- **Strict-Mode dev inflation:** the React-19 `useEffect` double-fire is suppressed in production by a `useRef` guard (`lib/use-track-once.ts`); `next dev` runs without that, so dev-environment events double-count.
- **Cross-origin spam:** `/api/track` returns `{"status":"ignored"}` (no DB write) for any request whose `Origin`/`Referer` isn't in `LENS_TRACK_ORIGINS` (or `LENS_CORS_ORIGINS` if track-specific isn't set). Configure `LENS_TRACK_ORIGINS=https://lens-fawn.vercel.app` in Vercel env to enforce.

### 5. Rate limiting + response cache (~half day)

**Problem**: deferred from earlier review (item #4 in the FastAPI review punch list). A bot loop could drain OpenRouter credits in minutes.

**Fix**: `api/_middleware.py` using Vercel KV (Upstash Redis):
- Per-IP rate limit (10 req/min)
- Response cache keyed on `endpoint + query.lower().strip()` with 24h TTL (per ADR D3)
- ~~Hard OpenRouter monthly cap set in OpenRouter dashboard~~ — done 2026-04-28: **weekly $5** cap configured (tighter than the planned monthly cap; revisit when the rate limit + cache land)

### 6. Surface degraded mode when embedding fails (~30 min)

**Problem**: Item #7 from the FastAPI review. `serve/explorer.search_papers` swallows embedding failures and silently returns FTS-only results. The API responds with `200 {results: ...}` indistinguishable from healthy hybrid search.

**Fix**: thread a `degraded: bool` flag through the response, computed at the API layer by pre-embedding and catching failures explicitly.

### 7. Misc polish (each ~30 min)

- Custom domain (Vercel `domains` tab; bring your own or buy one)
- README update with public URL + non-commercial note
- Vercel Analytics (1-line opt-in)
- ~~Set OpenRouter spending cap in dashboard~~ — done 2026-04-28 ($5/week)
- Add `lens-prod` row to `docs/web-deployment.md`'s state table once it exists

---

## Vercel deploy gotchas (learned the hard way 2026-04-26)

Each was a separate failed deploy. Captured here so a future deploy from scratch can avoid them.

1. **`vercel.json`'s `functions` block clashes with Vercel's FastAPI auto-detection.** Don't include it — let auto-detection handle the routing. `installCommand` is fine to override.
2. **`pip install`-time bundle for `lens-research` is ~7 GB** because `sentence-transformers` brings PyTorch with CUDA bindings (`cuda-bindings`, `cuda-pathfinder` — multi-GB libraries useless on CPU-only Lambda). Vercel's hard cap is 500 MB unzipped. Mitigation: heavy ML deps live in `[local]` extra; Vercel installs `--extra web --no-dev` only.
3. **`uv sync` defaults to editable install for the project package**, dropping a `.pth` file pointing at `/vercel/path0/src/lens` that doesn't survive Vercel's `/var/task/` runtime packaging. **Always pass `--no-editable`** in the Vercel `installCommand`.
4. **`requirements.txt` lives at repo root, not `api/`.** Vercel's Python builder reads it at root; placing it under `api/` is ignored.
5. **`numpy` must be a base dep**, not transitive via sentence-transformers. `lens.taxonomy.embedder` imports numpy at module level, so without it in base deps the API function 500s on import.
6. **`vercel logs` requires `--no-branch`** to see logs from any deployment regardless of git branch — without it, "No logs found" is confusing.
7. **TursoStore tests' teardown drops `papers`/`vocabulary` on `lens-dev`**, leaving the live API in a half-broken state. Re-publish the seed fixture or real corpus after running tests, OR (recommended) provision a separate `lens-prod` (next-step #1).

The current `vercel.json` captures the working install command:
```json
{
  "installCommand": "uv sync --frozen --extra web --no-dev --no-editable"
}
```

---

## Locked decisions (ADRs)

These are the architectural commitments confirmed 2026-04-26 after spike validation. Changes from here require a documented rationale.

| # | Decision | Choice | Rationale |
|---|---|---|---|
| **D1** | Embedding model (build + runtime) | `openai/text-embedding-3-small` (1536-dim) via OpenRouter | Battle-tested, ecosystem-friendly, ~$0 at LENS scale, compatible with future pgvector if we migrate to Neon. |
| **D2** | Request-time LLM | `deepseek/deepseek-v4-flash` ($0.14/M in, $0.28/M out) | New April 2026 release, 5.4× cheaper output than Gemini 3.1 Flash Lite Preview, 1M context, MoE 284B/13B-active, ~$2.10/mo at projected volume — well under $5/mo cap. |
| **D3** | Response cache | 24-hour TTL, key = `endpoint + query.lower().strip()` | Coarse but predictable; bounded staleness; lets caching actually pay. |
| **D4** | Repository structure | ~~Monorepo, single deploy: Next.js at root + `api/*.py` auto-detected by Vercel~~ — **superseded 2026-05-03**: split into two Vercel projects (`lens` web + `lens-api` Python). The Python bundle exceeded the 500 MB cap when bundled with Next.js artifacts; combined deploys could not be made to fit. See [Deployment topology](#deployment-topology-2026-05-03). | Atomic versioning, one deploy, what Vercel docs assume. |
| **D5** | Local-vs-Turso `Store` mode | Dual-backend: existing `Store` (sqlite-vec) for CLI/build/tests; new `TursoStore` (libSQL native) for Vercel runtime | Spike confirmed `file:` URLs lack libSQL vector functions; dual-backend isolates Turso-specific code and keeps tests offline. |

---

## Goals and constraints

1. **Public read-only website** — anyone can run `analyze`, `explain`, and `search` queries against the LENS knowledge base via HTTP.
2. **Near-zero-dollar prototype** — fits inside the free tiers of all platforms; the only paid line item is the OpenRouter credit balance for cheap-but-good LLM models at request time and during the weekly build (currently ~$20 of credit on the project account, expected burn rate ~$2–5/month at projected traffic).
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

#### Conventions verified by `scripts/publish_to_turso.py`

These were learned during the publish-script implementation and are baked into the script:

7. **Schema-discovery, not hardcoded DDL.** Local LENS DBs accumulate columns from historical migrations (e.g. `new_concept_description` from a deprecated migration) that aren't in the current `_COLUMN_MIGRATIONS` list in `store.py`. The publish script reads `PRAGMA table_info` per table and rebuilds DDL from observed columns, so it works against any version of the local schema without script edits.
8. **`AUTOINCREMENT` can't be detected via `PRAGMA table_info`** — only via `sqlite_schema.sql` parsing. The publish script drops the keyword for `tradeoff_extractions.rowid` etc.; this is fine because the read-only Turso path doesn't depend on the "never reuse a deleted rowid" guarantee.
9. **`DROP TABLE` on a libSQL FTS5 virtual table cascades to its shadow tables** (`*_data`, `*_idx`, `*_config`, `*_docsize`, `*_content`) — verified live against Turso 2026-04. The publish script's drop list doesn't need to enumerate them.
10. **Embedding-dimension mismatch is silent until query time** if not caught early. The publish script probes the first row of `<table>_vec.embedding` and verifies `len(blob) == 4 * embedding_dim` before attaching; mismatch fails loudly with the actual dim. This is the first line of defense against an embedding-model swap that nobody updated in config.
11. **No fallback for `--target prod` env vars.** `_resolve_target` requires explicit `TURSO_PROD_DATABASE_URL` and `TURSO_PROD_AUTH_TOKEN`; falling back to unprefixed `TURSO_DATABASE_URL` would let a dev-shell publish dev creds to "prod". `--target plain` exists for the unprefixed-vars case (e.g. inside GitHub Actions where there's only one DB per run).
12. **Verify FTS row counts, not just main-table counts.** A silent FTS5 rebuild failure leaves the index empty while the main table is fine; `verify_counts` cross-checks both so the build pipeline fails fast.

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
3. **OpenRouter spending cap** — set a hard limit in the OpenRouter dashboard. We currently use **$5/week** (configured 2026-04-28); a monthly cap is the more common shape but the weekly window bounds blast radius better while we have no rate limit. If exceeded, the API returns `402` and the Function falls back to a cached or static response.

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
| OpenRouter spend (cheap models) | hard cap **$5/week** (configured 2026-04-28; ≈ $20/mo ceiling) | ~$2/mo at 10K visitors | 10× |

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
| Wire GitHub Actions Phase 1 (`publish-turso.yml`, manual trigger, synthetic seed fixture) | 0.5 day | Validates publish path under CI without LLM costs; uses `scripts/build_seed_fixture.py` + `scripts/publish_to_turso.py` + `scripts/smoke_test_turso.py` |
| Wire GitHub Actions Phase 2 (full monitor cron with state preservation between runs) | 1 day | Adds `lens monitor`, requires OpenRouter credits + a state-pull step (release asset or similar) at start of each run |
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
- [x] ~~Create Turso account, create `lens-prod` and `lens-dev` databases, capture URL + auth tokens~~ — done 2026-04-26; credentials in `.env` (auto-loaded by `python-dotenv`)
- [x] ~~Validate libSQL native vectors + FTS5 against remote Turso (Spike 2)~~ — passed 2026-04-26
- [x] ~~Top up OpenRouter with $10 credit (one-time) to unlock 1000 req/day on free models *and* enable cheap paid models~~ — done; ~$20 balance available, which unblocks cloud embeddings, request-time LLM, and Phase 2 monitor cron
- [x] ~~Add `libsql-client` to `pyproject.toml` as a `[project.optional-dependencies] turso` extra~~ — done; also added `[web]` extra (fastapi + transitive turso) and `[local]` extra (sentence-transformers + sqlite-vec, kept off Vercel to fit the 500 MB bundle cap)
- [x] ~~Build new `TursoStore` class with read API matching `Store` (vector queries via `vector_top_k`, FTS5 unchanged)~~ — done (commit bf4664c)
- [x] ~~Write `scripts/publish_to_turso.py` to translate sqlite-vec schema → libSQL native and copy data~~ — done; end-to-end verified against real LENS DB (728 rows, ~100 s)
- [x] ~~Wire `TursoStore` into `serve/analyzer.py`, `serve/explainer.py`, `serve/explorer.py` via a small adapter~~ — done via the `ReadableStore` protocol (`src/lens/store/protocols.py`); 4 conformance tests in `tests/test_store_protocols.py` exercise both backends end-to-end
- [x] ~~Add `TursoStore` integration tests gated on `TURSO_DEV_*` env vars (skip when offline)~~ — done (commit bf4664c, 17 tests)
- [x] ~~Switch local config to `embeddings.provider: cloud` with OpenRouter base URL; verify `embedder.py` cloud path works end-to-end~~ — done 2026-04-26 (config has been on `cloud` for a while; the missing piece was matching dims)
- [x] ~~Re-embed the existing corpus once with the chosen embedding model (locks `EMBEDDING_DIM`)~~ — done 2026-04-26 via `scripts/reembed_corpus.py`; `EMBEDDING_DIM = 1536` in `models.py`, all 77 papers + 82 vocab re-embedded with `text-embedding-3-small`, both Turso DBs republished, prod redeployed (`dpl_…lens-9nnkguj08…`)

**Build pipeline**
- [x] ~~Write `scripts/publish_to_turso.py` (try/finally around the modify step)~~ — done; `pull_from_turso.py` deferred to Phase 2 monitor cron (will likely use a release-asset roundtrip instead of a true pull)
- [x] ~~Phase 1: `.github/workflows/publish-turso.yml` (manual trigger, synthetic seed fixture, no LLM cost)~~ — done; validates the publish chain under CI before the monitor cron lands
- [x] ~~Phase 2: `.github/workflows/monitor.yml` with weekly cron + `workflow_dispatch` — runs `lens monitor` end-to-end~~ — workflow added 2026-04-26; `corpus-snapshot` release bootstrapped from local `lens.db`; `OPENROUTER_API_KEY` added to GitHub Secrets. First real run will fire on the next Monday cron, or trigger manually via `gh workflow run monitor.yml`.
- [x] ~~Configure GitHub Secrets: `TURSO_DEV_DATABASE_URL`, `TURSO_DEV_AUTH_TOKEN`~~ — done (used by `publish-turso.yml` Phase 1). `TURSO_PROD_DATABASE_URL` + `TURSO_PROD_AUTH_TOKEN` added 2026-04-26 in preparation for Phase 2; `OPENROUTER_API_KEY` will land with the monitor workflow.
- [x] ~~Run the workflow once via `workflow_dispatch` to seed the prod DB~~ — Phase 1 workflow run validated end-to-end against `lens-dev` (workflow run id 24960286069); `lens-prod` seeded from local on 2026-04-26 (77 papers + 82 vocab + 65 matrix cells, 42.6 s)

**Web tier**
- [x] ~~Create `api/` directory at repo root with `analyze.py`, `explain.py`, `search.py` FastAPI handlers~~ — done as a single `api/index.py` (the recommended Vercel multi-route shape; one app, three routes); 15 unit tests in `tests/test_api_index.py` cover validation, dispatch, error paths, and dependency-injection wiring
- [ ] Wire per-IP rate limit + response cache via Vercel KV *(see step 5 in "Picking up from a fresh context")*
- [x] ~~Set OpenRouter monthly spending cap in OpenRouter dashboard~~ — done 2026-04-28; configured a **weekly** $5 cap instead of monthly (tighter; the prototype's lack of rate-limiting argues for shorter accumulation windows). Re-evaluate once `api/_middleware.py` lands.
- [x] ~~Scaffold Next.js project in `web/` (or separate repo) with 3 pages~~ — done 2026-04-28; built at the repo root (not `web/`) per ADR D4. Three pages + StatsBar + tracking, all calling `/api/*` same-origin. `next build` clean (4 static + 1 dynamic route)
- [x] ~~Deploy to Vercel; set env vars: `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN`, `OPENROUTER_API_KEY`~~ — done; live at `https://lens-fawn.vercel.app`. `KV_REST_API_*` deferred until rate-limit/cache step. The deploy required 7 iterations; full diagnostic log in commits `2195a9c..6d4dd49`.
- [x] ~~Smoke-test all three endpoints against prod Turso (warm and cold start)~~ — verified 2026-04-26; all 4 routes (`health`, `search`, `analyze`, `explain`) return 200 with real corpus data
- [ ] Verify `/tmp` libSQL replica behaviour under sustained traffic *(deferred — current deploy uses libSQL HTTP, not embedded replicas; revisit if cold-start latency becomes a real UX issue)*

**Polish**
- [ ] (Optional) Configure custom domain *(see step 7)*
- [ ] Add Vercel Analytics *(see step 7)*
- [ ] Update README with the public URL and a note that the site is non-commercial under Vercel Hobby ToS *(see step 7)*

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

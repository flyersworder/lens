# LENS Functionality Gaps — Design Spec

**Date:** 2026-04-12
**Scope:** Fix 10 identified gaps to make LENS fully functional

---

## 1. Add `acquire semantic` CLI Command

**Problem:** `semantic_scholar.py` module exists but has no CLI command.

**Change:** Add `acquire_app.command("semantic")` in `cli.py`.
- `--paper-id TEXT` — fetch SPECTER2 embedding for a single paper
- Without `--paper-id` — process all papers whose embedding is all zeros (detected by checking if `SUM(embedding) = 0` in the vec table, or papers missing from the vec table entirely)
- Calls `fetch_embedding()` / `fetch_embeddings_batch()` from `acquire/semantic_scholar.py`
- Updates `papers_vec` table via `store.upsert_embedding("papers", paper_id, embedding)`
- Logs events per paper updated

**Files:** `cli.py`

---

## 2. Remove `--interval` No-Op from Monitor

**Problem:** `--interval` flag on `lens monitor` says "not yet used" — misleading.

**Change:** Remove the `interval` parameter from the `monitor()` command.

**Files:** `cli.py`

---

## 3. Fix `year` Display Bug in `explore paper`

**Problem:** `explore paper` reads `result['year']` but the Paper model stores `date`, not `year`. The year line silently shows nothing.

**Change:** Replace `result.get('year')` references with `result.get('date', '')`.

**Files:** `cli.py`

---

## 4. Enhanced Monitor Pipeline

**Problem:** Monitor only does arxiv → extract → ideate. Missing: OpenAlex enrichment, taxonomy/matrix rebuild, LLM ideation.

**Change to `run_monitor_cycle()`:**

New parameters:
- `run_enrich: bool = True` — run OpenAlex enrichment on new papers
- `run_build: bool = True` — rebuild taxonomy + matrix after extraction
- `ideate_with_llm: bool = False` — use LLM to enrich ideation gaps
- `openalex_mailto: str = ""` — needed for enrichment API

New stages (in order):
1. Acquire — arxiv fetch (existing)
2. Enrich — OpenAlex enrichment on new papers (if `run_enrich` and `openalex_mailto` non-empty)
3. Extract — LLM extraction (existing)
4. Build — `build_vocabulary()` + `build_matrix()` (if `run_build`)
5. Ideate — `run_ideation()` or `run_ideation_with_llm()` based on flag

Return dict gains: `papers_enriched`, `taxonomy_built`, `matrix_built` keys.

**Change to CLI `monitor` command:**
- Add `--skip-enrich` and `--skip-build` flags (default False)
- Wire `monitor.ideate_llm` config to `ideate_with_llm` parameter
- Pass `openalex_mailto` from config

**Files:** `monitor/watcher.py`, `cli.py`

---

## 5. Wire Quality Score Computation

**Problem:** `quality_score()` exists in `acquire/quality.py` but is never called.

**Change:**
- After OpenAlex enrichment in `acquire openalex` CLI: compute and store `quality_score` for enriched papers
- After enrichment stage in monitor pipeline: same
- In `acquire seed` CLI: compute for seed papers (they have citations/venue)
- Call `store.update("papers", "quality_score = ?", "paper_id = ?", (score, pid))` for each

**Files:** `cli.py`, `monitor/watcher.py`

---

## 6. API Key Validation

**Problem:** Missing API key produces cryptic errors deep in the LLM client.

**Change:** Add `_require_llm_config(config)` helper in `cli.py`:
- Checks `config["llm"]["api_key"]` or `OPENROUTER_API_KEY` env var
- If missing, prints: "LLM API key not configured. Set via: lens config set llm.api_key YOUR_KEY or export OPENROUTER_API_KEY"
- Raises `typer.Exit(code=1)`
- Call before LLM usage in: `extract`, `analyze`, `explain`, `monitor` (when not `--trending`)

**Files:** `cli.py`

---

## 7. Logging Control

**Problem:** No way to control log verbosity from the CLI.

**Change:** Add a Typer callback on the main `app` with `--verbose / -v` option:
- Default (no flag): WARNING
- `-v`: INFO
- `-vv`: DEBUG
- Calls `logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")`

**Files:** `cli.py`

---

## 8. `lens status` Command

**Problem:** No quick overview of knowledge base health.

**Change:** New top-level `app.command("status")` that prints:

```
LENS Knowledge Base Status
==========================
Papers: 142 (complete: 130, pending: 8, failed: 4)
Vocabulary: 45 parameters, 32 principles, 12 arch slots, 8 agentic categories
Matrix: 89 cells, 247 total evidence
Top parameters: Inference Latency (18), Model Accuracy (15), ...
Taxonomy: v3 (last built: 2026-04-10)
Last event: 2026-04-11 14:30 (extract)
Issues: 3 orphans, 5 weak evidence, 2 missing embeddings
```

Runs cheap lint checks inline (orphans, weak_evidence, missing_embeddings — no near-duplicate computation). Queries DB directly for counts.

**Files:** `cli.py`

---

## Testing Strategy

All changes get unit tests:
- `acquire semantic`: mock `fetch_embeddings_batch`, test CLI invocation
- Monitor pipeline: mock all async stages, verify stage sequencing and skip flags
- Quality score: test computation is called after enrichment
- API key validation: test missing key exits with message
- Logging: test that `-v` sets INFO level
- Status: test output with populated and empty DB
- Year bug: covered by existing `explore paper` test update

---

## Non-Goals

- No daemon/scheduler for monitor (use system cron)
- No OpenAlex enrichment as a standalone stage in monitor for papers that were acquired previously (only new papers)
- No near-duplicate check in `status` (too expensive for a quick command)

# Scoop-Check (Idea-Card Novelty Verification) Design

## Overview

The `0.11.0` ideate stage generates structured, falsifiable Idea Cards, but nothing verifies whether a card's idea is actually novel. A validation batch (2026-07-15) confirmed the risk: several generated cards restate published work (KVQuant, CALM/early-exit, process-reward step verification) that does **not** appear in LENS's local corpus, so they read as novel when they are not. Surfacing unverified cards would cost user trust.

Scoop-check adds a **separate, idempotent pass** over persisted idea cards that queries a real prior-art source (Semantic Scholar), asks an LLM to judge novelty against the retrieved papers, and annotates each card with a verdict. It is deliberately decoupled from generation: a flaky external call can never break card creation, and the pass can be re-run to resume where it stopped.

This realizes the "scoop-check" Future Hook from the pattern-guided idea cards design; the `signature_terms` field pre-wired there is the query input here.

## Goals

1. **`search_semantic_scholar(query, limit, api_key)`** — a new relevance-search function against the Semantic Scholar `/paper/search` endpoint (the existing module only fetches embeddings by id).
2. **`judge_novelty(card, prior_art, llm_client)`** — an LLM judge returning `novel | overlaps | scooped`, the colliding paper(s), and a rationale.
3. **`run_scoop_check(store, llm_client, limit=None)`** — an idempotent pass over cards with `novelty_status = 'unchecked'`.
4. **Novelty columns on `idea_cards`** plus a **`lens scoop-check` CLI command**.

## Non-Goals

- No auto-run inside the scheduled monitor cron — the external API would add flakiness to an unattended job. Scoop-check is run deliberately.
- No deletion or hiding of scooped cards — v1 **annotates only** (`novelty_status`); filtering is a UI concern for later.
- No second source (arXiv/deepxiv/OpenAlex) — Semantic Scholar only for v1.
- No change to card generation (`run_ideation_with_llm`) or the matrix.
- No web endpoint / UI — surfacing verdicts is a separate future increment.

## Component: `search_semantic_scholar` (`src/lens/acquire/semantic_scholar.py`)

New async function alongside the existing embedding fetch, reusing `fetch_with_retry` (which already backs off on HTTP errors incl. 429):

- Endpoint: `https://api.semanticscholar.org/graph/v1/paper/search?query=<q>&limit=<n>&fields=title,abstract,year,citationCount,externalIds,url`
- **Free (unauthenticated) tier** — no API key. The tier is rate-limited (~1 req/s, shared); `fetch_with_retry` already backs off with jitter on `429`, which is sufficient for this low-volume batch pass. Signature keeps an unused `api_key: str | None = None` for parity with the sibling `fetch_embedding`, but no caller sets it and no env var is read.
- Returns a list of dicts: `{title, abstract, year, citations, arxiv_id, url}` (arxiv_id parsed from `externalIds.ArXiv` when present; `abstract`/`year` may be `None`).
- **Never raises** — on timeout, rate-limit exhaustion, or malformed response, logs a warning and returns `[]`. Papers with no abstract are dropped (nothing to judge against).

## Component: `judge_novelty` (`src/lens/knowledge/scoop_check.py`)

`judge_novelty(card: dict, prior_art: list[dict], llm_client) -> dict | None`

- Prompt: the card's `title`, `mechanism`, and `differentiation`, plus the top-k retrieved `{title, abstract}` (k ≈ 5). Instruction: decide whether the card's *core contribution* is already covered — distinguishing shared keywords from the same idea — and respond with a single JSON object.
- Expected JSON: `{"verdict": "novel|overlaps|scooped", "colliding_papers": ["<title>", ...], "rationale": "<one line>"}`.
  - `scooped` = the core idea is already published; `overlaps` = substantial related work but a distinct angle; `novel` = no close prior art in the retrieved set.
- Parse with `json.loads` → `json_repair.repair_json` fallback (mirrors `_parse_idea_card` and the extractor). Returns `None` on unusable output or a verdict outside the enum.

## Component: `run_scoop_check` (`src/lens/knowledge/scoop_check.py`)

`run_scoop_check(store, llm_client, limit=None, top_k=5) -> dict`

Per card where `novelty_status = 'unchecked'` (optionally capped at `limit`, lowest `id` first):

1. Build the query from `title` + `signature_terms` (space-joined).
2. `prior_art = await search_semantic_scholar(query, limit=top_k)`.
   - If `prior_art` is empty (no results or API failure): leave the card `unchecked` and continue — it will be retried on the next run. (An empty result is ambiguous between "genuinely nothing" and "API hiccup"; not marking it avoids a false `novel`.)
3. `verdict = await judge_novelty(card, prior_art, llm_client)`.
   - If `None` (unusable judge output): leave `unchecked`, continue.
4. Persist on the card: `novelty_status = verdict["verdict"]`, `prior_art` = the retrieved `{title, url, year}` list, `novelty_note` = rationale, `novelty_checked_at` = now. Guard the write (log + skip on DB error).

Returns `{"checked": n, "by_verdict": {novel, overlaps, scooped}}`.

## Schema (`src/lens/store/store.py`)

`idea_cards` already exists, so new columns go through both paths:

- Append to the `idea_cards` `CREATE TABLE` (fresh DBs):
  ```sql
  novelty_status     TEXT NOT NULL DEFAULT 'unchecked',
  prior_art          TEXT NOT NULL DEFAULT '[]',
  novelty_note       TEXT NOT NULL DEFAULT '',
  novelty_checked_at TEXT
  ```
- Add matching `_COLUMN_MIGRATIONS` entries (existing DBs get the columns via `ALTER TABLE`).
- Register the JSON column: `JSON_FIELDS["idea_cards"]` gains `"prior_art"`.

## CLI (`src/lens/cli.py`)

New `lens scoop-check` command:
- `_require_llm_config(config)` (needs the LLM judge).
- Builds `LLMClient` like `analyze`/`explain`, opens the store.
- `--limit N` to cap cards per run (default: all unchecked); `--top-k K` for retrieved papers.
- Semantic Scholar is queried on the free (unauthenticated) tier — no key is read or required.
- Prints a summary table (counts by verdict) and logs each `scooped`/`overlaps` card with its colliding paper.
- Logs a `scoop_check` event via `log_event` for the event log.

## Graceful Degradation

Every external touch fails soft: S2 errors → `[]` → card stays `unchecked`; judge errors → `None` → card stays `unchecked`; DB write errors → logged, card skipped. A run always completes and is safe to re-run; only cards that got a real verdict change state.

## Testing (`tests/test_scoop_check.py`; tmp_path real store, only httpx + llm_client stubbed)

1. `search_semantic_scholar` — monkeypatch the httpx GET to return a canned `{"data": [...]}`; assert mapping (arxiv_id from `externalIds`, abstract-less papers dropped). Plus an `@pytest.mark.integration` live test.
2. `judge_novelty` — stub `llm_client.complete` returns a canned verdict JSON → assert parsed dict; malformed / out-of-enum → `None`.
3. `run_scoop_check` happy path — seed two `idea_cards`, monkeypatch `search_semantic_scholar` + stub judge → both get a `novelty_status`, `prior_art`/`novelty_note` populated, summary counts correct.
4. **Idempotency** — a second `run_scoop_check` re-checks nothing (all now non-`unchecked`).
5. **Failure isolation** — search returns `[]` (or judge returns `None`) → card remains `unchecked`, run still returns a summary, no exception.

## Future Hooks (not built here)

- Surface `novelty_status` in a UI / API and let users filter out `scooped` cards (the payoff that makes cards trustworthy to show).
- Feed a `scooped` verdict back into generation (regenerate with the collision as an explicit constraint) — an audit/revise loop like ResearchStudio's Phase 3.
- Confidence calibration and a domain guard on generation (separate findings from the same validation batch).

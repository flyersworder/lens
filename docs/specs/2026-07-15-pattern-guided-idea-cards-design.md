# Pattern-Guided Idea Cards & Novelty Verification Design

> Covers two shipped increments: pattern-guided Idea Cards (0.11.0) and
> scoop-check novelty verification (0.12.0). Scoop-check was the "Future Hook"
> of the original idea-cards design; it is folded in here as the design of
> record. The former standalone `2026-07-15-scoop-check-design.md` is superseded.

## Overview

LENS discovers research gaps *structurally* — `find_sparse_cells` (parameter pairs with too few resolving principles) and `find_cross_pollination` (a principle resolving `(A, B)` that might transfer to an embedding-similar `(A', B)`). This TRIZ-style gap discovery is LENS's genuinely novel capability and is **not** changed by this work.

The weak link is the LLM step that turns a gap into an idea. Today `run_ideation_with_llm` (`src/lens/monitor/ideation.py:293`) issues one generic prompt per gap — *"propose a hypothesis for why this gap exists and how it might be resolved"* — with no paper evidence, no structure, no novelty check, and no falsifiable prediction. The result is an unverifiable free-text `llm_hypothesis` string.

This design grafts the rigor of Microsoft ResearchStudio's `idea-spark` onto LENS's structurally-discovered gaps: each gap is enriched with a **structured, pattern-grounded, falsifiable Idea Card** instead of a free-text hypothesis. It replaces only the weak enrichment step; the matrix math, acquisition, and extraction are untouched.

## Goals

1. **New `ideation_pattern` vocabulary kind** seeded with the 15 ideation patterns ResearchStudio induced from ~1,900 ICLR/ICML/NeurIPS papers.
2. **New `IdeaCard` model + `idea_cards` table** — a structured, queryable card linked to a gap.
3. **Upgrade `run_ideation_with_llm`** to select applicable pattern(s), ground the gap in the vocabulary it references, and generate one validated Idea Card per gap.
4. **Full back-compat** — the non-LLM `run_ideation` path and the `llm_hypothesis` column keep working unchanged.

## Non-Goals

- No change to `find_sparse_cells` / `find_cross_pollination` / the matrix.
- No new CLI command — this rides the existing monitor `ideate` stage (batch over all gaps).
- No fulltext fetching (RS Phase 0). Grounding is limited to vocabulary names/descriptions the gap already references.
- No novelty / scoop-check pass yet. The card carries `signature_terms` to pre-wire it, but the check itself is out of scope.
- No `advance/revise/abandon` audit loop.
- No embedding-based pattern retrieval — v1 injects the full 15-pattern catalog into the prompt.

## Pattern Catalog — new `ideation_pattern` vocabulary kind

Append 15 entries to `SEED_VOCABULARY` in `src/lens/taxonomy/vocabulary.py`, each `{"name", "kind": "ideation_pattern", "description"}`. Ids are derived by the existing `_slugify(name)` at load time. The `description` packs each pattern's operational signature and "when to apply" so it embeds meaningfully and reads well in the prompt.

| Name | Description (operational signature / when to apply) |
|---|---|
| Audit and Pivot an Assumption | Identify a load-bearing assumption in prior work, show it fails in some regime, and redesign around its negation. |
| Substitute the Operator or Representation | Replace a core operator (attention, convolution, tokenizer) or data representation with a better-suited alternative while keeping the surrounding system fixed. |
| Liberate a Fixed Generative Component | Take a component hard-coded or held fixed in prior work and make it learned, generative, or adaptive. |
| Design a Confound-Isolating Diagnostic | Construct a controlled experiment or benchmark that isolates one factor to explain an otherwise-confounded phenomenon. |
| Unify Heterogeneous Inputs into One Space | Map disparate modalities, tasks, or inputs into a single shared representation so one method handles all of them. |
| Reframe as a Solvable Object | Recast an ill-posed or intractable problem as an instance of a well-studied, solvable formulation. |
| Manufacture the Supervisory Signal | Invent a self-supervised or synthetic training signal where labels are unavailable or costly. |
| Encode Structure by Construction | Bake a known invariance or prior (symmetry, sparsity, hierarchy) directly into the architecture instead of learning it. |
| Prove Equivalence to Unify | Show two apparently distinct methods or objectives are mathematically equivalent, then exploit the unification. |
| Decompose for Differentiated Treatment | Split a heterogeneous problem into sub-parts and apply a specialized method to each. |
| Decompose and Delegate to Solvers | Break a task into sub-tasks routed to specialized existing solvers, tools, or agents. |
| Relax Discrete Search to Continuous | Replace a discrete or combinatorial search with a differentiable continuous relaxation to enable gradient optimization. |
| Adapt by Conditioning, Not Retraining | Achieve new behavior by conditioning inputs, prompts, or adapters instead of retraining weights. |
| Characterize a Limit, Then Surpass It | Formally characterize a fundamental limit or bound of current methods, then design a method that provably exceeds it. |
| Design a Property-Targeting Pretext Objective | Craft a pretraining objective specifically engineered to instill a targeted downstream property. |

**Safety — verified.** `find_sparse_cells` and `find_cross_pollination` query `WHERE kind = 'parameter'` (`ideation.py:25,65,79`), so the new kind never leaks into gap-finding. `build_vocabulary` embeds all vocab kinds uniformly, so patterns get embeddings for free (enabling future retrieval) with no special-casing.

## Data Model

### `IdeaCard` (Pydantic, `src/lens/store/models.py`)

Added alongside `IdeationGap`. `list[str]` fields are auto-serialized/deserialized by the store (existing convention).

```python
class IdeaCard(BaseModel):
    """A structured, pattern-grounded idea generated for an ideation gap."""

    id: int
    gap_id: int
    report_id: int
    title: str                      # <= ~15-word paper-level title
    pattern_ids: list[str]          # chosen ideation_pattern vocab ids
    hook: str                       # 1-2 sentence claim
    mechanism: str                  # how the pattern applies to the gap -> concrete artifact
    falsification: str              # minimal experiment + metric + distinguisher
    differentiation: list[str]      # how it differs from existing principles
    signature_terms: list[str]      # key terms; pre-wires a future scoop-check
    paper_ids: list[str]            # evidence used
    confidence: float               # LLM self-report, 0-1
    created_at: datetime
    taxonomy_version: int
```

### `idea_cards` table (`src/lens/store/store.py`)

A brand-new table → plain `CREATE TABLE IF NOT EXISTS` added to the schema block in `init_tables()`. The `_COLUMN_MIGRATIONS` path is only for `ALTER TABLE` on *existing* tables, so no migration entry is needed; existing databases pick up the new table on next `init_tables()`.

```sql
CREATE TABLE IF NOT EXISTS idea_cards (
    id              INTEGER PRIMARY KEY,
    gap_id          INTEGER NOT NULL,
    report_id       INTEGER NOT NULL,
    title           TEXT NOT NULL,
    pattern_ids     TEXT NOT NULL DEFAULT '[]',   -- JSON list
    hook            TEXT NOT NULL DEFAULT '',
    mechanism       TEXT NOT NULL DEFAULT '',
    falsification   TEXT NOT NULL DEFAULT '',
    differentiation TEXT NOT NULL DEFAULT '[]',   -- JSON list
    signature_terms TEXT NOT NULL DEFAULT '[]',   -- JSON list
    paper_ids       TEXT NOT NULL DEFAULT '[]',   -- JSON list
    confidence      REAL NOT NULL DEFAULT 0.0,
    created_at      TEXT NOT NULL,
    taxonomy_version INTEGER NOT NULL DEFAULT 0
);
```

`signature_terms` is populated now even though nothing consumes it. It is the exact input a future scoop-check needs, so idea #2 becomes additive rather than a schema change. The schema permits more than one card per gap; v1 emits one.

## Generation Logic

`run_ideation_with_llm` (`src/lens/monitor/ideation.py`) is rewritten to emit cards. `run_ideation` (structural gaps) and the synchronous non-LLM path are unchanged. Sequence per gap:

1. **Grounding context (light).** From the gap's `related_params` + `related_principles`, look up each vocab entry's `name` + `description` (already in memory from the `store.query("vocabulary")` the function does). No fulltext, no extra queries beyond what already runs.
2. **Pattern catalog.** Inject all 15 `ideation_pattern` entries (name + description) into the prompt.
3. **One LLM call.** System prompt: research analyst that applies a named ideation pattern to a matrix gap. User prompt: the gap description, its grounding context, and the pattern catalog. Ask for **a single JSON object** matching the `IdeaCard` field set (minus DB-assigned `id`/`gap_id`/`report_id`/timestamps), instructing the model to (a) pick 1-2 pattern names from the catalog, (b) return a falsification with a concrete minimal experiment and a distinguishing metric.
4. **Parse + validate.** Parse the response with `json_repair` (already a core dependency, same as the extractor), map chosen pattern names back to vocab ids via `_slugify`, drop any that don't resolve to a real `ideation_pattern`, assemble an `IdeaCard`, and validate with Pydantic.
5. **Persist.** Assign the next `id` (same `MAX(id)+1` idiom the function already uses for gaps), insert into `idea_cards`, and set the gap's `llm_hypothesis` to the card's `mechanism` for back-compat.
6. **Graceful degradation.** On malformed JSON, validation failure, or LLM error, log a warning and skip that gap's card (mirrors the current `except` branch) — the run still succeeds and the gap still exists.

The prompt asks for JSON and parses defensively; it does not depend on provider-specific `response_format`, so it works across the openai and litellm backends `complete()` supports. `**kwargs` passthrough in `complete()` means a JSON mode can be opted into later without an interface change.

## Back-Compat

- `run_ideation` and the non-LLM monitor path: unchanged.
- `IdeationGap.llm_hypothesis`: retained, now populated with the card's `mechanism`. Nothing that reads it breaks.
- `watcher.py` call site (`run_ideation_with_llm(store, llm_client)`): signature unchanged.
- New kind is additive; `lens status` (kind breakdown) shows `ideation_pattern` automatically. Surfacing cards in `status`/provenance is a later, optional add — out of scope here.

## Testing

`tmp_path` SQLite fixtures with a **real** store (per repo convention — no SQLite mocking). Only the `llm_client` boundary is stubbed (a small object whose `complete()` returns canned JSON).

1. **Seed** — `load_seed_vocabulary` inserts 15 `ideation_pattern` entries; `kind` counts are correct.
2. **Isolation** — after seeding patterns, `find_sparse_cells` still returns only parameter-pair gaps (new kind absent).
3. **Model + store round-trip** — an `IdeaCard` validates; inserting and reading back through the store preserves the JSON list fields.
4. **Happy path** — `run_ideation_with_llm` with a stub client returning a well-formed card JSON persists one `idea_cards` row per gap, with `pattern_ids` resolving to real `ideation_pattern` vocab ids and `llm_hypothesis` set to the mechanism.
5. **Malformed output** — stub returns non-JSON / a bad pattern name → the run completes, no card row is written for that gap, no exception propagates.

## Scoop-Check — Novelty Verification (built in 0.12.0)

Generated cards are structurally sound but unverified for novelty; a validation batch (2026-07-15) found several restating published work (KVQuant, CALM/early-exit) that is **not** in LENS's local corpus. Scoop-check is a **separate, idempotent pass** that flags prior art so cards aren't presented as novel when they aren't.

**Source decision.** The original Future Hook imagined a corpus hybrid-search, but the prior art that matters usually isn't in the local corpus, so scoop-check queries an *external* source. A live e2e found the free unauthenticated Semantic Scholar `/paper/search` tier returns `429` on every request; the shipped source is **OpenAlex** (free polite pool via `mailto`, reliable, relevant).

**Flow** — for each card with `novelty_status = 'unchecked'`:
1. `_gather_prior_art` runs one `search_openalex` (`acquire/openalex.py`) **per signature term** and unions the results (a single combined query dilutes relevance into off-domain surveys). Each search is restricted to **recent Computer-science works** (`from_publication_date` + CS concept filter) with light pacing; abstracts reconstructed from `abstract_inverted_index`; fail-soft (`[]` on error); keeps title-only works so a colliding paper without an abstract still reaches the judge.
2. `judge_novelty(card, prior_art, llm)` (`knowledge/scoop_check.py`) → `{verdict ∈ novel|overlaps|scooped, colliding_papers, rationale}`; parsed with `strip_code_fences` → `json.loads` → `json_repair`; unusable/out-of-enum output → `None`.
3. Persist `novelty_status`, `prior_art` (JSON), `novelty_note` (rationale + colliding papers), `novelty_checked_at`.

**Schema.** Four columns added to `idea_cards` (`novelty_status`, `prior_art`, `novelty_note`, `novelty_checked_at`) via `_TABLE_DDL` + `_COLUMN_MIGRATIONS`; `prior_art` registered in `JSON_FIELDS`.

**Guarantees.** Fully fail-soft and idempotent — a card only leaves `unchecked` on a real verdict (search-empty, judge-crash, or DB error all leave it for the next run). `--limit` caps *checked* cards (not a pre-slice), so persistently-failing cards can't starve unreached ones. **Not** wired into the monitor cron (keeps the external API out of the unattended job).

**Surface.** `lens scoop-check` (`--limit`, `--top-k`). Keyless. No web/UI yet.

**Known limitation.** Recall still depends on query wording — a card whose closest prior art uses different terminology can score `novel` with only adjacent work shown. The per-term + CS/recency retrieval markedly improved this (it now catches e.g. adaptive layer-skipping → SkipNet), and the `novel` verdicts that remain are grounded in *relevant* related work rather than junk. Under a large batch the free polite pool may rate-limit; the idempotent pass simply re-checks those cards on the next run.

## Future Hooks (not built here)

- **Embedding-based pattern retrieval:** patterns are already embedded by `build_vocabulary`; a later version can retrieve the top-k patterns by gap-description similarity instead of injecting all 15.
- **Card surfacing (UI):** an `/api/ideas` endpoint + an Ideas page that renders cards and lets users filter out `scooped` ones (now possible thanks to `novelty_status`). This is the increment that makes cards visible in the web app.
- **Audit/revise loop:** feed a `scooped` verdict back into generation to regenerate with the collision as an explicit constraint (like ResearchStudio's Phase 3).
- **Confidence calibration & domain guard** on generation (further findings from the validation batch).

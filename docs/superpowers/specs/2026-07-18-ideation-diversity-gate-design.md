# Ideation Diversity Gate — Design

**Date:** 2026-07-18
**Status:** Approved (design), lightweight inline TDD implementation.

## Problem

`run_ideation_with_llm` emits **one idea card per gap**. On the real corpus that is
408 cards — one per empty sparse matrix cell — and every fully-empty cell scores an
identical `1.0`, so there is no ranking. The LLM then collapses these distinct cells
onto ~15–20 recurring "moves" (differentiable NAS, scaling laws, warp scheduling), so
the 408 cards are heavily redundant at the *idea* level. This is impractical to
scoop-check (≈2000 free-pool OpenAlex calls) and a poor UI foundation.

**Root cause:** cell-level uniqueness ≠ idea-level uniqueness, and there is no cap.

## Goal

Reduce the run to ~40 *distinct* cards via a diversity gate between gap-finding and
card emission, without touching structural gap-finding (`run_ideation`) or the
card-parsing / provenance / graceful-degradation logic.

## Design

A single change to `run_ideation_with_llm` (`src/lens/monitor/ideation.py`). No new
tables, no pipeline restructure.

### Algorithm

1. `run_ideation(...)` → gaps (unchanged).
2. **Score floor:** drop gaps with `score < min_gap_score` (mainly prunes weak
   cross-pollination; fully-empty sparse cells score `1.0` and always pass).
3. **Diversified order:** sort surviving gaps by `score` desc, bucket by
   `related_params[0]` (the improving param), then round-robin across buckets — so
   the budget isn't spent on one parameter's cluster.
4. **Generate with incremental dedup + caps.** Iterate the diversified gaps:
   - LLM `complete` → parse card (existing logic, unchanged).
   - Compute a token set from `title + signature_terms` (lowercased `[a-z0-9]+`).
   - If token-Jaccard vs **any** already-kept card ≥ `dedup_threshold` → skip as a
     duplicate (log, continue).
   - Else persist (existing persist + hypothesis-writeback logic) and record its
     token set.
   - **Stop** when kept cards reach `max_cards`, or LLM calls reach the internal
     budget `max(max_cards * 3, 60)`.
5. Log a summary: cards emitted, gaps consumed, duplicates dropped, and — if the
   budget bound was hit — how many gaps went unprocessed (no silent truncation).

### Signature

```python
async def run_ideation_with_llm(
    store, llm_client,
    min_principles: int = 2, similarity_threshold: float = 0.75,
    max_cards: int = 40, min_gap_score: float = 0.0, dedup_threshold: float = 0.35,
) -> dict[str, Any]:
```

Function defaults (`min_gap_score=0.0`, `dedup_threshold=0.35`, `max_cards=40`) keep
bare callers and small-corpus unit tests working. Production values come from config.

### New pure helpers (unit-tested directly)

- `_card_token_set(title: str, signature_terms: list[str]) -> set[str]`
- `_jaccard(a: set[str], b: set[str]) -> float`  (empty∪empty → 0.0)
- `_diversified_gap_order(gaps: list[dict]) -> list[dict]`

### Config (`monitor`, `src/lens/config.py`)

| key | old | new | meaning |
|-----|-----|-----|---------|
| `ideate_top_n` | 10 (dead) | **40** | max distinct cards per run (→ `max_cards`) |
| `ideate_min_gap_score` | 0.5 (dead) | 0.5 | gap score floor (→ `min_gap_score`) |
| `ideate_dedup_threshold` | — | **0.35** | token-Jaccard duplicate threshold (→ `dedup_threshold`) |

`ideate_top_n` / `ideate_min_gap_score` already exist and are validated in
`config.py` but were never consumed — this wires them. Add validation for
`ideate_dedup_threshold` (0.0–1.0) alongside the existing checks.

### Wiring

Thread the three values `cli.py monitor` → `run_monitor_cycle` (three new kwargs with
config-matching defaults) → `run_ideation_with_llm`.

## Threshold note

The near-dup cluster (cards 401/403/405/408) shares ≈6 of ~16 tokens → Jaccard ≈ 0.37,
so the catch threshold sits at **~0.3–0.35**, not 0.5. Default 0.35; after the local
regen I'll eyeball the survivors and tune once, reporting the final distinct count
before scoop-check.

## Test impact

`tests/test_ideation.py`:
- **Unchanged (9 tests):** identical-mock tests still assert `len(cards) >= 1`; dedup
  collapsing identical cards to 1 keeps them green. Malformed/raise/db-fail →
  still 0 cards.
- **Update (2 tests):** `test_cross_pollination_card_uses_source_cell_provenance` and
  `test_cross_pollination_provenance_excludes_other_principle_cells` mock identical
  JSON for every gap; dedup would erase the cross-pollination card they assert on.
  Fix: distinct-per-call mock (unique `signature_terms`) + `max_cards=1000`, so the
  cross-poll card survives dedup and the cap. Provenance assertions unchanged.
- **New tests:** `_jaccard` / `_card_token_set` / `_diversified_gap_order` units;
  dedup-collapses-identical; max_cards-caps-output; min_gap_score-filters.

## Out of scope

Structural gap-finding, card schema, scoop-check, the Ideas UI. Embedding-based dedup
(considered, rejected for now — token Jaccard first, tune, revisit only if paraphrase
collapse survives).

# Vetted-Novelty Ideas Showcase — Design Spec

**Date:** 2026-07-19
**Status:** Approved (brainstorming complete); ready for implementation planning.

## Goal

Add a public `/ideas` page to the LENS web tier that showcases the machine-generated idea cards, with the **novelty verdict as the hero**: each card leads with whether it is `novel` / `overlaps` / `scooped` and shows the prior-art "receipt" it was checked against. Keep the showcase's data true over time by scoop-checking new cards in the weekly monitor cron.

## Framing decisions (from brainstorming)

- **Job of the page:** a *vetted-novelty showcase*, not a generic card browser. Novelty is the headline; the prior-art receipt is the trust-builder.
- **Card treatment:** *receipt-forward feed* — a single ranked feed (novel-first), one `/ideas` route, details expand inline (no per-card URLs).
- **Durability:** *durable + scoop-check in cron* — reverse the prior "scoop-check intentionally not in cron" decision so new cards get verified weekly; bootstrap the current verdicts into `corpus-snapshot` so they survive.
- **Default feed excludes `unchecked`** — a showcase only shows cards with a real verdict. Cron-generated cards appear once checked.

## Non-goals (YAGNI)

- No per-card pages / shareable card URLs (inline expand only).
- No server-side rendering — match the existing all-`"use client"` page pattern.
- No editing/moderation UI; the corpus is read-only in the web tier.
- No new dependencies (Tailwind only, no shadcn), no new secrets, no new Vercel env vars.
- No pagination — the corpus is ~33 cards; fetch all once and filter client-side.

---

## Component 1 — Data pipeline (durability + freshness)

### 1a. One-time bootstrap
Push the current local `~/.lens/data/lens.db` (which holds the 33 scoop-checked verdicts already live in `lens-prod`) to the `corpus-snapshot` GitHub release, so the next monitor cron starts from the checked state rather than the stale pre-scoop-check snapshot.

```bash
gh release upload corpus-snapshot ~/.lens/data/lens.db --repo flyersworder/lens --clobber
```

This is an ops action, not code. It must run before the first cron after this change (else that cron's scoop-check step would re-check every card from the old snapshot in one run, risking the OpenAlex daily budget).

### 1b. Recurring — new step in `.github/workflows/monitor.yml`
Insert **between** the existing *Run monitor* step and the *Publish to lens-prod* step:

```yaml
- name: Scoop-check new idea cards
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
  run: uv run lens scoop-check --max-terms 3
```

Rationale, and why this needs nothing else:
- **No new secrets/config.** The novelty judge uses the LLM already configured by the *Configure LENS for cloud LLM + embeddings* step; `acquire.openalex_mailto` is already set there (line 77). OpenAlex's free polite pool needs no key.
- **Bounded OpenAlex spend.** `scoop-check` is idempotent and only touches cards with `novelty_status='unchecked'`, so each week it checks only that week's *new* cards, not all 33 again. `--max-terms 3` caps per-card OpenAlex requests; the free tier is ~100 searches/day.
- **Self-maintaining durability.** The scoop-checked `lens.db` flows into both *Publish to lens-prod* and *Upload updated corpus snapshot* (`--clobber`), so verdicts persist into the snapshot automatically from this run onward. The 1a bootstrap only needs to happen once.
- **Reads the workflow's DB.** `lens scoop-check` uses `LENS_DATA_DIR` (`$LENS_DATA_DIR/lens.db`), the same DB `lens monitor` mutated — no `--local` flag needed (the CLI resolves the data dir from env).

### 1c. Doc updates
- `CLAUDE.md`: the Scoop-check bullet says "intentionally not run in the monitor cron" — change to reflect that it now runs weekly with `--max-terms 3`.
- `docs/web-deployment.md`: note the monitor step and that snapshot durability is now self-maintaining.

---

## Component 2 — API endpoint `GET /api/ideas`

Add to `api/index.py`, following the `_compute_stats` / `stats_endpoint` house style (independent SQL, degrade to safe empties, never 500).

### Response shape
```json
{
  "counts": { "novel": 25, "overlaps": 7, "scooped": 1, "total": 33 },
  "cards": [
    {
      "id": 4,
      "title": "Adaptive KV-Cache Pruning via Latency-Aware Token Importance Estimation",
      "hook": "Prune the KV cache by predicted per-token importance…",
      "mechanism": "…",
      "falsification": "…",
      "differentiation": ["…"],
      "signature_terms": ["kv-cache", "pruning"],
      "novelty_status": "novel",
      "prior_art": [ { "title": "Titanus", "url": "https://…", "year": 2025 } ],
      "novelty_note": "…",
      "grounded_paper_count": 3,
      "confidence": 0.7
    }
  ]
}
```

### Behavior
- **Source:** `SELECT` from `idea_cards WHERE novelty_status IN ('novel','overlaps','scooped')` (excludes `unchecked` and any other status). JSON columns (`differentiation`, `signature_terms`, `prior_art`, `paper_ids`) are stored as JSON text; parse with a fail-soft helper (bad JSON → `[]`).
- **`grounded_paper_count`** = length of the parsed `paper_ids` list (the UI shows "grounded in N papers", not the raw ids).
- **`counts`** = per-verdict tally over the same filtered set, plus `total`.
- **Sort:** verdict rank `novel(0) < overlaps(1) < scooped(2)`, then `confidence` descending, then `id` ascending (stable tiebreak).
- **Degradation:** on any query failure, log and return `{ "counts": { "novel": 0, "overlaps": 0, "scooped": 0, "total": 0 }, "cards": [] }` — never 500 (mirrors `_compute_stats`). A fresh DB with no `idea_cards` table degrades to the same empty response.
- **Dependency injection:** use the existing `StoreDep` pattern; read-only, no LLM, no embedding.

### Model change
Extend `IdeaCard` in `src/lens/store/models.py` with the four columns that already exist on the table:
```python
novelty_status: str = "unchecked"
prior_art: list[dict] = []
novelty_note: str = ""
novelty_checked_at: datetime | None = None
```
The endpoint may read via `query_sql` directly (as `_compute_stats` does) rather than through the model; the model extension keeps it honest and unblocks any typed use. `novelty_checked_at` is not surfaced by the API (kept minimal) but is added for model completeness.

---

## Component 3 — Frontend `/ideas` page

Match the established pattern: **`"use client"` page that fetches on mount** via a `lib/api` helper and tracks the view with `useTrackOnce`.

### 3a. `lib/api.ts` — add helper + types
```ts
export type IdeaCard = {
  id: number;
  title: string;
  hook: string;
  mechanism: string;
  falsification: string;
  differentiation: string[];
  signature_terms: string[];
  novelty_status: "novel" | "overlaps" | "scooped";
  prior_art: Array<{ title: string; url: string; year: number | null }>;
  novelty_note: string;
  grounded_paper_count: number;
  confidence: number;
};

export type IdeasResponse = {
  counts: { novel: number; overlaps: number; scooped: number; total: number };
  cards: IdeaCard[];
};

export const ideas = () => jget<IdeasResponse>("/api/ideas");
```

### 3b. `app/ideas/page.tsx` (client component)
- On mount: `ideas()` → store `{counts, cards}`; `useTrackOnce("view_ideas")`.
- Header: title line ("Ideas — N machine-generated, checked against prior art") + filter chips `[Novel n] [Overlaps n] [Scooped n]`. **Single-select toggle:** default = all cards shown (novel-first as returned); clicking a chip filters to that verdict; clicking the active chip again clears the filter. Filtering is client-side over the already-fetched list.
- Body: the returned `cards` rendered as `<IdeaCard>` in order (API already sorts novel-first).
- Empty state (`cards.length === 0`): "No vetted ideas yet." Loading state: a brief skeleton/spinner consistent with the other pages.
- Styling: Tailwind, reuse the palette/tokens from `layout.tsx` / existing pages (`ink`, `accent`, `zinc-*`).

### 3c. `app/components/IdeaCard.tsx` (client component)
Receipt-forward card, matching the approved mockup:
- **Top row:** verdict badge (🟢 NOVEL / 🟡 OVERLAPS / 🔴 SCOOPED, color-coded) + "✓ checked against {prior_art.length} papers" (hidden when `prior_art` empty).
- **Title** (prominent) + **hook** (muted).
- **Expandable sections** (collapsed by default, toggle on click): `mechanism`, `how to falsify` (`falsification`), `prior art` (list of `prior_art` entries; each title links to `url` when present, shows `year`; include `novelty_note` as the verdict rationale line).
- **Footer:** "grounded in {grounded_paper_count} papers · {signature_terms joined}".
- Track expands with `track("expand_idea", String(id))` (fire-and-forget).

### 3d. Nav — `app/layout.tsx`
Add `<a className="hover:text-white" href="/ideas">ideas</a>` to the `<nav>`, between `explain` and `usage`.

---

## Error / edge handling

| Case | Behavior |
|---|---|
| API failure / fresh DB / missing table | API returns empty envelope; page shows "No vetted ideas yet." |
| Card with empty `prior_art` | Card renders; the "checked against N papers" receipt is hidden. |
| Malformed JSON in a JSON column | Fail-soft parse → `[]`; card still renders. |
| `unchecked` / unknown `novelty_status` | Excluded by the API `WHERE` clause; never reaches the UI. |
| `track` call fails | Swallowed by the existing helper; no UX impact. |

---

## Testing

- **API (`tests/test_api_index.py`):** seed `idea_cards` with a mix of verdicts (incl. `unchecked`) →
  - response shape matches (`counts` + `cards`);
  - `unchecked` excluded; `counts.total` == returned card count;
  - sort order novel → overlaps → scooped, then confidence desc;
  - `grounded_paper_count` == length of `paper_ids`;
  - empty/failed DB degrades to the empty envelope (no 500).
- **Model (`tests/test_models.py` or existing model test):** `IdeaCard` round-trips the four novelty fields.
- **Frontend:** no test harness exists for the React pages in this repo (consistent with the other pages, which are untested); rely on `next build` type-checking + manual verification. Do not introduce a new frontend test framework for this feature.
- **Pipeline step:** validated by a manual `workflow_dispatch` dry-run of `monitor.yml` after the change (ops), not a unit test.

---

## Build order

The three components are loosely coupled and can land in one PR, built in this order:

1. **API + model** (`/api/ideas`, `IdeaCard` fields) + tests — independently testable; the 33 cards are already live in `lens-prod`, so the endpoint returns real data immediately.
2. **Frontend** (`lib/api` helper, `/ideas` page, `IdeaCard` component, nav) — demoable against the live API as soon as step 1 deploys.
3. **Pipeline** (monitor step + `CLAUDE.md` / doc edits) + the one-time `corpus-snapshot` bootstrap — independent of the UI; keeps the showcase fresh going forward.

The showcase is fully functional after steps 1–2 against current data; step 3 is what keeps it true after the next Monday cron.

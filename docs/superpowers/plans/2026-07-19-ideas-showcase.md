# Vetted-Novelty Ideas Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a public `/ideas` page that showcases the machine-generated idea cards with their prior-art novelty verdict as the hero, and keep the verdicts fresh by scoop-checking new cards in the weekly monitor cron.

**Architecture:** Three loosely-coupled components — a read-only `GET /api/ideas` FastAPI endpoint (reads `idea_cards` through the backend-agnostic `ReadableStore.query`), a `"use client"` Next.js `/ideas` page + `IdeaCard` component that fetches it on mount, and a one-line `scoop-check` step in `monitor.yml`. The endpoint returns real data immediately because the 33 scoop-checked cards are already live in `lens-prod`.

**Tech Stack:** Python 3.12 / FastAPI (Vercel Python runtime), Next.js 16 App Router + Tailwind (no shadcn), Pydantic v2, pytest, GitHub Actions.

## Global Constraints

- **Backend-agnostic store access:** the endpoint runs against `TursoStore` in prod and `LensStore` in tests. Use `store.query("idea_cards", where, params)` from the `ReadableStore` protocol — it auto-deserializes the JSON columns (`prior_art`, `signature_terms`, `paper_ids`, `differentiation`, `pattern_ids`) identically on both backends. Do **not** use `query_sql` (no JSON deserialization) and do **not** reference `LensStore`/`TursoStore` concretely.
- **Never 500:** the endpoint degrades to the empty envelope `{"counts": {"novel": 0, "overlaps": 0, "scooped": 0, "total": 0}, "cards": []}` on any failure — mirror `_compute_stats` / `stats_endpoint`.
- **Parameterized SQL only** — never string-interpolate values.
- **Verdict rank for sorting:** `novel`(0) < `overlaps`(1) < `scooped`(2), then `confidence` descending, then `id` ascending.
- **Exclude `unchecked`** (and any non-`{novel,overlaps,scooped}` status) from the feed — enforced in the SQL `WHERE`.
- **No new dependencies, no new secrets, no new Vercel env vars, no shadcn.** Tailwind design tokens already in the repo: `ink`, `ink-soft`, `ink-line`, `accent`, `accent-soft`, `zinc-*`.
- **Frontend pattern:** `"use client"`, fetch on mount via a `lib/api` helper, track the view with `useTrackOnce`.
- **Response `cards[]` fields (exact):** `id, title, hook, mechanism, falsification, differentiation, signature_terms, novelty_status, prior_art, novelty_note, grounded_paper_count, confidence`. `grounded_paper_count` = `len(paper_ids)`; raw `paper_ids` is NOT surfaced. `novelty_checked_at` is NOT surfaced.

---

### Task 1: Extend the `IdeaCard` model with novelty fields

**Files:**
- Modify: `src/lens/store/models.py` (the `IdeaCard` class, ends at `taxonomy_version: int`)
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `IdeaCard` now carries `novelty_status: str`, `prior_art: list[dict]`, `novelty_note: str`, `novelty_checked_at: datetime | None`, matching the four columns already on the `idea_cards` table (see `_COLUMN_MIGRATIONS` in `store.py`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_models.py`:

```python
def test_idea_card_novelty_fields():
    from lens.store.models import IdeaCard

    card = IdeaCard(
        id=1,
        gap_id=2,
        report_id=3,
        title="Adaptive KV-Cache Pruning",
        pattern_ids=[],
        hook="",
        mechanism="m",
        falsification="",
        differentiation=[],
        signature_terms=["kv-cache"],
        paper_ids=["2401.00001"],
        confidence=0.7,
        created_at=datetime(2026, 7, 19),
        taxonomy_version=0,
        novelty_status="novel",
        prior_art=[{"title": "Titanus", "url": "https://x", "year": 2025}],
        novelty_note="no core collision",
        novelty_checked_at=datetime(2026, 7, 19),
    )
    assert card.novelty_status == "novel"
    assert card.prior_art[0]["title"] == "Titanus"
    assert card.novelty_note == "no core collision"

    # Novelty fields are optional (backward-compat with pre-scoop-check rows).
    bare = IdeaCard(
        id=2,
        gap_id=1,
        report_id=1,
        title="Bare",
        pattern_ids=[],
        hook="",
        mechanism="",
        falsification="",
        differentiation=[],
        signature_terms=[],
        paper_ids=[],
        confidence=0.0,
        created_at=datetime(2026, 7, 19),
        taxonomy_version=0,
    )
    assert bare.novelty_status == "unchecked"
    assert bare.prior_art == []
    assert bare.novelty_note == ""
    assert bare.novelty_checked_at is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::test_idea_card_novelty_fields -v`
Expected: FAIL — `IdeaCard` rejects `novelty_status` (unexpected keyword / extra field).

- [ ] **Step 3: Write minimal implementation**

In `src/lens/store/models.py`, extend the `IdeaCard` class body (the fields currently end at `taxonomy_version: int`). Add after it:

```python
    taxonomy_version: int
    novelty_status: str = "unchecked"
    prior_art: list[dict] = []
    novelty_note: str = ""
    novelty_checked_at: datetime | None = None
```

(Keep the existing `taxonomy_version: int` line; the four new fields follow it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py::test_idea_card_novelty_fields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lens/store/models.py tests/test_models.py
git commit -m "feat(models): add novelty fields to IdeaCard"
```

---

### Task 2: `GET /api/ideas` endpoint

**Files:**
- Modify: `api/index.py` (add the endpoint after the `/api/stats` block, near the other read-only GETs)
- Test: `tests/test_api_index.py`

**Interfaces:**
- Consumes: the `StoreDep` alias and `get_store` dependency already defined in `api/index.py`; `store.query(table, where, params)` from `ReadableStore`.
- Produces: `GET /api/ideas` → `{"counts": {"novel": int, "overlaps": int, "scooped": int, "total": int}, "cards": [ {…12 fields…} ]}`, sorted novel-first.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api_index.py` (uses the existing `client` fixture and `MagicMock`, both already imported):

```python
def _idea_row(card_id: int, status: str, confidence: float = 0.5,
              paper_ids=None, prior_art=None) -> dict:
    """A store.query('idea_cards', ...) row (JSON columns already deserialized)."""
    return {
        "id": card_id,
        "gap_id": 1,
        "report_id": 1,
        "title": f"Card {card_id}",
        "pattern_ids": [],
        "hook": "hook",
        "mechanism": "mechanism",
        "falsification": "falsify",
        "differentiation": ["diff"],
        "signature_terms": ["t1", "t2"],
        "paper_ids": ["p1", "p2"] if paper_ids is None else paper_ids,
        "confidence": confidence,
        "created_at": "2026-07-19T00:00:00+00:00",
        "taxonomy_version": 0,
        "novelty_status": status,
        "prior_art": [{"title": "Prior", "url": "u", "year": 2024}]
        if prior_art is None else prior_art,
        "novelty_note": "note",
    }


def test_ideas_returns_sorted_envelope(client):
    c, fake_store, _ = client
    fake_store.query = MagicMock(return_value=[
        _idea_row(3, "scooped", 0.9),
        _idea_row(1, "novel", 0.6),
        _idea_row(2, "novel", 0.8),
        _idea_row(4, "overlaps", 0.5),
    ])
    r = c.get("/api/ideas")
    assert r.status_code == 200
    body = r.json()
    assert body["counts"] == {"novel": 2, "overlaps": 1, "scooped": 1, "total": 4}
    # novel-first, then confidence desc, then id asc → [2, 1, 4, 3]
    assert [card["id"] for card in body["cards"]] == [2, 1, 4, 3]
    first = body["cards"][0]
    assert first["novelty_status"] == "novel"
    assert first["grounded_paper_count"] == 2       # len(paper_ids)
    assert "paper_ids" not in first                 # raw ids not surfaced
    assert first["prior_art"][0]["title"] == "Prior"
    assert first["signature_terms"] == ["t1", "t2"]


def test_ideas_excludes_unchecked_in_query(client):
    c, fake_store, _ = client
    captured = {}

    def fake_query(table, where="", params=None):
        captured["table"] = table
        captured["where"] = where
        captured["params"] = params
        return []

    fake_store.query = fake_query
    r = c.get("/api/ideas")
    assert r.status_code == 200
    assert captured["table"] == "idea_cards"
    assert "novelty_status IN" in captured["where"]
    assert set(captured["params"]) == {"novel", "overlaps", "scooped"}
    assert r.json() == {
        "counts": {"novel": 0, "overlaps": 0, "scooped": 0, "total": 0},
        "cards": [],
    }


def test_ideas_degrades_on_query_failure(client):
    c, fake_store, _ = client
    fake_store.query = MagicMock(side_effect=RuntimeError("no idea_cards table"))
    r = c.get("/api/ideas")
    assert r.status_code == 200            # never 500
    body = r.json()
    assert body["cards"] == []
    assert body["counts"]["total"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_index.py -k ideas -v`
Expected: FAIL — 404 (route not registered).

- [ ] **Step 3: Write minimal implementation**

In `api/index.py`, add after the `stats_endpoint` / `_compute_stats` block:

```python
@app.get("/api/ideas")
def ideas_endpoint(store: StoreDep) -> dict[str, Any]:
    """Vetted idea cards for the /ideas showcase.

    Returns only cards with a real novelty verdict (novel / overlaps /
    scooped); ``unchecked`` cards are excluded via the WHERE clause.
    Sorted novel-first, then by confidence. Degrades to an empty
    envelope on any failure — never 500 (mirrors ``_compute_stats``).
    """
    empty = {
        "counts": {"novel": 0, "overlaps": 0, "scooped": 0, "total": 0},
        "cards": [],
    }
    try:
        rows = store.query(
            "idea_cards",
            "novelty_status IN (?, ?, ?)",
            ("novel", "overlaps", "scooped"),
        )
    except Exception:
        logger.exception("ideas: query failed")
        return empty

    rank = {"novel": 0, "overlaps": 1, "scooped": 2}
    rows.sort(
        key=lambda c: (
            rank.get(str(c.get("novelty_status")), 9),
            -float(c.get("confidence") or 0.0),
            c.get("id") or 0,
        )
    )

    counts = {"novel": 0, "overlaps": 0, "scooped": 0}
    cards: list[dict[str, Any]] = []
    for c in rows:
        status = str(c.get("novelty_status"))
        if status not in counts:
            continue
        counts[status] += 1
        cards.append(
            {
                "id": c.get("id"),
                "title": c.get("title", ""),
                "hook": c.get("hook", ""),
                "mechanism": c.get("mechanism", ""),
                "falsification": c.get("falsification", ""),
                "differentiation": c.get("differentiation") or [],
                "signature_terms": c.get("signature_terms") or [],
                "novelty_status": status,
                "prior_art": c.get("prior_art") or [],
                "novelty_note": c.get("novelty_note", ""),
                "grounded_paper_count": len(c.get("paper_ids") or []),
                "confidence": float(c.get("confidence") or 0.0),
            }
        )
    counts["total"] = len(cards)
    return {"counts": counts, "cards": cards}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api_index.py -k ideas -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full API test module + type check**

Run: `uv run pytest tests/test_api_index.py -q && uv run ty check api/index.py`
Expected: all pass, no new type errors.

- [ ] **Step 6: Commit**

```bash
git add api/index.py tests/test_api_index.py
git commit -m "feat(api): GET /api/ideas — vetted idea cards for the showcase"
```

---

### Task 3: `/ideas` frontend page

**Files:**
- Modify: `lib/api.ts` (add `IdeaCard` type, `IdeasResponse` type, `ideas()` helper)
- Create: `app/components/IdeaCard.tsx`
- Create: `app/ideas/page.tsx`
- Modify: `app/layout.tsx` (nav link)
- Modify: `app/usage/page.tsx` (add labels for the two new events)

**Interfaces:**
- Consumes: `GET /api/ideas` (Task 2); `jget` and `track` from `lib/api.ts`; `useTrackOnce` from `lib/use-track-once.ts`.
- Produces: a working `/ideas` route in the site nav.

No unit tests: this repo's React pages are untested by convention (see `app/*/page.tsx` — none have tests). Verification is `next build` (type-check + route compile). Do not introduce a frontend test framework.

- [ ] **Step 1: Add the API helper + types to `lib/api.ts`**

Append to the "endpoints" section of `lib/api.ts` (after the `stats` export block):

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

- [ ] **Step 2: Create `app/components/IdeaCard.tsx`**

```tsx
"use client";

import { useState } from "react";
import { track, type IdeaCard as IdeaCardT } from "@/lib/api";

const BADGE: Record<string, { label: string; dot: string; text: string }> = {
  novel: { label: "NOVEL", dot: "🟢", text: "text-green-400" },
  overlaps: { label: "OVERLAPS", dot: "🟡", text: "text-yellow-400" },
  scooped: { label: "SCOOPED", dot: "🔴", text: "text-red-400" },
};

export function IdeaCard({ card }: { card: IdeaCardT }) {
  const [open, setOpen] = useState<string | null>(null);
  const badge = BADGE[card.novelty_status] ?? BADGE.novel;
  const priorCount = card.prior_art.length;

  const toggle = (section: string) => {
    setOpen((cur) => (cur === section ? null : section));
    track("expand_idea", String(card.id));
  };

  return (
    <article className="rounded-lg border border-ink-line bg-ink-soft/60 p-5 space-y-3">
      <div className="flex items-center justify-between gap-3">
        <span className={`font-mono text-xs tracking-wide ${badge.text}`}>
          {badge.dot} {badge.label}
        </span>
        {priorCount > 0 && (
          <span className="text-xs text-zinc-500">
            ✓ checked against {priorCount} paper{priorCount === 1 ? "" : "s"}
          </span>
        )}
      </div>

      <h3 className="text-lg font-semibold text-white leading-snug">
        {card.title}
      </h3>
      {card.hook && <p className="text-sm text-zinc-400">{card.hook}</p>}

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {card.mechanism && (
          <button
            className="text-accent hover:text-accent-soft"
            onClick={() => toggle("mechanism")}
          >
            ▸ mechanism
          </button>
        )}
        {card.falsification && (
          <button
            className="text-accent hover:text-accent-soft"
            onClick={() => toggle("falsify")}
          >
            ▸ how to falsify
          </button>
        )}
        {priorCount > 0 && (
          <button
            className="text-accent hover:text-accent-soft"
            onClick={() => toggle("prior")}
          >
            ▸ prior art
          </button>
        )}
      </div>

      {open === "mechanism" && (
        <p className="text-sm text-zinc-300 border-l-2 border-ink-line pl-3">
          {card.mechanism}
        </p>
      )}
      {open === "falsify" && (
        <p className="text-sm text-zinc-300 border-l-2 border-ink-line pl-3">
          {card.falsification}
        </p>
      )}
      {open === "prior" && (
        <div className="text-sm text-zinc-300 border-l-2 border-ink-line pl-3 space-y-1">
          {card.novelty_note && (
            <p className="italic text-zinc-400">{card.novelty_note}</p>
          )}
          <ul className="space-y-1">
            {card.prior_art.map((p, i) => (
              <li key={i}>
                {p.url ? (
                  <a
                    href={p.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-accent hover:text-accent-soft"
                  >
                    {p.title}
                  </a>
                ) : (
                  <span>{p.title}</span>
                )}
                {p.year ? <span className="text-zinc-500"> ({p.year})</span> : null}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="font-mono text-[11px] text-zinc-500">
        grounded in {card.grounded_paper_count} paper
        {card.grounded_paper_count === 1 ? "" : "s"}
        {card.signature_terms.length > 0 &&
          ` · ${card.signature_terms.join(", ")}`}
      </div>
    </article>
  );
}
```

- [ ] **Step 3: Create `app/ideas/page.tsx`**

```tsx
"use client";

import { useEffect, useState } from "react";
import { ideas, type IdeasResponse } from "@/lib/api";
import { IdeaCard } from "../components/IdeaCard";
import { useTrackOnce } from "@/lib/use-track-once";

type Filter = "novel" | "overlaps" | "scooped" | null;

const CHIPS: Array<{ key: Exclude<Filter, null>; label: string }> = [
  { key: "novel", label: "Novel" },
  { key: "overlaps", label: "Overlaps" },
  { key: "scooped", label: "Scooped" },
];

export default function IdeasPage() {
  useTrackOnce("view_ideas");
  const [data, setData] = useState<IdeasResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>(null);

  useEffect(() => {
    ideas()
      .then(setData)
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);

  const cards = data
    ? filter
      ? data.cards.filter((c) => c.novelty_status === filter)
      : data.cards
    : [];

  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">
          Ideas
        </h1>
        <p className="max-w-2xl text-sm sm:text-base text-zinc-400">
          Research ideas generated from the LENS corpus, each checked against
          real prior art via OpenAlex. The verdict says whether the core idea
          looks novel, overlaps existing work, or is already covered.
        </p>
        {data && (
          <div className="flex flex-wrap gap-2">
            {CHIPS.map((chip) => {
              const n = data.counts[chip.key];
              const active = filter === chip.key;
              return (
                <button
                  key={chip.key}
                  onClick={() => setFilter(active ? null : chip.key)}
                  className={`rounded-full border px-3 py-1 font-mono text-xs transition ${
                    active
                      ? "border-accent bg-accent/15 text-accent"
                      : "border-ink-line text-zinc-400 hover:text-white"
                  }`}
                >
                  {chip.label} {n}
                </button>
              );
            })}
          </div>
        )}
      </section>

      {err && <p className="text-sm text-red-400">Error: {err}</p>}

      {data && cards.length === 0 && !err && (
        <div className="rounded-lg border border-ink-line bg-ink-soft/60 p-6 text-zinc-300">
          No vetted ideas yet.
        </div>
      )}

      <div className="space-y-4">
        {cards.map((card) => (
          <IdeaCard key={card.id} card={card} />
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Add the nav link in `app/layout.tsx`**

In the `<nav>` block, add an `ideas` link between `explain` and `usage`:

```tsx
            <a className="hover:text-white" href="/explain">explain</a>
            <a className="hover:text-white" href="/ideas">ideas</a>
            <a className="hover:text-white" href="/usage">usage</a>
```

- [ ] **Step 5: Add event labels in `app/usage/page.tsx`**

Extend the `EVENT_LABELS` map so the new events read nicely on the usage page:

```tsx
  view_explain_concept: "Concept page view",
  view_ideas: "Ideas page view",
  expand_idea: "Idea card expanded",
  search: "Search submitted",
```

(Insert the two new entries; keep the surrounding entries.)

- [ ] **Step 6: Verify the build type-checks and compiles the route**

Run: `npm run build`
Expected: build succeeds; the route list includes `/ideas`. (This is the type-check gate — a TS error in any new file fails the build.)

- [ ] **Step 7: Commit**

```bash
git add lib/api.ts app/components/IdeaCard.tsx app/ideas/page.tsx app/layout.tsx app/usage/page.tsx
git commit -m "feat(web): /ideas showcase page with novelty-verdict cards"
```

---

### Task 4: Scoop-check in the monitor cron + docs + one-time bootstrap

**Files:**
- Modify: `.github/workflows/monitor.yml` (new step between *Run monitor* and *Publish to lens-prod*)
- Modify: `CLAUDE.md` (Scoop-check bullet)
- Modify: `docs/web-deployment.md` (monitor pipeline note)

**Interfaces:**
- Consumes: `lens scoop-check --max-terms N` (already shipped in 0.13.0); `OPENROUTER_API_KEY` secret (already used by the *Run monitor* step); `acquire.openalex_mailto` (already configured in the *Configure LENS* step).
- Produces: weekly novelty verdicts on newly-generated cards; self-maintaining durability because the scoop-checked `lens.db` is both published and re-uploaded to `corpus-snapshot`.

- [ ] **Step 1: Add the workflow step**

In `.github/workflows/monitor.yml`, insert between the `Run monitor (…)` step and the `Publish to lens-prod` step:

```yaml
      - name: Scoop-check new idea cards
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: uv run lens scoop-check --max-terms 3
```

- [ ] **Step 2: Verify the workflow YAML still parses**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/monitor.yml'))"`
Expected: no error (valid YAML).

- [ ] **Step 3: Update `CLAUDE.md`**

In the Scoop-check bullet, replace the clause `intentionally not run in the monitor cron` with:

> now runs in the weekly monitor cron (`--max-terms 3`, after `ideate`, before publish); because it is idempotent and only touches `unchecked` cards, each run checks just that week's new cards.

- [ ] **Step 4: Update `docs/web-deployment.md`**

In the monitor-pipeline description (the `## Live state snapshot` durability note added on 2026-07-19), append one sentence: the monitor cron now runs `lens scoop-check --max-terms 3` between *Run monitor* and *Publish*, so verdicts are re-uploaded into `corpus-snapshot` and durability is self-maintaining after the one-time bootstrap.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/monitor.yml CLAUDE.md docs/web-deployment.md
git commit -m "ci(monitor): scoop-check new idea cards weekly; make novelty durable"
```

- [ ] **Step 6: One-time `corpus-snapshot` bootstrap (manual ops — run once)**

> **This is a manual action, not code. Run it once before the next Monday cron, from a machine with the local `~/.lens/data/lens.db` (holding the 33 verdicts) and `gh` authenticated.** It replaces the `corpus-snapshot` asset so the next cron starts from the checked state instead of overwriting the live verdicts.

```bash
gh release upload corpus-snapshot ~/.lens/data/lens.db --repo flyersworder/lens --clobber
```

Verify: `gh release view corpus-snapshot --repo flyersworder/lens --json assets -q '.assets[].name'` lists `lens.db`.

---

## Self-Review

- **Spec coverage:** Component 1 (pipeline) → Task 4; Component 2 (API + model) → Tasks 1–2; Component 3 (frontend) → Task 3. Error/empty handling → Task 2 (degrade) + Task 3 (empty state). Testing surface → Task 1 (model), Task 2 (API), Task 3 (build gate, no unit tests per repo convention). Bootstrap → Task 4 Step 6.
- **Placeholders:** none — every code step contains complete code.
- **Type consistency:** endpoint returns exactly the 12 `cards[]` fields the `IdeaCard` TS type declares; `grounded_paper_count`/`confidence` numeric; `prior_art` shape `{title,url,year}` matches between the Python endpoint (passes through `store.query` deserialized dicts) and the TS type. `novelty_status` union `novel|overlaps|scooped` is guaranteed by the endpoint's `WHERE` + `counts` filter.

## Build order & independence

Tasks 1→2→3 are the showcase and work against the already-live 33 cards; Task 4 is independent and keeps it fresh. Land in one PR (or ship 1–3 first, 4 immediately after). Task 4 Step 6 (bootstrap) must run before the next Monday 06:00 UTC cron to preserve the current verdicts.

# Scoop-Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an idempotent novelty-verification pass that queries Semantic Scholar for prior art on each generated idea card, LLM-judges whether the idea is already published, and annotates the card.

**Architecture:** A new `search_semantic_scholar` (relevance search) + a `judge_novelty` LLM judge + a `run_scoop_check` pass over `idea_cards` rows with `novelty_status='unchecked'`, plus four new columns and a `lens scoop-check` CLI command. Decoupled from card generation; every external touch fails soft and the pass is safe to re-run.

**Tech Stack:** Python 3.12, httpx (async), Semantic Scholar free tier, `json-repair`, `pytest`/`pytest-asyncio` with `tmp_path` real-store fixtures.

## Global Constraints

- Python `>=3.12`; type-check clean under `ty`; pre-commit hooks (`prek`: ruff E/F/I/UP/B/SIM @ line-length 99, ruff-format) must pass — **never** `git commit --no-verify`. If a line exceeds 99 chars, wrap it (implicit string concatenation).
- Semantic Scholar is used on the **free unauthenticated tier** — no API key is read or required.
- Every external touch fails soft: S2 error/empty → `[]`; judge unusable → `None`; DB write error → logged + skipped. A card only leaves `unchecked` when it gets a real verdict. The pass always completes and is idempotent.
- JSON list/dict columns must be registered in `JSON_FIELDS` (store.py:36). `store.update` does NOT auto-serialize — `json.dumps` JSON values manually; `store.query` auto-deserializes registered columns.
- Parse LLM JSON with `strip_code_fences` (from `lens.llm.utils`) → `json.loads` → `json_repair.repair_json(text, return_objects=True)` fallback, mirroring `extractor.py` / `_parse_idea_card`.
- No mocking of SQLite — real `LensStore` on `tmp_path`; stub only `httpx`/`fetch_with_retry` and the `llm_client`.
- Use `?` placeholders; never string-interpolate values into SQL.
- Verdict enum is exactly `unchecked | novel | overlaps | scooped`.

---

### Task 1: `search_semantic_scholar` relevance search

**Files:**
- Modify: `src/lens/acquire/semantic_scholar.py` (add function after `fetch_embedding`)
- Test: `tests/test_semantic_scholar.py` (create if absent)

**Interfaces:**
- Consumes: `S2_API_URL`, `RATE_LIMIT_SECONDS`, `fetch_with_retry(client, url, headers=None)` (all already in the module).
- Produces: `async search_semantic_scholar(query: str, limit: int = 5, api_key: str | None = None) -> list[dict]` where each dict is `{title, abstract, year, citations, arxiv_id, url}`; abstract-less papers dropped; never raises (returns `[]` on failure).

- [ ] **Step 1: Write the failing tests**

Create/append `tests/test_semantic_scholar.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_search_semantic_scholar_parses(monkeypatch):
    import lens.acquire.semantic_scholar as s2

    class FakeResp:
        def json(self):
            return {
                "data": [
                    {
                        "title": "GQA",
                        "abstract": "grouped-query attention reduces KV heads",
                        "year": 2023,
                        "citationCount": 100,
                        "externalIds": {"ArXiv": "2305.13245"},
                        "url": "http://x",
                    },
                    {
                        "title": "No abstract",
                        "abstract": None,
                        "year": 2024,
                        "citationCount": 1,
                        "externalIds": {},
                        "url": "",
                    },
                ]
            }

    async def fake_fetch(client, url, headers=None):
        return FakeResp()

    monkeypatch.setattr(s2, "fetch_with_retry", fake_fetch)
    monkeypatch.setattr(s2, "RATE_LIMIT_SECONDS", 0)

    res = await s2.search_semantic_scholar("quantization", limit=5)
    assert len(res) == 1  # abstract-less paper dropped
    assert res[0]["arxiv_id"] == "2305.13245"
    assert res[0]["citations"] == 100
    assert res[0]["title"] == "GQA"


@pytest.mark.asyncio
async def test_search_semantic_scholar_fails_soft(monkeypatch):
    import lens.acquire.semantic_scholar as s2

    async def boom(client, url, headers=None):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(s2, "fetch_with_retry", boom)
    monkeypatch.setattr(s2, "RATE_LIMIT_SECONDS", 0)

    assert await s2.search_semantic_scholar("anything") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_semantic_scholar.py -v`
Expected: FAIL — `AttributeError: module 'lens.acquire.semantic_scholar' has no attribute 'search_semantic_scholar'`.

- [ ] **Step 3: Implement `search_semantic_scholar`**

In `src/lens/acquire/semantic_scholar.py`, add after `fetch_embedding` (add `from urllib.parse import quote_plus` to the imports at the top):

```python
async def search_semantic_scholar(
    query: str,
    limit: int = 5,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Search Semantic Scholar for prior art matching a text query.

    Free (unauthenticated) tier. Never raises — returns [] on timeout,
    rate-limit exhaustion, or a malformed response. Papers without an
    abstract are dropped (nothing to judge against).
    """
    fields = "title,abstract,year,citationCount,externalIds,url"
    url = f"{S2_API_URL}/search?query={quote_plus(query)}&limit={limit}&fields={fields}"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    data: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await fetch_with_retry(client, url, headers=headers)
            data = resp.json()
        except Exception as e:
            logger.warning("Semantic Scholar search failed for %r: %s", query, e)
            return []
        finally:
            await asyncio.sleep(RATE_LIMIT_SECONDS)

    papers: list[dict[str, Any]] = []
    for item in data.get("data") or []:
        abstract = item.get("abstract")
        if not abstract:
            continue
        ext = item.get("externalIds") or {}
        papers.append(
            {
                "title": item.get("title") or "",
                "abstract": abstract,
                "year": item.get("year"),
                "citations": item.get("citationCount") or 0,
                "arxiv_id": ext.get("ArXiv", ""),
                "url": item.get("url") or "",
            }
        )
    return papers
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_semantic_scholar.py -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add src/lens/acquire/semantic_scholar.py tests/test_semantic_scholar.py
git commit -m "feat(acquire): add Semantic Scholar relevance search for prior art"
```

---

### Task 2: Novelty columns on `idea_cards`

**Files:**
- Modify: `src/lens/store/store.py` (idea_cards DDL in `_TABLE_DDL`; `_COLUMN_MIGRATIONS`; `JSON_FIELDS`)
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `idea_cards` gains `novelty_status TEXT DEFAULT 'unchecked'`, `prior_art TEXT DEFAULT '[]'` (JSON list), `novelty_note TEXT DEFAULT ''`, `novelty_checked_at TEXT`. Existing DBs get them via `ALTER TABLE`. `JSON_FIELDS["idea_cards"]` includes `"prior_art"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_store.py` (ensure `from datetime import UTC, datetime` and `import sqlite3` are imported at the top):

```python
def test_idea_cards_novelty_columns_roundtrip(tmp_path):
    from datetime import UTC, datetime

    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "t.db"))
    store.init_tables()
    store.add_rows(
        "idea_cards",
        [
            {
                "id": 1,
                "gap_id": 1,
                "report_id": 1,
                "title": "T",
                "pattern_ids": [],
                "hook": "",
                "mechanism": "m",
                "falsification": "",
                "differentiation": [],
                "signature_terms": ["quantization"],
                "paper_ids": [],
                "confidence": 0.5,
                "created_at": datetime.now(UTC),
                "taxonomy_version": 0,
                "prior_art": [{"title": "GQA", "url": "http://x", "year": 2023}],
                "novelty_status": "scooped",
                "novelty_note": "already published",
            }
        ],
    )
    c = store.query("idea_cards")[0]
    assert c["novelty_status"] == "scooped"
    assert c["prior_art"] == [{"title": "GQA", "url": "http://x", "year": 2023}]
    assert c["novelty_note"] == "already published"


def test_idea_cards_novelty_migration_on_old_table(tmp_path):
    """A pre-scoop-check idea_cards table gains the novelty columns on init."""
    from lens.store.store import LensStore

    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE idea_cards ("
        "id INTEGER PRIMARY KEY, gap_id INTEGER NOT NULL, report_id INTEGER NOT NULL, "
        "title TEXT NOT NULL, pattern_ids TEXT NOT NULL DEFAULT '[]', hook TEXT NOT NULL DEFAULT '', "
        "mechanism TEXT NOT NULL DEFAULT '', falsification TEXT NOT NULL DEFAULT '', "
        "differentiation TEXT NOT NULL DEFAULT '[]', signature_terms TEXT NOT NULL DEFAULT '[]', "
        "paper_ids TEXT NOT NULL DEFAULT '[]', confidence REAL NOT NULL DEFAULT 0.0, "
        "created_at TEXT NOT NULL, taxonomy_version INTEGER NOT NULL DEFAULT 0)"
    )
    conn.commit()
    conn.close()

    store = LensStore(db)
    store.init_tables()
    cols = {row[1] for row in store.conn.execute("PRAGMA table_info(idea_cards)")}
    assert {"novelty_status", "prior_art", "novelty_note", "novelty_checked_at"} <= cols
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py::test_idea_cards_novelty_columns_roundtrip tests/test_store.py::test_idea_cards_novelty_migration_on_old_table -v`
Expected: FAIL — roundtrip fails with `sqlite3.OperationalError: table idea_cards has no column named novelty_status`; migration fails the `<= cols` assertion.

- [ ] **Step 3: Add the columns and register the JSON field**

In `src/lens/store/store.py`:

1. Extend the `idea_cards` `CREATE TABLE` in `_TABLE_DDL` — change its tail from `taxonomy_version INTEGER NOT NULL DEFAULT 0\n    )` to:

```python
        taxonomy_version INTEGER NOT NULL DEFAULT 0,
        novelty_status TEXT NOT NULL DEFAULT 'unchecked',
        prior_art TEXT NOT NULL DEFAULT '[]',
        novelty_note TEXT NOT NULL DEFAULT '',
        novelty_checked_at TEXT
    )""",
```

2. Add to `_COLUMN_MIGRATIONS` (after the last entry):

```python
    ("idea_cards", "novelty_status", "TEXT NOT NULL DEFAULT 'unchecked'"),
    ("idea_cards", "prior_art", "TEXT NOT NULL DEFAULT '[]'"),
    ("idea_cards", "novelty_note", "TEXT NOT NULL DEFAULT ''"),
    ("idea_cards", "novelty_checked_at", "TEXT"),
```

3. Add `"prior_art"` to the `idea_cards` set in `JSON_FIELDS`:

```python
    "idea_cards": {"pattern_ids", "differentiation", "signature_terms", "paper_ids", "prior_art"},
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py::test_idea_cards_novelty_columns_roundtrip tests/test_store.py::test_idea_cards_novelty_migration_on_old_table -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add src/lens/store/store.py tests/test_store.py
git commit -m "feat(store): add novelty columns to idea_cards"
```

---

### Task 3: `judge_novelty` + `run_scoop_check`

**Files:**
- Create: `src/lens/knowledge/scoop_check.py`
- Test: `tests/test_scoop_check.py`

**Interfaces:**
- Consumes: `search_semantic_scholar` (Task 1); `idea_cards` novelty columns (Task 2); `strip_code_fences` (`lens.llm.utils`); `store.query`/`store.update`; `llm_client.complete(messages) -> str`.
- Produces:
  - `async judge_novelty(card: dict, prior_art: list[dict], llm_client) -> dict | None` → `{"verdict": "novel|overlaps|scooped", "colliding_papers": list[str], "rationale": str}` or `None`.
  - `async run_scoop_check(store, llm_client, limit: int | None = None, top_k: int = 5) -> dict` → `{"checked": int, "by_verdict": {"novel": int, "overlaps": int, "scooped": int}}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scoop_check.py`:

```python
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from lens.store.store import LensStore


def _seed_card(store, card_id, terms):
    store.add_rows(
        "idea_cards",
        [
            {
                "id": card_id,
                "gap_id": card_id,
                "report_id": 1,
                "title": f"Idea {card_id}",
                "pattern_ids": [],
                "hook": "",
                "mechanism": "some mechanism",
                "falsification": "",
                "differentiation": [],
                "signature_terms": terms,
                "paper_ids": [],
                "confidence": 0.5,
                "created_at": datetime.now(UTC),
                "taxonomy_version": 0,
            }
        ],
    )


@pytest.fixture
def store(tmp_path):
    s = LensStore(str(tmp_path / "t.db"))
    s.init_tables()
    return s


@pytest.mark.asyncio
async def test_judge_novelty_parses_verdict():
    from lens.knowledge.scoop_check import judge_novelty

    llm = AsyncMock()
    llm.complete.return_value = json.dumps(
        {"verdict": "scooped", "colliding_papers": ["GQA"], "rationale": "same idea"}
    )
    out = await judge_novelty(
        {"title": "t", "mechanism": "m", "differentiation": []},
        [{"title": "GQA", "abstract": "grouped query attention", "year": 2023}],
        llm,
    )
    assert out["verdict"] == "scooped"
    assert out["colliding_papers"] == ["GQA"]


@pytest.mark.asyncio
async def test_judge_novelty_rejects_bad_verdict():
    from lens.knowledge.scoop_check import judge_novelty

    llm = AsyncMock()
    llm.complete.return_value = json.dumps({"verdict": "maybe", "rationale": "x"})
    assert await judge_novelty({"title": "t", "mechanism": "m"}, [], llm) is None

    llm.complete.return_value = "not json at all {{{"
    assert await judge_novelty({"title": "t", "mechanism": "m"}, [], llm) is None


@pytest.mark.asyncio
async def test_run_scoop_check_annotates_and_is_idempotent(store, monkeypatch):
    import lens.knowledge.scoop_check as sc
    from lens.knowledge.scoop_check import run_scoop_check

    _seed_card(store, 1, ["quantization"])
    _seed_card(store, 2, ["attention"])

    async def fake_search(query, limit=5):
        return [{"title": "Prior", "abstract": "a", "year": 2023, "url": "http://p"}]

    monkeypatch.setattr(sc, "search_semantic_scholar", fake_search)

    llm = AsyncMock()
    llm.complete.return_value = json.dumps(
        {"verdict": "scooped", "colliding_papers": ["Prior"], "rationale": "match"}
    )

    summary = await run_scoop_check(store, llm)
    assert summary["checked"] == 2
    assert summary["by_verdict"]["scooped"] == 2
    cards = store.query("idea_cards")
    assert all(c["novelty_status"] == "scooped" for c in cards)
    assert all(c["prior_art"] == [{"title": "Prior", "url": "http://p", "year": 2023}] for c in cards)

    # Idempotent: nothing left unchecked -> second run checks 0.
    summary2 = await run_scoop_check(store, llm)
    assert summary2["checked"] == 0


@pytest.mark.asyncio
async def test_run_scoop_check_leaves_unchecked_on_empty_prior_art(store, monkeypatch):
    import lens.knowledge.scoop_check as sc
    from lens.knowledge.scoop_check import run_scoop_check

    _seed_card(store, 1, ["quantization"])

    async def empty_search(query, limit=5):
        return []

    monkeypatch.setattr(sc, "search_semantic_scholar", empty_search)
    llm = AsyncMock()

    summary = await run_scoop_check(store, llm)
    assert summary["checked"] == 0
    assert store.query("idea_cards")[0]["novelty_status"] == "unchecked"
    llm.complete.assert_not_awaited()  # no judge call when there's no prior art
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scoop_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lens.knowledge.scoop_check'`.

- [ ] **Step 3: Implement `scoop_check.py`**

Create `src/lens/knowledge/scoop_check.py`:

```python
"""Scoop-check: verify idea-card novelty against Semantic Scholar prior art."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from lens.acquire.semantic_scholar import search_semantic_scholar
from lens.llm.utils import strip_code_fences
from lens.store.store import LensStore

logger = logging.getLogger(__name__)

_VERDICTS = {"novel", "overlaps", "scooped"}

NOVELTY_SYSTEM_PROMPT = (
    "You are a research novelty auditor. Given a proposed research idea and a list "
    "of existing papers, decide whether the idea's CORE contribution is already "
    "covered. Distinguish shared keywords from the same contribution. Respond with "
    "a single JSON object and nothing else."
)


def _str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def _format_prior_art(prior_art: list[dict[str, Any]]) -> str:
    lines = []
    for i, p in enumerate(prior_art, 1):
        yr = f" ({p['year']})" if p.get("year") else ""
        lines.append(f"{i}. {p.get('title', '')}{yr}\n   {(p.get('abstract') or '')[:500]}")
    return "\n".join(lines) if lines else "(no prior art found)"


async def judge_novelty(
    card: dict[str, Any],
    prior_art: list[dict[str, Any]],
    llm_client: Any,
) -> dict[str, Any] | None:
    """Ask the LLM whether the card's core idea is already covered by prior art."""
    user = (
        "Proposed idea:\n"
        f"  Title: {card.get('title', '')}\n"
        f"  Mechanism: {card.get('mechanism', '')}\n"
        f"  Differentiation: {'; '.join(card.get('differentiation') or [])}\n\n"
        f"Existing papers:\n{_format_prior_art(prior_art)}\n\n"
        'Return JSON: {"verdict": "novel|overlaps|scooped", '
        '"colliding_papers": ["<paper title>", ...], "rationale": "<one sentence>"}\n'
        "verdict meanings: scooped = core idea already published; "
        "overlaps = substantial related work but a distinct angle; "
        "novel = no close prior art in the list."
    )
    messages = [
        {"role": "system", "content": NOVELTY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        text = await llm_client.complete(messages)
    except Exception:
        logger.warning("Novelty judge LLM call failed")
        return None

    text = strip_code_fences(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        from json_repair import repair_json

        try:
            data = repair_json(text, return_objects=True)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in _VERDICTS:
        return None
    return {
        "verdict": verdict,
        "colliding_papers": _str_list(data.get("colliding_papers")),
        "rationale": str(data.get("rationale", "")).strip(),
    }


async def run_scoop_check(
    store: LensStore,
    llm_client: Any,
    limit: int | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Novelty-check every idea card with novelty_status='unchecked'.

    Idempotent and fail-soft: a card only leaves 'unchecked' when it gets a
    real verdict; search/judge/DB failures leave it for the next run.
    """
    cards = store.query("idea_cards", "novelty_status = ?", ("unchecked",))
    cards = sorted(cards, key=lambda c: c["id"])
    if limit is not None:
        cards = cards[:limit]

    now = datetime.now(UTC).isoformat()
    counts = {"novel": 0, "overlaps": 0, "scooped": 0}
    checked = 0

    for card in cards:
        terms = card.get("signature_terms") or []
        query = " ".join([card.get("title", ""), *terms]).strip()
        prior_art = await search_semantic_scholar(query, limit=top_k)
        if not prior_art:
            logger.info("No prior art for card %d; leaving unchecked", card["id"])
            continue

        verdict = await judge_novelty(card, prior_art, llm_client)
        if verdict is None:
            logger.warning("Unusable novelty judgment for card %d; leaving unchecked", card["id"])
            continue

        stored_art = [
            {"title": p.get("title", ""), "url": p.get("url", ""), "year": p.get("year")}
            for p in prior_art
        ]
        try:
            store.update(
                "idea_cards",
                "novelty_status = ?, prior_art = ?, novelty_note = ?, novelty_checked_at = ?",
                "id = ?",
                (verdict["verdict"], json.dumps(stored_art), verdict["rationale"], now, card["id"]),
            )
        except Exception:
            logger.warning("Failed to persist novelty verdict for card %d", card["id"])
            continue
        counts[verdict["verdict"]] += 1
        checked += 1

    logger.info("Scoop-check: %d cards checked %s", checked, counts)
    return {"checked": checked, "by_verdict": counts}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scoop_check.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Commit**

```bash
git add src/lens/knowledge/scoop_check.py tests/test_scoop_check.py
git commit -m "feat(knowledge): scoop-check novelty judge and pass over idea_cards"
```

---

### Task 4: `lens scoop-check` CLI command

**Files:**
- Modify: `src/lens/cli.py` (new `@app.command()`)
- Test: `tests/test_cli_scoop_check.py`

**Interfaces:**
- Consumes: `run_scoop_check` (Task 3); `load_config`, `_get_config_path`, `_require_llm_config`, `_get_data_dir`, `_llm_kwargs`, `LLMClient`, `log_event` (all in `cli.py`).
- Produces: a `scoop-check` CLI command taking `--limit` and `--top-k`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_scoop_check.py`:

```python
from unittest.mock import patch

from typer.testing import CliRunner

from lens.cli import app

runner = CliRunner()


def test_scoop_check_command_invokes_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    async def fake_run(store, client, limit=None, top_k=5):
        return {"checked": 3, "by_verdict": {"novel": 1, "overlaps": 1, "scooped": 1}}

    # Avoid building a real LLM client / hitting the network.
    with (
        patch("lens.cli.LLMClient"),
        patch("lens.knowledge.scoop_check.run_scoop_check", side_effect=fake_run),
        patch("lens.cli._get_data_dir", return_value=tmp_path),
    ):
        result = runner.invoke(app, ["scoop-check", "--limit", "3"])

    assert result.exit_code == 0
    assert "3" in result.stdout  # checked count surfaced
    assert "scooped" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_scoop_check.py -v`
Expected: FAIL — `scoop-check` is not a registered command (Typer exits non-zero: "No such command").

- [ ] **Step 3: Add the CLI command**

In `src/lens/cli.py`, add a new command (place it after the `analyze`/`explain` commands):

```python
@app.command()
def scoop_check(
    limit: int | None = typer.Option(None, "--limit", help="Max cards to check this run."),
    top_k: int = typer.Option(5, "--top-k", help="Prior-art papers to retrieve per card."),
) -> None:
    """Verify idea-card novelty against Semantic Scholar prior art."""
    config = load_config(_get_config_path())
    _require_llm_config(config)
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.knowledge.scoop_check import run_scoop_check
    from lens.llm.client import LLMClient

    client = LLMClient(model=config["llm"]["default_model"], **_llm_kwargs(config))
    session_id = str(uuid4())[:8]

    summary = asyncio.run(run_scoop_check(store, client, limit=limit, top_k=top_k))

    rprint(f"\n[bold]Scoop-check:[/bold] {summary['checked']} card(s) checked")
    for verdict, n in summary["by_verdict"].items():
        rprint(f"  {verdict}: {n}")
    log_event(
        store,
        "scoop_check",
        "scoop_check.run",
        detail=summary,
        session_id=session_id,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_scoop_check.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lens/cli.py tests/test_cli_scoop_check.py
git commit -m "feat(cli): add lens scoop-check command"
```

---

### Task 5: Docs + full-suite verification

**Files:**
- Modify: `CHANGELOG.md`, `CLAUDE.md`, `pyproject.toml` (version), `uv.lock`

- [ ] **Step 1: Bump version and document**

In `pyproject.toml`, bump `version = "0.11.0"` → `version = "0.12.0"`. Add a `## 0.12.0 (2026-07-15)` `### Added` section to `CHANGELOG.md` describing scoop-check (Semantic Scholar prior-art search, LLM novelty judge, `novelty_status` on idea cards, `lens scoop-check`). Add a one-line **Scoop-check** bullet to `CLAUDE.md` near the Idea cards bullet.

- [ ] **Step 2: Run the full non-integration suite**

Run: `uv run pytest -m "not integration" -q`
Expected: PASS (previous 325 + new tests; the version bump auto-syncs `uv.lock` when hooks run — stage `uv.lock` if the `ty` hook modifies it).

- [ ] **Step 3: Type check + hooks**

Run: `prek run --all-files`
Expected: all hooks PASS. If the `ty`/`uv` hook regenerates `uv.lock` for the version bump, `git add uv.lock` and re-commit (never `--no-verify`).

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md CLAUDE.md pyproject.toml uv.lock
git commit -m "docs: bump to 0.12.0, changelog + docs for scoop-check"
```

---

## Self-Review

**Spec coverage:**
- `search_semantic_scholar` (free tier, fail-soft, abstract-drop) → Task 1. ✅
- Novelty columns + JSON_FIELDS + migration → Task 2. ✅
- `judge_novelty` (enum-validated, json_repair) + `run_scoop_check` (idempotent, fail-soft, `unchecked` on empty) → Task 3. ✅
- `lens scoop-check` CLI (`--limit`/`--top-k`, keyless, event log) → Task 4. ✅
- Version/changelog/docs → Task 5 (release hygiene, learned from 0.11.0). ✅
- Annotate-only, no monitor-cron wiring → honored (no monitor changes in any task). ✅

**Placeholder scan:** No TBD/TODO; every code step is complete.

**Type consistency:** `search_semantic_scholar` returns `{title, abstract, year, citations, arxiv_id, url}`; `run_scoop_check` reads `title/url/year` from those and stores `{title, url, year}` — matches the roundtrip test in Task 2. `judge_novelty` returns `{verdict, colliding_papers, rationale}`; `run_scoop_check` reads `verdict["verdict"]`/`["rationale"]` — consistent. `novelty_status` values are drawn from `_VERDICTS`, matching the `unchecked | novel | overlaps | scooped` enum in the columns.

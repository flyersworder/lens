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
    assert out is not None
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

    monkeypatch.setattr(sc, "search_openalex", fake_search)

    llm = AsyncMock()
    llm.complete.return_value = json.dumps(
        {"verdict": "scooped", "colliding_papers": ["Prior"], "rationale": "match"}
    )

    summary = await run_scoop_check(store, llm)
    assert summary["checked"] == 2
    assert summary["by_verdict"]["scooped"] == 2
    cards = store.query("idea_cards")
    assert all(c["novelty_status"] == "scooped" for c in cards)
    assert all(
        c["prior_art"] == [{"title": "Prior", "url": "http://p", "year": 2023}] for c in cards
    )

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

    monkeypatch.setattr(sc, "search_openalex", empty_search)
    llm = AsyncMock()

    summary = await run_scoop_check(store, llm)
    assert summary["checked"] == 0
    assert store.query("idea_cards")[0]["novelty_status"] == "unchecked"
    llm.complete.assert_not_awaited()  # no judge call when there's no prior art


@pytest.mark.asyncio
async def test_run_scoop_check_survives_judge_crash(store, monkeypatch):
    import lens.knowledge.scoop_check as sc
    from lens.knowledge.scoop_check import run_scoop_check

    _seed_card(store, 1, ["quantization"])

    async def fake_search(query, limit=5):
        return [{"title": "Prior", "abstract": "a", "year": 2023, "url": "http://p"}]

    monkeypatch.setattr(sc, "search_openalex", fake_search)

    llm = AsyncMock()
    llm.complete.return_value = None  # non-str -> strip_code_fences would crash

    summary = await run_scoop_check(store, llm)  # must not raise
    assert summary["checked"] == 0
    assert store.query("idea_cards")[0]["novelty_status"] == "unchecked"


@pytest.mark.asyncio
async def test_run_scoop_check_persists_colliding_papers(store, monkeypatch):
    import lens.knowledge.scoop_check as sc
    from lens.knowledge.scoop_check import run_scoop_check

    _seed_card(store, 1, ["quantization"])

    async def fake_search(query, limit=5):
        return [{"title": "KVQuant", "abstract": "a", "year": 2024, "url": "http://p"}]

    monkeypatch.setattr(sc, "search_openalex", fake_search)
    llm = AsyncMock()
    llm.complete.return_value = json.dumps(
        {"verdict": "scooped", "colliding_papers": ["KVQuant"], "rationale": "same idea"}
    )

    await run_scoop_check(store, llm)
    note = store.query("idea_cards")[0]["novelty_note"]
    assert "KVQuant" in note  # colliding paper surfaced in the persisted note


@pytest.mark.asyncio
async def test_run_scoop_check_skips_empty_query(store, monkeypatch):
    import lens.knowledge.scoop_check as sc
    from lens.knowledge.scoop_check import run_scoop_check

    # A card with no title and no signature_terms -> empty query.
    _seed_card(store, 1, [])
    store.update("idea_cards", "title = ?", "id = ?", ("", 1))

    searched = []

    async def fake_search(query, limit=5):
        searched.append(query)
        return [{"title": "X", "abstract": "a", "year": 2024, "url": "u"}]

    monkeypatch.setattr(sc, "search_openalex", fake_search)
    llm = AsyncMock()

    summary = await run_scoop_check(store, llm)
    assert searched == []  # no search call for the empty-query card
    assert summary["checked"] == 0
    assert store.query("idea_cards")[0]["novelty_status"] == "unchecked"


@pytest.mark.asyncio
async def test_gather_prior_art_searches_per_term_and_dedups(monkeypatch):
    import lens.knowledge.scoop_check as sc

    calls = []

    async def fake_search(query, limit=5):
        calls.append(query)
        return [
            {"title": f"Paper for {query}", "abstract": "a", "year": 2024, "url": "u"},
            {"title": "Shared Survey", "abstract": "s", "year": 2020, "url": "u2"},
        ]

    monkeypatch.setattr(sc, "search_openalex", fake_search)
    card = {"title": "T", "signature_terms": ["alpha", "beta", "gamma"]}
    art = await sc._gather_prior_art(card, per_term_limit=3, max_total=8)

    assert calls == ["alpha", "beta", "gamma"]  # one focused search per term
    titles = [p["title"] for p in art]
    assert titles.count("Shared Survey") == 1  # deduped across terms
    assert "Paper for alpha" in titles and "Paper for gamma" in titles


@pytest.mark.asyncio
async def test_gather_prior_art_falls_back_to_title(monkeypatch):
    import lens.knowledge.scoop_check as sc

    calls = []

    async def fake_search(query, limit=5):
        calls.append(query)
        return []

    monkeypatch.setattr(sc, "search_openalex", fake_search)
    await sc._gather_prior_art({"title": "Only Title", "signature_terms": []})
    assert calls == ["Only Title"]  # falls back to the title when no terms


@pytest.mark.asyncio
async def test_run_scoop_check_limit_skips_failing_cards(store, monkeypatch):
    """--limit caps CHECKED cards, so a persistently-failing low-id card does
    not consume the cap and starve higher-id cards (review finding)."""
    import lens.knowledge.scoop_check as sc
    from lens.knowledge.scoop_check import run_scoop_check

    _seed_card(store, 1, ["starve"])  # lowest id, always returns no prior art
    _seed_card(store, 2, ["quantization"])
    _seed_card(store, 3, ["attention"])

    async def fake_search(query, limit=5):
        if "starve" in query:
            return []
        return [{"title": "Prior", "abstract": "a", "year": 2023, "url": "u"}]

    monkeypatch.setattr(sc, "search_openalex", fake_search)
    llm = AsyncMock()
    llm.complete.return_value = json.dumps(
        {"verdict": "scooped", "colliding_papers": ["Prior"], "rationale": "m"}
    )

    summary = await run_scoop_check(store, llm, limit=1)
    assert summary["checked"] == 1  # the failing low-id card didn't consume the cap
    by_id = {c["id"]: c["novelty_status"] for c in store.query("idea_cards")}
    assert by_id[1] == "unchecked"  # failing card left for retry
    assert by_id[2] == "scooped"  # next card reached and checked despite limit=1
    assert by_id[3] == "unchecked"  # cap reached after one real check

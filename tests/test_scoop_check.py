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

    monkeypatch.setattr(sc, "search_semantic_scholar", empty_search)
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

    monkeypatch.setattr(sc, "search_semantic_scholar", fake_search)

    llm = AsyncMock()
    llm.complete.return_value = None  # non-str -> strip_code_fences would crash

    summary = await run_scoop_check(store, llm)  # must not raise
    assert summary["checked"] == 0
    assert store.query("idea_cards")[0]["novelty_status"] == "unchecked"

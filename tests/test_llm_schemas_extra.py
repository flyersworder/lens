"""Schemas for the non-extraction structured-output call sites."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lens.llm.structured import strict_schema


@pytest.fixture
def analysis_store_for_nomatch(tmp_path):
    """A store with seed vocabulary, so the classify enum is non-empty."""
    from lens.store.store import LensStore
    from lens.taxonomy.vocabulary import load_seed_vocabulary

    store = LensStore(str(tmp_path / "nomatch.db"))
    store.init_tables()
    load_seed_vocabulary(store)
    return store


def test_novelty_verdict_is_a_closed_enum():
    """The verdict is one of three values; strict mode can enforce that directly.

    Previously an out-of-range verdict was detected only after the call and
    silently discarded the whole response.
    """
    from lens.llm.schemas import NoveltyVerdict

    schema = strict_schema(NoveltyVerdict)
    verdict = schema["properties"]["verdict"]
    assert set(verdict["enum"]) == {"novel", "overlaps", "scooped"}

    with pytest.raises(ValidationError):
        NoveltyVerdict.model_validate(
            {"verdict": "maybe", "colliding_papers": [], "rationale": "x"}
        )


@pytest.mark.asyncio
async def test_judge_novelty_routes_through_schema_constrained_call():
    """judge_novelty asks for a NoveltyVerdict, not free text."""
    import json

    from conftest import make_llm_client

    from lens.knowledge.scoop_check import judge_novelty

    payload = json.dumps(
        {"verdict": "scooped", "colliding_papers": ["GQA"], "rationale": "same idea"}
    )
    client, fake = make_llm_client([payload], supports_schema=True)

    out = await judge_novelty(
        {"title": "t", "mechanism": "m", "differentiation": []},
        [{"title": "GQA", "abstract": "grouped query attention", "year": 2023}],
        client,
    )

    assert out is not None
    assert out["verdict"] == "scooped"
    assert out["colliding_papers"] == ["GQA"]
    assert fake.calls == [True]  # enforced schema, no fallback


@pytest.mark.asyncio
async def test_judge_novelty_still_fails_soft_on_unusable_response():
    """Fail-soft is load-bearing: a card must stay 'unchecked' unless a real
    verdict came back, so a persistently bad response yields None, not a raise."""
    from conftest import make_llm_client

    from lens.knowledge.scoop_check import judge_novelty

    client, _ = make_llm_client(["not json at all {{{", "still not json"])

    assert await judge_novelty({"title": "t", "mechanism": "m"}, [], client) is None


def test_choice_model_constrains_fields_to_runtime_options():
    """Vocabulary names are only known at runtime, so the enum is built per call.

    Without this the model can name a parameter that isn't in the corpus, which
    resolves to None and silently degrades the answer after a paid call.
    """
    from lens.llm.structured import choice_model

    Model = choice_model(
        "TradeoffClassification",
        improving=["Latency", "Accuracy"],
        worsening=["Latency", "Accuracy"],
    )

    schema = strict_schema(Model)
    assert set(schema["properties"]["improving"]["enum"]) == {"Latency", "Accuracy"}

    ok = Model.model_validate({"improving": "Latency", "worsening": "Accuracy"})
    assert ok.model_dump()["improving"] == "Latency"

    with pytest.raises(ValidationError):
        Model.model_validate({"improving": "Hallucinated", "worsening": "Accuracy"})


@pytest.mark.asyncio
async def test_candidate_selection_is_constrained_to_valid_indices():
    """The selector can only name a candidate that exists.

    Previously the reply was parsed with int() and clamped, so an out-of-range or
    non-numeric answer silently became candidate 1 after a paid call.
    """
    from conftest import make_llm_client

    from lens.serve.explainer import _select_candidate

    client, fake = make_llm_client(['{"choice": "2"}'], supports_schema=True)
    idx = await _select_candidate(client, "q", ["a", "b", "c"])
    assert idx == 1
    assert fake.calls == [True]

    # An index outside the candidate list never validates; fall back to the first.
    client, _ = make_llm_client(['{"choice": "9"}', '{"choice": "9"}'])
    assert await _select_candidate(client, "q", ["a", "b"]) == 0


@pytest.mark.asyncio
async def test_candidate_selection_skips_llm_for_a_single_candidate():
    """One candidate needs no call at all."""
    from conftest import make_llm_client

    from lens.serve.explainer import _select_candidate

    client, fake = make_llm_client([])
    assert await _select_candidate(client, "q", ["only"]) == 0
    assert fake.calls == []


def test_single_option_choice_emits_enum_not_const():
    """`const` is not in the documented strict-mode keyword set.

    Pydantic renders a one-value Literal as `const`, which risks a 400 from the
    very endpoints this is meant to constrain.
    """
    from lens.llm.structured import choice_model

    schema = strict_schema(choice_model("One", slot=["Attention Mechanism"]))
    prop = schema["properties"]["slot"]
    assert "const" not in prop
    assert prop["enum"] == ["Attention Mechanism"]


def test_schema_builder_does_not_eat_a_property_named_default():
    """`default` is stripped as a keyword, but a *property* called default is data.

    Removing it from `properties` while `required` still lists it produces a
    schema referencing a field that does not exist.
    """
    from pydantic import BaseModel

    class HasDefault(BaseModel):
        default: str
        other: str

    schema = strict_schema(HasDefault)
    assert set(schema["required"]) == set(schema["properties"])
    assert "default" in schema["properties"]


@pytest.mark.asyncio
async def test_analyze_can_still_answer_no_match(analysis_store_for_nomatch):
    """Pinning both fields to an enum removed the 'not a tradeoff' escape hatch.

    Without an explicit sentinel the model is forced to name two real parameters,
    so an off-topic query gets a confidently wrong tradeoff instead of nothing.
    """
    import json

    from conftest import make_llm_client

    from lens.serve.analyzer import _NO_MATCH, analyze

    payload = json.dumps({"improving": _NO_MATCH, "worsening": _NO_MATCH})
    client, _ = make_llm_client([payload], supports_schema=True)

    result = await analyze("what is the weather today", analysis_store_for_nomatch, client)

    assert result["improving"] is None
    assert result["worsening"] is None
    assert result["principles"] == []

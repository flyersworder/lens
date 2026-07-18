"""Tests for the ideation gap analysis pipeline."""

from unittest.mock import AsyncMock

import pytest

from lens.store.models import EMBEDDING_DIM
from lens.taxonomy.vocabulary import load_seed_vocabulary


def _make_embedding(values: list[float]) -> list[float]:
    """Make a full-dimension embedding from a short list (zero-padded)."""
    return values + [0.0] * (EMBEDDING_DIM - len(values))


@pytest.fixture
def ideation_store(tmp_path):
    from lens.knowledge.matrix import build_matrix
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    # load_seed_vocabulary also calls init_tables
    load_seed_vocabulary(store)

    # Insert embeddings for three vocabulary parameters so cross-pollination works.
    # inference-latency and throughput are similar (both speed-related):
    #   inference-latency: [1.0, 0.0, ...]
    #   model-accuracy:    [0.0, 1.0, ...]
    #   training-cost:     [0.9, 0.1, ...] — similar to inference-latency
    store.upsert_embedding("vocabulary", "inference-latency", _make_embedding([1.0, 0.0, 0.0]))
    store.upsert_embedding("vocabulary", "model-accuracy", _make_embedding([0.0, 1.0, 0.0]))
    store.upsert_embedding("vocabulary", "training-cost", _make_embedding([0.9, 0.1, 0.0]))

    # One matrix cell: inference-latency improves, model-accuracy worsens, quantization
    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p1",
                "improves": "Inference Latency",
                "worsens": "Model Accuracy",
                "technique": "Quantization",
                "context": "test",
                "confidence": 0.9,
                "evidence_quote": "test quote",
                "new_concepts": {},
            },
        ],
    )
    build_matrix(store)

    return store


def test_find_sparse_cells(ideation_store):
    from lens.monitor.ideation import find_sparse_cells

    gaps = find_sparse_cells(ideation_store, min_principles=2)
    assert len(gaps) >= 1
    assert any(
        g["improving_param_id"] == "inference-latency"
        and g["worsening_param_id"] == "model-accuracy"
        for g in gaps
    )


def test_find_sparse_cells_no_gaps(ideation_store):
    """With min_principles=0, no pair qualifies as sparse."""
    from lens.monitor.ideation import find_sparse_cells

    gaps = find_sparse_cells(ideation_store, min_principles=0)
    assert len(gaps) == 0


def test_find_sparse_cells_includes_zero_evidence(ideation_store):
    """Pairs with zero principles should be reported as gaps."""
    from lens.monitor.ideation import find_sparse_cells

    # Seed vocabulary has 12 parameters. With min_principles=1, only the one
    # cell that has 1 principle (inference-latency -> model-accuracy with quantization)
    # has enough, so all other directed pairs (12*11 - 1 = 131) have zero principles.
    gaps = find_sparse_cells(ideation_store, min_principles=1)
    assert len(gaps) == 12 * 11 - 1
    zero_gaps = [g for g in gaps if g["count"] == 0]
    assert len(zero_gaps) == 12 * 11 - 1


def test_find_cross_pollination(ideation_store):
    from lens.monitor.ideation import find_cross_pollination

    # training-cost has embedding [0.9, 0.1, ...] which is similar to
    # inference-latency [1.0, 0.0, ...]. The matrix cell has
    # inference-latency -> model-accuracy with quantization.
    # So quantization should be suggested for training-cost -> model-accuracy.
    candidates = find_cross_pollination(ideation_store, similarity_threshold=0.7)
    assert len(candidates) >= 1
    improving_ids = {c["improving_param_id"] for c in candidates}
    assert "training-cost" in improving_ids


def test_run_ideation(ideation_store):
    from lens.monitor.ideation import run_ideation

    report = run_ideation(ideation_store)
    assert report["gap_count"] >= 1
    assert len(report["gaps"]) >= 1

    gaps = ideation_store.query("ideation_gaps")
    assert len(gaps) >= 1

    reports = ideation_store.query("ideation_reports")
    assert len(reports) == 1


@pytest.mark.asyncio
async def test_run_ideation_with_llm_generates_cards(ideation_store):
    import json

    from lens.monitor.ideation import run_ideation_with_llm

    mock_client = AsyncMock()
    mock_client.complete.return_value = json.dumps(
        {
            "title": "Quantization-aware throughput scheduling",
            "patterns": ["Substitute the Operator or Representation"],
            "hook": "Swap the decode operator to trade accuracy for latency.",
            "mechanism": "Replace dense attention with a quantized kernel selected per layer.",
            "falsification": "Measure tokens/sec vs perplexity on WikiText-103; "
            "the quantized variant should raise throughput >20% at <1% perplexity loss.",
            "differentiation": ["Unlike static quantization, adapts per-layer at decode time"],
            "signature_terms": ["quantization", "throughput", "attention"],
            "confidence": 0.7,
        }
    )

    report = await run_ideation_with_llm(ideation_store, mock_client)

    cards = ideation_store.query("idea_cards")
    assert len(cards) >= 1
    assert report["idea_cards"]
    c = cards[0]
    assert c["title"] == "Quantization-aware throughput scheduling"
    assert c["pattern_ids"] == ["substitute-the-operator-or-representation"]
    assert c["signature_terms"] == ["quantization", "throughput", "attention"]
    assert 0.0 <= c["confidence"] <= 1.0

    # Back-compat: the originating gap's hypothesis is populated with the mechanism.
    gaps = ideation_store.query("ideation_gaps")
    assert any((g["llm_hypothesis"] or "").startswith("Replace dense attention") for g in gaps)


@pytest.mark.asyncio
async def test_run_ideation_with_llm_malformed_json_skips(ideation_store):
    from lens.monitor.ideation import run_ideation_with_llm

    mock_client = AsyncMock()
    mock_client.complete.return_value = "not json at all {{{"

    report = await run_ideation_with_llm(ideation_store, mock_client)

    # No card is written, but the run still succeeds and gaps still exist.
    assert ideation_store.query("idea_cards") == []
    assert report["idea_cards"] == []
    assert report["gap_count"] >= 1
    # Back-compat (Finding 4): unparseable prose is kept as the free-text
    # hypothesis rather than dropped entirely.
    gaps = ideation_store.query("ideation_gaps")
    assert any((g["llm_hypothesis"] or "") == "not json at all {{{" for g in gaps)


@pytest.mark.asyncio
async def test_run_ideation_with_llm_unknown_pattern_keeps_hypothesis_only(ideation_store):
    """A card whose patterns don't resolve to seeded ids is NOT persisted as a
    pattern-guided card (Finding 5); its mechanism is kept as the hypothesis."""
    import json

    from lens.monitor.ideation import run_ideation_with_llm

    mock_client = AsyncMock()
    mock_client.complete.return_value = json.dumps(
        {
            "title": "X",
            "patterns": None,  # unresolvable -> pattern_ids == []
            "hook": "h",
            "mechanism": "m",
            "falsification": "f",
            "differentiation": [],
            "signature_terms": [],
            "confidence": 0.4,
        }
    )

    report = await run_ideation_with_llm(ideation_store, mock_client)

    # No pattern-less card is written, and the run still completes gracefully.
    assert report["idea_cards"] == []
    assert ideation_store.query("idea_cards") == []
    # The mechanism is preserved as the gap's free-text hypothesis.
    gaps = ideation_store.query("ideation_gaps")
    assert any((g["llm_hypothesis"] or "") == "m" for g in gaps)


@pytest.mark.asyncio
async def test_run_ideation_with_llm_complete_raises_skips(ideation_store):
    from lens.monitor.ideation import run_ideation_with_llm

    mock_client = AsyncMock()
    mock_client.complete.side_effect = RuntimeError("boom")

    report = await run_ideation_with_llm(ideation_store, mock_client)

    assert ideation_store.query("idea_cards") == []
    assert report["idea_cards"] == []


@pytest.mark.asyncio
async def test_cross_pollination_card_uses_source_cell_provenance(ideation_store):
    """Cross-pollination cards should be attributed to the source cell that
    motivated the transfer, not the (empty) target matrix cell."""
    # Distinct card per call so the diversity gate keeps the cross-pollination
    # card instead of deduping it against an identical sparse-cell card. confidence
    # is fixed at 1.5 to exercise the clamp on every card.
    import itertools
    import json

    from lens.monitor.ideation import run_ideation_with_llm

    counter = itertools.count()

    def _distinct(*_a, **_k):
        i = next(counter)
        return json.dumps(
            {
                "title": f"Quantization-aware throughput scheduling {i}",
                "patterns": ["Substitute the Operator or Representation"],
                "hook": "Swap the decode operator to trade accuracy for latency.",
                "mechanism": "Replace dense attention with a quantized kernel selected per layer.",
                "falsification": "Measure tokens/sec vs perplexity on WikiText-103; "
                "the quantized variant should raise throughput >20% at <1% perplexity loss.",
                "differentiation": ["Unlike static quantization, adapts per-layer at decode time"],
                "signature_terms": [f"quantization{i}", f"throughput{i}", f"attention{i}"],
                "confidence": 1.5,
            }
        )

    mock_client = AsyncMock()
    mock_client.complete.side_effect = _distinct

    await run_ideation_with_llm(ideation_store, mock_client, max_cards=1000)

    gap_type_by_id = {g["id"]: g["gap_type"] for g in ideation_store.query("ideation_gaps")}
    cards = ideation_store.query("idea_cards")
    cross_pollination_cards = [
        c for c in cards if gap_type_by_id.get(c["gap_id"]) == "cross_pollination"
    ]

    assert cross_pollination_cards
    for card in cross_pollination_cards:
        assert card["paper_ids"] == ["p1"]

    # Finding 2: confidence is clamped to [0, 1] even when the LLM returns 1.5.
    assert all(card["confidence"] == 1.0 for card in cards)


def _valid_card_json(**overrides):
    import json

    card = {
        "title": "T",
        "patterns": ["Substitute the Operator or Representation"],
        "hook": "h",
        "mechanism": "m",
        "falsification": "f",
        "differentiation": [],
        "signature_terms": [],
        "confidence": 0.5,
    }
    card.update(overrides)
    return json.dumps(card)


@pytest.mark.asyncio
async def test_cross_pollination_provenance_excludes_other_principle_cells(ideation_store):
    """Provenance is the exact source cell, not every cell the principle resolves."""
    from lens.knowledge.matrix import build_matrix
    from lens.monitor.ideation import run_ideation_with_llm

    # A second, unrelated Quantization cell (Model Size vs Training Cost, paper p2).
    # Model Size has no embedding, so it yields no cross-pollination candidate; it
    # only widens the set of papers backing the Quantization principle.
    ideation_store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p2",
                "improves": "Model Size",
                "worsens": "Training Cost",
                "technique": "Quantization",
                "context": "test",
                "confidence": 0.9,
                "evidence_quote": "test quote",
                "new_concepts": {},
            },
        ],
    )
    build_matrix(ideation_store)

    # Distinct card per call so the cross-pollination card survives the diversity
    # gate rather than being deduped against an identical sparse-cell card.
    import itertools

    counter = itertools.count()
    mock_client = AsyncMock()
    mock_client.complete.side_effect = lambda *_a, **_k: _valid_card_json(
        title=f"T{(i := next(counter))}", signature_terms=[f"term{i}"]
    )

    await run_ideation_with_llm(ideation_store, mock_client, max_cards=1000)

    gap_type_by_id = {g["id"]: g["gap_type"] for g in ideation_store.query("ideation_gaps")}
    cross_cards = [
        c
        for c in ideation_store.query("idea_cards")
        if gap_type_by_id.get(c["gap_id"]) == "cross_pollination"
    ]
    assert cross_cards
    for card in cross_cards:
        # Source cell (inference-latency, model-accuracy, quantization) -> ["p1"].
        # p2's unrelated quantization cell must NOT leak into the provenance.
        assert card["paper_ids"] == ["p1"]


@pytest.mark.asyncio
async def test_seeds_ideation_patterns_when_missing(ideation_store):
    """A pre-0.11.0 DB with no ideation patterns is seeded on demand rather than
    silently disabling all enrichment (Finding 1)."""
    from lens.monitor.ideation import run_ideation_with_llm

    ideation_store.delete("vocabulary", "kind = ?", ("ideation_pattern",))
    assert ideation_store.query("vocabulary", "kind = ?", ("ideation_pattern",)) == []

    mock_client = AsyncMock()
    mock_client.complete.return_value = _valid_card_json()

    report = await run_ideation_with_llm(ideation_store, mock_client)

    assert len(ideation_store.query("vocabulary", "kind = ?", ("ideation_pattern",))) == 15
    assert report["idea_cards"]


@pytest.mark.asyncio
async def test_db_write_failure_skips_gap_gracefully(ideation_store, monkeypatch):
    """A DB error while persisting one card is logged and skipped, not fatal."""
    from lens.monitor.ideation import run_ideation_with_llm

    mock_client = AsyncMock()
    mock_client.complete.return_value = _valid_card_json()

    orig_add_rows = ideation_store.add_rows

    def failing_add_rows(table, rows, **kwargs):
        if table == "idea_cards":
            raise RuntimeError("disk full")
        return orig_add_rows(table, rows, **kwargs)

    monkeypatch.setattr(ideation_store, "add_rows", failing_add_rows)

    report = await run_ideation_with_llm(ideation_store, mock_client)
    assert report["idea_cards"] == []


@pytest.mark.asyncio
async def test_differentiation_dict_is_dropped(ideation_store):
    """A non-list differentiation (e.g. a dict) is dropped, not repr-stringified (Finding 6)."""
    from lens.monitor.ideation import run_ideation_with_llm

    mock_client = AsyncMock()
    mock_client.complete.return_value = _valid_card_json(
        differentiation={"vs_static": "adapts per layer"}
    )

    await run_ideation_with_llm(ideation_store, mock_client)
    cards = ideation_store.query("idea_cards")
    assert cards
    for card in cards:
        assert card["differentiation"] == []


# --- Diversity gate: pure helpers ---


def test_card_token_set_tokenizes_title_and_terms():
    from lens.monitor.ideation import _card_token_set

    tokens = _card_token_set(
        "Differentiable Architecture Search",
        ["Structural Sparsity", "Gumbel-Softmax"],
    )
    assert tokens == {
        "differentiable",
        "architecture",
        "search",
        "structural",
        "sparsity",
        "gumbel",
        "softmax",
    }


def test_jaccard_basic_and_empty():
    from lens.monitor.ideation import _jaccard

    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a", "b", "c", "d"}, {"a", "b"}) == 0.5
    assert _jaccard({"a"}, {"b"}) == 0.0
    # Empty union must not divide by zero.
    assert _jaccard(set(), set()) == 0.0


def test_diversified_gap_order_round_robins_improving_param():
    from lens.monitor.ideation import _diversified_gap_order

    # Three gaps on param "a", one on "b". Round-robin must surface "b" before
    # the first bucket is exhausted, and sort by score desc within a bucket.
    gaps = [
        {"id": 1, "related_params": ["a", "x"], "score": 0.5},
        {"id": 2, "related_params": ["a", "y"], "score": 0.9},
        {"id": 3, "related_params": ["a", "z"], "score": 0.7},
        {"id": 4, "related_params": ["b", "x"], "score": 0.4},
    ]
    ordered = [g["id"] for g in _diversified_gap_order(gaps)]
    # Pass 1: best of "a" (id 2, score 0.9), then best of "b" (id 4).
    assert ordered[:2] == [2, 4]
    # Remaining "a" gaps follow in score order.
    assert ordered[2:] == [3, 1]


def _counting_mock(fmt="term{i}"):
    """AsyncMock whose complete() returns a *distinct* card per call, so the
    diversity gate keeps every card (no dedup collapse)."""
    import itertools

    counter = itertools.count()

    def _next(*_args, **_kwargs):
        i = next(counter)
        return _valid_card_json(title=f"Idea {i}", signature_terms=[fmt.format(i=i)])

    mock = AsyncMock()
    mock.complete.side_effect = _next
    return mock


@pytest.mark.asyncio
async def test_dedup_collapses_identical_cards(ideation_store):
    """Many gaps that yield identical cards collapse to a single distinct card."""
    from lens.monitor.ideation import run_ideation_with_llm

    mock_client = AsyncMock()
    mock_client.complete.return_value = _valid_card_json(
        title="Same Idea", signature_terms=["quantization", "throughput"]
    )

    await run_ideation_with_llm(ideation_store, mock_client)
    assert len(ideation_store.query("idea_cards")) == 1


@pytest.mark.asyncio
async def test_max_cards_caps_output(ideation_store):
    """With distinct cards available, max_cards bounds how many are emitted."""
    from lens.monitor.ideation import run_ideation_with_llm

    await run_ideation_with_llm(ideation_store, _counting_mock(), max_cards=3)
    assert len(ideation_store.query("idea_cards")) == 3


@pytest.mark.asyncio
async def test_min_gap_score_filters_all_gaps(ideation_store):
    """A score floor above every gap's score emits no cards (and spends no LLM calls)."""
    from lens.monitor.ideation import run_ideation_with_llm

    mock = _counting_mock()
    await run_ideation_with_llm(ideation_store, mock, min_gap_score=2.0)
    assert ideation_store.query("idea_cards") == []
    mock.complete.assert_not_called()

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


@pytest.mark.asyncio
async def test_run_ideation_with_llm_null_patterns_does_not_crash(ideation_store):
    import json

    from lens.monitor.ideation import run_ideation_with_llm

    mock_client = AsyncMock()
    mock_client.complete.return_value = json.dumps(
        {
            "title": "X",
            "patterns": None,
            "hook": "h",
            "mechanism": "m",
            "falsification": "f",
            "differentiation": [],
            "signature_terms": [],
            "confidence": 0.4,
        }
    )

    report = await run_ideation_with_llm(ideation_store, mock_client)

    assert isinstance(report["idea_cards"], list)
    cards = ideation_store.query("idea_cards")
    assert len(cards) >= 1
    for card in cards:
        assert card["pattern_ids"] == []


@pytest.mark.asyncio
async def test_run_ideation_with_llm_complete_raises_skips(ideation_store):
    from lens.monitor.ideation import run_ideation_with_llm

    mock_client = AsyncMock()
    mock_client.complete.side_effect = RuntimeError("boom")

    report = await run_ideation_with_llm(ideation_store, mock_client)

    assert ideation_store.query("idea_cards") == []
    assert report["idea_cards"] == []


@pytest.mark.asyncio
async def test_cross_pollination_card_uses_principle_provenance(ideation_store):
    """Cross-pollination cards should be attributed to the papers backing the
    transferable principle, not the (empty) target matrix cell."""
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
            "confidence": 1.5,
        }
    )

    await run_ideation_with_llm(ideation_store, mock_client)

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

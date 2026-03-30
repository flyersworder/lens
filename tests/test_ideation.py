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
async def test_run_ideation_with_llm(ideation_store):
    from lens.monitor.ideation import run_ideation_with_llm

    mock_client = AsyncMock()
    mock_client.complete.return_value = (
        "This gap suggests that quantization techniques "
        "could be applied to throughput optimization."
    )

    report = await run_ideation_with_llm(
        ideation_store,
        mock_client,
    )
    assert report["gap_count"] >= 1
    gaps_with_hyp = [g for g in report["gaps"] if g.get("llm_hypothesis")]
    assert len(gaps_with_hyp) >= 1

"""Tests for the contradiction matrix builder (vocabulary-based)."""

from lens.knowledge.matrix import build_matrix, get_ranked_matrix
from lens.store.store import LensStore
from lens.taxonomy.vocabulary import load_seed_vocabulary


def _seed_store_with_extractions(store, extractions):
    """Helper to set up a store with vocabulary + extractions."""
    load_seed_vocabulary(store)
    if extractions:
        store.add_rows("tradeoff_extractions", extractions)


def test_build_matrix_basic(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    _seed_store_with_extractions(
        store,
        [
            {
                "paper_id": "p1",
                "improves": "Inference Latency",
                "worsens": "Model Accuracy",
                "technique": "Quantization",
                "context": "test",
                "confidence": 0.9,
                "evidence_quote": "quote",
                "new_concepts": {},
            },
            {
                "paper_id": "p2",
                "improves": "Inference Latency",
                "worsens": "Model Accuracy",
                "technique": "Quantization",
                "context": "test2",
                "confidence": 0.7,
                "evidence_quote": "quote2",
                "new_concepts": {},
            },
        ],
    )

    build_matrix(store)

    cells = store.query("matrix_cells")
    assert len(cells) == 1
    assert cells[0]["improving_param_id"] == "inference-latency"
    assert cells[0]["worsening_param_id"] == "model-accuracy"
    assert cells[0]["principle_id"] == "quantization"
    assert cells[0]["count"] == 2
    assert abs(cells[0]["avg_confidence"] - 0.8) < 0.01


def test_build_matrix_filters_low_confidence(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    _seed_store_with_extractions(
        store,
        [
            {
                "paper_id": "p1",
                "improves": "Inference Latency",
                "worsens": "Model Accuracy",
                "technique": "Quantization",
                "context": "test",
                "confidence": 0.3,  # Below 0.5 threshold
                "evidence_quote": "quote",
                "new_concepts": {},
            },
        ],
    )

    build_matrix(store)

    cells = store.query("matrix_cells")
    assert len(cells) == 0


def test_build_matrix_empty(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    load_seed_vocabulary(store)
    build_matrix(store)
    cells = store.query("matrix_cells")
    assert len(cells) == 0


def test_build_matrix_strips_new_prefix(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    load_seed_vocabulary(store)
    # First add the NEW: concept to vocabulary
    store.add_rows(
        "vocabulary",
        [
            {
                "id": "energy-efficiency",
                "name": "Energy Efficiency",
                "kind": "parameter",
                "description": "Power consumption",
                "source": "extracted",
                "first_seen": "2026-03-29",
                "paper_count": 0,
                "avg_confidence": 0.0,
            }
        ],
    )
    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p1",
                "improves": "NEW: Energy Efficiency",
                "worsens": "Model Accuracy",
                "technique": "Quantization",
                "context": "test",
                "confidence": 0.8,
                "evidence_quote": "quote",
                "new_concepts": {"Energy Efficiency": "Power consumption"},
            },
        ],
    )

    build_matrix(store)

    cells = store.query("matrix_cells")
    assert len(cells) == 1
    assert cells[0]["improving_param_id"] == "energy-efficiency"


def test_get_ranked_matrix(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    _seed_store_with_extractions(
        store,
        [
            {
                "paper_id": "p1",
                "improves": "Inference Latency",
                "worsens": "Model Accuracy",
                "technique": "Quantization",
                "context": "test",
                "confidence": 0.9,
                "evidence_quote": "quote",
                "new_concepts": {},
            },
            {
                "paper_id": "p2",
                "improves": "Inference Latency",
                "worsens": "Model Accuracy",
                "technique": "Knowledge Distillation",
                "context": "test2",
                "confidence": 0.7,
                "evidence_quote": "quote2",
                "new_concepts": {},
            },
        ],
    )

    build_matrix(store)
    result = get_ranked_matrix(store, top_k=1)

    assert len(result) == 1  # top_k=1 per pair
    assert result[0]["improving_param_id"] == "inference-latency"

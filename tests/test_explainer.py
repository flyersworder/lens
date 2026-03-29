"""Tests for the explain functionality."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from lens.store.models import EMBEDDING_DIM


@pytest.fixture
def explain_store(tmp_path):
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "inference-latency",
                "name": "Inference Latency",
                "kind": "parameter",
                "description": "Speed of inference",
                "source": "seed",
                "first_seen": "2026-01-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
                "embedding": [1.0] + [0.0] * (EMBEDDING_DIM - 1),
            },
        ],
    )
    store.add_rows(
        "vocabulary",
        [
            {
                "id": "quantization",
                "name": "Quantization",
                "kind": "principle",
                "description": "Reduce precision",
                "source": "seed",
                "first_seen": "2026-01-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
                "embedding": [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2),
            },
        ],
    )
    store.add_rows(
        "matrix_cells",
        [
            {
                "improving_param_id": "inference-latency",
                "worsening_param_id": "model-accuracy",
                "principle_id": "quantization",
                "count": 3,
                "avg_confidence": 0.85,
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
            },
        ],
    )
    store.add_rows(
        "taxonomy_versions",
        [
            {
                "version_id": 1,
                "created_at": "2026-03-21T00:00:00",
                "paper_count": 10,
                "param_count": 1,
                "principle_count": 1,
                "slot_count": 0,
                "variant_count": 0,
                "pattern_count": 0,
            },
        ],
    )
    return store


def test_resolve_concept(explain_store):
    from lens.serve.explainer import resolve_concept

    with patch("lens.serve.explainer.embed_strings") as mock_embed:
        mock_embed.return_value = np.array([[1.0] + [0.0] * (EMBEDDING_DIM - 1)])
        result = resolve_concept(
            query="inference latency",
            store=explain_store,
        )
    assert result is not None
    assert result["resolved_name"] == "Inference Latency"
    assert result["resolved_type"] == "parameter"


def test_resolve_concept_principle(explain_store):
    from lens.serve.explainer import resolve_concept

    with patch("lens.serve.explainer.embed_strings") as mock_embed:
        mock_embed.return_value = np.array([[0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)])
        result = resolve_concept(
            query="quantization",
            store=explain_store,
        )
    assert result is not None
    assert result["resolved_name"] == "Quantization"
    assert result["resolved_type"] == "principle"


def test_graph_walk(explain_store):
    from lens.serve.explainer import graph_walk

    walk = graph_walk(
        resolved_type="parameter",
        resolved_id="inference-latency",
        store=explain_store,
    )
    assert "identity" in walk
    assert walk["identity"]["name"] == "Inference Latency"
    assert "tradeoffs" in walk


@pytest.mark.asyncio
async def test_explain_full(explain_store):
    from lens.serve.explainer import explain

    mock_client = AsyncMock()
    mock_client.complete.return_value = (
        "Inference Latency refers to the speed at which an LLM generates output tokens."
    )

    with patch("lens.serve.explainer.embed_strings") as mock_embed:
        mock_embed.return_value = np.array([[1.0] + [0.0] * (EMBEDDING_DIM - 1)])
        result = await explain(
            query="inference latency",
            store=explain_store,
            llm_client=mock_client,
        )
    assert result is not None
    assert result.resolved_name == "Inference Latency"
    assert result.resolved_type == "parameter"
    assert len(result.narrative) > 0
    assert isinstance(result.tradeoffs, list)
    assert isinstance(result.connections, list)
    assert isinstance(result.paper_refs, list)
    assert isinstance(result.alternatives, list)
    assert isinstance(result.evolution, list)

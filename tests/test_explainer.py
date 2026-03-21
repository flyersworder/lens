"""Tests for the explain functionality."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest


@pytest.fixture
def explain_store(tmp_path):
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    store.add_rows(
        "parameters",
        [
            {
                "id": 1,
                "name": "Inference Latency",
                "description": "Speed of inference",
                "raw_strings": ["latency"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [1.0] + [0.0] * 767,
            },
        ],
    )
    store.add_rows(
        "principles",
        [
            {
                "id": 50001,
                "name": "Quantization",
                "description": "Reduce precision",
                "sub_techniques": ["int8", "int4"],
                "raw_strings": ["quantization"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.0, 1.0] + [0.0] * 766,
            },
        ],
    )
    store.add_rows(
        "matrix_cells",
        [
            {
                "improving_param_id": 1,
                "worsening_param_id": 2,
                "principle_id": 50001,
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
            },
        ],
    )
    return store


def test_resolve_concept(explain_store):
    from lens.serve.explainer import resolve_concept

    with patch("lens.serve.explainer.embed_strings") as mock_embed:
        mock_embed.return_value = np.array([[1.0] + [0.0] * 767])
        result = resolve_concept(
            query="inference latency",
            store=explain_store,
            taxonomy_version=1,
        )
    assert result is not None
    assert result["resolved_name"] == "Inference Latency"
    assert result["resolved_type"] == "parameter"


def test_resolve_concept_principle(explain_store):
    from lens.serve.explainer import resolve_concept

    with patch("lens.serve.explainer.embed_strings") as mock_embed:
        mock_embed.return_value = np.array([[0.0, 1.0] + [0.0] * 766])
        result = resolve_concept(
            query="quantization",
            store=explain_store,
            taxonomy_version=1,
        )
    assert result is not None
    assert result["resolved_name"] == "Quantization"
    assert result["resolved_type"] == "principle"


def test_graph_walk(explain_store):
    from lens.serve.explainer import graph_walk

    walk = graph_walk(
        resolved_type="parameter",
        resolved_id=1,
        store=explain_store,
        taxonomy_version=1,
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
        mock_embed.return_value = np.array([[1.0] + [0.0] * 767])
        result = await explain(
            query="inference latency",
            store=explain_store,
            llm_client=mock_client,
            taxonomy_version=1,
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

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
            {
                "id": "quantization",
                "name": "Quantization",
                "kind": "principle",
                "description": "Reduce precision of weights",
                "source": "seed",
                "first_seen": "2026-01-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
                "embedding": [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2),
            },
            {
                "id": "quantization-method",
                "name": "Quantization Method",
                "kind": "arch_slot",
                "description": "Techniques for reducing numerical precision",
                "source": "seed",
                "first_seen": "2026-01-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
                "embedding": [0.05, 0.99] + [0.0] * (EMBEDDING_DIM - 2),
            },
            {
                "id": "attention-mechanism",
                "name": "Attention Mechanism",
                "kind": "arch_slot",
                "description": "How the model attends to input",
                "source": "seed",
                "first_seen": "2026-01-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
                "embedding": [0.0, 0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 3),
            },
            {
                "id": "reasoning",
                "name": "Reasoning",
                "kind": "agentic_category",
                "description": "Multi-step logical inference",
                "source": "seed",
                "first_seen": "2026-01-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
                "embedding": [0.0, 0.0, 0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 4),
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
                "taxonomy_version": 0,
            },
        ],
    )
    store.add_rows(
        "architecture_extractions",
        [
            {
                "paper_id": "p1",
                "component_slot": "Attention Mechanism",
                "variant_name": "FlashAttention-2",
                "replaces": "FlashAttention",
                "key_properties": "better parallelism",
                "confidence": 0.9,
                "new_concepts": {},
            },
            {
                "paper_id": "p2",
                "component_slot": "Attention Mechanism",
                "variant_name": "GQA",
                "replaces": None,
                "key_properties": "fewer KV heads",
                "confidence": 0.85,
                "new_concepts": {},
            },
        ],
    )
    store.add_rows(
        "agentic_extractions",
        [
            {
                "paper_id": "p1",
                "pattern_name": "ReAct",
                "category": "Reasoning",
                "structure": "interleaves reasoning and acting",
                "use_case": "multi-step QA",
                "components": ["LLM", "tools"],
                "confidence": 0.9,
                "new_concepts": {},
            },
        ],
    )
    store.rebuild_vocabulary_fts()
    return store


def test_find_candidates(explain_store):
    from lens.serve.explainer import find_candidates

    with patch("lens.serve.explainer.embed_strings") as mock_embed:
        mock_embed.return_value = np.array([[1.0] + [0.0] * (EMBEDDING_DIM - 1)])
        candidates = find_candidates(query="inference latency", store=explain_store)
    assert len(candidates) >= 1
    names = {c["name"] for c in candidates}
    assert "Inference Latency" in names


def test_find_candidates_returns_multiple(explain_store):
    """Searching for 'quantization' should return both the principle and arch_slot."""
    from lens.serve.explainer import find_candidates

    with patch("lens.serve.explainer.embed_strings") as mock_embed:
        mock_embed.return_value = np.array([[0.02, 1.0] + [0.0] * (EMBEDDING_DIM - 2)])
        candidates = find_candidates(query="quantization", store=explain_store, top_k=3)
    names = {c["name"] for c in candidates}
    assert "Quantization" in names
    assert "Quantization Method" in names


def test_graph_walk_parameter(explain_store):
    from lens.serve.explainer import graph_walk

    walk = graph_walk(
        resolved_type="parameter", resolved_id="inference-latency", store=explain_store
    )
    assert walk["identity"]["name"] == "Inference Latency"
    assert walk["identity"]["type"] == "parameter"
    assert len(walk["tradeoffs"]) == 1
    assert len(walk["connections"]) > 0


def test_graph_walk_arch_slot(explain_store):
    from lens.serve.explainer import graph_walk

    walk = graph_walk(
        resolved_type="arch_slot", resolved_id="attention-mechanism", store=explain_store
    )
    assert walk["identity"]["name"] == "Attention Mechanism"
    assert walk["identity"]["type"] == "arch_slot"
    assert len(walk["variants"]) == 2
    names = {v["variant_name"] for v in walk["variants"]}
    assert "FlashAttention-2" in names
    assert "GQA" in names


def test_graph_walk_agentic_category(explain_store):
    from lens.serve.explainer import graph_walk

    walk = graph_walk(
        resolved_type="agentic_category", resolved_id="reasoning", store=explain_store
    )
    assert walk["identity"]["name"] == "Reasoning"
    assert walk["identity"]["type"] == "agentic_category"
    assert len(walk["patterns"]) == 1
    assert walk["patterns"][0]["pattern_name"] == "ReAct"


@pytest.mark.asyncio
async def test_explain_with_llm_selection(explain_store):
    """The LLM selects the best candidate and explains it."""
    from lens.serve.explainer import explain

    mock_client = AsyncMock()
    # First call: LLM selection returns "1" (first candidate)
    # Second call: LLM synthesis returns the explanation
    mock_client.complete.side_effect = [
        "1",
        "Inference Latency is the time taken to generate output tokens.",
    ]

    with patch("lens.serve.explainer.embed_strings") as mock_embed:
        mock_embed.return_value = np.array([[1.0] + [0.0] * (EMBEDDING_DIM - 1)])
        result = await explain(
            query="inference latency",
            store=explain_store,
            llm_client=mock_client,
        )
    assert result is not None
    assert "Latency" in result.narrative or "time" in result.narrative
    assert len(result.alternatives) >= 0


@pytest.mark.asyncio
async def test_explain_arch_slot(explain_store):
    """Explain correctly handles arch_slot concepts."""
    from lens.serve.explainer import explain

    mock_client = AsyncMock()
    mock_client.complete.side_effect = [
        "1",
        "The Attention Mechanism determines how the model weighs input tokens.",
    ]

    with patch("lens.serve.explainer.embed_strings") as mock_embed:
        mock_embed.return_value = np.array([[0.0, 0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 3)])
        result = await explain(
            query="attention mechanism",
            store=explain_store,
            llm_client=mock_client,
        )
    assert result is not None
    assert "Attention" in result.narrative

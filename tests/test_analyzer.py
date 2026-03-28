"""Tests for the analyze functionality."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from lens.store.models import EMBEDDING_DIM


@pytest.fixture
def analysis_store(tmp_path):
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "parameters",
        [
            {
                "id": 1,
                "name": "Inference Latency",
                "description": "Speed",
                "raw_strings": ["latency", "inference speed"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "id": 2,
                "name": "Model Accuracy",
                "description": "Quality",
                "raw_strings": ["accuracy", "model quality"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
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
                "sub_techniques": ["int8"],
                "raw_strings": ["quantization"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "id": 50002,
                "name": "Distillation",
                "description": "Compress model",
                "sub_techniques": ["kd"],
                "raw_strings": ["distillation"],
                "paper_ids": ["p2"],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
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
                "count": 5,
                "avg_confidence": 0.9,
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
            },
            {
                "improving_param_id": 1,
                "worsening_param_id": 2,
                "principle_id": 50002,
                "count": 3,
                "avg_confidence": 0.8,
                "paper_ids": ["p2"],
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
                "param_count": 2,
                "principle_count": 2,
                "slot_count": 0,
                "variant_count": 0,
                "pattern_count": 0,
            },
        ],
    )
    return store


@pytest.mark.asyncio
async def test_analyze_tradeoff(analysis_store):
    from lens.serve.analyzer import analyze

    mock_client = AsyncMock()
    mock_client.complete.return_value = (
        '{"improving": "Inference Latency", "worsening": "Model Accuracy"}'
    )

    result = await analyze(
        query="reduce latency without hurting accuracy",
        store=analysis_store,
        llm_client=mock_client,
        taxonomy_version=1,
    )
    assert result is not None
    assert len(result["principles"]) >= 1
    assert result["principles"][0]["name"] in ["Quantization", "Distillation"]


@pytest.mark.asyncio
async def test_analyze_no_match(analysis_store):
    from lens.serve.analyzer import analyze

    mock_client = AsyncMock()
    mock_client.complete.return_value = (
        '{"improving": "Unknown Param", "worsening": "Other Param"}'
    )

    result = await analyze(
        query="something with no matching parameters",
        store=analysis_store,
        llm_client=mock_client,
        taxonomy_version=1,
    )
    assert result is not None
    assert len(result["principles"]) == 0


@pytest.fixture
def arch_agentic_store(tmp_path):
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test2.db"))
    store.init_tables()

    store.add_rows(
        "taxonomy_versions",
        [
            {
                "version_id": 1,
                "created_at": "2026-03-21T00:00:00",
                "paper_count": 5,
                "param_count": 0,
                "principle_count": 0,
                "slot_count": 2,
                "variant_count": 2,
                "pattern_count": 2,
            }
        ],
    )

    store.add_rows(
        "architecture_slots",
        [
            {
                "id": 1,
                "name": "Attention Mechanism",
                "description": "How tokens attend to each other",
                "taxonomy_version": 1,
            },
            {
                "id": 2,
                "name": "Feed-Forward Network",
                "description": "MLP layers in the transformer block",
                "taxonomy_version": 1,
            },
        ],
    )

    store.add_rows(
        "architecture_variants",
        [
            {
                "id": 101,
                "slot_id": 1,
                "name": "Multi-Head Attention",
                "replaces": [],
                "properties": "Standard scaled dot-product attention with multiple heads",
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.1] * 768,
            },
            {
                "id": 102,
                "slot_id": 2,
                "name": "Mixture of Experts",
                "replaces": [],
                "properties": "Sparse gating for conditional computation",
                "paper_ids": ["p2"],
                "taxonomy_version": 1,
                "embedding": [0.2] * 768,
            },
        ],
    )

    store.add_rows(
        "agentic_patterns",
        [
            {
                "id": 201,
                "name": "ReAct",
                "category": "reasoning",
                "description": "Reason and act in interleaved steps",
                "components": ["reasoner", "actor", "memory"],
                "use_cases": ["tool use", "question answering"],
                "paper_ids": ["p3"],
                "taxonomy_version": 1,
                "embedding": [0.3] * 768,
            },
            {
                "id": 202,
                "name": "Chain of Thought",
                "category": "reasoning",
                "description": "Step-by-step reasoning before answering",
                "components": ["reasoning chain"],
                "use_cases": ["math", "logic"],
                "paper_ids": ["p4"],
                "taxonomy_version": 1,
                "embedding": [0.4] * 768,
            },
        ],
    )

    return store


@pytest.mark.asyncio
async def test_analyze_architecture(arch_agentic_store):
    from lens.serve.analyzer import analyze_architecture

    mock_client = AsyncMock()
    mock_client.complete.return_value = '{"slot": "Attention Mechanism"}'

    fake_embedding = np.array([[0.1] * 768])
    with patch("lens.serve.analyzer.embed_strings", return_value=fake_embedding):
        result = await analyze_architecture(
            query="How does attention work in transformers?",
            store=arch_agentic_store,
            llm_client=mock_client,
            taxonomy_version=1,
        )

    assert result is not None
    assert result["query"] == "How does attention work in transformers?"
    assert "slot" in result
    assert "variants" in result
    assert isinstance(result["variants"], list)
    assert len(result["variants"]) >= 1
    first = result["variants"][0]
    assert "name" in first
    assert "properties" in first


@pytest.mark.asyncio
async def test_analyze_architecture_llm_failure(arch_agentic_store):
    """When LLM fails to identify a slot, all variants are returned."""
    from lens.serve.analyzer import analyze_architecture

    mock_client = AsyncMock()
    mock_client.complete.side_effect = Exception("LLM error")

    fake_embedding = np.array([[0.1] * 768])
    with patch("lens.serve.analyzer.embed_strings", return_value=fake_embedding):
        result = await analyze_architecture(
            query="transformer architecture overview",
            store=arch_agentic_store,
            llm_client=mock_client,
            taxonomy_version=1,
        )

    assert result is not None
    assert result["slot"] is None
    assert isinstance(result["variants"], list)


@pytest.mark.asyncio
async def test_analyze_agentic(arch_agentic_store):
    from lens.serve.analyzer import analyze_agentic

    mock_client = AsyncMock()

    fake_embedding = np.array([[0.3] * 768])
    with patch("lens.serve.analyzer.embed_strings", return_value=fake_embedding):
        result = await analyze_agentic(
            query="step-by-step reasoning for complex tasks",
            store=arch_agentic_store,
            llm_client=mock_client,
            taxonomy_version=1,
        )

    assert result is not None
    assert result["query"] == "step-by-step reasoning for complex tasks"
    assert "patterns" in result
    assert isinstance(result["patterns"], list)
    assert len(result["patterns"]) >= 1
    first = result["patterns"][0]
    assert "name" in first
    assert "components" in first
    assert "use_cases" in first

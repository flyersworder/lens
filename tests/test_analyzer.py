"""Tests for the analyze functionality."""

from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def analysis_store(tmp_path):
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
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
                "embedding": [0.0] * 768,
            },
            {
                "id": 2,
                "name": "Model Accuracy",
                "description": "Quality",
                "raw_strings": ["accuracy", "model quality"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.0] * 768,
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
                "embedding": [0.0] * 768,
            },
            {
                "id": 50002,
                "name": "Distillation",
                "description": "Compress model",
                "sub_techniques": ["kd"],
                "raw_strings": ["distillation"],
                "paper_ids": ["p2"],
                "taxonomy_version": 1,
                "embedding": [0.0] * 768,
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

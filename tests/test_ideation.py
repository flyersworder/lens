"""Tests for the ideation gap analysis pipeline."""

from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def ideation_store(tmp_path):
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    store.add_rows(
        "parameters",
        [
            {
                "id": 1,
                "name": "Latency",
                "description": "Speed",
                "raw_strings": ["latency"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [1.0, 0.0, 0.0] + [0.0] * 765,
            },
            {
                "id": 2,
                "name": "Accuracy",
                "description": "Quality",
                "raw_strings": ["accuracy"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.0, 1.0, 0.0] + [0.0] * 765,
            },
            {
                "id": 3,
                "name": "Throughput",
                "description": "Speed variant",
                "raw_strings": ["throughput"],
                "paper_ids": ["p2"],
                "taxonomy_version": 1,
                "embedding": [0.9, 0.1, 0.0] + [0.0] * 765,
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
        ],
    )
    store.add_rows(
        "taxonomy_versions",
        [
            {
                "version_id": 1,
                "created_at": "2026-03-21T00:00:00",
                "paper_count": 10,
                "param_count": 3,
                "principle_count": 1,
            },
        ],
    )
    return store


def test_find_sparse_cells(ideation_store):
    from lens.monitor.ideation import find_sparse_cells

    gaps = find_sparse_cells(ideation_store, taxonomy_version=1, min_principles=2)
    assert len(gaps) >= 1
    assert any(g["improving_param_id"] == 1 and g["worsening_param_id"] == 2 for g in gaps)


def test_find_sparse_cells_no_gaps(ideation_store):
    from lens.monitor.ideation import find_sparse_cells

    gaps = find_sparse_cells(ideation_store, taxonomy_version=1, min_principles=1)
    assert len(gaps) == 0


def test_find_cross_pollination(ideation_store):
    from lens.monitor.ideation import find_cross_pollination

    candidates = find_cross_pollination(
        ideation_store, taxonomy_version=1, similarity_threshold=0.7
    )
    assert len(candidates) >= 1


def test_run_ideation(ideation_store):
    from lens.monitor.ideation import run_ideation

    report = run_ideation(ideation_store, taxonomy_version=1)
    assert report["gap_count"] >= 1
    assert len(report["gaps"]) >= 1

    gaps_df = ideation_store.get_table("ideation_gaps").to_polars()
    assert len(gaps_df) >= 1

    reports_df = ideation_store.get_table("ideation_reports").to_polars()
    assert len(reports_df) == 1


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
        taxonomy_version=1,
    )
    assert report["gap_count"] >= 1
    gaps_with_hyp = [g for g in report["gaps"] if g.get("llm_hypothesis")]
    assert len(gaps_with_hyp) >= 1

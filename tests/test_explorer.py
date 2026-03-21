"""Tests for the explore functionality."""

import pytest


@pytest.fixture
def populated_store(tmp_path):
    """Store with taxonomy and matrix data for exploration."""
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
                "raw_strings": ["latency"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.0] * 768,
            },
            {
                "id": 2,
                "name": "Model Accuracy",
                "description": "Quality",
                "raw_strings": ["accuracy"],
                "paper_ids": ["p1", "p2"],
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
                "sub_techniques": ["int8", "int4"],
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
                "avg_confidence": 0.85,
                "paper_ids": ["p1", "p2"],
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
                "principle_count": 1,
                "slot_count": 0,
                "variant_count": 0,
                "pattern_count": 0,
            },
        ],
    )
    return store


def test_list_parameters(populated_store):
    from lens.serve.explorer import list_parameters

    params = list_parameters(populated_store, taxonomy_version=1)
    assert len(params) == 2
    names = {p["name"] for p in params}
    assert "Inference Latency" in names


def test_list_principles(populated_store):
    from lens.serve.explorer import list_principles

    principles = list_principles(populated_store, taxonomy_version=1)
    assert len(principles) == 1
    assert principles[0]["name"] == "Quantization"


def test_get_matrix_cell(populated_store):
    from lens.serve.explorer import get_matrix_cell

    cell = get_matrix_cell(populated_store, 1, 2, taxonomy_version=1)
    assert cell is not None
    assert len(cell) >= 1
    assert cell[0]["principle_id"] == 50001


def test_get_matrix_cell_not_found(populated_store):
    from lens.serve.explorer import get_matrix_cell

    cell = get_matrix_cell(populated_store, 99, 99, taxonomy_version=1)
    assert len(cell) == 0


def test_get_paper(populated_store):
    from lens.serve.explorer import get_paper

    populated_store.add_papers(
        [
            {
                "paper_id": "p1",
                "arxiv_id": "p1",
                "title": "Test Paper",
                "abstract": "Abstract",
                "authors": ["Author"],
                "date": "2024-01-01",
                "venue": None,
                "citations": 0,
                "quality_score": 0.0,
                "extraction_status": "complete",
                "embedding": [0.0] * 768,
            }
        ]
    )
    paper = get_paper(populated_store, "p1")
    assert paper is not None
    assert paper["title"] == "Test Paper"


def test_get_paper_not_found(populated_store):
    from lens.serve.explorer import get_paper

    paper = get_paper(populated_store, "nonexistent")
    assert paper is None


def test_list_matrix_overview(populated_store):
    from lens.serve.explorer import list_matrix_overview

    overview = list_matrix_overview(populated_store, taxonomy_version=1)
    assert len(overview) >= 1

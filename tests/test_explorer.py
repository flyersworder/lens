"""Tests for the explore functionality."""

import pytest

from lens.store.models import EMBEDDING_DIM


@pytest.fixture
def populated_store(tmp_path):
    """Store with taxonomy and matrix data for exploration."""
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
                "raw_strings": ["latency"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "id": 2,
                "name": "Model Accuracy",
                "description": "Quality",
                "raw_strings": ["accuracy"],
                "paper_ids": ["p1", "p2"],
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
                "sub_techniques": ["int8", "int4"],
                "raw_strings": ["quantization"],
                "paper_ids": ["p1"],
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
                "embedding": [0.0] * EMBEDDING_DIM,
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


@pytest.fixture
def arch_store(tmp_path):
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    store.add_rows(
        "architecture_slots",
        [
            {
                "id": 1,
                "name": "Attention",
                "description": "Attention mechanism",
                "taxonomy_version": 1,
            },
            {
                "id": 2,
                "name": "Positional Encoding",
                "description": "Position info",
                "taxonomy_version": 1,
            },
        ],
    )
    store.add_rows(
        "architecture_variants",
        [
            {
                "id": 10,
                "slot_id": 1,
                "name": "Multi-Head Attention",
                "replaces": [],
                "properties": "parallel heads",
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "id": 11,
                "slot_id": 1,
                "name": "Grouped-Query Attention",
                "replaces": [10],
                "properties": "shared KV cache",
                "paper_ids": ["p2"],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "id": 12,
                "slot_id": 2,
                "name": "RoPE",
                "replaces": [],
                "properties": "relative position",
                "paper_ids": ["p3"],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
        ],
    )
    store.add_rows(
        "agentic_patterns",
        [
            {
                "id": 20,
                "name": "ReAct",
                "category": "Reasoning",
                "description": "Reasoning and acting",
                "components": ["LLM", "tools"],
                "use_cases": ["tool use"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "id": 21,
                "name": "Reflexion",
                "category": "Reflection",
                "description": "Self-critique loop",
                "components": ["actor", "evaluator"],
                "use_cases": ["code generation"],
                "paper_ids": ["p2"],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            },
        ],
    )
    store.add_rows(
        "papers",
        [
            {
                "paper_id": "p1",
                "title": "T1",
                "abstract": "A1",
                "authors": [],
                "date": "2017-06-01",
                "arxiv_id": "p1",
                "citations": 0,
                "quality_score": 0.0,
                "extraction_status": "complete",
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "paper_id": "p2",
                "title": "T2",
                "abstract": "A2",
                "authors": [],
                "date": "2023-05-01",
                "arxiv_id": "p2",
                "citations": 0,
                "quality_score": 0.0,
                "extraction_status": "complete",
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "paper_id": "p3",
                "title": "T3",
                "abstract": "A3",
                "authors": [],
                "date": "2021-04-01",
                "arxiv_id": "p3",
                "citations": 0,
                "quality_score": 0.0,
                "extraction_status": "complete",
                "embedding": [0.0] * EMBEDDING_DIM,
            },
        ],
    )
    store.add_rows(
        "taxonomy_versions",
        [
            {
                "version_id": 1,
                "created_at": "2026-03-21T00:00:00",
                "paper_count": 3,
                "param_count": 0,
                "principle_count": 0,
                "slot_count": 2,
                "variant_count": 3,
                "pattern_count": 2,
            },
        ],
    )
    return store


def test_list_architecture_slots(arch_store):
    from lens.serve.explorer import list_architecture_slots

    slots = list_architecture_slots(arch_store, taxonomy_version=1)
    assert len(slots) == 2
    attn = next(s for s in slots if s["name"] == "Attention")
    assert attn["variant_count"] == 2


def test_list_architecture_variants(arch_store):
    from lens.serve.explorer import list_architecture_variants

    variants = list_architecture_variants(arch_store, slot_name="Attention", taxonomy_version=1)
    assert len(variants) == 2
    names = {v["name"] for v in variants}
    assert "Multi-Head Attention" in names
    assert "Grouped-Query Attention" in names


def test_list_agentic_patterns(arch_store):
    from lens.serve.explorer import list_agentic_patterns

    patterns = list_agentic_patterns(arch_store, taxonomy_version=1)
    assert len(patterns) == 2


def test_list_agentic_patterns_by_category(arch_store):
    from lens.serve.explorer import list_agentic_patterns

    patterns = list_agentic_patterns(arch_store, taxonomy_version=1, category="Reasoning")
    assert len(patterns) == 1
    assert patterns[0]["name"] == "ReAct"


def test_get_architecture_timeline(arch_store):
    from lens.serve.explorer import get_architecture_timeline

    timeline = get_architecture_timeline(arch_store, slot_name="Attention", taxonomy_version=1)
    assert len(timeline) == 2
    # p1 (2017) should come before p2 (2023)
    assert timeline[0]["name"] == "Multi-Head Attention"
    assert timeline[1]["name"] == "Grouped-Query Attention"
    assert "earliest_date" in timeline[0]

"""Tests for the explore functionality."""

import pytest

from lens.store.models import EMBEDDING_DIM
from lens.taxonomy.vocabulary import load_seed_vocabulary


@pytest.fixture
def populated_store(tmp_path):
    """Store with vocabulary and matrix data for exploration.

    Returns a tuple (store, ids_dict) where ids_dict contains test IDs.
    """
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    load_seed_vocabulary(store)

    # Get two parameter IDs from the loaded vocabulary
    params = store.query("vocabulary", "kind = ?", ("parameter",))
    princs = store.query("vocabulary", "kind = ?", ("principle",))
    assert len(params) >= 2, "Need at least 2 parameters in seed vocabulary"
    assert len(princs) >= 1, "Need at least 1 principle in seed vocabulary"

    param_id_1 = params[0]["id"]
    param_id_2 = params[1]["id"]
    princ_id = princs[0]["id"]

    store.add_rows(
        "matrix_cells",
        [
            {
                "improving_param_id": param_id_1,
                "worsening_param_id": param_id_2,
                "principle_id": princ_id,
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
                "param_count": len(params),
                "principle_count": len(princs),
                "slot_count": 0,
                "variant_count": 0,
                "pattern_count": 0,
            },
        ],
    )
    ids = {"param_id_1": param_id_1, "param_id_2": param_id_2, "princ_id": princ_id}
    return store, ids


def test_list_parameters(populated_store):
    from lens.serve.explorer import list_parameters

    store, _ = populated_store
    params = list_parameters(store)
    assert len(params) >= 2
    names = {p["name"] for p in params}
    # Seed vocabulary contains parameters like "Inference Latency"
    assert any("Latency" in n or "Throughput" in n or "Accuracy" in n for n in names)


def test_list_principles(populated_store):
    from lens.serve.explorer import list_principles

    store, _ = populated_store
    principles = list_principles(store)
    assert len(principles) >= 1


def test_get_matrix_cell(populated_store):
    from lens.serve.explorer import get_matrix_cell

    store, ids = populated_store
    param_id_1 = ids["param_id_1"]
    param_id_2 = ids["param_id_2"]
    princ_id = ids["princ_id"]

    cell = get_matrix_cell(store, param_id_1, param_id_2)
    assert cell is not None
    assert len(cell) >= 1
    assert cell[0]["principle_id"] == princ_id


def test_get_matrix_cell_not_found(populated_store):
    from lens.serve.explorer import get_matrix_cell

    store, _ = populated_store
    cell = get_matrix_cell(store, "nonexistent-param", "also-nonexistent")
    assert len(cell) == 0


def test_get_paper(populated_store):
    from lens.serve.explorer import get_paper

    store, _ = populated_store
    store.add_papers(
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
    paper = get_paper(store, "p1")
    assert paper is not None
    assert paper["title"] == "Test Paper"


def test_get_paper_not_found(populated_store):
    from lens.serve.explorer import get_paper

    store, _ = populated_store
    paper = get_paper(store, "nonexistent")
    assert paper is None


def test_list_matrix_overview(populated_store):
    from lens.serve.explorer import list_matrix_overview

    store, _ = populated_store
    overview = list_matrix_overview(store)
    assert len(overview) >= 1


@pytest.fixture
def arch_store(tmp_path):
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    load_seed_vocabulary(store)

    # Insert architecture extractions using canonical slot names from vocabulary
    store.add_rows(
        "architecture_extractions",
        [
            {
                "paper_id": "p1",
                "component_slot": "Attention Mechanism",
                "variant_name": "Multi-Head Attention",
                "replaces": None,
                "key_properties": "parallel heads",
                "confidence": 0.9,
            },
            {
                "paper_id": "p2",
                "component_slot": "Attention Mechanism",
                "variant_name": "Grouped-Query Attention",
                "replaces": "Multi-Head Attention",
                "key_properties": "shared KV cache",
                "confidence": 0.85,
            },
            {
                "paper_id": "p3",
                "component_slot": "Positional Encoding",
                "variant_name": "RoPE",
                "replaces": None,
                "key_properties": "relative position",
                "confidence": 0.9,
            },
        ],
    )

    # Insert agentic extractions using canonical category names from vocabulary
    store.add_rows(
        "agentic_extractions",
        [
            {
                "paper_id": "p1",
                "pattern_name": "ReAct",
                "category": "Reasoning",
                "structure": "interleaved reasoning and acting",
                "use_case": "tool use",
                "components": ["LLM", "tools"],
                "confidence": 0.9,
            },
            {
                "paper_id": "p2",
                "pattern_name": "Reflexion",
                "category": "Self-Reflection",
                "structure": "self-critique loop",
                "use_case": "code generation",
                "components": ["actor", "evaluator"],
                "confidence": 0.85,
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
    return store


def test_list_architecture_slots(arch_store):
    from lens.serve.explorer import list_architecture_slots

    slots = list_architecture_slots(arch_store)
    # Should have all arch_slot entries from seed vocabulary
    assert len(slots) >= 2
    attn = next(s for s in slots if s["name"] == "Attention Mechanism")
    assert attn["variant_count"] == 2
    pos = next(s for s in slots if s["name"] == "Positional Encoding")
    assert pos["variant_count"] == 1


def test_list_architecture_variants(arch_store):
    from lens.serve.explorer import list_architecture_variants

    variants = list_architecture_variants(arch_store, slot_name="Attention Mechanism")
    assert len(variants) == 2
    names = {v["variant_name"] for v in variants}
    assert "Multi-Head Attention" in names
    assert "Grouped-Query Attention" in names


def test_list_agentic_patterns(arch_store):
    from lens.serve.explorer import list_agentic_patterns

    patterns = list_agentic_patterns(arch_store)
    assert len(patterns) == 2


def test_list_agentic_patterns_by_category(arch_store):
    from lens.serve.explorer import list_agentic_patterns

    patterns = list_agentic_patterns(arch_store, category="Reasoning")
    assert len(patterns) == 1
    assert patterns[0]["pattern_name"] == "ReAct"


def test_get_architecture_timeline(arch_store):
    from lens.serve.explorer import get_architecture_timeline

    timeline = get_architecture_timeline(arch_store, slot_name="Attention Mechanism")
    assert len(timeline) == 2
    # p1 (2017) should come before p2 (2023)
    assert timeline[0]["variant_name"] == "Multi-Head Attention"
    assert timeline[1]["variant_name"] == "Grouped-Query Attention"
    assert "earliest_date" in timeline[0]

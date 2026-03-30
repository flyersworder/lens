"""Tests for the analyze functionality."""

from unittest.mock import AsyncMock

import pytest

from lens.taxonomy.vocabulary import load_seed_vocabulary


@pytest.fixture
def analysis_store(tmp_path):
    """Returns (store, names_dict) where names_dict has latency/accuracy/quant/distil names."""
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    load_seed_vocabulary(store)

    # Get two parameter IDs and one principle ID from vocabulary
    params = store.query("vocabulary", "kind = ?", ("parameter",))
    princs = store.query("vocabulary", "kind = ?", ("principle",))
    assert len(params) >= 2
    assert len(princs) >= 2

    # Find by exact name or fall back to first entries
    param_map = {p["name"]: p["id"] for p in params}
    princ_map = {p["name"]: p["id"] for p in princs}

    latency_id = param_map.get("Inference Latency", params[0]["id"])
    accuracy_id = param_map.get("Model Accuracy", params[1]["id"])
    quant_id = princ_map.get("Quantization", princs[0]["id"])
    distil_id = princ_map.get("Knowledge Distillation", princs[1]["id"])

    latency_name = next(p["name"] for p in params if p["id"] == latency_id)
    accuracy_name = next(p["name"] for p in params if p["id"] == accuracy_id)
    quant_name = next(p["name"] for p in princs if p["id"] == quant_id)
    distil_name = next(p["name"] for p in princs if p["id"] == distil_id)

    store.add_rows(
        "matrix_cells",
        [
            {
                "improving_param_id": latency_id,
                "worsening_param_id": accuracy_id,
                "principle_id": quant_id,
                "count": 5,
                "avg_confidence": 0.9,
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
            },
            {
                "improving_param_id": latency_id,
                "worsening_param_id": accuracy_id,
                "principle_id": distil_id,
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
                "param_count": len(params),
                "principle_count": len(princs),
                "slot_count": 0,
                "variant_count": 0,
                "pattern_count": 0,
            },
        ],
    )

    names = {
        "latency": latency_name,
        "accuracy": accuracy_name,
        "quant": quant_name,
        "distil": distil_name,
    }
    return store, names


@pytest.mark.asyncio
async def test_analyze_tradeoff(analysis_store):
    from lens.serve.analyzer import analyze

    store, names = analysis_store
    latency_name = names["latency"]
    accuracy_name = names["accuracy"]
    quant_name = names["quant"]
    distil_name = names["distil"]

    mock_client = AsyncMock()
    mock_client.complete.return_value = (
        f'{{"improving": "{latency_name}", "worsening": "{accuracy_name}"}}'
    )

    result = await analyze(
        query="reduce latency without hurting accuracy",
        store=store,
        llm_client=mock_client,
    )
    assert result is not None
    assert len(result["principles"]) >= 1
    assert result["principles"][0]["name"] in [quant_name, distil_name]


@pytest.mark.asyncio
async def test_analyze_no_match(analysis_store):
    from lens.serve.analyzer import analyze

    store, _ = analysis_store
    mock_client = AsyncMock()
    mock_client.complete.return_value = (
        '{"improving": "Unknown Param", "worsening": "Other Param"}'
    )

    result = await analyze(
        query="something with no matching parameters",
        store=store,
        llm_client=mock_client,
    )
    assert result is not None
    assert len(result["principles"]) == 0


@pytest.fixture
def arch_agentic_store(tmp_path):
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test2.db"))
    store.init_tables()

    load_seed_vocabulary(store)

    # Insert architecture extractions with canonical slot names
    store.add_rows(
        "architecture_extractions",
        [
            {
                "paper_id": "p1",
                "component_slot": "Attention Mechanism",
                "variant_name": "Multi-Head Attention",
                "replaces": None,
                "key_properties": "Standard scaled dot-product attention with multiple heads",
                "confidence": 0.9,
            },
            {
                "paper_id": "p2",
                "component_slot": "FFN",
                "variant_name": "Mixture of Experts",
                "replaces": None,
                "key_properties": "Sparse gating for conditional computation",
                "confidence": 0.85,
            },
        ],
    )

    # Insert agentic extractions with canonical category names
    store.add_rows(
        "agentic_extractions",
        [
            {
                "paper_id": "p3",
                "pattern_name": "ReAct",
                "category": "Reasoning",
                "structure": "interleaved reasoning and acting",
                "use_case": "tool use, question answering",
                "components": ["reasoner", "actor", "memory"],
                "confidence": 0.9,
            },
            {
                "paper_id": "p4",
                "pattern_name": "Chain of Thought",
                "category": "Reasoning",
                "structure": "step-by-step reasoning chain",
                "use_case": "math, logic",
                "components": ["reasoning chain"],
                "confidence": 0.85,
            },
        ],
    )

    return store


@pytest.mark.asyncio
async def test_analyze_architecture(arch_agentic_store):
    from lens.serve.analyzer import analyze_architecture

    mock_client = AsyncMock()
    mock_client.complete.return_value = '{"slot": "Attention Mechanism"}'

    result = await analyze_architecture(
        query="How does attention work in transformers?",
        store=arch_agentic_store,
        llm_client=mock_client,
    )

    assert result is not None
    assert result["query"] == "How does attention work in transformers?"
    assert result["slot"] == "Attention Mechanism"
    assert "variants" in result
    assert isinstance(result["variants"], list)
    assert len(result["variants"]) == 1
    first = result["variants"][0]
    assert first["variant_name"] == "Multi-Head Attention"
    assert "properties" in first


@pytest.mark.asyncio
async def test_analyze_architecture_llm_failure(arch_agentic_store):
    """When LLM fails to identify a slot, all variants are returned."""
    from lens.serve.analyzer import analyze_architecture

    mock_client = AsyncMock()
    mock_client.complete.side_effect = Exception("LLM error")

    result = await analyze_architecture(
        query="transformer architecture overview",
        store=arch_agentic_store,
        llm_client=mock_client,
    )

    assert result is not None
    assert result["slot"] is None
    assert isinstance(result["variants"], list)
    # All extractions returned when no slot identified
    assert len(result["variants"]) == 2


@pytest.mark.asyncio
async def test_analyze_agentic(arch_agentic_store):
    from lens.serve.analyzer import analyze_agentic

    mock_client = AsyncMock()
    mock_client.complete.return_value = '{"category": "Reasoning"}'

    result = await analyze_agentic(
        query="step-by-step reasoning for complex tasks",
        store=arch_agentic_store,
        llm_client=mock_client,
    )

    assert result is not None
    assert result["query"] == "step-by-step reasoning for complex tasks"
    assert "patterns" in result
    assert isinstance(result["patterns"], list)
    assert len(result["patterns"]) == 2
    names = {p["pattern_name"] for p in result["patterns"]}
    assert "ReAct" in names
    assert "Chain of Thought" in names
    first = result["patterns"][0]
    assert "components" in first
    assert "use_case" in first

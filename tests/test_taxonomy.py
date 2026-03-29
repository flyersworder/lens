"""Tests for the taxonomy pipeline."""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from lens.store.models import EMBEDDING_DIM


def test_embed_strings_returns_array():
    from lens.taxonomy.embedder import embed_strings

    embeddings = embed_strings(["inference latency", "model accuracy"])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0


def test_embed_strings_empty():
    from lens.taxonomy.embedder import embed_strings

    embeddings = embed_strings([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 0


def test_embed_strings_deterministic():
    from lens.taxonomy.embedder import embed_strings

    e1 = embed_strings(["test string"])
    e2 = embed_strings(["test string"])
    np.testing.assert_array_almost_equal(e1, e2)


def test_cluster_embeddings():
    from lens.taxonomy.clusterer import cluster_embeddings

    rng = np.random.RandomState(42)
    cluster_a = rng.randn(10, 50) + np.array([5.0] + [0.0] * 49)
    cluster_b = rng.randn(10, 50) + np.array([0.0, 5.0] + [0.0] * 48)
    embeddings = np.vstack([cluster_a, cluster_b])

    labels = cluster_embeddings(embeddings, min_cluster_size=3)
    assert len(labels) == 20
    unique = set(labels)
    unique.discard(-1)
    assert len(unique) >= 2


def test_cluster_embeddings_small_dataset():
    from lens.taxonomy.clusterer import cluster_embeddings

    embeddings = np.random.randn(5, 50)
    labels = cluster_embeddings(embeddings, min_cluster_size=2)
    assert len(labels) == 5


def test_cluster_embeddings_fallback_to_kmeans():
    from lens.taxonomy.clusterer import cluster_embeddings

    # All identical points — HDBSCAN assigns all to noise, should fallback
    embeddings = np.ones((20, 50))
    labels = cluster_embeddings(embeddings, min_cluster_size=3, target_clusters=3)
    assert len(labels) == 20
    unique = set(labels)
    unique.discard(-1)
    assert len(unique) >= 1


@pytest.mark.asyncio
async def test_label_clusters():
    from lens.taxonomy.labeler import label_clusters

    clusters = {
        0: ["inference latency", "inference speed", "generation time"],
        1: ["model accuracy", "benchmark performance", "task accuracy"],
    }
    mock_client = AsyncMock()
    mock_client.complete.side_effect = [
        '{"name": "Inference Latency", "description": "Speed of generating output tokens"}',
        '{"name": "Model Accuracy", "description": "Performance on evaluation benchmarks"}',
    ]

    labels = await label_clusters(clusters, mock_client)
    assert len(labels) == 2
    assert labels[0]["name"] == "Inference Latency"
    assert labels[1]["name"] == "Model Accuracy"
    assert "description" in labels[0]


@pytest.mark.asyncio
async def test_label_clusters_handles_malformed():
    from lens.taxonomy.labeler import label_clusters

    clusters = {0: ["test string"]}
    mock_client = AsyncMock()
    mock_client.complete.return_value = "not json"

    labels = await label_clusters(clusters, mock_client)
    assert len(labels) == 1
    assert labels[0]["name"] is not None


def test_build_tradeoff_taxonomy(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy import build_tradeoff_taxonomy
    from lens.taxonomy.vocabulary import load_seed_vocabulary

    store = LensStore(str(tmp_path / "test.db"))
    load_seed_vocabulary(store)

    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p1",
                "improves": "Inference Latency",
                "worsens": "Model Accuracy",
                "technique": "NEW: Pruning",
                "context": "test",
                "confidence": 0.85,
                "evidence_quote": "quote",
                "new_concept_description": "Removing unnecessary model weights",
            },
        ],
    )

    stats = build_tradeoff_taxonomy(store)
    assert stats["new_entries"] == 1

    rows = store.query("vocabulary", "id = ?", ("pruning",))
    assert len(rows) == 1
    assert rows[0]["source"] == "extracted"
    assert rows[0]["paper_count"] == 1


def test_get_next_version(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy.versioning import get_next_version

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    assert get_next_version(store) == 1


def test_next_id_empty_table(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy import _next_id

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    assert _next_id(store, "parameters") == 1


def test_next_id_with_existing_data(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy import _next_id

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    store.add_rows(
        "parameters",
        [
            {
                "id": 42,
                "name": "Test",
                "description": "d",
                "raw_strings": ["t"],
                "paper_ids": ["p1"],
                "taxonomy_version": 1,
                "embedding": [0.0] * EMBEDDING_DIM,
            }
        ],
    )
    assert _next_id(store, "parameters") == 43


@pytest.mark.asyncio
async def test_normalize_slots():
    import json

    from lens.taxonomy.labeler import normalize_slots

    raw_strings = ["attention mechanism", "self-attention", "positional encoding", "pos embedding"]
    mock_client = AsyncMock()
    mock_client.complete.return_value = json.dumps(
        {
            "attention mechanism": "Attention",
            "self-attention": "Attention",
            "positional encoding": "Positional Encoding",
            "pos embedding": "Positional Encoding",
        }
    )
    mapping = await normalize_slots(raw_strings, mock_client)
    assert mapping["attention mechanism"] == "Attention"
    assert mapping["self-attention"] == "Attention"
    assert mapping["positional encoding"] == "Positional Encoding"


@pytest.mark.asyncio
async def test_normalize_slots_malformed_fallback():
    from lens.taxonomy.labeler import normalize_slots

    raw_strings = ["attention mechanism"]
    mock_client = AsyncMock()
    mock_client.complete.return_value = "not json"
    mapping = await normalize_slots(raw_strings, mock_client)
    assert mapping["attention mechanism"] == "Attention Mechanism"


@pytest.mark.asyncio
async def test_summarize_variant_properties_single():
    from lens.taxonomy.labeler import summarize_variant_properties

    mock_client = AsyncMock()
    result = await summarize_variant_properties(
        ["Uses sparse attention"], "Sparse Transformer", mock_client
    )
    assert result == "Uses sparse attention"
    mock_client.complete.assert_not_called()


@pytest.mark.asyncio
async def test_summarize_variant_properties_multiple():
    from lens.taxonomy.labeler import summarize_variant_properties

    expected = (
        "Combines sparse attention with local context windows"
        " for efficient long-sequence modeling."
    )
    mock_client = AsyncMock()
    mock_client.complete.return_value = expected
    props = ["uses sparse attention", "local context windows", "efficient for long sequences"]
    result = await summarize_variant_properties(props, "Sparse Transformer", mock_client)
    assert result == expected


@pytest.mark.asyncio
async def test_summarize_variant_properties_fallback():
    from lens.taxonomy.labeler import summarize_variant_properties

    mock_client = AsyncMock()
    mock_client.complete.side_effect = Exception("LLM error")
    props = ["prop a", "prop b", "prop a"]
    result = await summarize_variant_properties(props, "Some Variant", mock_client)
    # Fallback: join unique properties
    assert "prop a" in result
    assert "prop b" in result


@pytest.mark.asyncio
async def test_label_clusters_with_category():
    import json

    from lens.taxonomy.labeler import label_clusters_with_category

    clusters = {0: ["ReAct", "react pattern", "reasoning and acting"]}
    structures = {0: ["LLM agent with tool use and reasoning loop"]}
    mock_client = AsyncMock()
    mock_client.complete.return_value = json.dumps(
        {
            "name": "ReAct",
            "description": "Reasoning and acting pattern for tool-using agents",
            "category": "Reasoning",
        }
    )
    labels = await label_clusters_with_category(clusters, structures, mock_client)
    assert labels[0]["name"] == "ReAct"
    assert labels[0]["category"] == "Reasoning"
    assert "description" in labels[0]


@pytest.mark.asyncio
async def test_build_architecture_taxonomy(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy import build_architecture_taxonomy

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    store.add_rows(
        "architecture_extractions",
        [
            {
                "paper_id": "p1",
                "component_slot": "attention mechanism",
                "variant_name": "multi-head attention",
                "replaces": None,
                "key_properties": "parallel heads",
                "confidence": 0.9,
            },
            {
                "paper_id": "p2",
                "component_slot": "attention mechanism",
                "variant_name": "grouped-query attention",
                "replaces": "multi-head attention",
                "key_properties": "shared KV cache",
                "confidence": 0.85,
            },
            {
                "paper_id": "p3",
                "component_slot": "positional encoding",
                "variant_name": "RoPE",
                "replaces": None,
                "key_properties": "relative position",
                "confidence": 0.9,
            },
        ],
    )
    mock_client = AsyncMock()
    mock_client.complete.return_value = '{"name": "Test", "description": "test"}'
    result = await build_architecture_taxonomy(
        store, mock_client, min_cluster_size=2, version_id=1
    )
    assert "slot_entries" in result
    assert "variant_entries" in result
    slots = store.query("architecture_slots")
    assert len(slots) >= 1
    variants = store.query("architecture_variants")
    assert len(variants) >= 1


@pytest.mark.asyncio
async def test_build_agentic_taxonomy(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy import build_agentic_taxonomy

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    store.add_rows(
        "agentic_extractions",
        [
            {
                "paper_id": "p1",
                "pattern_name": "ReAct",
                "structure": "reasoning and acting loop",
                "use_case": "tool use",
                "components": ["LLM", "tools", "memory"],
                "confidence": 0.9,
            },
            {
                "paper_id": "p2",
                "pattern_name": "Reflexion",
                "structure": "self-critique loop",
                "use_case": "code generation",
                "components": ["actor", "evaluator", "memory"],
                "confidence": 0.85,
            },
        ],
    )
    mock_client = AsyncMock()
    mock_client.complete.return_value = (
        '{"name": "Test Pattern", "description": "test", "category": "Reasoning"}'
    )
    pattern_entries = await build_agentic_taxonomy(
        store, mock_client, min_cluster_size=2, version_id=1
    )
    assert len(pattern_entries) >= 1
    patterns = store.query("agentic_patterns")
    assert len(patterns) >= 1
    assert "category" in patterns[0]


@pytest.mark.asyncio
async def test_label_clusters_with_category_fallback():
    from lens.taxonomy.labeler import label_clusters_with_category

    clusters = {0: ["some concept"]}
    structures = {0: ["some structure description"]}
    mock_client = AsyncMock()
    mock_client.complete.return_value = "not json"
    labels = await label_clusters_with_category(clusters, structures, mock_client)
    assert labels[0]["name"] is not None
    assert labels[0]["category"] == "Uncategorized"
    assert "description" in labels[0]

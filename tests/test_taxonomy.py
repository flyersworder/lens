"""Tests for the taxonomy pipeline."""

import numpy as np


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
                "new_concepts": {"Pruning": "Removing unnecessary model weights"},
            },
        ],
    )

    stats = build_tradeoff_taxonomy(store)
    assert stats["new_entries"] == 1

    rows = store.query("vocabulary", "id = ?", ("pruning",))
    assert len(rows) == 1
    assert rows[0]["source"] == "extracted"
    assert rows[0]["paper_count"] == 1


def test_build_vocabulary(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy.vocabulary import build_vocabulary, load_seed_vocabulary

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
                "new_concepts": {"Pruning": "Removing unnecessary weights"},
            },
        ],
    )
    store.add_rows(
        "architecture_extractions",
        [
            {
                "paper_id": "p1",
                "component_slot": "Attention Mechanism",
                "variant_name": "GQA",
                "replaces": None,
                "key_properties": "fewer KV heads",
                "confidence": 0.9,
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
                "structure": "interleave",
                "use_case": "QA",
                "components": ["LLM"],
                "confidence": 0.8,
                "new_concepts": {},
            },
        ],
    )
    stats = build_vocabulary(store)
    assert stats["new_entries"] == 1  # Pruning
    vocab = store.query("vocabulary")
    pruning = [v for v in vocab if v["id"] == "pruning"]
    assert len(pruning) == 1
    attn = [v for v in vocab if v["id"] == "attention-mechanism"]
    assert attn[0]["paper_count"] == 1
    reasoning = [v for v in vocab if v["id"] == "reasoning"]
    assert reasoning[0]["paper_count"] == 1


def test_get_next_version(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy.versioning import get_next_version

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    assert get_next_version(store) == 1

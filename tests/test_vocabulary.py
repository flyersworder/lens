"""Tests for canonical vocabulary."""

import pytest
from pydantic import ValidationError

from lens.store.models import VocabularyEntry
from lens.store.store import LensStore
from lens.taxonomy.vocabulary import SEED_VOCABULARY, load_seed_vocabulary, process_new_concepts


def test_vocabulary_entry_validates():
    entry = VocabularyEntry(
        id="inference-latency",
        name="Inference Latency",
        kind="parameter",
        description="Time required to generate output from input at deployment",
        source="seed",
        first_seen="2026-03-29",
        paper_count=0,
        avg_confidence=0.0,
    )
    assert entry.id == "inference-latency"
    assert entry.kind == "parameter"


def test_vocabulary_entry_kind_validation():
    with pytest.raises(ValidationError):
        VocabularyEntry(
            id="bad",
            name="Bad",
            kind="invalid",
            description="test",
            source="seed",
            first_seen="2026-03-29",
            paper_count=0,
            avg_confidence=0.0,
        )


def test_vocabulary_table_exists(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    rows = store.query("vocabulary")
    assert rows == []


def test_vocabulary_entry_arch_slot():
    entry = VocabularyEntry(
        id="attention-mechanism",
        name="Attention Mechanism",
        kind="arch_slot",
        description="How the model attends to different parts of the input",
        source="seed",
        first_seen="2026-03-29",
        paper_count=0,
        avg_confidence=0.0,
    )
    assert entry.kind == "arch_slot"


def test_vocabulary_entry_agentic_category():
    entry = VocabularyEntry(
        id="reasoning",
        name="Reasoning",
        kind="agentic_category",
        description="Patterns for multi-step logical inference and problem solving",
        source="seed",
        first_seen="2026-03-29",
        paper_count=0,
        avg_confidence=0.0,
    )
    assert entry.kind == "agentic_category"


def test_seed_vocabulary_has_expected_entries():
    params = [e for e in SEED_VOCABULARY if e["kind"] == "parameter"]
    principles = [e for e in SEED_VOCABULARY if e["kind"] == "principle"]
    arch_slots = [e for e in SEED_VOCABULARY if e["kind"] == "arch_slot"]
    agentic_categories = [e for e in SEED_VOCABULARY if e["kind"] == "agentic_category"]
    assert len(params) == 12
    assert len(principles) == 12
    assert len(arch_slots) == 10
    assert len(agentic_categories) == 6


def test_load_seed_vocabulary(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    count = load_seed_vocabulary(store)
    assert count == 40

    rows = store.query("vocabulary")
    assert len(rows) == 40
    latency = [r for r in rows if r["id"] == "inference-latency"]
    assert len(latency) == 1
    assert latency[0]["name"] == "Inference Latency"
    assert latency[0]["kind"] == "parameter"
    assert latency[0]["source"] == "seed"


def test_load_seed_vocabulary_is_idempotent(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    load_seed_vocabulary(store)
    count = load_seed_vocabulary(store)
    assert count == 0
    rows = store.query("vocabulary")
    assert len(rows) == 40


def test_process_new_concepts_accepts_new_entries(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    load_seed_vocabulary(store)

    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "paper1",
                "improves": "NEW: Energy Efficiency",
                "worsens": "Model Accuracy",
                "technique": "Quantization",
                "context": "test",
                "confidence": 0.8,
                "evidence_quote": "test quote",
                "new_concept_description": "Power consumption relative to compute throughput",
            },
        ],
    )

    stats = process_new_concepts(store)
    assert stats["new_entries"] == 1

    rows = store.query("vocabulary", "id = ?", ("energy-efficiency",))
    assert len(rows) == 1
    assert rows[0]["name"] == "Energy Efficiency"
    assert rows[0]["kind"] == "parameter"
    assert rows[0]["source"] == "extracted"
    assert rows[0]["description"] == "Power consumption relative to compute throughput"


def test_process_new_concepts_updates_paper_count(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    load_seed_vocabulary(store)

    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "paper1",
                "improves": "Inference Latency",
                "worsens": "Model Accuracy",
                "technique": "Quantization",
                "context": "test",
                "confidence": 0.9,
                "evidence_quote": "quote1",
                "new_concept_description": None,
            },
            {
                "paper_id": "paper2",
                "improves": "Inference Latency",
                "worsens": "Training Cost",
                "technique": "Quantization",
                "context": "test",
                "confidence": 0.7,
                "evidence_quote": "quote2",
                "new_concept_description": None,
            },
        ],
    )

    process_new_concepts(store)

    latency = store.query("vocabulary", "id = ?", ("inference-latency",))
    assert latency[0]["paper_count"] == 2
    assert latency[0]["avg_confidence"] == 0.8  # (0.9 + 0.7) / 2

    quant = store.query("vocabulary", "id = ?", ("quantization",))
    assert quant[0]["paper_count"] == 2
    assert quant[0]["avg_confidence"] == 0.8


def test_process_new_concepts_handles_architecture(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    load_seed_vocabulary(store)
    store.add_rows(
        "architecture_extractions",
        [
            {
                "paper_id": "p1",
                "component_slot": "Attention Mechanism",
                "variant_name": "FlashAttention-2",
                "replaces": None,
                "key_properties": "better parallelism",
                "confidence": 0.9,
                "new_concept_description": None,
            },
            {
                "paper_id": "p2",
                "component_slot": "NEW: Embedding Layer",
                "variant_name": "Rotary Embeddings",
                "replaces": None,
                "key_properties": "relative position",
                "confidence": 0.85,
                "new_concept_description": "Token embedding and projection layer",
            },
        ],
    )
    stats = process_new_concepts(store)
    assert stats["new_entries"] == 1
    rows = store.query("vocabulary", "id = ?", ("embedding-layer",))
    assert len(rows) == 1
    assert rows[0]["kind"] == "arch_slot"
    assert rows[0]["source"] == "extracted"
    attn = store.query("vocabulary", "id = ?", ("attention-mechanism",))
    assert attn[0]["paper_count"] == 1


def test_process_new_concepts_handles_agentic(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    load_seed_vocabulary(store)
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
                "new_concept_description": None,
            },
            {
                "paper_id": "p2",
                "pattern_name": "LATS",
                "category": "NEW: Search",
                "structure": "tree search",
                "use_case": "complex reasoning",
                "components": ["LLM", "MCTS"],
                "confidence": 0.8,
                "new_concept_description": "Patterns using systematic search over solution spaces",
            },
        ],
    )
    stats = process_new_concepts(store)
    assert stats["new_entries"] == 1
    rows = store.query("vocabulary", "id = ?", ("search",))
    assert len(rows) == 1
    assert rows[0]["kind"] == "agentic_category"
    reasoning = store.query("vocabulary", "id = ?", ("reasoning",))
    assert reasoning[0]["paper_count"] == 1


def test_end_to_end_all_extraction_types(tmp_path):
    """Integration: seed vocab -> extract all types -> process -> matrix."""
    from lens.knowledge.matrix import build_matrix

    store = LensStore(str(tmp_path / "test.db"))
    count = load_seed_vocabulary(store)
    assert count == 40

    # Tradeoff extraction
    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p1",
                "improves": "Inference Latency",
                "worsens": "Model Accuracy",
                "technique": "Quantization",
                "context": "4-bit on 7B models",
                "confidence": 0.9,
                "evidence_quote": "2x speedup with 4-bit.",
                "new_concept_description": None,
            },
        ],
    )

    # Architecture extraction
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
                "new_concept_description": None,
            },
            {
                "paper_id": "p2",
                "component_slot": "NEW: Tokenizer",
                "variant_name": "BPE-dropout",
                "replaces": None,
                "key_properties": "regularization via subword sampling",
                "confidence": 0.8,
                "new_concept_description": "Text tokenization and subword segmentation methods",
            },
        ],
    )

    # Agentic extraction
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
                "confidence": 0.85,
                "new_concept_description": None,
            },
        ],
    )

    # Process all
    stats = process_new_concepts(store)
    assert stats["new_entries"] == 1  # Tokenizer

    vocab = store.query("vocabulary")
    assert any(v["id"] == "tokenizer" and v["kind"] == "arch_slot" for v in vocab)
    assert any(v["id"] == "attention-mechanism" and v["paper_count"] == 1 for v in vocab)
    assert any(v["id"] == "reasoning" and v["paper_count"] == 1 for v in vocab)

    # Matrix (tradeoffs only)
    build_matrix(store)
    cells = store.query("matrix_cells")
    assert len(cells) == 1
    assert cells[0]["improving_param_id"] == "inference-latency"

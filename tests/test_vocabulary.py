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


def test_seed_vocabulary_has_expected_entries():
    params = [e for e in SEED_VOCABULARY if e["kind"] == "parameter"]
    principles = [e for e in SEED_VOCABULARY if e["kind"] == "principle"]
    assert len(params) == 12
    assert len(principles) == 12


def test_load_seed_vocabulary(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    count = load_seed_vocabulary(store)
    assert count == 24

    rows = store.query("vocabulary")
    assert len(rows) == 24
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
    assert len(rows) == 24


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

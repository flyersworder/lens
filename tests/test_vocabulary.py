"""Tests for canonical vocabulary."""

import pytest
from pydantic import ValidationError

from lens.store.models import VocabularyEntry
from lens.store.store import LensStore
from lens.taxonomy.vocabulary import SEED_VOCABULARY, load_seed_vocabulary


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

"""Tests for canonical vocabulary."""

import pytest
from pydantic import ValidationError

from lens.store.models import VocabularyEntry
from lens.store.store import LensStore


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

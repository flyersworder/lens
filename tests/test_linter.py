"""Tests for the LENS knowledge base linter."""

from lens.knowledge.linter import check_contradictions, check_orphan_vocabulary, fix_orphans
from lens.store.store import LensStore
from lens.taxonomy.vocabulary import load_seed_vocabulary


def test_lint_orphan_vocabulary(tmp_path):
    """Extracted entries with paper_count=0 should be flagged as orphans."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    # Add an extracted entry with no papers
    store.add_rows(
        "vocabulary",
        [
            {
                "id": "orphan-concept",
                "name": "Orphan Concept",
                "kind": "parameter",
                "description": "A concept with no evidence",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
            }
        ],
    )

    orphans = check_orphan_vocabulary(store)
    assert len(orphans) == 1
    assert orphans[0]["id"] == "orphan-concept"


def test_lint_orphan_ignores_seed(tmp_path):
    """Seed vocabulary with paper_count=0 should NOT be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    orphans = check_orphan_vocabulary(store)
    assert len(orphans) == 0


def test_lint_fix_orphans(tmp_path):
    """fix_orphans() should delete orphan entries."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "orphan-concept",
                "name": "Orphan Concept",
                "kind": "parameter",
                "description": "A concept with no evidence",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
            }
        ],
    )

    deleted = fix_orphans(store)
    assert len(deleted) == 1
    assert deleted[0] == "orphan-concept"

    remaining = store.query("vocabulary", "id = ?", ("orphan-concept",))
    assert len(remaining) == 0


def test_lint_contradictions(tmp_path):
    """Opposing matrix cells (A->B and B->A) should be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    store.add_rows(
        "matrix_cells",
        [
            {
                "improving_param_id": "inference-latency",
                "worsening_param_id": "model-accuracy",
                "principle_id": "quantization",
                "count": 2,
                "avg_confidence": 0.8,
                "paper_ids": ["p1", "p2"],
                "taxonomy_version": 1,
            },
            {
                "improving_param_id": "model-accuracy",
                "worsening_param_id": "inference-latency",
                "principle_id": "quantization",
                "count": 2,
                "avg_confidence": 0.7,
                "paper_ids": ["p3", "p4"],
                "taxonomy_version": 1,
            },
        ],
    )

    contradictions = check_contradictions(store)
    assert len(contradictions) == 1
    pair = contradictions[0]
    assert set(pair["params"]) == {"inference-latency", "model-accuracy"}
    assert pair["principle_id"] == "quantization"


def test_lint_contradictions_ignores_weak(tmp_path):
    """Single-paper contradictions (count < 2) should not be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    store.add_rows(
        "matrix_cells",
        [
            {
                "improving_param_id": "inference-latency",
                "worsening_param_id": "model-accuracy",
                "principle_id": "quantization",
                "count": 2,
                "avg_confidence": 0.8,
                "paper_ids": ["p1", "p2"],
                "taxonomy_version": 1,
            },
            {
                "improving_param_id": "model-accuracy",
                "worsening_param_id": "inference-latency",
                "principle_id": "quantization",
                "count": 1,
                "avg_confidence": 0.7,
                "paper_ids": ["p3"],
                "taxonomy_version": 1,
            },
        ],
    )

    contradictions = check_contradictions(store)
    assert len(contradictions) == 0

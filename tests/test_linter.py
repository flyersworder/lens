"""Tests for the LENS knowledge base linter."""

import numpy as np

from lens.knowledge.linter import (
    check_contradictions,
    check_missing_embeddings,
    check_near_duplicates,
    check_orphan_vocabulary,
    check_stale_extractions,
    check_unverified_extractions,
    check_weak_evidence,
    fix_duplicates,
    fix_orphans,
    fix_stale_extractions,
    lint,
)
from lens.store.models import EMBEDDING_DIM
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


def test_lint_weak_evidence(tmp_path):
    """Entries with paper_count=1 or low confidence should be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "strong-concept",
                "name": "Strong Concept",
                "kind": "parameter",
                "description": "Well-supported",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 5,
                "avg_confidence": 0.8,
            },
            {
                "id": "weak-one-paper",
                "name": "Weak One Paper",
                "kind": "parameter",
                "description": "Only one paper",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 1,
                "avg_confidence": 0.9,
            },
            {
                "id": "weak-low-conf",
                "name": "Weak Low Conf",
                "kind": "principle",
                "description": "Low confidence",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 3,
                "avg_confidence": 0.3,
            },
        ],
    )

    findings = check_weak_evidence(store, confidence_threshold=0.5)
    ids = {f["id"] for f in findings}
    assert "weak-one-paper" in ids
    assert "weak-low-conf" in ids
    assert "strong-concept" not in ids


def test_lint_missing_embeddings(tmp_path):
    """Vocabulary entries without a vec row should be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "no-vec",
                "name": "No Vec Entry",
                "kind": "parameter",
                "description": "Missing embedding",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 2,
                "avg_confidence": 0.8,
            }
        ],
    )

    findings = check_missing_embeddings(store)
    assert len(findings) == 1
    assert findings[0]["id"] == "no-vec"


def test_lint_missing_embeddings_none_when_present(tmp_path):
    """Vocabulary entries WITH a vec row should not be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "has-vec",
                "name": "Has Vec Entry",
                "kind": "parameter",
                "description": "Has embedding",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 2,
                "avg_confidence": 0.8,
                "embedding": [0.1] * 768,
            }
        ],
    )

    findings = check_missing_embeddings(store)
    assert len(findings) == 0


def test_lint_stale_extractions(tmp_path):
    """Papers with non-complete status should be flagged."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "papers",
        [
            {
                "paper_id": "complete-paper",
                "title": "Good Paper",
                "abstract": "All done",
                "authors": ["A"],
                "date": "2026-01-01",
                "arxiv_id": "2601.00001",
                "extraction_status": "complete",
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "paper_id": "failed-paper",
                "title": "Bad Paper",
                "abstract": "Failed",
                "authors": ["B"],
                "date": "2026-01-01",
                "arxiv_id": "2601.00002",
                "extraction_status": "failed",
                "embedding": [0.0] * EMBEDDING_DIM,
            },
            {
                "paper_id": "pending-paper",
                "title": "Waiting Paper",
                "abstract": "Pending",
                "authors": ["C"],
                "date": "2026-01-01",
                "arxiv_id": "2601.00003",
                "extraction_status": "pending",
                "embedding": [0.0] * EMBEDDING_DIM,
            },
        ],
    )

    findings = check_stale_extractions(store)
    ids = {f["paper_id"] for f in findings}
    assert "failed-paper" in ids
    assert "pending-paper" in ids
    assert "complete-paper" not in ids


def test_lint_stale_extractions_last_event(tmp_path):
    """Stale findings should include last_event timestamp from event_log."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "papers",
        [
            {
                "paper_id": "stuck-paper",
                "title": "Stuck",
                "abstract": "Stuck",
                "authors": ["A"],
                "date": "2026-01-01",
                "arxiv_id": "2601.00010",
                "extraction_status": "failed",
                "embedding": [0.0] * EMBEDDING_DIM,
            },
        ],
    )

    # Log an event for this paper
    from lens.knowledge.events import log_event

    log_event(
        store,
        "extract",
        "extraction.failed",
        target_type="paper",
        target_id="stuck-paper",
    )

    findings = check_stale_extractions(store)
    assert len(findings) == 1
    assert findings[0]["last_event"] is not None

    # Paper with no events should have last_event=None
    store.add_rows(
        "papers",
        [
            {
                "paper_id": "no-events-paper",
                "title": "No Events",
                "abstract": "None",
                "authors": ["B"],
                "date": "2026-01-01",
                "arxiv_id": "2601.00011",
                "extraction_status": "pending",
                "embedding": [0.0] * EMBEDDING_DIM,
            },
        ],
    )

    findings = check_stale_extractions(store)
    no_event_finding = next(f for f in findings if f["paper_id"] == "no-events-paper")
    assert no_event_finding["last_event"] is None


def test_lint_fix_stale_requeues(tmp_path):
    """fix_stale_extractions() should reset status to pending."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "papers",
        [
            {
                "paper_id": "failed-paper",
                "title": "Bad Paper",
                "abstract": "Failed",
                "authors": ["B"],
                "date": "2026-01-01",
                "arxiv_id": "2601.00002",
                "extraction_status": "failed",
                "embedding": [0.0] * EMBEDDING_DIM,
            }
        ],
    )

    requeued = fix_stale_extractions(store)
    assert requeued == ["failed-paper"]

    paper = store.query("papers", "paper_id = ?", ("failed-paper",))
    assert paper[0]["extraction_status"] == "pending"


def test_lint_near_duplicates(tmp_path):
    """Vocabulary entries with very similar embeddings should be paired."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    # Create two nearly identical embeddings
    base_emb = np.random.RandomState(42).randn(EMBEDDING_DIM).astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb)
    similar_emb = (
        base_emb + np.random.RandomState(43).randn(EMBEDDING_DIM).astype(np.float32) * 0.01
    )
    similar_emb = similar_emb / np.linalg.norm(similar_emb)
    different_emb = np.random.RandomState(99).randn(EMBEDDING_DIM).astype(np.float32)
    different_emb = different_emb / np.linalg.norm(different_emb)

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "concept-a",
                "name": "Concept A",
                "kind": "parameter",
                "description": "First concept",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 3,
                "avg_confidence": 0.8,
                "embedding": base_emb.tolist(),
            },
            {
                "id": "concept-a-variant",
                "name": "Concept A Variant",
                "kind": "parameter",
                "description": "Nearly identical to first",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 1,
                "avg_confidence": 0.7,
                "embedding": similar_emb.tolist(),
            },
            {
                "id": "concept-b",
                "name": "Concept B",
                "kind": "parameter",
                "description": "Completely different",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 2,
                "avg_confidence": 0.9,
                "embedding": different_emb.tolist(),
            },
        ],
    )

    pairs = check_near_duplicates(store, similarity_threshold=0.92)
    assert len(pairs) == 1
    pair_ids = {pairs[0]["id_a"], pairs[0]["id_b"]}
    assert pair_ids == {"concept-a", "concept-a-variant"}


def test_lint_near_duplicates_different_kinds(tmp_path):
    """Near-duplicates across different kinds should NOT be paired."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    base_emb = np.random.RandomState(42).randn(EMBEDDING_DIM).astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb)

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "param-x",
                "name": "Param X",
                "kind": "parameter",
                "description": "A parameter",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 2,
                "avg_confidence": 0.8,
                "embedding": base_emb.tolist(),
            },
            {
                "id": "principle-x",
                "name": "Principle X",
                "kind": "principle",
                "description": "A principle",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 2,
                "avg_confidence": 0.8,
                "embedding": base_emb.tolist(),
            },
        ],
    )

    pairs = check_near_duplicates(store, similarity_threshold=0.92)
    assert len(pairs) == 0


def test_lint_unverified_extractions(tmp_path):
    """Extractions marked unverified/blocked should be aggregated per paper."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p1",
                "improves": "a",
                "worsens": "b",
                "technique": "t",
                "context": "",
                "confidence": 0.3,
                "evidence_quote": "short",
                "new_concepts": {},
                "verification_status": "unverified",
            },
            {
                "paper_id": "p1",
                "improves": "c",
                "worsens": "d",
                "technique": "t",
                "context": "",
                "confidence": 0.9,
                "evidence_quote": "a long and substantive quote",
                "new_concepts": {},
                "verification_status": "verified",
            },
            {
                "paper_id": "p2",
                "improves": "a",
                "worsens": "b",
                "technique": "t",
                "context": "",
                "confidence": 0.2,
                "evidence_quote": "x",
                "new_concepts": {},
                "verification_status": "blocked",
            },
        ],
    )

    findings = check_unverified_extractions(store)
    by_paper = {f["paper_id"]: f for f in findings}

    # p1 has one unverified tradeoff; verified rows are ignored.
    assert by_paper["p1"]["unverified"] == 1
    assert by_paper["p1"]["blocked"] == 0
    assert by_paper["p1"]["by_kind"] == {"tradeoff": {"unverified": 1, "blocked": 0}}

    # p2 has one blocked tradeoff.
    assert by_paper["p2"]["blocked"] == 1
    assert by_paper["p2"]["unverified"] == 0
    assert by_paper["p2"]["by_kind"] == {"tradeoff": {"unverified": 0, "blocked": 1}}


def test_lint_unverified_extractions_splits_by_kind(tmp_path):
    """A single paper with fragile rows across two tables keeps counts per kind."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p1",
                "improves": "a",
                "worsens": "b",
                "technique": "t",
                "context": "",
                "confidence": 0.3,
                "evidence_quote": "short",
                "new_concepts": {},
                "verification_status": "unverified",
            }
            for _ in range(5)
        ],
    )
    store.add_rows(
        "architecture_extractions",
        [
            {
                "paper_id": "p1",
                "component_slot": "s",
                "variant_name": "v",
                "replaces": None,
                "key_properties": "k",
                "confidence": 0.2,
                "new_concepts": {},
                "verification_status": "blocked",
            }
        ],
    )

    findings = check_unverified_extractions(store)
    assert len(findings) == 1
    assert findings[0]["by_kind"] == {
        "tradeoff": {"unverified": 5, "blocked": 0},
        "architecture": {"unverified": 0, "blocked": 1},
    }


def test_lint_unverified_extractions_empty(tmp_path):
    """No findings when every extraction is verified/inferred."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p1",
                "improves": "a",
                "worsens": "b",
                "technique": "t",
                "context": "",
                "confidence": 0.9,
                "evidence_quote": "a long and substantive quote",
                "new_concepts": {},
                "verification_status": "verified",
            }
        ],
    )

    assert check_unverified_extractions(store) == []


def test_lint_orchestrator_runs_all_checks(tmp_path):
    """lint() should run all checks and return a LintReport."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "orphan-test",
                "name": "Orphan Test",
                "kind": "parameter",
                "description": "No papers",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
            }
        ],
    )

    report = lint(store)
    assert len(report.orphans) == 1
    assert report.orphans[0]["id"] == "orphan-test"
    assert isinstance(report.contradictions, list)
    assert isinstance(report.weak_evidence, list)
    assert isinstance(report.missing_embeddings, list)
    assert isinstance(report.stale_extractions, list)
    assert isinstance(report.near_duplicates, list)
    assert report.fixes_applied == []


def test_lint_with_fix(tmp_path):
    """lint(fix=True) should apply fixes and record them."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "orphan-fix-test",
                "name": "Orphan Fix Test",
                "kind": "parameter",
                "description": "Will be deleted",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
            }
        ],
    )

    report = lint(store, fix=True)
    assert len(report.fixes_applied) >= 1
    assert any(f["action"] == "orphan.deleted" for f in report.fixes_applied)

    remaining = store.query("vocabulary", "id = ?", ("orphan-fix-test",))
    assert len(remaining) == 0


def test_lint_check_filter(tmp_path):
    """lint(checks=['orphans']) should only run the orphan check."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "orphan-filter",
                "name": "Orphan Filter",
                "kind": "parameter",
                "description": "No papers",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
            }
        ],
    )

    store.add_rows(
        "papers",
        [
            {
                "paper_id": "stale-p",
                "title": "Stale",
                "abstract": "Stale",
                "authors": ["A"],
                "date": "2026-01-01",
                "arxiv_id": "2601.00099",
                "extraction_status": "failed",
                "embedding": [0.0] * EMBEDDING_DIM,
            }
        ],
    )

    report = lint(store, checks=["orphans"])
    assert len(report.orphans) == 1
    assert report.stale_extractions == []


def test_lint_fix_merge_duplicates(tmp_path):
    """fix_duplicates() should merge lower-count entry into higher-count."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    base_emb = np.random.RandomState(42).randn(EMBEDDING_DIM).astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb)
    similar_emb = (
        base_emb + np.random.RandomState(43).randn(EMBEDDING_DIM).astype(np.float32) * 0.01
    )
    similar_emb = similar_emb / np.linalg.norm(similar_emb)

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "keeper-concept",
                "name": "Keeper Concept",
                "kind": "parameter",
                "description": "Has more papers",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 5,
                "avg_confidence": 0.8,
                "embedding": base_emb.tolist(),
            },
            {
                "id": "duplicate-concept",
                "name": "Duplicate Concept",
                "kind": "parameter",
                "description": "Has fewer papers",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 2,
                "avg_confidence": 0.7,
                "embedding": similar_emb.tolist(),
            },
        ],
    )

    # Add a tradeoff extraction referencing the duplicate
    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p1",
                "improves": "Duplicate Concept",
                "worsens": "Model Accuracy",
                "technique": "Quantization",
                "context": "test",
                "confidence": 0.9,
                "evidence_quote": "quote",
                "new_concepts": {},
            }
        ],
    )

    pairs = check_near_duplicates(store, similarity_threshold=0.92)
    assert len(pairs) == 1

    merges = fix_duplicates(store, pairs)
    assert len(merges) == 1
    assert merges[0]["keeper_id"] == "keeper-concept"
    assert merges[0]["duplicate_id"] == "duplicate-concept"

    # Duplicate should be gone
    remaining = store.query("vocabulary", "id = ?", ("duplicate-concept",))
    assert len(remaining) == 0

    # Keeper should have merged stats
    keeper = store.query("vocabulary", "id = ?", ("keeper-concept",))
    assert keeper[0]["paper_count"] == 7  # 5 + 2

    # Extraction should reference keeper now
    extractions = store.query("tradeoff_extractions")
    assert extractions[0]["improves"] == "Keeper Concept"

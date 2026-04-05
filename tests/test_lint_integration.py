# tests/test_lint_integration.py
"""Integration tests for lint + event log."""

from lens.knowledge.events import log_event, query_events
from lens.knowledge.linter import lint
from lens.store.store import LensStore
from lens.taxonomy.vocabulary import load_seed_vocabulary


def test_extract_events_logged(tmp_path):
    """Verify event logging infrastructure works for extract events."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    log_event(
        store,
        "extract",
        "extraction.completed",
        target_type="paper",
        target_id="test-paper-1",
        detail={"tradeoffs": 2, "architecture": 1, "agentic": 0},
        session_id="test-session",
    )

    events = query_events(store, kind="extract")
    assert len(events) == 1
    assert events[0]["action"] == "extraction.completed"
    assert events[0]["target_id"] == "test-paper-1"


def test_lint_log_events_recorded(tmp_path):
    """Running lint should record lint.* events in the event log."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "integ-orphan",
                "name": "Integration Orphan",
                "kind": "parameter",
                "description": "Test",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
            }
        ],
    )

    lint(store, session_id="integ-sess")

    events = query_events(store, kind="lint", session_id="integ-sess")
    assert len(events) >= 1
    assert any(e["action"] == "orphan.found" for e in events)


def test_fix_events_recorded(tmp_path):
    """Running lint --fix should record fix.* events in the event log."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "vocabulary",
        [
            {
                "id": "fix-orphan",
                "name": "Fix Orphan",
                "kind": "parameter",
                "description": "Will be fixed",
                "source": "extracted",
                "first_seen": "2026-04-01",
                "paper_count": 0,
                "avg_confidence": 0.0,
            }
        ],
    )

    lint(store, fix=True, session_id="fix-sess")

    fix_events = query_events(store, kind="fix", session_id="fix-sess")
    assert len(fix_events) >= 1
    assert any(e["action"] == "orphan.deleted" for e in fix_events)


def test_lint_no_false_positives_on_fresh_data(tmp_path):
    """Lint on a clean DB with seed vocabulary should report no orphans."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    load_seed_vocabulary(store)

    report = lint(store, checks=["orphans"])
    assert len(report.orphans) == 0

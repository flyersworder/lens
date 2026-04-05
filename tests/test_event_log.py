"""Tests for the event log system."""

from lens.knowledge.events import log_event, query_events
from lens.store.store import LensStore


def test_event_log_table_exists(tmp_path):
    """The event_log table should be created by init_tables()."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()
    rows = store.query_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='event_log'"
    )
    assert len(rows) == 1
    assert rows[0]["name"] == "event_log"


def test_log_event_writes(tmp_path):
    """log_event() should insert a row into event_log."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    log_event(
        store,
        kind="ingest",
        action="paper.added",
        target_type="paper",
        target_id="test-paper-1",
        detail={"title": "Test Paper", "source": "arxiv"},
        session_id="sess-001",
    )

    rows = store.query("event_log")
    assert len(rows) == 1
    assert rows[0]["kind"] == "ingest"
    assert rows[0]["action"] == "paper.added"
    assert rows[0]["target_type"] == "paper"
    assert rows[0]["target_id"] == "test-paper-1"
    assert rows[0]["detail"] == {"title": "Test Paper", "source": "arxiv"}
    assert rows[0]["session_id"] == "sess-001"
    assert rows[0]["timestamp"]  # non-empty


def test_log_event_session_grouping(tmp_path):
    """Events with the same session_id should be queryable together."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    log_event(store, kind="ingest", action="paper.added", target_id="p1", session_id="sess-A")
    log_event(
        store, kind="extract", action="extraction.completed", target_id="p1", session_id="sess-A"
    )
    log_event(store, kind="ingest", action="paper.added", target_id="p2", session_id="sess-B")

    events = query_events(store, session_id="sess-A")
    assert len(events) == 2
    assert all(e["session_id"] == "sess-A" for e in events)


def test_query_events_filters(tmp_path):
    """query_events() should support kind, since, and limit filters."""
    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    log_event(store, kind="ingest", action="paper.added", target_id="p1")
    log_event(store, kind="extract", action="extraction.completed", target_id="p1")
    log_event(store, kind="ingest", action="paper.added", target_id="p2")

    # Filter by kind
    ingest_events = query_events(store, kind="ingest")
    assert len(ingest_events) == 2

    # Limit
    limited = query_events(store, limit=1)
    assert len(limited) == 1

    # Since (all events are from today, so a past date returns all)
    all_events = query_events(store, since="2020-01-01")
    assert len(all_events) == 3

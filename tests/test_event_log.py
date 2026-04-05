"""Tests for the event log system."""

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

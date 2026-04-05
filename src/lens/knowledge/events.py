# src/lens/knowledge/events.py
"""Unified event log for LENS — records all mutations for audit and lint."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from lens.store.store import LensStore


def log_event(
    store: LensStore,
    kind: str,
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    detail: dict | None = None,
    session_id: str | None = None,
) -> None:
    """Append one event to the event_log table."""
    store.add_rows(
        "event_log",
        [
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "kind": kind,
                "action": action,
                "target_type": target_type,
                "target_id": target_id,
                "detail": detail or {},
                "session_id": session_id,
            }
        ],
    )


def query_events(
    store: LensStore,
    kind: str | None = None,
    since: str | None = None,
    limit: int = 20,
    session_id: str | None = None,
) -> list[dict]:
    """Query event_log with optional filters. Returns newest-first."""
    clauses: list[str] = []
    params: list[str | int] = []

    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if since:
        clauses.append("timestamp >= ?")
        params.append(since)
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)

    where = " AND ".join(clauses) if clauses else "1 = 1"
    sql = f"SELECT * FROM event_log WHERE {where} ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = store.query_sql(sql, tuple(params))
    for row in rows:
        if isinstance(row.get("detail"), str):
            row["detail"] = json.loads(row["detail"])
    return rows

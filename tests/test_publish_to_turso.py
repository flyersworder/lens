"""Unit tests for the publish_to_turso helpers.

Focused on the retry seam — the actual end-to-end publish is covered
implicitly by the weekly monitor job (and would require a real Turso DB
to test honestly, which we don't do here).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

# publish_to_turso lives in scripts/ (not src/), so import it directly
# from the file path rather than via a package.
_SPEC = importlib.util.spec_from_file_location(
    "publish_to_turso",
    Path(__file__).resolve().parent.parent / "scripts" / "publish_to_turso.py",
)
assert _SPEC is not None and _SPEC.loader is not None
publish_to_turso = importlib.util.module_from_spec(_SPEC)
sys.modules["publish_to_turso"] = publish_to_turso
_SPEC.loader.exec_module(publish_to_turso)


class _FakeClient:
    """Records batch() calls and raises a scripted sequence of exceptions."""

    def __init__(self, exceptions_then_success: list[BaseException | None]):
        # Each entry is an exception to raise, or None to succeed.
        self._script = list(exceptions_then_success)
        self.calls = 0

    def batch(self, statements: list[object]) -> None:
        self.calls += 1
        if not self._script:
            return
        nxt = self._script.pop(0)
        if nxt is not None:
            raise nxt


def test_batch_with_retry_succeeds_first_try():
    client = _FakeClient([None])
    publish_to_turso._batch_with_retry(client, ["stmt"], op="copy papers")
    assert client.calls == 1


def test_batch_with_retry_recovers_from_timeout():
    # Two transient timeouts, then success on the third attempt.
    client = _FakeClient([TimeoutError(), TimeoutError(), None])
    publish_to_turso._batch_with_retry(client, ["stmt"], op="attach embeddings", base_delay=0.0)
    assert client.calls == 3


def test_batch_with_retry_gives_up_after_max_attempts():
    # Always times out — should raise after exhausting attempts.
    client = _FakeClient([TimeoutError()] * 10)
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        publish_to_turso._batch_with_retry(
            client, ["stmt"], op="copy", base_delay=0.0, max_attempts=3
        )
    assert client.calls == 3


def test_batch_with_retry_does_not_retry_on_libsql_error():
    """Server-side errors (schema/constraint) must surface immediately."""

    class FakeLibsqlError(Exception):
        """Stand-in for libsql_client.LibsqlError."""

    client = _FakeClient([FakeLibsqlError("constraint violation")])
    with pytest.raises(FakeLibsqlError):
        publish_to_turso._batch_with_retry(client, ["stmt"], op="copy", base_delay=0.0)
    # No retry — fail fast on deterministic errors.
    assert client.calls == 1


def test_batch_with_retry_does_not_retry_on_value_error():
    """Programming errors (e.g., bad SQL) must not be masked as flakes."""
    client = _FakeClient([ValueError("bad statement")])
    with pytest.raises(ValueError):
        publish_to_turso._batch_with_retry(client, ["stmt"], op="copy", base_delay=0.0)
    assert client.calls == 1


class _RecordingClient:
    """Fake libSQL client that records execute() DDL and batch() inserts."""

    def __init__(self):
        self.ddl: list[str] = []
        self.batched: list = []

    def execute(self, sql: str) -> None:
        self.ddl.append(sql)

    def batch(self, statements: list) -> None:
        self.batched.extend(statements)


def _seed_local_db_with_card(tmp_path):
    """Build a real local LENS sqlite DB holding one scoop-checked idea card."""
    from datetime import UTC, datetime

    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "lens.db"))
    store.init_tables()
    store.add_rows(
        "idea_cards",
        [
            {
                "id": 1,
                "gap_id": 7,
                "report_id": 3,
                "title": "Adaptive KV-Cache Pruning",
                "pattern_ids": [],
                "hook": "",
                "mechanism": "prune by latency-aware importance",
                "falsification": "",
                "differentiation": [],
                "signature_terms": ["kv-cache", "pruning"],
                "paper_ids": [],
                "confidence": 0.7,
                "created_at": datetime.now(UTC),
                "taxonomy_version": 0,
                "novelty_status": "scooped",
                "prior_art": [{"title": "Titanus", "url": "u", "year": 2025}],
                "novelty_note": "collides with Titanus",
            }
        ],
    )
    return str(tmp_path / "lens.db")


def test_idea_cards_is_published(tmp_path):
    """idea_cards (with novelty columns) must be created and copied to Turso,
    otherwise the whole scoop-check pipeline never reaches production."""
    assert "idea_cards" in publish_to_turso.TABLES_TO_COPY

    db_path = _seed_local_db_with_card(tmp_path)
    local = publish_to_turso.open_local(db_path)
    try:
        client = _RecordingClient()

        publish_to_turso.create_remote_schema(local, client, publish_to_turso.EMBEDDING_DIM)
        idea_ddl = [d for d in client.ddl if d.startswith("CREATE TABLE idea_cards")]
        assert idea_ddl, "no CREATE TABLE idea_cards emitted"
        # Novelty columns must survive into the remote schema.
        assert "novelty_status" in idea_ddl[0]
        assert "prior_art" in idea_ddl[0]
        # idea_cards has no embeddings, so no F32_BLOB column.
        assert "F32_BLOB" not in idea_ddl[0]

        copied = publish_to_turso.copy_table(local, client, "idea_cards")
        assert copied == 1
        # The card's novelty verdict is carried in the copied row.
        flat_args = [a for stmt in client.batched for a in stmt.args]
        assert "scooped" in flat_args
    finally:
        local.close()

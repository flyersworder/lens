"""Tests for scripts/sync_idea_verdicts.py.

Both sides are real databases: the local side is a LensStore, the "remote"
is the same schema opened through libsql_client over a ``file:`` URL — the
client the script uses against Turso.
"""

from __future__ import annotations

import pytest

from lens.store.store import LensStore

libsql_client = pytest.importorskip("libsql_client")


def _card(card_id: int, **overrides) -> dict:
    card = {
        "id": card_id,
        "gap_id": 1,
        "report_id": 1,
        "title": f"Idea {card_id}",
        "created_at": "2026-10-26T06:00:00+00:00",
    }
    card.update(overrides)
    return card


@pytest.fixture
def stores(tmp_path):
    local = LensStore(str(tmp_path / "local.db"))
    local.init_tables()
    remote_path = tmp_path / "remote.db"
    LensStore(str(remote_path)).init_tables()  # the identical schema
    remote = libsql_client.create_client_sync(url=f"file:{remote_path}")
    yield local, remote, remote_path
    remote.close()


def _remote_verdict(remote, card_id: int) -> tuple:
    rs = remote.execute(
        "SELECT novelty_status, prior_art, novelty_note, novelty_checked_at "
        "FROM idea_cards WHERE id = ?",
        [card_id],
    )
    return tuple(rs.rows[0])


def _seed_remote(remote, *cards: dict) -> None:
    for c in cards:
        remote.execute(
            "INSERT INTO idea_cards (id, gap_id, report_id, title, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [c["id"], c["gap_id"], c["report_id"], c["title"], c["created_at"]],
        )


def test_pushes_checked_verdicts_and_is_idempotent(stores):
    from scripts.sync_idea_verdicts import sync_verdicts

    local, remote, _ = stores
    checked = _card(
        1,
        novelty_status="novel",
        prior_art=[{"title": "Prior", "year": 2025}],
        novelty_note="No direct match.",
        novelty_checked_at="2026-10-30T06:10:00+00:00",
    )
    local.add_rows("idea_cards", [checked])
    _seed_remote(remote, _card(1))

    assert sync_verdicts(local.conn, remote) == 1
    status, prior_art, note, checked_at = _remote_verdict(remote, 1)
    assert (status, note, checked_at) == ("novel", "No direct match.", "2026-10-30T06:10:00+00:00")
    assert "Prior" in prior_art

    # Second run finds nothing to change.
    assert sync_verdicts(local.conn, remote) == 0


def test_never_pushes_unchecked_cards(stores):
    """An unchecked local card must not clobber a remote verdict."""
    from scripts.sync_idea_verdicts import sync_verdicts

    local, remote, _ = stores
    local.add_rows("idea_cards", [_card(2)])
    _seed_remote(remote, _card(2))
    remote.execute("UPDATE idea_cards SET novelty_status = 'scooped' WHERE id = 2")

    assert sync_verdicts(local.conn, remote) == 0
    assert _remote_verdict(remote, 2)[0] == "scooped"


def test_skips_cards_missing_remotely(stores):
    """Cards not yet published are left for the next full publish, not inserted."""
    from scripts.sync_idea_verdicts import sync_verdicts

    local, remote, _ = stores
    local.add_rows(
        "idea_cards",
        [_card(3, novelty_status="novel", novelty_checked_at="2026-10-30T06:00:00+00:00")],
    )

    assert sync_verdicts(local.conn, remote) == 0
    assert remote.execute("SELECT COUNT(*) FROM idea_cards").rows[0][0] == 0


def test_leaves_other_columns_untouched(stores):
    from scripts.sync_idea_verdicts import sync_verdicts

    local, remote, _ = stores
    local.add_rows(
        "idea_cards",
        [
            _card(
                4,
                title="Local title",
                novelty_status="overlaps",
                novelty_checked_at="2026-10-30T06:00:00+00:00",
            )
        ],
    )
    _seed_remote(remote, _card(4, title="Remote title"))

    assert sync_verdicts(local.conn, remote) == 1
    rs = remote.execute("SELECT title, novelty_status FROM idea_cards WHERE id = 4")
    assert tuple(rs.rows[0]) == ("Remote title", "overlaps")


def test_dry_run_reports_without_writing(stores):
    from scripts.sync_idea_verdicts import sync_verdicts

    local, remote, _ = stores
    local.add_rows(
        "idea_cards",
        [_card(5, novelty_status="novel", novelty_checked_at="2026-10-30T06:00:00+00:00")],
    )
    _seed_remote(remote, _card(5))

    assert sync_verdicts(local.conn, remote, dry_run=True) == 1
    assert _remote_verdict(remote, 5)[0] == "unchecked"

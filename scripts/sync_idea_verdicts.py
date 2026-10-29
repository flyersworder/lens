"""Push idea-card novelty verdicts from a local LENS DB to Turso, in place.

``publish_to_turso.py`` drop-recreates every remote table, which leaves the
live API degraded for the ~20 minutes it takes. A scoop-check-only run
changes just four columns on a few dozen ``idea_cards`` rows, so this script
``UPDATE``s exactly those rows and touches nothing else — no schema drop, no
downtime.

* Rows match by ``id`` (the full publish copies ids verbatim).
* Only cards that have a verdict locally (``novelty_status != 'unchecked'``)
  are pushed, so an unchecked local card never clobbers a remote verdict.
* Only rows whose verdict differs from the remote are written, so a re-run
  is a no-op. Cards missing remotely are left for the next full publish.

Usage::

    # Run as a module so ``scripts.publish_to_turso`` is importable.
    uv run python -m scripts.sync_idea_verdicts --target prod \\
        --local ~/.lens/data/lens.db
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import libsql_client
from scripts.publish_to_turso import _batch_with_retry, _resolve_target

from lens.store.turso_store import _normalize_url

logger = logging.getLogger(__name__)

VERDICT_COLUMNS = ("novelty_status", "prior_art", "novelty_note", "novelty_checked_at")
_SELECT = f"SELECT id, {', '.join(VERDICT_COLUMNS)} FROM idea_cards"


def sync_verdicts(
    local: sqlite3.Connection,
    remote: libsql_client.ClientSync,
    batch_size: int = 20,
    dry_run: bool = False,
) -> int:
    """Copy changed verdicts from ``local`` to ``remote``; return rows updated.

    With ``dry_run`` the remote is only read: the count is what *would* change.
    """
    remote_rows = {row[0]: tuple(row[1:]) for row in remote.execute(_SELECT).rows}
    local_rows = local.execute(f"{_SELECT} WHERE novelty_status != 'unchecked'").fetchall()

    missing = 0
    changed: list[tuple] = []
    for row in local_rows:
        card_id, verdict = row[0], tuple(row[1:])
        if card_id not in remote_rows:
            missing += 1
        elif remote_rows[card_id] != verdict:
            changed.append((card_id, verdict))
    if missing:
        logger.warning(f"{missing} checked card(s) not on remote; left for the next full publish")

    if dry_run:
        logger.info(f"dry run: would update {len(changed)} idea-card verdict(s)")
        return len(changed)

    assignments = ", ".join(f"{col} = ?" for col in VERDICT_COLUMNS)
    statements = [
        libsql_client.Statement(
            f"UPDATE idea_cards SET {assignments} WHERE id = ?", [*verdict, card_id]
        )
        for card_id, verdict in changed
    ]
    for start in range(0, len(statements), batch_size):
        _batch_with_retry(remote, statements[start : start + batch_size], op="sync verdicts")
    logger.info(f"updated {len(changed)} idea-card verdict(s) on remote")
    return len(changed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push idea-card novelty verdicts to Turso without a full publish."
    )
    parser.add_argument(
        "--local",
        default=str(Path.home() / ".lens" / "data" / "lens.db"),
        help="Path to local sqlite-vec DB (default: ~/.lens/data/lens.db)",
    )
    parser.add_argument(
        "--target",
        choices=("dev", "prod", "plain"),
        default="dev",
        help="Which Turso DB to update (default: dev)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read the remote and report what would change, without writing.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    url, token = _resolve_target(args.target)

    local = sqlite3.connect(f"file:{Path(args.local).expanduser()}?mode=ro", uri=True)
    remote = libsql_client.create_client_sync(url=_normalize_url(url), auth_token=token)
    try:
        sync_verdicts(local, remote, dry_run=args.dry_run)
    finally:
        remote.close()
        local.close()


if __name__ == "__main__":
    main()

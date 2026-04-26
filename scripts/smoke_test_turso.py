"""Smoke-test a published Turso DB by querying it via TursoStore.

Used by the GitHub Actions publish workflow to verify the end-to-end
chain works: that the publish script wrote what the read backend
expects to read. Fails loudly (non-zero exit) if any expectation
is violated.

Targets the seed fixture built by ``scripts/build_seed_fixture.py`` —
2 papers, 2 vocabulary entries, 1 tradeoff extraction, 1 matrix cell.

Usage::

    uv run python scripts/smoke_test_turso.py --target dev
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from lens.store.turso_store import TursoStore


def _load_env_local() -> None:
    """Pick up TURSO_* env vars from .env.local if present (local dev only)."""
    env_local = Path(__file__).resolve().parent.parent / ".env.local"
    if env_local.exists():
        for line in env_local.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _resolve_target(target: str) -> tuple[str, str]:
    """Mirror of publish_to_turso.py's resolver, kept independent for clarity."""
    if target == "dev":
        url = os.environ.get("TURSO_DEV_DATABASE_URL")
        token = os.environ.get("TURSO_DEV_AUTH_TOKEN")
    elif target == "prod":
        url = os.environ.get("TURSO_PROD_DATABASE_URL")
        token = os.environ.get("TURSO_PROD_AUTH_TOKEN")
    else:
        url = os.environ.get("TURSO_DATABASE_URL")
        token = os.environ.get("TURSO_AUTH_TOKEN")
    if not url or not token:
        sys.exit(f"error: TURSO credentials for target '{target}' not found")
    return url, token


def smoke_test(url: str, auth_token: str) -> int:
    """Run smoke checks against the published seed fixture.

    Returns the number of failed checks (0 = all pass).
    """
    store = TursoStore(url=url, auth_token=auth_token)
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal failed
        if condition:
            print(f"  [OK]   {name}")
        else:
            print(f"  [FAIL] {name}: {detail}")
            failed += 1

    try:
        # 1. Basic table queries
        papers = store.query("papers")
        check(
            "papers count == 2",
            len(papers) == 2,
            f"got {len(papers)}",
        )
        check(
            "papers contain seed:0001",
            any(p["paper_id"] == "seed:0001" for p in papers),
            "seed:0001 not found",
        )
        check(
            "papers.authors deserialized as list",
            isinstance(papers[0].get("authors"), list),
            f"authors type: {type(papers[0].get('authors')).__name__}",
        )

        vocab = store.query("vocabulary")
        check("vocabulary count == 2", len(vocab) == 2, f"got {len(vocab)}")
        check(
            "vocabulary kinds include principle and parameter",
            {v["kind"] for v in vocab} == {"principle", "parameter"},
            f"kinds: {[v['kind'] for v in vocab]}",
        )

        tradeoffs = store.query("tradeoff_extractions")
        check(
            "tradeoff_extractions count == 1",
            len(tradeoffs) == 1,
            f"got {len(tradeoffs)}",
        )

        matrix = store.query("matrix_cells")
        check(
            "matrix_cells count == 1",
            len(matrix) == 1,
            f"got {len(matrix)}",
        )

        # 2. WHERE clause
        attn = store.query("vocabulary", "id = ?", ("attention",))
        check(
            "vocabulary WHERE id = 'attention' returns 1 row",
            len(attn) == 1 and attn[0]["name"] == "Attention",
            f"got {attn}",
        )

        # 3. FTS path: query with a token that's in seed:0001's title.
        fts = store.search_papers(query="attention", limit=5)
        check(
            "FTS search 'attention' returns at least 1 result",
            len(fts) >= 1,
            "no FTS results",
        )

        # 4. Vector path: querying with the embedding of a known paper
        #    should return that paper as the nearest neighbor.
        target_paper = next(p for p in papers if p["paper_id"] == "seed:0001")
        # Convert the F32 BLOB back to a list[float].
        import struct

        emb_bytes = target_paper["embedding"]
        emb = list(struct.unpack(f"{len(emb_bytes) // 4}f", emb_bytes))
        nearest = store.vector_search("papers", emb, limit=2)
        check(
            "vector_search nearest neighbor is seed:0001",
            len(nearest) >= 1 and nearest[0]["paper_id"] == "seed:0001",
            f"got {[r['paper_id'] for r in nearest]}",
        )

        # 5. JSON deserialization on matrix cells
        check(
            "matrix_cells.paper_ids deserialized as list",
            isinstance(matrix[0]["paper_ids"], list),
            f"got {type(matrix[0]['paper_ids']).__name__}",
        )
    finally:
        store.close()

    return failed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test a published Turso LENS DB via TursoStore."
    )
    parser.add_argument(
        "--target",
        choices=("dev", "prod", "plain"),
        default="dev",
        help="Which Turso target to verify (default: dev).",
    )
    args = parser.parse_args()

    _load_env_local()
    url, token = _resolve_target(args.target)
    print(f"smoke-testing target: {url}")
    failed = smoke_test(url, token)
    if failed:
        print(f"\nFAILED: {failed} check(s) did not pass")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

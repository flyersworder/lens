"""Build a tiny synthetic LENS database for CI smoke tests and local debug.

The output DB has the canonical LENS schema (via :class:`LensStore.init_tables`)
populated with a deterministic toy corpus: 2 papers, 2 vocabulary entries,
1 tradeoff extraction, 1 matrix cell, and matching embeddings. Random
vectors are seeded so output is byte-identical across runs.

This avoids committing a binary ``lens.db`` to the repo and lets the
publish workflow validate the full chain without needing real LLM
extraction or a saved corpus.

Usage::

    uv run python scripts/build_seed_fixture.py /tmp/seed.db
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from lens.store.models import EMBEDDING_DIM
from lens.store.store import LensStore

SEED = 42  # fixed for reproducibility


def _random_unit_embedding(dim: int, rng: random.Random) -> list[float]:
    """Return a random unit-norm float vector. Cheap; we don't need numpy here."""
    raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


def build(output_path: str, *, embedding_dim: int = EMBEDDING_DIM) -> None:
    """Create the seed DB at ``output_path`` (overwrites any existing file)."""
    out = Path(output_path)
    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    store = LensStore(str(out))
    store.init_tables()

    rng = random.Random(SEED)

    # Two papers along orthogonal embedding axes for predictable nearest-neighbor.
    papers = [
        {
            "paper_id": "seed:0001",
            "title": "Attention Is All You Need (seed)",
            "abstract": "Synthetic seed paper exercising the publish path.",
            "authors": ["Seed Author A", "Seed Author B"],
            "venue": "Synthetic",
            "date": "2017-06-12",
            "arxiv_id": "1706.03762",
            "citations": 1,
            "quality_score": 0.5,
            "extraction_status": "extracted",
            "embedding": _random_unit_embedding(embedding_dim, rng),
        },
        {
            "paper_id": "seed:0002",
            "title": "BERT Synthetic",
            "abstract": "Second synthetic seed paper.",
            "authors": ["Seed Author C"],
            "venue": "Synthetic",
            "date": "2018-10-11",
            "arxiv_id": "1810.04805",
            "citations": 1,
            "quality_score": 0.5,
            "extraction_status": "extracted",
            "embedding": _random_unit_embedding(embedding_dim, rng),
        },
    ]
    store.add_rows("papers", papers)

    vocab = [
        {
            "id": "attention",
            "name": "Attention",
            "kind": "principle",
            "description": "Synthetic seed vocab entry for attention.",
            "source": "seed",
            "first_seen": "2017-06-12",
            "paper_count": 1,
            "avg_confidence": 0.9,
            "embedding": _random_unit_embedding(embedding_dim, rng),
        },
        {
            "id": "accuracy",
            "name": "Accuracy",
            "kind": "parameter",
            "description": "Synthetic seed vocab entry for accuracy.",
            "source": "seed",
            "first_seen": "2018-10-11",
            "paper_count": 1,
            "avg_confidence": 0.8,
            "embedding": _random_unit_embedding(embedding_dim, rng),
        },
    ]
    store.add_rows("vocabulary", vocab)

    # One toy extraction + one toy matrix cell so the cross-table tests
    # have something to read.
    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "seed:0001",
                "improves": "accuracy",
                "worsens": "computational-complexity",
                "technique": "self-attention",
                "context": "Replacing recurrence with attention.",
                "confidence": 0.9,
                "evidence_quote": "Synthetic evidence quote.",
                "new_concepts": {},
                "verification_status": "verified",
            }
        ],
    )

    store.add_rows(
        "matrix_cells",
        [
            {
                "improving_param_id": "accuracy",
                "worsening_param_id": "computational-complexity",
                "principle_id": "attention",
                "count": 1,
                "avg_confidence": 0.9,
                "paper_ids": ["seed:0001"],
                "taxonomy_version": 1,
            }
        ],
    )

    # Rebuild FTS so the resulting file is fully ready for downstream copy.
    store.rebuild_papers_fts()
    store.rebuild_vocabulary_fts()
    store.conn.close()

    size_kb = os.path.getsize(out) / 1024
    print(f"built seed fixture: {out} ({size_kb:.1f} KB, embedding_dim={embedding_dim})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a tiny synthetic LENS database for CI smoke tests."
    )
    parser.add_argument(
        "output",
        help="Output path for the seed DB (will be overwritten if it exists).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=EMBEDDING_DIM,
        help=f"Embedding vector dimension (default: {EMBEDDING_DIM})",
    )
    args = parser.parse_args()
    build(args.output, embedding_dim=args.embedding_dim)


if __name__ == "__main__":
    sys.exit(main())

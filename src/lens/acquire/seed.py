"""Seed paper loader — reads enriched YAML manifest and stores papers.

The manifest includes pre-fetched metadata (title, abstract, authors, date)
so seed acquisition requires NO external API calls. Embeddings are generated
locally during taxonomy build.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from lens.acquire.quality import quality_score
from lens.store.models import EMBEDDING_DIM
from lens.store.store import LensStore

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path(__file__).parent.parent / "data" / "seed_papers.yaml"


def load_seed_manifest(manifest_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load the seed paper manifest from YAML."""
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("papers", [])


async def acquire_seed(
    store: LensStore,
    manifest_path: Path | str | None = None,
) -> int:
    """Acquire seed papers from the enriched manifest.

    Uses pre-fetched metadata from the YAML manifest — no API calls needed.
    Papers already in the database are skipped.

    Returns the number of papers successfully acquired.
    """
    manifest = load_seed_manifest(manifest_path)
    logger.info("Acquiring %d seed papers", len(manifest))

    # Check which papers are already stored
    try:
        existing = store.query("papers")
        existing_ids = {p["paper_id"] for p in existing}
    except Exception:
        existing_ids = set()

    papers_to_store: list[dict[str, Any]] = []

    for entry in manifest:
        arxiv_id = entry["arxiv_id"]
        if arxiv_id in existing_ids:
            logger.info("Skipping %s — already stored", arxiv_id)
            continue

        paper = {
            "paper_id": arxiv_id,
            "arxiv_id": arxiv_id,
            "title": entry.get("title", ""),
            "abstract": entry.get("abstract", ""),
            "authors": entry.get("authors", []),
            "date": entry.get("date", ""),
            "venue": None,
            "citations": 0,
            "quality_score": 0.0,
            "extraction_status": "pending",
            "embedding": [0.0] * EMBEDDING_DIM,
        }

        papers_to_store.append(paper)

    # Compute quality scores
    for paper in papers_to_store:
        paper["quality_score"] = quality_score(
            citations=paper.get("citations", 0),
            venue=paper.get("venue"),
            paper_date=paper.get("date", "2020-01-01"),
        )

    # Store in DB
    if papers_to_store:
        store.add_papers(papers_to_store)
        logger.info("Stored %d seed papers", len(papers_to_store))

    return len(papers_to_store)

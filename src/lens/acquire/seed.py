"""Seed paper loader — reads YAML manifest and orchestrates acquisition."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
import yaml

from lens.acquire.arxiv import ARXIV_API_URL, parse_arxiv_response
from lens.acquire.http import fetch_with_retry
from lens.acquire.openalex import enrich_with_openalex
from lens.acquire.quality import quality_score
from lens.acquire.semantic_scholar import fetch_embedding
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


async def _fetch_paper_metadata(arxiv_id: str) -> dict[str, Any] | None:
    """Fetch paper metadata from arxiv for a single paper with retry."""
    from urllib.parse import quote

    url = f"{ARXIV_API_URL}?id_list={quote(arxiv_id)}&max_results=1"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await fetch_with_retry(client, url)
            papers = parse_arxiv_response(resp.text)
            return papers[0] if papers else None
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning("Failed to fetch arxiv metadata for %s: %s", arxiv_id, e)
        except Exception as e:
            logger.warning("Unexpected error fetching %s: %s", arxiv_id, e)
            return None
        finally:
            await asyncio.sleep(3.0)  # rate limit


def _normalize_embedding(embedding: list[float], dim: int = EMBEDDING_DIM) -> list[float]:
    """Pad or truncate embedding to target dimensionality."""
    if len(embedding) < dim:
        return embedding + [0.0] * (dim - len(embedding))
    return embedding[:dim]


async def acquire_seed(
    store: LensStore,
    manifest_path: Path | str | None = None,
) -> int:
    """Acquire seed papers: fetch from arxiv, enrich, embed, and store.

    Returns the number of papers successfully acquired.
    """
    manifest = load_seed_manifest(manifest_path)
    logger.info(f"Acquiring {len(manifest)} seed papers")

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
            logger.info(f"Skipping {arxiv_id} — already stored")
            continue

        # Fetch metadata from arxiv
        paper = await _fetch_paper_metadata(arxiv_id)
        if not paper:
            logger.warning(f"Skipping {arxiv_id} — not found on arxiv")
            continue

        # Fetch SPECTER2 embedding from Semantic Scholar
        emb_result = await fetch_embedding(arxiv_id)
        if emb_result and emb_result.get("embedding"):
            paper["embedding"] = _normalize_embedding(emb_result["embedding"])
        else:
            paper["embedding"] = [0.0] * EMBEDDING_DIM
            logger.warning(f"No SPECTER2 embedding for {arxiv_id}")

        papers_to_store.append(paper)

    # Enrich with OpenAlex (batch)
    if papers_to_store:
        papers_to_store = await enrich_with_openalex(papers_to_store)

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
        logger.info(f"Stored {len(papers_to_store)} seed papers")

    return len(papers_to_store)

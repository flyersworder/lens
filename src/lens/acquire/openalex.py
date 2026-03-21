"""OpenAlex API client for paper metadata enrichment.

Provides citation counts and venue information.
Rate limit: polite pool (~10 req/s with mailto parameter).
Retry with exponential backoff per spec.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from lens.acquire.http import fetch_with_retry

logger = logging.getLogger(__name__)

OPENALEX_API_URL = "https://api.openalex.org/works"
DEFAULT_MAILTO = "lens-project@example.com"


def _extract_arxiv_id_from_doi(doi: str | None) -> str | None:
    """Extract arxiv ID from an OpenAlex DOI like 'https://doi.org/10.48550/arXiv.1706.03762'."""
    if not doi:
        return None
    match = re.search(r"arXiv\.(\d{4}\.\d{4,5})", doi, re.IGNORECASE)
    return match.group(1) if match else None


def parse_openalex_works(works: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse OpenAlex work objects into enrichment dicts keyed by arxiv_id."""
    results = []
    for work in works:
        venue = None
        loc = work.get("primary_location")
        if loc and isinstance(loc, dict):
            source = loc.get("source")
            if source and isinstance(source, dict):
                venue = source.get("display_name")

        arxiv_id = _extract_arxiv_id_from_doi(work.get("doi"))

        results.append(
            {
                "arxiv_id": arxiv_id,
                "openalex_id": work.get("id", ""),
                "citations": work.get("cited_by_count", 0),
                "venue": venue,
            }
        )
    return results


async def enrich_with_openalex(
    papers: list[dict[str, Any]],
    batch_size: int = 50,
    mailto: str = "",
) -> list[dict[str, Any]]:
    """Enrich paper dicts with OpenAlex citation counts and venue info.

    Matches by arxiv_id extracted from OpenAlex DOI field.
    Papers not found in OpenAlex are returned unchanged.
    """
    arxiv_ids = [p["arxiv_id"] for p in papers if p.get("arxiv_id")]
    if not arxiv_ids:
        return papers

    all_works: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i : i + batch_size]
            effective_mailto = mailto or DEFAULT_MAILTO
            doi_filter = "|".join(f"https://doi.org/10.48550/arXiv.{aid}" for aid in batch)
            url = (
                f"{OPENALEX_API_URL}?filter=doi:{doi_filter}"
                f"&mailto={effective_mailto}&per-page={batch_size}"
            )
            try:
                resp = await fetch_with_retry(client, url)
                data = resp.json()
                all_works.extend(data.get("results", []))
            except httpx.HTTPError as e:
                logger.warning("OpenAlex batch fetch failed for %d papers: %s", len(batch), e)
                continue

    # Parse and build lookup by arxiv_id
    enrichments = parse_openalex_works(all_works)
    enrichment_by_id: dict[str, dict] = {}
    for e in enrichments:
        if e.get("arxiv_id"):
            enrichment_by_id[e["arxiv_id"]] = e

    # Apply enrichment — match by arxiv_id
    for paper in papers:
        aid = paper.get("arxiv_id", "")
        if aid in enrichment_by_id:
            e = enrichment_by_id[aid]
            paper["citations"] = e["citations"]
            if e.get("venue"):
                paper["venue"] = e["venue"]

    return papers

"""OpenAlex API client for paper metadata enrichment.

Provides citation counts and venue information.
Rate limit: polite pool (~10 req/s with mailto parameter).
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

OPENALEX_API_URL = "https://api.openalex.org/works"
MAILTO = "lens-project@example.com"


def parse_openalex_works(works: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse OpenAlex work objects into enrichment dicts."""
    results = []
    for work in works:
        venue = None
        loc = work.get("primary_location")
        if loc and isinstance(loc, dict):
            source = loc.get("source")
            if source and isinstance(source, dict):
                venue = source.get("display_name")

        results.append({
            "openalex_id": work.get("id", ""),
            "citations": work.get("cited_by_count", 0),
            "venue": venue,
        })
    return results


def build_url_for_arxiv_ids(arxiv_ids: list[str], per_page: int = 100) -> str:
    """Build an OpenAlex filter URL for a batch of arxiv IDs."""
    ids_filter = "|".join(f"https://arxiv.org/abs/{aid}" for aid in arxiv_ids)
    return f"{OPENALEX_API_URL}?filter=locations.source.id:{quote(ids_filter)}&mailto={MAILTO}&per-page={per_page}"


async def enrich_with_openalex(
    papers: list[dict[str, Any]],
    batch_size: int = 50,
) -> list[dict[str, Any]]:
    """Enrich paper dicts with OpenAlex citation counts and venue info.

    Papers not found in OpenAlex are returned unchanged.
    """
    arxiv_ids = [p["arxiv_id"] for p in papers if p.get("arxiv_id")]
    if not arxiv_ids:
        return papers

    all_works: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i : i + batch_size]
            ids_filter = "|".join(f"https://arxiv.org/abs/{aid}" for aid in batch)
            url = f"{OPENALEX_API_URL}?filter=locations.source.id:{ids_filter}&mailto={MAILTO}&per-page={batch_size}"
            try:
                resp = await client.get(url)
                if resp.status_code >= 400:
                    continue
                data = resp.json()
                all_works.extend(data.get("results", []))
            except (httpx.HTTPError, KeyError):
                continue

    # Parse raw OpenAlex works into enrichment dicts
    enrichments = parse_openalex_works(all_works)

    # Apply enrichment to papers
    for paper in papers:
        for e in enrichments:
            if e.get("citations", 0) > paper.get("citations", 0):
                paper["citations"] = e["citations"]
                if e.get("venue"):
                    paper["venue"] = e["venue"]
                break

    return papers

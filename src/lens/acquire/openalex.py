"""OpenAlex API client for paper metadata enrichment.

Provides citation counts and venue information.
Rate limit: polite pool (~10 req/s with mailto parameter).
Retry with exponential backoff per spec.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date
from typing import Any
from urllib.parse import quote_plus

import httpx

from lens.acquire.http import fetch_with_retry

logger = logging.getLogger(__name__)

# Light pacing between search requests: scoop-check fires several per card, and
# the free polite pool rate-limits bursts. Tests set this to 0.
SEARCH_PACING_SECONDS = 0.2

OPENALEX_API_URL = "https://api.openalex.org/works"
DEFAULT_MAILTO = "lens-project@example.com"
# OpenAlex "Computer science" concept — scoop-check searches LLM-research prior
# art, so restricting to recent CS works keeps generic terms (e.g. "adaptive
# quantization") from dredging up control-theory / signal-processing papers.
CS_CONCEPT_ID = "C41008148"
# Prior-art window, expressed as a span rather than a fixed start year. A
# hardcoded year looks stable but silently widens the window every January —
# "recent work" is relative to now, so the span is the part that must be frozen.
DEFAULT_LOOKBACK_YEARS = 8


def default_from_year(today: date | None = None) -> int:
    """First publication year included in a prior-art search.

    ``today`` overrides the reference date; it defaults to the current date.
    """
    reference = today if today is not None else date.today()
    return reference.year - DEFAULT_LOOKBACK_YEARS


def _reconstruct_abstract(inverted_index: Any) -> str:
    """Rebuild plain-text abstract from OpenAlex's abstract_inverted_index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        if isinstance(idxs, list):
            positions.extend((i, word) for i in idxs)
    positions.sort()
    return " ".join(word for _, word in positions)


async def search_openalex(
    query: str,
    limit: int = 5,
    mailto: str = "",
    from_year: int | None = None,
) -> list[dict[str, Any]]:
    """Search OpenAlex for prior art matching a text query (polite free pool).

    Restricted to recent Computer-science works so generic query terms don't
    surface off-domain / decades-old papers. Never raises — returns [] on
    timeout / HTTP error / malformed body; works without a title are dropped.

    ``from_year`` defaults to a rolling :data:`DEFAULT_LOOKBACK_YEARS` window
    ending at the current year; pass it explicitly to pin the window.
    """
    effective_mailto = mailto or DEFAULT_MAILTO
    effective_from_year = from_year if from_year is not None else default_from_year()
    fields = "title,abstract_inverted_index,publication_year,doi,id"
    filters = f"from_publication_date:{effective_from_year}-01-01,concepts.id:{CS_CONCEPT_ID}"
    url = (
        f"{OPENALEX_API_URL}?search={quote_plus(query)}"
        f"&per-page={limit}&mailto={effective_mailto}&select={fields}&filter={filters}"
    )

    data: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await fetch_with_retry(client, url)
            data = resp.json()
        except Exception as e:
            logger.warning("OpenAlex search failed for %r: %s", query, e)
            return []
        finally:
            await asyncio.sleep(SEARCH_PACING_SECONDS)

    if not isinstance(data, dict):
        return []

    papers: list[dict[str, Any]] = []
    for work in data.get("results") or []:
        title = work.get("title") or ""
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        # Keep title-only works: a colliding paper with no abstract is still
        # visible to the judge by its title (dropping it risks a false "novel").
        if not title and not abstract:
            continue
        papers.append(
            {
                "title": title,
                "abstract": abstract,
                "year": work.get("publication_year"),
                "arxiv_id": _extract_arxiv_id_from_doi(work.get("doi")),
                "url": work.get("doi") or work.get("id") or "",
            }
        )
    return papers


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

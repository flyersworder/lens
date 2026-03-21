"""arxiv API client for paper discovery.

Uses the arxiv Atom feed API. Rate limit: 1 request per 3 seconds.
Includes retry with exponential backoff (spec requirement).
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote

import httpx

from lens.acquire.http import fetch_with_retry

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
RATE_LIMIT_SECONDS = 3.0

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def _extract_arxiv_id(id_url: str) -> str:
    """Extract arxiv ID from full URL like 'http://arxiv.org/abs/1706.03762v7'."""
    match = re.search(r"(\d{4}\.\d{4,5})", id_url)
    if match:
        return match.group(1)
    match = re.search(r"abs/(.+?)(?:v\d+)?$", id_url)
    return match.group(1) if match else id_url


def parse_arxiv_response(xml_text: str) -> list[dict[str, Any]]:
    """Parse an arxiv Atom feed response into paper dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("Failed to parse arxiv response as XML (possibly a rate-limit page)")
        return []
    papers = []
    for entry in root.findall("atom:entry", NS):
        id_el = entry.find("atom:id", NS)
        title_el = entry.find("atom:title", NS)
        summary_el = entry.find("atom:summary", NS)
        published_el = entry.find("atom:published", NS)

        if id_el is None or title_el is None:
            continue

        arxiv_id = _extract_arxiv_id(id_el.text or "")
        authors = []
        for a in entry.findall("atom:author", NS):
            name_el = a.find("atom:name", NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text)
        published = ((published_el.text or "") if published_el is not None else "")[:10]

        papers.append(
            {
                "paper_id": arxiv_id,
                "arxiv_id": arxiv_id,
                "title": " ".join((title_el.text or "").split()),
                "abstract": (
                    " ".join((summary_el.text or "").split()) if summary_el is not None else ""
                ),
                "authors": authors,
                "date": published,
                "venue": None,
                "citations": 0,
                "quality_score": 0.0,
                "extraction_status": "pending",
            }
        )
    return papers


def build_query_url(
    query: str,
    categories: list[str],
    since: str | None = None,
    start: int = 0,
    max_results: int = 100,
) -> str:
    """Build an arxiv API query URL."""
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    search = f"({cat_query}) AND all:{query}"
    if since:
        date_clean = since.replace("-", "")
        search += f" AND submittedDate:[{date_clean}0000 TO 99991231]"
    encoded = quote(search)
    return (
        f"{ARXIV_API_URL}?search_query={encoded}"
        f"&start={start}&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )


async def fetch_arxiv_papers(
    query: str,
    categories: list[str],
    since: str | None = None,
    max_results: int = 100,
) -> list[dict[str, Any]]:
    """Fetch papers from arxiv API with rate limiting and retry."""
    url = build_query_url(query, categories, since=since, max_results=max_results)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await fetch_with_retry(client, url)
    await asyncio.sleep(RATE_LIMIT_SECONDS)
    return parse_arxiv_response(resp.text)

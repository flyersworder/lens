"""Semantic Scholar API client for SPECTER2 embeddings.

Rate limit: 1 request per 3 seconds. Retry with exponential backoff per spec.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import quote_plus

import httpx

from lens.acquire.http import fetch_with_retry

logger = logging.getLogger(__name__)

S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper"
RATE_LIMIT_SECONDS = 3.0


def parse_embedding_response(data: dict[str, Any]) -> dict[str, Any]:
    """Parse a Semantic Scholar paper response for embedding data."""
    external_ids = data.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv", "")

    embedding = None
    emb_data = data.get("embedding")
    if emb_data and isinstance(emb_data, dict):
        vector = emb_data.get("vector")
        if vector:
            embedding = vector

    return {
        "arxiv_id": arxiv_id,
        "semantic_scholar_id": data.get("paperId", ""),
        "embedding": embedding,
    }


async def fetch_embedding(
    arxiv_id: str,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Fetch SPECTER2 embedding for a paper from Semantic Scholar."""
    url = f"{S2_API_URL}/ArXiv:{arxiv_id}?fields=externalIds,title,embedding"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await fetch_with_retry(client, url, headers=headers)
            data = resp.json()
            return parse_embedding_response(data)
        except Exception as e:
            logger.warning("Failed to fetch S2 embedding for %s: %s", arxiv_id, e)
            return None
        finally:
            await asyncio.sleep(RATE_LIMIT_SECONDS)


async def search_semantic_scholar(
    query: str,
    limit: int = 5,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Search Semantic Scholar for prior art matching a text query.

    Free (unauthenticated) tier. Never raises — returns [] on timeout,
    rate-limit exhaustion, or a malformed response. Papers without an
    abstract are dropped (nothing to judge against).
    """
    fields = "title,abstract,year,citationCount,externalIds,url"
    url = f"{S2_API_URL}/search?query={quote_plus(query)}&limit={limit}&fields={fields}"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    data: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await fetch_with_retry(client, url, headers=headers)
            data = resp.json()
        except Exception as e:
            logger.warning("Semantic Scholar search failed for %r: %s", query, e)
            return []
        finally:
            await asyncio.sleep(RATE_LIMIT_SECONDS)

    if not isinstance(data, dict):
        logger.warning("Semantic Scholar returned a non-dict body for %r", query)
        return []

    papers: list[dict[str, Any]] = []
    for item in data.get("data") or []:
        abstract = item.get("abstract")
        if not abstract:
            continue
        ext = item.get("externalIds") or {}
        papers.append(
            {
                "title": item.get("title") or "",
                "abstract": abstract,
                "year": item.get("year"),
                "citations": item.get("citationCount") or 0,
                "arxiv_id": ext.get("ArXiv", ""),
                "url": item.get("url") or "",
            }
        )
    return papers


async def fetch_embeddings_batch(
    arxiv_ids: list[str],
    api_key: str | None = None,
) -> dict[str, list[float] | None]:
    """Fetch SPECTER2 embeddings for multiple papers sequentially."""
    results: dict[str, list[float] | None] = {}
    for aid in arxiv_ids:
        result = await fetch_embedding(aid, api_key=api_key)
        if result and result.get("embedding"):
            results[aid] = result["embedding"]
        else:
            results[aid] = None
    return results

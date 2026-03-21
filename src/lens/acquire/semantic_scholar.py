"""Semantic Scholar API client for SPECTER2 embeddings.

Rate limit: 1 request per 3 seconds. Retry with exponential backoff per spec.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

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


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=30))
async def _fetch_with_retry(client: httpx.AsyncClient, url: str, headers: dict) -> httpx.Response:
    """Fetch with exponential backoff and jitter."""
    resp = await client.get(url, headers=headers)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise httpx.HTTPStatusError(
            f"HTTP {resp.status_code}", request=resp.request, response=resp
        )
    return resp


async def fetch_embedding(
    arxiv_id: str,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Fetch SPECTER2 embedding for a paper from Semantic Scholar."""
    url = f"{S2_API_URL}/ArXiv:{arxiv_id}?fields=externalIds,title,embedding"
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await _fetch_with_retry(client, url, headers)
            if resp.status_code == 404:
                return None
            if resp.status_code >= 400:
                return None
            data = resp.json()
            return parse_embedding_response(data)
        except httpx.HTTPError as e:
            logger.warning("Failed to fetch S2 embedding for %s: %s", arxiv_id, e)
            return None
        finally:
            await asyncio.sleep(RATE_LIMIT_SECONDS)


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

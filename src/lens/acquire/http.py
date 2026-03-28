"""Shared HTTP utilities for acquisition clients."""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=30))
async def fetch_with_retry(
    client: httpx.AsyncClient, url: str, **kwargs: object
) -> httpx.Response:
    """Fetch a URL with exponential backoff and jitter.

    Retries on any HTTP error status (>= 400), including rate limits (429).
    """
    resp = await client.get(url, **kwargs)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
    if resp.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"HTTP {resp.status_code}",
            request=resp.request,
            response=resp,
        )
    return resp

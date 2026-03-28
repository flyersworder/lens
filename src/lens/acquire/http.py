"""Shared HTTP utilities for acquisition clients."""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=30))
async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Fetch a URL with exponential backoff and jitter.

    Retries on any HTTP error status (>= 400), including rate limits (429).
    """
    kwargs: dict[str, Any] = {}
    if headers is not None:
        kwargs["headers"] = headers
    resp = await client.get(url, **kwargs)
    if resp.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"HTTP {resp.status_code}",
            request=resp.request,
            response=resp,
        )
    return resp

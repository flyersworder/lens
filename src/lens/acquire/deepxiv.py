"""DeepXiv paper search and retrieval (optional dependency).

Requires: uv sync --extra deepxiv
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from lens.acquire.quality import quality_score

logger = logging.getLogger(__name__)

try:
    from deepxiv_sdk import Reader

    HAS_DEEPXIV = True
except ImportError:
    HAS_DEEPXIV = False

    class Reader:  # type: ignore[no-redef]
        """Placeholder when deepxiv-sdk is not installed."""

        token: str | None = None

        def __init__(self, token: str | None = None) -> None: ...

        def search(self, **kwargs: Any) -> dict[str, Any]:
            return {}

        def brief(self, arxiv_id: str) -> dict[str, Any]:
            return {}


def _register_and_save_token() -> str | None:
    """Auto-register a DeepXiv token and save it to the project .env."""
    from deepxiv_sdk.cli import auto_register_token, save_token

    token, _ = auto_register_token()
    if token:
        save_token(token, is_global=False)  # save to project .env
    return token


def _get_reader() -> Reader:
    """Create a Reader instance, auto-registering a token if needed."""
    if not HAS_DEEPXIV:
        raise RuntimeError("deepxiv-sdk is not installed. Run: uv sync --extra deepxiv")
    from dotenv import load_dotenv

    load_dotenv()
    token = os.environ.get("DEEPXIV_TOKEN")
    if not token:
        logger.info("No DEEPXIV_TOKEN found, auto-registering...")
        token = _register_and_save_token()
    return Reader(token=token) if token else Reader()


def _call_with_token_refresh(reader: Reader, method: str, *args: Any, **kwargs: Any) -> Any:
    """Call a Reader method, auto-refreshing the token on AuthenticationError."""
    from deepxiv_sdk import AuthenticationError

    try:
        return getattr(reader, method)(*args, **kwargs)
    except AuthenticationError:
        logger.info("Token expired, auto-registering a new one...")
        token = _register_and_save_token()
        if not token:
            raise
        reader.token = token
        return getattr(reader, method)(*args, **kwargs)


def _extract_date(publish_at: str | None) -> str:
    """Extract YYYY-MM-DD from a datetime string."""
    if not publish_at:
        return "1970-01-01"
    match = re.match(r"(\d{4}-\d{2}-\d{2})", publish_at)
    return match.group(1) if match else "1970-01-01"


def _extract_authors(authors: list[dict | str] | None) -> list[str]:
    """Extract author name strings from DeepXiv author data."""
    if not authors:
        return []
    names = []
    for a in authors:
        name = a.get("name", "") if isinstance(a, dict) else str(a)
        if name:
            names.append(name)
    return names


def search_deepxiv(
    query: str,
    categories: list[str] | None = None,
    since: str | None = None,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Search DeepXiv and return LENS Paper-shaped dicts.

    Parameters
    ----------
    query:
        Search query string.
    categories:
        Optional arXiv category filters (e.g. ["cs.AI", "cs.CL"]).
    since:
        Only papers after this date (YYYY-MM-DD).
    max_results:
        Maximum number of papers to return.
    """
    reader = _get_reader()
    kwargs: dict[str, Any] = {
        "query": query,
        "size": max_results,
        "search_mode": "hybrid",
    }
    if categories:
        kwargs["categories"] = categories
    if since:
        kwargs["date_from"] = since

    response = _call_with_token_refresh(reader, "search", **kwargs)
    results = response.get("results", [])

    papers = []
    for r in results:
        arxiv_id = r.get("arxiv_id")
        if not arxiv_id:
            continue
        date = _extract_date(r.get("publish_at") or r.get("published"))
        citations = r.get("citation", 0) or 0
        venue = r.get("venue") or r.get("journal_name")
        papers.append(
            {
                "paper_id": arxiv_id,
                "arxiv_id": arxiv_id,
                "title": r.get("title", ""),
                "abstract": r.get("abstract", ""),
                "authors": _extract_authors(r.get("authors")),
                "date": date,
                "venue": venue,
                "citations": citations,
                "quality_score": quality_score(citations, venue, date),
                "extraction_status": "pending",
            }
        )
    return papers


def fetch_deepxiv_paper(arxiv_id: str) -> dict[str, Any]:
    """Fetch a single paper with rich metadata via DeepXiv brief().

    Returns a LENS Paper-shaped dict with keywords and github_url populated.
    """
    reader = _get_reader()
    data = _call_with_token_refresh(reader, "brief", arxiv_id)

    date = _extract_date(data.get("publish_at"))
    citations = data.get("citations", 0) or 0
    venue = data.get("venue") or data.get("journal_name")

    return {
        "paper_id": arxiv_id,
        "arxiv_id": arxiv_id,
        "title": data.get("title", ""),
        "abstract": data.get("abstract", ""),
        "authors": _extract_authors(data.get("authors")),
        "date": date,
        "venue": venue,
        "citations": citations,
        "quality_score": quality_score(citations, venue, date),
        "extraction_status": "pending",
        "keywords": data.get("keywords") or [],
        "github_url": data.get("github_url"),
    }

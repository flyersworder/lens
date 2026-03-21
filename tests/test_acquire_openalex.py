"""Tests for OpenAlex API client."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_openalex_works():
    from lens.acquire.openalex import parse_openalex_works
    data = json.loads((FIXTURE_DIR / "openalex_response.json").read_text())
    results = parse_openalex_works(data["results"])
    assert len(results) == 1
    r = results[0]
    assert r["citations"] == 120000
    assert "Neural Information Processing" in (r["venue"] or "")


def test_parse_openalex_null_venue():
    from lens.acquire.openalex import parse_openalex_works
    works = [{"id": "W1", "doi": None, "title": "T", "cited_by_count": 5,
              "publication_date": "2024-01-01", "primary_location": None, "authorships": []}]
    results = parse_openalex_works(works)
    assert results[0]["venue"] is None


def test_build_openalex_url_from_arxiv_ids():
    from lens.acquire.openalex import build_url_for_arxiv_ids
    url = build_url_for_arxiv_ids(["1706.03762", "2401.12345"])
    assert "filter" in url.lower() or "arxiv" in url.lower()


@pytest.mark.asyncio
async def test_enrich_papers_with_openalex():
    import httpx
    from lens.acquire.openalex import enrich_with_openalex

    fixture = (FIXTURE_DIR / "openalex_response.json").read_text()
    mock_response = httpx.Response(200, text=fixture, headers={"content-type": "application/json"})

    with patch("lens.acquire.openalex.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        papers = [{"arxiv_id": "1706.03762", "citations": 0, "venue": None}]
        enriched = await enrich_with_openalex(papers)
        assert enriched[0]["citations"] == 120000

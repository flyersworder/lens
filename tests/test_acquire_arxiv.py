"""Tests for arxiv API client."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_arxiv_response():
    from lens.acquire.arxiv import parse_arxiv_response
    xml_text = (FIXTURE_DIR / "arxiv_response.xml").read_text()
    papers = parse_arxiv_response(xml_text)
    assert len(papers) == 1
    p = papers[0]
    assert p["arxiv_id"] == "1706.03762"
    assert p["title"] == "Attention Is All You Need"
    assert "Ashish Vaswani" in p["authors"]
    assert p["date"] == "2017-06-12"
    assert "dominant sequence" in p["abstract"]


def test_parse_arxiv_extracts_paper_id():
    from lens.acquire.arxiv import parse_arxiv_response
    xml_text = (FIXTURE_DIR / "arxiv_response.xml").read_text()
    papers = parse_arxiv_response(xml_text)
    assert papers[0]["paper_id"] == "1706.03762"


def test_build_arxiv_query_url():
    from lens.acquire.arxiv import build_query_url
    url = build_query_url(query="LLM", categories=["cs.CL", "cs.LG"], max_results=10)
    assert "search_query" in url
    assert "cs.CL" in url
    assert "max_results=10" in url


def test_build_arxiv_query_url_with_since():
    from lens.acquire.arxiv import build_query_url
    url = build_query_url(query="LLM", categories=["cs.CL"], since="2024-01-01", max_results=50)
    assert "submittedDate" in url or "2024" in url


@pytest.mark.asyncio
async def test_fetch_arxiv_papers():
    """Test fetch with mocked HTTP response."""
    import httpx
    from lens.acquire.arxiv import fetch_arxiv_papers

    xml_text = (FIXTURE_DIR / "arxiv_response.xml").read_text()
    mock_response = httpx.Response(200, text=xml_text)

    with patch("lens.acquire.arxiv.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        papers = await fetch_arxiv_papers(query="attention", categories=["cs.CL"], max_results=10)
        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "1706.03762"

"""Tests for Semantic Scholar API client."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_semantic_paper():
    from lens.acquire.semantic_scholar import parse_embedding_response
    data = json.loads((FIXTURE_DIR / "semantic_response.json").read_text())
    result = parse_embedding_response(data)
    assert result["arxiv_id"] == "1706.03762"
    assert result["embedding"] is not None
    assert len(result["embedding"]) > 0


def test_parse_semantic_paper_no_embedding():
    from lens.acquire.semantic_scholar import parse_embedding_response
    data = {"paperId": "abc", "externalIds": {"ArXiv": "2401.12345"}, "title": "T", "embedding": None}
    result = parse_embedding_response(data)
    assert result["embedding"] is None


@pytest.mark.asyncio
async def test_fetch_embeddings():
    import httpx
    from lens.acquire.semantic_scholar import fetch_embedding

    fixture = (FIXTURE_DIR / "semantic_response.json").read_text()
    mock_response = httpx.Response(200, text=fixture, headers={"content-type": "application/json"})

    with patch("lens.acquire.semantic_scholar.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        result = await fetch_embedding("1706.03762")
        assert result is not None
        assert result["arxiv_id"] == "1706.03762"
        assert result["embedding"] is not None

"""Tests for OpenAlex API client."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_openalex_works():
    from lens.acquire.openalex import parse_openalex_works

    data = json.loads((FIXTURE_DIR / "openalex_response.json").read_text())
    results = parse_openalex_works(data["results"])
    assert len(results) == 1
    r = results[0]
    assert r["citations"] == 120000
    assert "Neural Information Processing" in (r["venue"] or "")
    assert r["arxiv_id"] == "1706.03762"


def test_parse_openalex_null_venue():
    from lens.acquire.openalex import parse_openalex_works

    works = [
        {
            "id": "W1",
            "doi": None,
            "title": "T",
            "cited_by_count": 5,
            "publication_date": "2024-01-01",
            "primary_location": None,
            "authorships": [],
        }
    ]
    results = parse_openalex_works(works)
    assert results[0]["venue"] is None
    assert results[0]["arxiv_id"] is None


def test_extract_arxiv_id_from_doi():
    from lens.acquire.openalex import _extract_arxiv_id_from_doi

    assert _extract_arxiv_id_from_doi("https://doi.org/10.48550/arXiv.1706.03762") == "1706.03762"
    assert _extract_arxiv_id_from_doi("https://doi.org/10.1234/something-else") is None
    assert _extract_arxiv_id_from_doi(None) is None


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
        assert "Neural Information Processing" in enriched[0]["venue"]


@pytest.mark.asyncio
async def test_enrich_no_match_leaves_paper_unchanged():
    import httpx

    from lens.acquire.openalex import enrich_with_openalex

    # Response has a different paper's DOI — should not match
    response_data = {
        "results": [
            {
                "id": "W1",
                "doi": "https://doi.org/10.48550/arXiv.9999.99999",
                "title": "Other",
                "cited_by_count": 500,
                "publication_date": "2024-01-01",
                "primary_location": None,
                "authorships": [],
            }
        ]
    }
    mock_response = httpx.Response(
        200, text=json.dumps(response_data), headers={"content-type": "application/json"}
    )

    with patch("lens.acquire.openalex.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.get.return_value = mock_response
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        papers = [{"arxiv_id": "1706.03762", "citations": 0, "venue": None}]
        enriched = await enrich_with_openalex(papers)
        assert enriched[0]["citations"] == 0  # unchanged
        assert enriched[0]["venue"] is None  # unchanged


def test_reconstruct_abstract():
    from lens.acquire.openalex import _reconstruct_abstract

    # word -> positions; reconstruction orders by position.
    inv = {"Adaptive": [0], "KV": [1], "cache": [2], "quantization": [3]}
    assert _reconstruct_abstract(inv) == "Adaptive KV cache quantization"
    assert _reconstruct_abstract(None) == ""
    assert _reconstruct_abstract("not a dict") == ""


@pytest.mark.asyncio
async def test_search_openalex_parses(monkeypatch):
    import lens.acquire.openalex as oa

    class FakeResp:
        def json(self):
            return {
                "results": [
                    {
                        "title": "QAQ: Quality Adaptive Quantization for LLM KV Cache",
                        "abstract_inverted_index": {"Adaptive": [0], "quantization": [1]},
                        "publication_year": 2024,
                        "doi": "https://doi.org/10.48550/arXiv.2403.04643",
                        "id": "https://openalex.org/W123",
                    },
                    {
                        "title": "Title-only work (no abstract)",
                        "abstract_inverted_index": None,
                        "publication_year": 2023,
                        "doi": None,
                        "id": "https://openalex.org/W456",
                    },
                    {
                        "title": None,  # no title AND no abstract -> dropped
                        "abstract_inverted_index": None,
                        "publication_year": 2022,
                        "doi": None,
                        "id": "https://openalex.org/W789",
                    },
                ]
            }

    async def fake_fetch(client, url, headers=None):
        return FakeResp()

    monkeypatch.setattr(oa, "fetch_with_retry", fake_fetch)
    res = await oa.search_openalex("kv cache quantization", limit=5)
    assert len(res) == 2  # full + title-only kept; fully-empty work dropped
    assert res[0]["abstract"] == "Adaptive quantization"
    assert res[0]["year"] == 2024
    assert res[0]["arxiv_id"] == "2403.04643"
    assert res[0]["url"] == "https://doi.org/10.48550/arXiv.2403.04643"
    assert res[1]["title"] == "Title-only work (no abstract)"
    assert res[1]["abstract"] == ""  # kept despite no abstract


@pytest.mark.asyncio
async def test_search_openalex_fails_soft(monkeypatch):
    import lens.acquire.openalex as oa

    async def boom(client, url, headers=None):
        raise RuntimeError("429")

    monkeypatch.setattr(oa, "fetch_with_retry", boom)
    assert await oa.search_openalex("anything") == []

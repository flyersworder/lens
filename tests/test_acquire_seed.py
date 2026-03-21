"""Tests for seed paper loader."""

import pytest
import yaml


def test_load_seed_manifest():
    from lens.acquire.seed import load_seed_manifest

    papers = load_seed_manifest()
    assert len(papers) >= 10
    assert all("arxiv_id" in p for p in papers)
    assert all("title" in p for p in papers)


def test_load_seed_manifest_custom_path(tmp_path):
    from lens.acquire.seed import load_seed_manifest

    manifest = tmp_path / "custom_seeds.yaml"
    manifest.write_text(
        yaml.dump(
            {
                "papers": [
                    {"arxiv_id": "9999.99999", "title": "Test Paper", "category": "test"},
                ]
            }
        )
    )
    papers = load_seed_manifest(manifest)
    assert len(papers) == 1
    assert papers[0]["arxiv_id"] == "9999.99999"


def test_seed_manifest_has_categories():
    from lens.acquire.seed import load_seed_manifest

    papers = load_seed_manifest()
    categories = {p.get("category") for p in papers}
    assert "foundational" in categories
    assert "agentic" in categories


@pytest.mark.asyncio
async def test_acquire_seed_papers(tmp_path):
    """Test seed acquisition with mocked API clients."""
    import json
    from unittest.mock import AsyncMock, patch

    import httpx

    from lens.acquire.seed import acquire_seed
    from lens.store.store import LensStore

    manifest = tmp_path / "seeds.yaml"
    manifest.write_text(
        yaml.dump(
            {
                "papers": [
                    {
                        "arxiv_id": "1706.03762",
                        "title": "Attention Is All You Need",
                        "category": "foundational",
                    },
                ]
            }
        )
    )

    arxiv_xml = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/1706.03762v1</id>
        <title>Attention Is All You Need</title>
        <summary>Test abstract</summary>
        <published>2017-06-12T00:00:00Z</published>
        <author><name>Vaswani</name></author>
      </entry>
    </feed>"""

    s2_json = json.dumps(
        {
            "paperId": "abc",
            "externalIds": {"ArXiv": "1706.03762"},
            "title": "Attention",
            "embedding": {"model": "specter2", "vector": [0.1] * 768},
        }
    )

    openalex_json = json.dumps(
        {
            "results": [
                {
                    "id": "W1",
                    "doi": None,
                    "title": "Attention",
                    "cited_by_count": 100000,
                    "publication_date": "2017-06-12",
                    "primary_location": {"source": {"display_name": "NeurIPS"}},
                    "authorships": [],
                }
            ]
        }
    )

    async def mock_get(url, **kwargs):
        url_str = str(url)
        if "openalex" in url_str:
            return httpx.Response(
                200, text=openalex_json, headers={"content-type": "application/json"}
            )
        elif "semanticscholar" in url_str:
            return httpx.Response(200, text=s2_json, headers={"content-type": "application/json"})
        elif "arxiv" in url_str:
            return httpx.Response(200, text=arxiv_xml)
        return httpx.Response(404)

    mock_client = AsyncMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("lens.acquire.seed.httpx.AsyncClient", return_value=mock_client),
        patch("lens.acquire.semantic_scholar.httpx.AsyncClient", return_value=mock_client),
        patch("lens.acquire.openalex.httpx.AsyncClient", return_value=mock_client),
    ):
        store = LensStore(str(tmp_path / "test.lance"))
        store.init_tables()
        count = await acquire_seed(store, manifest_path=manifest)
        assert count >= 1
        papers = store.get_table("papers").to_polars()
        assert len(papers) >= 1

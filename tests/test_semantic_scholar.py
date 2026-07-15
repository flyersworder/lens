import pytest


@pytest.mark.asyncio
async def test_search_semantic_scholar_parses(monkeypatch):
    import lens.acquire.semantic_scholar as s2

    class FakeResp:
        def json(self):
            return {
                "data": [
                    {
                        "title": "GQA",
                        "abstract": "grouped-query attention reduces KV heads",
                        "year": 2023,
                        "citationCount": 100,
                        "externalIds": {"ArXiv": "2305.13245"},
                        "url": "http://x",
                    },
                    {
                        "title": "No abstract",
                        "abstract": None,
                        "year": 2024,
                        "citationCount": 1,
                        "externalIds": {},
                        "url": "",
                    },
                ]
            }

    async def fake_fetch(client, url, headers=None):
        return FakeResp()

    monkeypatch.setattr(s2, "fetch_with_retry", fake_fetch)
    monkeypatch.setattr(s2, "RATE_LIMIT_SECONDS", 0)

    res = await s2.search_semantic_scholar("quantization", limit=5)
    assert len(res) == 1  # abstract-less paper dropped
    assert res[0]["arxiv_id"] == "2305.13245"
    assert res[0]["citations"] == 100
    assert res[0]["title"] == "GQA"


@pytest.mark.asyncio
async def test_search_semantic_scholar_fails_soft(monkeypatch):
    import lens.acquire.semantic_scholar as s2

    async def boom(client, url, headers=None):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(s2, "fetch_with_retry", boom)
    monkeypatch.setattr(s2, "RATE_LIMIT_SECONDS", 0)

    assert await s2.search_semantic_scholar("anything") == []


@pytest.mark.asyncio
async def test_search_semantic_scholar_non_dict_body(monkeypatch):
    import lens.acquire.semantic_scholar as s2

    class FakeResp:
        def json(self):
            return None  # valid JSON, but not a dict

    async def fake_fetch(client, url, headers=None):
        return FakeResp()

    monkeypatch.setattr(s2, "fetch_with_retry", fake_fetch)
    monkeypatch.setattr(s2, "RATE_LIMIT_SECONDS", 0)

    assert await s2.search_semantic_scholar("anything") == []

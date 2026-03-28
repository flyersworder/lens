"""Tests for the monitor pipeline."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_run_monitor_cycle(tmp_path):
    from lens.monitor.watcher import run_monitor_cycle
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    store.add_rows(
        "taxonomy_versions",
        [
            {
                "version_id": 1,
                "created_at": "2026-03-21T00:00:00",
                "paper_count": 0,
                "param_count": 0,
                "principle_count": 0,
                "slot_count": 0,
                "variant_count": 0,
                "pattern_count": 0,
            },
        ],
    )

    mock_llm = AsyncMock()
    mock_llm.complete.return_value = '{"tradeoffs": [], "architecture": [], "agentic": []}'

    import httpx

    arxiv_xml = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.99999v1</id>
        <title>New LLM Paper</title>
        <summary>A new paper about LLMs.</summary>
        <published>2024-01-15T00:00:00Z</published>
        <author><name>Author</name></author>
      </entry>
    </feed>"""

    mock_http = AsyncMock()
    mock_http.get.return_value = httpx.Response(200, text=arxiv_xml)
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "lens.acquire.arxiv.httpx.AsyncClient",
        return_value=mock_http,
    ):
        result = await run_monitor_cycle(
            store=store,
            llm_client=mock_llm,
            query="LLM",
            categories=["cs.CL"],
            max_results=10,
        )

    assert result["papers_acquired"] >= 0
    assert "papers_extracted" in result


@pytest.mark.asyncio
async def test_run_monitor_cycle_no_taxonomy(tmp_path):
    from lens.monitor.watcher import run_monitor_cycle
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    mock_llm = AsyncMock()
    mock_http = AsyncMock()
    mock_http.get.return_value = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "lens.acquire.arxiv.httpx.AsyncClient",
        return_value=mock_http,
    ):
        result = await run_monitor_cycle(
            store=store,
            llm_client=mock_llm,
            query="LLM",
            categories=["cs.CL"],
            max_results=10,
        )

    assert "papers_acquired" in result

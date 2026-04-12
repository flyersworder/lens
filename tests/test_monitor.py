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


@pytest.mark.asyncio
async def test_monitor_cycle_with_build(tmp_path, monkeypatch):
    """Monitor cycle with run_build=True should call build_vocabulary and build_matrix."""
    from unittest.mock import AsyncMock

    from lens.monitor.watcher import run_monitor_cycle
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    mock_llm = AsyncMock()

    async def fake_fetch(**kwargs):
        return []

    monkeypatch.setattr("lens.monitor.watcher.fetch_arxiv_papers", fake_fetch)

    from lens.taxonomy.vocabulary import load_seed_vocabulary

    load_seed_vocabulary(store)

    result = await run_monitor_cycle(
        store,
        mock_llm,
        run_build=True,
        run_ideation_flag=False,
    )
    assert "taxonomy_built" in result
    assert result["taxonomy_built"] is True


@pytest.mark.asyncio
async def test_monitor_cycle_skip_build(tmp_path, monkeypatch):
    """Monitor cycle with run_build=False should skip taxonomy and matrix build."""
    from unittest.mock import AsyncMock

    from lens.monitor.watcher import run_monitor_cycle
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    mock_llm = AsyncMock()

    async def fake_fetch(**kwargs):
        return []

    monkeypatch.setattr("lens.monitor.watcher.fetch_arxiv_papers", fake_fetch)

    result = await run_monitor_cycle(
        store,
        mock_llm,
        run_build=False,
        run_ideation_flag=False,
    )
    assert result["taxonomy_built"] is False
    assert result["matrix_built"] is False


@pytest.mark.asyncio
async def test_monitor_cycle_skip_enrich(tmp_path, monkeypatch):
    """Monitor cycle with run_enrich=False should skip OpenAlex enrichment."""
    from unittest.mock import AsyncMock

    from lens.monitor.watcher import run_monitor_cycle
    from lens.store.models import EMBEDDING_DIM
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.db"))
    store.init_tables()

    mock_llm = AsyncMock()

    paper = {
        "paper_id": "test-001",
        "title": "Test Paper",
        "abstract": "Test abstract.",
        "authors": ["Author"],
        "date": "2025-01-01",
        "arxiv_id": "2501.00001",
        "extraction_status": "pending",
        "embedding": [0.0] * EMBEDDING_DIM,
    }

    async def fake_fetch(**kwargs):
        return [paper]

    monkeypatch.setattr("lens.monitor.watcher.fetch_arxiv_papers", fake_fetch)

    async def fake_extract(store, client, concurrency=3, session_id=None):
        return 0

    monkeypatch.setattr("lens.monitor.watcher.extract_papers", fake_extract)

    result = await run_monitor_cycle(
        store,
        mock_llm,
        run_enrich=False,
        run_build=False,
        run_ideation_flag=False,
    )
    assert result["papers_enriched"] == 0

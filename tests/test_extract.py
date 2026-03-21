"""Tests for the extraction pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_build_extraction_prompt_abstract_only():
    from lens.extract.prompts import build_extraction_prompt

    prompt = build_extraction_prompt(
        title="Attention Is All You Need",
        abstract="We propose a new simple network architecture...",
    )
    assert "Attention Is All You Need" in prompt
    assert "network architecture" in prompt
    assert "tradeoff" in prompt.lower() or "TradeoffExtraction" in prompt
    assert "ArchitectureExtraction" in prompt or "architecture" in prompt.lower()
    assert "AgenticExtraction" in prompt or "agentic" in prompt.lower()
    assert "confidence" in prompt.lower()


def test_build_extraction_prompt_with_full_text():
    from lens.extract.prompts import build_extraction_prompt

    prompt = build_extraction_prompt(
        title="Test Paper",
        abstract="Short abstract.",
        full_text="This is the full text of the paper with much more detail...",
    )
    assert "full text" in prompt.lower() or "This is the full text" in prompt


def test_build_extraction_prompt_confidence_anchors():
    from lens.extract.prompts import build_extraction_prompt

    prompt = build_extraction_prompt(title="T", abstract="A")
    assert "0.9" in prompt or "explicitly stated" in prompt.lower()
    assert "0.5" in prompt


def test_build_extraction_prompt_empty_list_instruction():
    from lens.extract.prompts import build_extraction_prompt

    prompt = build_extraction_prompt(title="T", abstract="A")
    assert "empty" in prompt.lower()


def test_extraction_response_schema():
    from lens.extract.prompts import EXTRACTION_RESPONSE_SCHEMA

    schema = EXTRACTION_RESPONSE_SCHEMA
    assert "tradeoffs" in schema
    assert "architecture" in schema
    assert "agentic" in schema


def test_parse_extraction_response():
    from lens.extract.extractor import parse_extraction_response

    fixture = (FIXTURE_DIR / "extraction_response.json").read_text()
    result = parse_extraction_response(fixture, paper_id="2005.14165")
    assert result is not None
    tradeoffs, architecture, agentic = result

    assert len(tradeoffs) == 1
    assert tradeoffs[0]["paper_id"] == "2005.14165"
    assert tradeoffs[0]["improves"] == "model quality across benchmarks"
    assert tradeoffs[0]["confidence"] == 0.92

    assert len(architecture) == 1
    assert architecture[0]["component_slot"] == "architecture class"
    assert architecture[0]["replaces"] is None

    assert len(agentic) == 0


def test_parse_extraction_response_malformed():
    from lens.extract.extractor import parse_extraction_response

    result = parse_extraction_response("not json at all", paper_id="test")
    assert result is None


def test_parse_extraction_response_partial():
    from lens.extract.extractor import parse_extraction_response

    partial = (
        '{"tradeoffs": [{"improves": "a", "worsens": "b", "technique": "c",'
        ' "context": "", "confidence": 0.8, "evidence_quote": "q"}]}'
    )
    result = parse_extraction_response(partial, paper_id="test")
    assert result is not None
    tradeoffs, architecture, agentic = result
    assert len(tradeoffs) == 1
    assert len(architecture) == 0
    assert len(agentic) == 0


def test_parse_extraction_response_strips_markdown_fences():
    from lens.extract.extractor import parse_extraction_response

    fenced = '```json\n{"tradeoffs": [], "architecture": [], "agentic": []}\n```'
    result = parse_extraction_response(fenced, paper_id="test")
    assert result is not None


@pytest.mark.asyncio
async def test_extract_paper():
    from lens.extract.extractor import extract_paper

    fixture = (FIXTURE_DIR / "extraction_response.json").read_text()
    mock_client = AsyncMock()
    mock_client.complete.return_value = fixture

    result = await extract_paper(
        paper_id="2005.14165",
        title="Language Models are Few-Shot Learners",
        abstract="We demonstrate that scaling up language models...",
        llm_client=mock_client,
    )
    assert result is not None
    tradeoffs, architecture, agentic = result
    assert len(tradeoffs) == 1
    assert len(architecture) == 1


@pytest.mark.asyncio
async def test_extract_paper_retries_on_malformed():
    from lens.extract.extractor import extract_paper

    fixture = (FIXTURE_DIR / "extraction_response.json").read_text()
    mock_client = AsyncMock()
    mock_client.complete.side_effect = ["not json", fixture]

    result = await extract_paper(
        paper_id="2005.14165",
        title="Test",
        abstract="Test abstract",
        llm_client=mock_client,
    )
    assert result is not None
    assert mock_client.complete.call_count == 2


@pytest.mark.asyncio
async def test_extract_paper_returns_none_after_retries():
    from lens.extract.extractor import extract_paper

    mock_client = AsyncMock()
    mock_client.complete.return_value = "still not json"

    result = await extract_paper(
        paper_id="test",
        title="Test",
        abstract="Test",
        llm_client=mock_client,
    )
    assert result is None
    assert mock_client.complete.call_count == 2


@pytest.mark.asyncio
async def test_extract_papers_batch(tmp_path):
    import polars as pl

    from lens.extract.extractor import extract_papers
    from lens.store.store import LensStore

    fixture = (FIXTURE_DIR / "extraction_response.json").read_text()
    mock_client = AsyncMock()
    mock_client.complete.return_value = fixture

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    store.add_papers(
        [
            {
                "paper_id": "2005.14165",
                "arxiv_id": "2005.14165",
                "title": "Language Models are Few-Shot Learners",
                "abstract": "We demonstrate that scaling up language models...",
                "authors": ["Brown"],
                "date": "2020-05-28",
                "venue": None,
                "citations": 0,
                "quality_score": 0.5,
                "extraction_status": "pending",
                "embedding": [0.0] * 768,
            }
        ]
    )

    count = await extract_papers(store, mock_client, concurrency=1)
    assert count == 1

    # Check extractions were stored
    tradeoffs = store.get_table("tradeoff_extractions").to_polars()
    assert len(tradeoffs) == 1

    arch = store.get_table("architecture_extractions").to_polars()
    assert len(arch) == 1

    # Check paper status updated to 'complete'
    papers = store.get_table("papers").to_polars()
    status = papers.filter(pl.col("paper_id") == "2005.14165")["extraction_status"][0]
    assert status == "complete"


@pytest.mark.asyncio
async def test_extract_papers_skips_completed(tmp_path):
    from lens.extract.extractor import extract_papers
    from lens.store.store import LensStore

    mock_client = AsyncMock()

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    store.add_papers(
        [
            {
                "paper_id": "already_done",
                "arxiv_id": "already_done",
                "title": "Already extracted",
                "abstract": "Test",
                "authors": [],
                "date": "2024-01-01",
                "venue": None,
                "citations": 0,
                "quality_score": 0.0,
                "extraction_status": "complete",
                "embedding": [0.0] * 768,
            }
        ]
    )

    count = await extract_papers(store, mock_client, concurrency=1)
    assert count == 0
    mock_client.complete.assert_not_called()

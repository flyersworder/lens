"""Tests for the extraction pipeline."""

from pathlib import Path

import pytest
from conftest import make_llm_client

from lens.store.models import EMBEDDING_DIM

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
    """The response schema is now derived from the Pydantic model.

    Replaces the hand-written EXTRACTION_RESPONSE_SCHEMA string, so the prompt,
    the request and the validation gate cannot drift apart.
    """
    from lens.llm.schemas import ExtractionResponse
    from lens.llm.structured import strict_schema

    schema = strict_schema(ExtractionResponse)

    assert set(schema["required"]) == {"tradeoffs", "architecture", "agentic"}


def test_compute_verification_status():
    from lens.extract.extractor import compute_verification_status

    # High confidence + substantive quote.
    assert compute_verification_status(0.9, "this is a long enough quote") == "verified"
    # High confidence, no quote field (arch/agentic case).
    assert compute_verification_status(0.9) == "verified"
    # High confidence but tiny quote — demoted to inferred.
    assert compute_verification_status(0.9, "x") == "inferred"
    # Medium confidence.
    assert compute_verification_status(0.6, "some quote here") == "inferred"
    # Low confidence.
    assert compute_verification_status(0.3, "some quote here") == "unverified"


@pytest.mark.asyncio
async def test_extract_paper():
    from lens.extract.extractor import extract_paper

    fixture = (FIXTURE_DIR / "extraction_response.json").read_text()
    client, fake = make_llm_client([fixture])

    result = await extract_paper(
        paper_id="2005.14165",
        title="Language Models are Few-Shot Learners",
        abstract="We demonstrate that scaling up language models...",
        llm_client=client,
    )
    assert result is not None
    tradeoffs, architecture, agentic = result
    assert len(tradeoffs) == 1
    assert len(architecture) == 1
    # Probed for schema support, then fell back on this stub endpoint.
    assert fake.calls == [True, False]


@pytest.mark.asyncio
async def test_extract_paper_retries_on_malformed():
    """A response that cannot be validated earns one corrective retry."""
    from lens.extract.extractor import extract_paper

    fixture = (FIXTURE_DIR / "extraction_response.json").read_text()
    client, fake = make_llm_client(["not json", fixture])

    result = await extract_paper(
        paper_id="2005.14165",
        title="Test",
        abstract="Test abstract",
        llm_client=client,
    )

    assert result is not None
    tradeoffs, _, _ = result
    assert tradeoffs[0]["paper_id"] == "2005.14165"


@pytest.mark.asyncio
async def test_extract_paper_returns_none_after_retries():
    """Persistent garbage yields None rather than a partial extraction."""
    from lens.extract.extractor import extract_paper

    client, _ = make_llm_client(["still not json", "also not json"])

    result = await extract_paper(paper_id="test", title="Test", abstract="Test", llm_client=client)
    assert result is None


@pytest.mark.asyncio
async def test_extract_paper_handles_llm_exception():
    """A transport failure is swallowed into None, not raised to the caller."""
    from lens.extract.extractor import extract_paper

    client, _ = make_llm_client([RuntimeError("API down"), RuntimeError("API down")])

    result = await extract_paper(paper_id="test", title="Test", abstract="Test", llm_client=client)
    assert result is None


@pytest.mark.asyncio
async def test_extract_papers_batch(tmp_path):
    from lens.extract.extractor import extract_papers
    from lens.store.store import LensStore

    fixture = (FIXTURE_DIR / "extraction_response.json").read_text()
    client, _fake = make_llm_client([fixture])

    store = LensStore(str(tmp_path / "test.db"))
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
                "embedding": [0.0] * EMBEDDING_DIM,
            }
        ]
    )

    count = await extract_papers(store, client, concurrency=1)
    assert count == 1

    # Check extractions were stored
    tradeoffs = store.query("tradeoff_extractions")
    assert len(tradeoffs) == 1

    arch = store.query("architecture_extractions")
    assert len(arch) == 1

    # Check paper status updated to 'complete'
    papers = store.query("papers", "paper_id = ?", ("2005.14165",))
    assert papers[0]["extraction_status"] == "complete"


@pytest.mark.asyncio
async def test_extract_papers_skips_completed(tmp_path):
    from lens.extract.extractor import extract_papers
    from lens.store.store import LensStore

    # No payloads: any LLM call at all would raise, proving none was made.
    client, fake = make_llm_client([])

    store = LensStore(str(tmp_path / "test.db"))
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
                "embedding": [0.0] * EMBEDDING_DIM,
            }
        ]
    )

    count = await extract_papers(store, client, concurrency=1)
    assert count == 0
    # Real evidence the endpoint was never touched, rather than a mock assertion
    # that would now pass vacuously because the code calls complete_structured.
    assert fake.calls == []

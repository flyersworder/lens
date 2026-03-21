"""Tests for the extraction pipeline."""


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

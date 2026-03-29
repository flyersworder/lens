"""Tests for guided extraction prompt."""

from lens.extract.prompts import build_extraction_prompt


def test_prompt_includes_vocabulary():
    vocabulary = [
        {"name": "Inference Latency", "kind": "parameter"},
        {"name": "Model Accuracy", "kind": "parameter"},
        {"name": "Quantization", "kind": "principle"},
    ]
    prompt = build_extraction_prompt(
        title="Test Paper",
        abstract="Test abstract",
        vocabulary=vocabulary,
    )
    assert "Inference Latency" in prompt
    assert "Model Accuracy" in prompt
    assert "Quantization" in prompt
    assert "Parameters:" in prompt
    assert "Principles:" in prompt
    assert "NEW:" in prompt


def test_prompt_without_vocabulary_still_works():
    prompt = build_extraction_prompt(
        title="Test Paper",
        abstract="Test abstract",
    )
    assert "Test Paper" in prompt
    assert "tradeoffs" in prompt.lower()

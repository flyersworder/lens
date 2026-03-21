# Plan 3: Extract — LLM Extraction Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the LLM extraction pipeline that processes papers and extracts tradeoffs, architecture contributions, and agentic patterns into structured Layer 1 data.

**Architecture:** A single LLM call per paper extracts all three tuple types simultaneously. The prompt provides the paper text (title + abstract, or full text for PDFs) and requests structured JSON output matching the Pydantic extraction models. litellm abstracts the LLM provider. Concurrent processing with configurable limits. Retry with stricter prompt on malformed output. Partial results stored on second failure.

**Tech Stack:** litellm (LLM abstraction), existing Pydantic models (TradeoffExtraction, ArchitectureExtraction, AgenticExtraction), asyncio for concurrency

**Spec:** `docs/specs/design.md` (Stage 2: Extract, lines 307-330; Error Handling, line 644; Config, lines 537-540)

---

## File Structure

```
src/lens/
├── extract/
│   ├── __init__.py           # Public API: extract_papers, extract_paper
│   ├── extractor.py          # Core extraction logic — LLM call, parsing, retry
│   └── prompts.py            # Extraction prompt template
├── llm/
│   ├── __init__.py           # Re-exports
│   └── client.py             # LLM client wrapper around litellm
tests/
├── test_extract.py           # Extraction pipeline tests
├── test_llm_client.py        # LLM client tests
└── fixtures/
    └── extraction_response.json  # Recorded LLM extraction output
```

---

### Task 1: LLM Client Wrapper

**Files:**
- Create: `src/lens/llm/__init__.py`
- Create: `src/lens/llm/client.py`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_client.py
"""Tests for LLM client wrapper."""

import pytest
from unittest.mock import AsyncMock, patch


def test_llm_client_init():
    from lens.llm.client import LLMClient

    client = LLMClient(model="test/model")
    assert client.model == "test/model"


def test_llm_client_default_model():
    from lens.llm.client import LLMClient

    client = LLMClient()
    assert client.model == "openrouter/anthropic/claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_llm_client_complete():
    from lens.llm.client import LLMClient

    client = LLMClient(model="test/model")
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = '{"result": "test"}'

    with patch("lens.llm.client.litellm.acompletion", return_value=mock_response):
        result = await client.complete(
            messages=[{"role": "user", "content": "test"}]
        )
        assert result == '{"result": "test"}'


@pytest.mark.asyncio
async def test_llm_client_complete_with_system():
    from lens.llm.client import LLMClient

    client = LLMClient(model="test/model")
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "response"

    with patch("lens.llm.client.litellm.acompletion", return_value=mock_response) as mock_call:
        await client.complete(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        )
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args
        assert call_kwargs.kwargs["model"] == "test/model"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_llm_client.py -v
```

- [ ] **Step 3: Implement LLM client**

```python
# src/lens/llm/__init__.py
"""LENS LLM abstraction layer."""

from lens.llm.client import LLMClient

__all__ = ["LLMClient"]
```

```python
# src/lens/llm/client.py
"""LLM client wrapper around litellm.

Provides a simple async interface for LLM completions.
Supports any provider via litellm's model routing.
"""

from __future__ import annotations

from typing import Any

import litellm

DEFAULT_MODEL = "openrouter/anthropic/claude-sonnet-4-6"


class LLMClient:
    """Async LLM client using litellm for provider abstraction.

    Args:
        model: litellm model string (e.g., 'openrouter/anthropic/claude-sonnet-4-6').
        temperature: Sampling temperature (0.0 for deterministic extraction).
        max_tokens: Maximum response tokens.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send a completion request and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional litellm parameters.

        Returns:
            The assistant's response text.
        """
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_llm_client.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/lens/llm/ tests/test_llm_client.py
git commit -m "feat: add LLM client wrapper around litellm"
```

---

### Task 2: Extraction Prompt

**Files:**
- Create: `src/lens/extract/__init__.py`
- Create: `src/lens/extract/prompts.py`
- Create: `tests/test_extract.py` (prompt tests first, extractor tests in Task 3)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_extract.py
"""Tests for the extraction pipeline."""

import json


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
    # Spec: confidence anchors 0.9+, 0.7-0.9, 0.5-0.7, <0.5
    assert "0.9" in prompt or "explicitly stated" in prompt.lower()
    assert "0.5" in prompt


def test_build_extraction_prompt_empty_list_instruction():
    from lens.extract.prompts import build_extraction_prompt

    prompt = build_extraction_prompt(title="T", abstract="A")
    # Spec: "explicitly instructs the LLM to return empty lists rather than fabricate"
    assert "empty" in prompt.lower()


def test_extraction_response_schema():
    from lens.extract.prompts import EXTRACTION_RESPONSE_SCHEMA

    schema = EXTRACTION_RESPONSE_SCHEMA
    assert "tradeoffs" in schema
    assert "architecture" in schema
    assert "agentic" in schema
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_extract.py -v
```

- [ ] **Step 3: Implement extraction prompt**

```python
# src/lens/extract/__init__.py
"""LENS extraction pipeline."""
```

```python
# src/lens/extract/prompts.py
"""Extraction prompt template for LLM-based paper analysis.

Adapted from Trapp & Warschat (2024) single-prompt contradiction extraction.
A single prompt per paper extracts all three tuple types:
tradeoffs, architecture contributions, and agentic patterns.
"""

from __future__ import annotations

EXTRACTION_RESPONSE_SCHEMA = """{
  "tradeoffs": [
    {
      "improves": "what the technique improves (e.g., 'mathematical reasoning accuracy')",
      "worsens": "what gets worse as a result (e.g., 'inference time per token')",
      "technique": "the technique or method used (e.g., 'chain-of-thought with self-consistency')",
      "context": "conditions or constraints mentioned",
      "confidence": 0.85,
      "evidence_quote": "relevant sentence from the paper"
    }
  ],
  "architecture": [
    {
      "component_slot": "the architecture component category (e.g., 'attention mechanism')",
      "variant_name": "the specific variant introduced (e.g., 'grouped-query attention')",
      "replaces": "what it replaces or generalizes (null if novel)",
      "key_properties": "key properties or advantages",
      "confidence": 0.9
    }
  ],
  "agentic": [
    {
      "pattern_name": "name of the agent pattern (e.g., 'reflexion')",
      "structure": "high-level structure (e.g., 'single agent with self-critique loop')",
      "use_case": "primary use case",
      "components": ["list", "of", "components"],
      "confidence": 0.8
    }
  ]
}"""


def build_extraction_prompt(
    title: str,
    abstract: str,
    full_text: str | None = None,
) -> str:
    """Build the extraction prompt for a single paper.

    Args:
        title: Paper title.
        abstract: Paper abstract.
        full_text: Optional full paper text (for seed/high-quality papers or PDFs).

    Returns:
        The complete prompt string to send to the LLM.
    """
    paper_content = f"Title: {title}\n\nAbstract: {abstract}"
    if full_text:
        paper_content += f"\n\nFull text:\n{full_text}"

    return f"""You are an expert in LLM research. Analyze the following paper and extract structured information.

## Paper
{paper_content}

## Task
Extract three types of structured information from this paper. Return a JSON object with three arrays: "tradeoffs", "architecture", and "agentic". Each array may be empty — return empty arrays rather than fabricating extractions when the paper does not contain relevant information for that category.

### 1. Tradeoffs (TradeoffExtraction)
Identify engineering tradeoffs: when improving one aspect worsens another.
- "improves": what the technique/method improves
- "worsens": what gets worse as a consequence
- "technique": the specific technique or method
- "context": conditions, benchmarks, or constraints mentioned
- "confidence": your confidence score (see scale below)
- "evidence_quote": a relevant sentence from the paper

### 2. Architecture Contributions (ArchitectureExtraction)
Identify novel or notable architecture components.
- "component_slot": the category (e.g., attention mechanism, positional encoding, normalization, FFN, activation function, MoE routing)
- "variant_name": the specific variant name
- "replaces": what it replaces/generalizes (null if entirely novel)
- "key_properties": key properties or advantages
- "confidence": your confidence score

### 3. Agentic Patterns (AgenticExtraction)
Identify LLM agent design patterns.
- "pattern_name": name of the pattern
- "structure": high-level description of the agent structure
- "use_case": primary use case or application
- "components": list of key components (e.g., ["actor", "evaluator", "memory"])
- "confidence": your confidence score

## Confidence Scale
- 0.9-1.0: Explicitly stated in the paper text
- 0.7-0.9: Strongly implied by the results or methodology
- 0.5-0.7: Inferred from context but not directly stated
- Below 0.5: Speculative — include only if potentially valuable

## Response Format
Return ONLY valid JSON matching this schema:
{EXTRACTION_RESPONSE_SCHEMA}

Do not include any text outside the JSON object."""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_extract.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/lens/extract/ tests/test_extract.py
git commit -m "feat: add extraction prompt template with confidence anchors"
```

---

### Task 3: Core Extractor

**Files:**
- Create: `src/lens/extract/extractor.py`
- Create: `tests/fixtures/extraction_response.json`
- Modify: `tests/test_extract.py` — add extractor tests

- [ ] **Step 1: Create extraction response fixture**

```json
{
  "tradeoffs": [
    {
      "improves": "model quality across benchmarks",
      "worsens": "training compute cost",
      "technique": "scaling model parameters",
      "context": "GPT-3 175B vs smaller variants",
      "confidence": 0.92,
      "evidence_quote": "We find that scaling up language models greatly improves task-agnostic, few-shot performance"
    }
  ],
  "architecture": [
    {
      "component_slot": "architecture class",
      "variant_name": "autoregressive transformer decoder",
      "replaces": null,
      "key_properties": "decoder-only transformer with learned positional embeddings, 175B parameters",
      "confidence": 0.95
    }
  ],
  "agentic": []
}
```

- [ ] **Step 2: Write failing extractor tests** (append to `tests/test_extract.py`)

```python
# Append to tests/test_extract.py

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_parse_extraction_response():
    from lens.extract.extractor import parse_extraction_response

    fixture = (FIXTURE_DIR / "extraction_response.json").read_text()
    tradeoffs, architecture, agentic = parse_extraction_response(fixture, paper_id="2005.14165")

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

    # Only tradeoffs present, missing other keys
    partial = '{"tradeoffs": [{"improves": "a", "worsens": "b", "technique": "c", "context": "", "confidence": 0.8, "evidence_quote": "q"}]}'
    tradeoffs, architecture, agentic = parse_extraction_response(partial, paper_id="test")
    assert len(tradeoffs) == 1
    assert len(architecture) == 0
    assert len(agentic) == 0


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
    # First call returns garbage, second returns valid JSON
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
    assert mock_client.complete.call_count == 2  # initial + 1 retry
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_extract.py -v
```

- [ ] **Step 4: Implement core extractor**

```python
# src/lens/extract/extractor.py
"""Core extraction logic — LLM call, JSON parsing, retry.

For each paper, sends a single prompt to the LLM and parses the structured
JSON response into extraction tuples. Retries once with a stricter prompt
on malformed output.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from lens.extract.prompts import build_extraction_prompt
from lens.llm.client import LLMClient
from lens.store.models import AgenticExtraction, ArchitectureExtraction, TradeoffExtraction

logger = logging.getLogger(__name__)

# Type alias for extraction results
ExtractionTuple = tuple[
    list[dict[str, Any]],  # tradeoffs
    list[dict[str, Any]],  # architecture
    list[dict[str, Any]],  # agentic
]


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        # Find first { and last } for robust extraction
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
    return text


def _validate_tradeoff(raw: dict, paper_id: str) -> dict | None:
    """Validate a tradeoff dict against the Pydantic model."""
    raw["paper_id"] = paper_id
    try:
        obj = TradeoffExtraction(**raw)
        return obj.model_dump()
    except (ValidationError, TypeError):
        logger.warning(f"Invalid tradeoff extraction for {paper_id}: {raw}")
        return None


def _validate_architecture(raw: dict, paper_id: str) -> dict | None:
    """Validate an architecture dict against the Pydantic model."""
    raw["paper_id"] = paper_id
    try:
        obj = ArchitectureExtraction(**raw)
        return obj.model_dump()
    except (ValidationError, TypeError):
        logger.warning(f"Invalid architecture extraction for {paper_id}: {raw}")
        return None


def _validate_agentic(raw: dict, paper_id: str) -> dict | None:
    """Validate an agentic dict against the Pydantic model."""
    raw["paper_id"] = paper_id
    try:
        obj = AgenticExtraction(**raw)
        return obj.model_dump()
    except (ValidationError, TypeError):
        logger.warning(f"Invalid agentic extraction for {paper_id}: {raw}")
        return None


def parse_extraction_response(
    response_text: str,
    paper_id: str,
) -> ExtractionTuple | None:
    """Parse LLM JSON response into validated extraction dicts.

    Returns a tuple of (tradeoffs, architecture, agentic) lists,
    or None if the response is not valid JSON.
    Individual invalid entries are skipped with a warning.
    """
    text = _strip_code_fences(response_text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    tradeoffs = [
        v for t in data.get("tradeoffs", [])
        if (v := _validate_tradeoff(t, paper_id)) is not None
    ]
    architecture = [
        v for a in data.get("architecture", [])
        if (v := _validate_architecture(a, paper_id)) is not None
    ]
    agentic = [
        v for ag in data.get("agentic", [])
        if (v := _validate_agentic(ag, paper_id)) is not None
    ]

    return tradeoffs, architecture, agentic


async def extract_paper(
    paper_id: str,
    title: str,
    abstract: str,
    llm_client: LLMClient,
    full_text: str | None = None,
) -> ExtractionTuple | None:
    """Extract tradeoffs, architecture, and agentic patterns from a paper.

    Makes one LLM call. On malformed JSON, retries once with a stricter prompt.
    Returns None if both attempts fail.

    Args:
        paper_id: Unique paper identifier.
        title: Paper title.
        abstract: Paper abstract.
        llm_client: LLM client for completions.
        full_text: Optional full paper text.

    Returns:
        Tuple of (tradeoffs, architecture, agentic) extraction lists, or None.
    """
    prompt = build_extraction_prompt(title, abstract, full_text=full_text)
    messages = [{"role": "user", "content": prompt}]

    # First attempt
    try:
        response = await llm_client.complete(messages)
    except Exception:
        logger.warning(f"LLM call failed for paper {paper_id}")
        return None

    result = parse_extraction_response(response, paper_id)
    if result is not None:
        return result

    # Retry with stricter prompt
    logger.info(f"Retrying extraction for {paper_id} with stricter prompt")
    retry_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
        {
            "role": "user",
            "content": (
                "Your previous response was not valid JSON. "
                "Please respond with ONLY a valid JSON object matching the schema. "
                "No markdown, no explanation, just the JSON."
            ),
        },
    ]

    try:
        response = await llm_client.complete(retry_messages)
    except Exception:
        logger.warning(f"LLM retry failed for paper {paper_id}")
        return None

    return parse_extraction_response(response, paper_id)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_extract.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/lens/extract/extractor.py tests/test_extract.py tests/fixtures/extraction_response.json
git commit -m "feat: add core extractor with JSON parsing, retry, and LLM integration"
```

---

### Task 4: Batch Extraction Pipeline and Storage

**Files:**
- Modify: `src/lens/extract/extractor.py` — add `extract_papers` batch function
- Modify: `src/lens/extract/__init__.py` — add exports
- Modify: `tests/test_extract.py` — add batch + storage tests

- [ ] **Step 1: Write failing tests** (append to `tests/test_extract.py`)

```python
# Append to tests/test_extract.py


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

    # Add a test paper
    store.add_papers([{
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
    }])

    count = await extract_papers(store, mock_client, concurrency=1)
    assert count == 1

    # Check extractions were stored
    tradeoffs = store.get_table("tradeoff_extractions").to_polars()
    assert len(tradeoffs) == 1

    arch = store.get_table("architecture_extractions").to_polars()
    assert len(arch) == 1

    # Check paper status updated to 'complete'
    papers = store.get_table("papers").to_polars()
    assert papers.filter(pl.col("paper_id") == "2005.14165")["extraction_status"][0] == "complete"


@pytest.mark.asyncio
async def test_extract_papers_skips_completed(tmp_path):
    from lens.extract.extractor import extract_papers
    from lens.store.store import LensStore

    mock_client = AsyncMock()

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    store.add_papers([{
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
    }])

    count = await extract_papers(store, mock_client, concurrency=1)
    assert count == 0
    mock_client.complete.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_extract.py::test_extract_papers_batch -v
```

- [ ] **Step 3: Implement batch extraction**

Add to `src/lens/extract/extractor.py`:

```python
import asyncio
import polars as pl
from lens.store.store import LensStore


def _delete_old_extractions(store: LensStore, paper_id: str) -> None:
    """Delete previous extractions for a paper (idempotent re-extraction)."""
    for table_name in ["tradeoff_extractions", "architecture_extractions", "agentic_extractions"]:
        try:
            store.get_table(table_name).delete(f"paper_id = '{paper_id}'")
        except Exception:
            pass  # table may be empty, delete with no matches is fine


def _update_paper_status(store: LensStore, paper_id: str, status: str) -> None:
    """Update a paper's extraction_status."""
    try:
        store.get_table("papers").update(
            where=f"paper_id = '{paper_id}'",
            values={"extraction_status": status},
        )
    except Exception:
        logger.warning(f"Failed to update status for {paper_id}")


async def extract_papers(
    store: LensStore,
    llm_client: LLMClient,
    concurrency: int = 5,
    paper_id: str | None = None,
) -> int:
    """Extract knowledge from all pending papers in the store.

    Idempotent: re-extraction deletes old results before storing new ones.
    Updates extraction_status to 'complete' or 'incomplete'.

    Args:
        store: LensStore instance.
        llm_client: LLM client for completions.
        concurrency: Max concurrent LLM calls.
        paper_id: If set, only extract this specific paper (even if already complete).

    Returns:
        Number of papers successfully extracted.
    """
    papers_df = store.get_table("papers").to_polars()

    if paper_id:
        papers_df = papers_df.filter(pl.col("paper_id") == paper_id)
    else:
        papers_df = papers_df.filter(
            pl.col("extraction_status").is_in(["pending", "incomplete"])
        )

    if len(papers_df) == 0:
        logger.info("No papers to extract")
        return 0

    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(row: dict) -> bool:
        async with semaphore:
            pid = row["paper_id"]

            # Delete old extractions for idempotent re-extraction
            _delete_old_extractions(store, pid)

            result = await extract_paper(
                paper_id=pid,
                title=row["title"],
                abstract=row["abstract"],
                llm_client=llm_client,
            )

            if result is None:
                _update_paper_status(store, pid, "incomplete")
                logger.warning(f"Extraction failed for {pid}")
                return False

            tradeoffs, architecture, agentic = result
            if tradeoffs:
                store.add_rows("tradeoff_extractions", tradeoffs)
            if architecture:
                store.add_rows("architecture_extractions", architecture)
            if agentic:
                store.add_rows("agentic_extractions", agentic)

            _update_paper_status(store, pid, "complete")
            logger.info(
                f"Extracted {pid}: "
                f"{len(tradeoffs)} tradeoffs, "
                f"{len(architecture)} arch, "
                f"{len(agentic)} agentic"
            )
            return True

    tasks = [process_one(row) for row in papers_df.to_dicts()]
    results = await asyncio.gather(*tasks)
    success_count = sum(1 for r in results if r)

    logger.info(f"Extracted {success_count}/{len(papers_df)} papers")
    return success_count
```

- [ ] **Step 4: Update extract __init__.py**

```python
# src/lens/extract/__init__.py
"""LENS extraction pipeline."""

from lens.extract.extractor import extract_paper, extract_papers, parse_extraction_response

__all__ = ["extract_paper", "extract_papers", "parse_extraction_response"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_extract.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/lens/extract/ tests/test_extract.py
git commit -m "feat: add batch extraction pipeline with concurrency and storage"
```

---

### Task 5: Wire Up CLI Extract Command

**Files:**
- Modify: `src/lens/cli.py` — replace extract stub
- Modify: `src/lens/__init__.py` — add extract exports (optional)

- [ ] **Step 1: Replace the extract stub in cli.py**

Replace the `extract` command with a real implementation that:
- Loads config to get `extract_model` (or uses `--model` override)
- Creates `LensStore` and `LLMClient`
- Calls `asyncio.run(extract_papers(store, client, concurrency, paper_id))`
- Prints the result count

```python
@app.command()
def extract(
    paper_id: str | None = typer.Option(None, "--paper-id", help="Extract specific paper."),
    model: str | None = typer.Option(None, "--model", help="LLM model override."),
    concurrency: int = typer.Option(5, "--concurrency", help="Concurrent LLM calls."),
) -> None:
    """Extract tradeoffs, architecture, and agentic patterns from papers."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    llm_model = model or config["llm"]["extract_model"]

    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.extract.extractor import extract_papers
    from lens.llm.client import LLMClient

    client = LLMClient(model=llm_model)
    count = asyncio.run(extract_papers(store, client, concurrency=concurrency, paper_id=paper_id))
    rprint(f"[green]Extracted {count} papers[/green]")
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest -v
```

All tests should pass (previous 70 + new ~12 = ~82 total).

- [ ] **Step 3: Verify CLI**

```bash
uv run lens extract --help
```

- [ ] **Step 4: Commit**

```bash
git add src/lens/cli.py
git commit -m "feat: wire up extract CLI command with model and concurrency options"
```

---

## Summary

After completing this plan, LENS can:
- **Call any LLM** via litellm (Claude, Gemini, OpenAI, etc.) through a clean async client
- **Extract structured data** from papers: tradeoffs, architecture contributions, agentic patterns
- **Parse LLM responses** with JSON validation, markdown fence stripping, and retry on malformed output
- **Process papers in batch** with configurable concurrency
- **Store extractions** in LanceDB (tradeoff_extractions, architecture_extractions, agentic_extractions)
- **Skip already-extracted papers** (idempotent)

The `lens extract` CLI command is functional with `--model`, `--paper-id`, and `--concurrency` options.

**Next:** Plan 4 (Taxonomy + Structure) — clustering extractions into parameters/principles, building the contradiction matrix and catalogs.

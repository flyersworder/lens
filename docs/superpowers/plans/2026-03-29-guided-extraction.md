# Guided Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace HDBSCAN clustering with LLM-guided extraction using a canonical vocabulary for the tradeoff pipeline.

**Architecture:** A new `vocabulary` table replaces `parameters` and `principles`. The extraction prompt injects the vocabulary so the LLM outputs canonical names directly. `build_taxonomy()` splits into three focused functions. All downstream consumers (serve, ideation, CLI) migrate from parameters/principles to vocabulary.

**Tech Stack:** Python, SQLite + sqlite-vec, Pydantic, Typer, openai SDK (via LLMClient)

**Spec:** `docs/superpowers/specs/2026-03-29-guided-extraction-design.md`

---

### Task 1: Add VocabularyEntry model and vocabulary table schema

**Files:**
- Modify: `src/lens/store/models.py`
- Modify: `src/lens/store/store.py`
- Test: `tests/test_vocabulary.py` (create)

- [ ] **Step 1: Write the failing test for VocabularyEntry model**

Create `tests/test_vocabulary.py`:

```python
"""Tests for canonical vocabulary."""

from lens.store.models import VocabularyEntry


def test_vocabulary_entry_validates():
    entry = VocabularyEntry(
        id="inference-latency",
        name="Inference Latency",
        kind="parameter",
        description="Time required to generate output from input at deployment",
        source="seed",
        first_seen="2026-03-29",
        paper_count=0,
        avg_confidence=0.0,
    )
    assert entry.id == "inference-latency"
    assert entry.kind == "parameter"


def test_vocabulary_entry_kind_validation():
    import pytest

    with pytest.raises(Exception):
        VocabularyEntry(
            id="bad",
            name="Bad",
            kind="invalid",
            description="test",
            source="seed",
            first_seen="2026-03-29",
            paper_count=0,
            avg_confidence=0.0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vocabulary.py -v`
Expected: FAIL — `VocabularyEntry` does not exist

- [ ] **Step 3: Add VocabularyEntry model**

In `src/lens/store/models.py`, add after the `Principle` class (after line 121):

```python
class VocabularyEntry(BaseModel):
    """A canonical parameter or principle in the vocabulary."""

    id: str
    name: str
    kind: str  # "parameter" or "principle"
    description: str
    source: str  # "seed" or "extracted"
    first_seen: str
    paper_count: int = 0
    avg_confidence: float = 0.0
    embedding: list[float] = []

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, v: str) -> str:
        if v not in ("parameter", "principle"):
            raise ValueError(f"kind must be 'parameter' or 'principle', got '{v}'")
        return v

    @field_validator("source")
    @classmethod
    def _check_source(cls, v: str) -> str:
        if v not in ("seed", "extracted"):
            raise ValueError(f"source must be 'seed' or 'extracted', got '{v}'")
        return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vocabulary.py -v`
Expected: PASS

- [ ] **Step 5: Write failing test for vocabulary table creation**

Append to `tests/test_vocabulary.py`:

```python
from lens.store.store import LensStore


def test_vocabulary_table_exists(tmp_path):
    store = LensStore(tmp_path / "test.db")
    # Should be able to query the vocabulary table without error
    rows = store.query("vocabulary")
    assert rows == []
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_vocabulary.py::test_vocabulary_table_exists -v`
Expected: FAIL — table `vocabulary` does not exist

- [ ] **Step 7: Add vocabulary table DDL and update VEC_TABLES and JSON_FIELDS**

In `src/lens/store/store.py`:

Add to `_TABLE_DDL` list (after the `agentic_patterns` table, before `matrix_cells`):

```python
    """CREATE TABLE IF NOT EXISTS vocabulary (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        description TEXT NOT NULL,
        source TEXT NOT NULL,
        first_seen TEXT NOT NULL,
        paper_count INTEGER NOT NULL DEFAULT 0,
        avg_confidence REAL NOT NULL DEFAULT 0.0
    )""",
```

Update `VEC_TABLES` — add vocabulary entry:

```python
VEC_TABLES: dict[str, tuple[str, str]] = {
    "papers": ("paper_id", "TEXT"),
    "parameters": ("id", "INTEGER"),
    "principles": ("id", "INTEGER"),
    "vocabulary": ("id", "TEXT"),
    "architecture_variants": ("id", "INTEGER"),
    "agentic_patterns": ("id", "INTEGER"),
}
```

Note: Keep `parameters` and `principles` in VEC_TABLES for now — they will be removed in Task 8 after all consumers are migrated.

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_vocabulary.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/lens/store/models.py src/lens/store/store.py tests/test_vocabulary.py
git commit -m "feat: add VocabularyEntry model and vocabulary table schema"
```

---

### Task 2: Add seed vocabulary data and `vocab init` logic

**Files:**
- Create: `src/lens/taxonomy/vocabulary.py`
- Test: `tests/test_vocabulary.py` (append)

- [ ] **Step 1: Write failing test for seed vocabulary loading**

Append to `tests/test_vocabulary.py`:

```python
from lens.taxonomy.vocabulary import SEED_VOCABULARY, load_seed_vocabulary


def test_seed_vocabulary_has_expected_entries():
    params = [e for e in SEED_VOCABULARY if e["kind"] == "parameter"]
    principles = [e for e in SEED_VOCABULARY if e["kind"] == "principle"]
    assert len(params) == 12
    assert len(principles) == 12


def test_load_seed_vocabulary(tmp_path):
    store = LensStore(tmp_path / "test.db")
    count = load_seed_vocabulary(store)
    assert count == 24

    rows = store.query("vocabulary")
    assert len(rows) == 24
    # Check a known entry
    latency = [r for r in rows if r["id"] == "inference-latency"]
    assert len(latency) == 1
    assert latency[0]["name"] == "Inference Latency"
    assert latency[0]["kind"] == "parameter"
    assert latency[0]["source"] == "seed"


def test_load_seed_vocabulary_is_idempotent(tmp_path):
    store = LensStore(tmp_path / "test.db")
    load_seed_vocabulary(store)
    count = load_seed_vocabulary(store)
    assert count == 0  # nothing new inserted

    rows = store.query("vocabulary")
    assert len(rows) == 24
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vocabulary.py::test_seed_vocabulary_has_expected_entries -v`
Expected: FAIL — cannot import `SEED_VOCABULARY`

- [ ] **Step 3: Create vocabulary module with seed data and loader**

Create `src/lens/taxonomy/vocabulary.py`:

```python
"""Canonical vocabulary for guided extraction — seed data and management."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    """Convert a display name to a URL-safe slug ID."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


SEED_VOCABULARY: list[dict[str, str]] = [
    # Parameters
    {"name": "Inference Latency", "kind": "parameter",
     "description": "Time required to generate output from input at deployment"},
    {"name": "Model Accuracy", "kind": "parameter",
     "description": "Quality of model predictions on target tasks"},
    {"name": "Training Cost", "kind": "parameter",
     "description": "Compute, time, and financial cost to train or fine-tune"},
    {"name": "Model Size", "kind": "parameter",
     "description": "Number of parameters in the model"},
    {"name": "Memory Usage", "kind": "parameter",
     "description": "RAM and VRAM required during inference or training"},
    {"name": "Context Length", "kind": "parameter",
     "description": "Maximum input sequence length the model can process"},
    {"name": "Safety/Alignment", "kind": "parameter",
     "description": "Degree to which model outputs align with human values and intent"},
    {"name": "Reasoning Capability", "kind": "parameter",
     "description": "Ability to perform multi-step logical or mathematical reasoning"},
    {"name": "Data Efficiency", "kind": "parameter",
     "description": "Amount of training data needed to reach target performance"},
    {"name": "Generalization", "kind": "parameter",
     "description": "Ability to perform well on unseen tasks or domains"},
    {"name": "Interpretability", "kind": "parameter",
     "description": "Degree to which model decisions can be understood by humans"},
    {"name": "Robustness", "kind": "parameter",
     "description": "Resilience to adversarial inputs, noise, and distribution shift"},
    # Principles
    {"name": "Knowledge Distillation", "kind": "principle",
     "description": "Training a smaller model to mimic a larger teacher model"},
    {"name": "Quantization", "kind": "principle",
     "description": "Reducing numerical precision of model weights and activations"},
    {"name": "Sparse Attention/MoE", "kind": "principle",
     "description": "Activating only a subset of parameters or attention heads per input"},
    {"name": "RAG", "kind": "principle",
     "description": "Augmenting generation with retrieved external knowledge"},
    {"name": "Chain-of-Thought", "kind": "principle",
     "description": "Prompting or training models to produce intermediate reasoning steps"},
    {"name": "Preference Optimization (RLHF/DPO)", "kind": "principle",
     "description": "Aligning model outputs to human preferences via reward signals"},
    {"name": "Parameter-Efficient Fine-Tuning (LoRA/QLoRA)", "kind": "principle",
     "description": "Adapting models by training a small number of added parameters"},
    {"name": "Speculative Decoding", "kind": "principle",
     "description": "Using a fast draft model to propose tokens verified by a larger model"},
    {"name": "Flash Attention", "kind": "principle",
     "description": "Memory-efficient attention computation via tiling and recomputation"},
    {"name": "Positional Encoding Innovation", "kind": "principle",
     "description": "Novel methods for representing token position in sequences"},
    {"name": "Scaling", "kind": "principle",
     "description": "Increasing model size, data, or compute to improve performance"},
    {"name": "Multi-Agent Collaboration", "kind": "principle",
     "description": "Multiple LLM agents coordinating to solve complex tasks"},
]


def load_seed_vocabulary(store: LensStore) -> int:
    """Load seed vocabulary into the database. Idempotent — skips existing entries.

    Returns the number of newly inserted entries.
    """
    existing = store.query("vocabulary")
    existing_ids = {r["id"] for r in existing}
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    new_rows: list[dict[str, Any]] = []
    for entry in SEED_VOCABULARY:
        entry_id = _slugify(entry["name"])
        if entry_id in existing_ids:
            continue
        new_rows.append({
            "id": entry_id,
            "name": entry["name"],
            "kind": entry["kind"],
            "description": entry["description"],
            "source": "seed",
            "first_seen": today,
            "paper_count": 0,
            "avg_confidence": 0.0,
        })

    if new_rows:
        store.add_rows("vocabulary", new_rows)
        logger.info("Loaded %d seed vocabulary entries", len(new_rows))

    return len(new_rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vocabulary.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/lens/taxonomy/vocabulary.py tests/test_vocabulary.py
git commit -m "feat: add seed vocabulary data and loader"
```

---

### Task 3: Add `process_new_concepts()` and vocabulary stats update

**Files:**
- Modify: `src/lens/taxonomy/vocabulary.py`
- Test: `tests/test_vocabulary.py` (append)

- [ ] **Step 1: Write failing test for process_new_concepts**

Append to `tests/test_vocabulary.py`:

```python
from lens.taxonomy.vocabulary import process_new_concepts


def test_process_new_concepts_accepts_new_entries(tmp_path):
    store = LensStore(tmp_path / "test.db")
    load_seed_vocabulary(store)

    # Simulate extractions with a NEW: concept
    store.add_rows("tradeoff_extractions", [
        {
            "paper_id": "paper1",
            "improves": "NEW: Energy Efficiency",
            "worsens": "Model Accuracy",
            "technique": "Quantization",
            "context": "test",
            "confidence": 0.8,
            "evidence_quote": "test quote",
            "new_concept_description": "Power consumption relative to compute throughput",
        },
    ])

    stats = process_new_concepts(store)
    assert stats["new_entries"] == 1

    rows = store.query("vocabulary", "id = ?", ("energy-efficiency",))
    assert len(rows) == 1
    assert rows[0]["name"] == "Energy Efficiency"
    assert rows[0]["kind"] == "parameter"
    assert rows[0]["source"] == "extracted"
    assert rows[0]["description"] == "Power consumption relative to compute throughput"


def test_process_new_concepts_updates_paper_count(tmp_path):
    store = LensStore(tmp_path / "test.db")
    load_seed_vocabulary(store)

    store.add_rows("tradeoff_extractions", [
        {
            "paper_id": "paper1",
            "improves": "Inference Latency",
            "worsens": "Model Accuracy",
            "technique": "Quantization",
            "context": "test",
            "confidence": 0.9,
            "evidence_quote": "quote1",
            "new_concept_description": None,
        },
        {
            "paper_id": "paper2",
            "improves": "Inference Latency",
            "worsens": "Training Cost",
            "technique": "Quantization",
            "context": "test",
            "confidence": 0.7,
            "evidence_quote": "quote2",
            "new_concept_description": None,
        },
    ])

    process_new_concepts(store)

    latency = store.query("vocabulary", "id = ?", ("inference-latency",))
    assert latency[0]["paper_count"] == 2
    assert latency[0]["avg_confidence"] == 0.8  # (0.9 + 0.7) / 2

    quant = store.query("vocabulary", "id = ?", ("quantization",))
    assert quant[0]["paper_count"] == 2
    assert quant[0]["avg_confidence"] == 0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vocabulary.py::test_process_new_concepts_accepts_new_entries -v`
Expected: FAIL — cannot import `process_new_concepts`

- [ ] **Step 3: Update tradeoff_extractions table to include new_concept_description column**

In `src/lens/store/store.py`, update the `tradeoff_extractions` DDL:

```python
    """CREATE TABLE IF NOT EXISTS tradeoff_extractions (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id TEXT NOT NULL,
        improves TEXT NOT NULL,
        worsens TEXT NOT NULL,
        technique TEXT NOT NULL,
        context TEXT NOT NULL,
        confidence REAL NOT NULL,
        evidence_quote TEXT NOT NULL,
        new_concept_description TEXT
    )""",
```

In `src/lens/store/models.py`, update the `TradeoffExtraction` class:

```python
class TradeoffExtraction(BaseModel):
    """A single tradeoff extracted from a paper."""

    paper_id: str
    improves: str
    worsens: str
    technique: str
    context: str
    confidence: float
    evidence_quote: str
    new_concept_description: str | None = None
```

- [ ] **Step 4: Implement process_new_concepts**

Add to `src/lens/taxonomy/vocabulary.py`:

```python
def process_new_concepts(store: LensStore) -> dict[str, int]:
    """Scan extractions for NEW: concepts, accept them, and update vocabulary stats.

    Returns dict with keys: new_entries, updated_entries.
    """
    extractions = store.query("tradeoff_extractions")
    if not extractions:
        return {"new_entries": 0, "updated_entries": 0}

    existing = store.query("vocabulary")
    existing_by_name: dict[str, dict[str, Any]] = {r["name"]: r for r in existing}
    existing_ids: set[str] = {r["id"] for r in existing}
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    # Collect all concept references: name -> list of (paper_id, confidence, kind)
    references: dict[str, list[tuple[str, float, str]]] = {}
    new_concepts: dict[str, dict[str, str]] = {}  # name -> {kind, description}

    for ext in extractions:
        for field, kind in [
            ("improves", "parameter"),
            ("worsens", "parameter"),
            ("technique", "principle"),
        ]:
            raw_value = ext[field]
            if raw_value.startswith("NEW: "):
                name = raw_value[5:].strip()
                if name not in new_concepts:
                    desc = ext.get("new_concept_description") or f"Extracted concept: {name}"
                    new_concepts[name] = {"kind": kind, "description": desc}
            else:
                name = raw_value

            references.setdefault(name, []).append(
                (ext["paper_id"], ext["confidence"], kind)
            )

    # Insert new concepts
    new_rows: list[dict[str, Any]] = []
    for name, info in new_concepts.items():
        entry_id = _slugify(name)
        if entry_id in existing_ids:
            continue
        new_rows.append({
            "id": entry_id,
            "name": name,
            "kind": info["kind"],
            "description": info["description"],
            "source": "extracted",
            "first_seen": today,
            "paper_count": 0,
            "avg_confidence": 0.0,
        })
        existing_ids.add(entry_id)
        existing_by_name[name] = new_rows[-1]

    if new_rows:
        store.add_rows("vocabulary", new_rows)
        logger.info("Accepted %d new vocabulary entries", len(new_rows))

    # Update paper_count and avg_confidence for all referenced concepts
    updated = 0
    for name, refs in references.items():
        if name not in existing_by_name:
            continue
        entry = existing_by_name[name]
        entry_id = entry["id"]
        unique_papers = {r[0] for r in refs}
        avg_conf = sum(r[1] for r in refs) / len(refs)
        store.conn.execute(
            "UPDATE vocabulary SET paper_count = ?, avg_confidence = ? WHERE id = ?",
            (len(unique_papers), round(avg_conf, 4), entry_id),
        )
        updated += 1

    store.conn.commit()
    return {"new_entries": len(new_rows), "updated_entries": updated}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_vocabulary.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/lens/taxonomy/vocabulary.py src/lens/store/models.py src/lens/store/store.py tests/test_vocabulary.py
git commit -m "feat: add process_new_concepts and new_concept_description field"
```

---

### Task 4: Update extraction prompt with guided vocabulary

**Files:**
- Modify: `src/lens/extract/prompts.py`
- Modify: `src/lens/extract/extractor.py`
- Test: `tests/test_extraction.py` (append or create)

- [ ] **Step 1: Write failing test for vocabulary injection in prompt**

Create or append to `tests/test_extraction.py`:

```python
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
    assert "NEW:" in prompt  # Instructions for new concepts


def test_prompt_without_vocabulary_still_works():
    prompt = build_extraction_prompt(
        title="Test Paper",
        abstract="Test abstract",
    )
    assert "Test Paper" in prompt
    assert "tradeoffs" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_extraction.py::test_prompt_includes_vocabulary -v`
Expected: FAIL — `build_extraction_prompt` does not accept `vocabulary` parameter

- [ ] **Step 3: Update prompt builder to accept and inject vocabulary**

In `src/lens/extract/prompts.py`, update the `EXTRACTION_RESPONSE_SCHEMA` to include the new field:

```python
EXTRACTION_RESPONSE_SCHEMA = """{
  "tradeoffs": [
    {
      "improves": "what the technique improves",
      "worsens": "what gets worse as a result",
      "technique": "the technique or method used",
      "context": "conditions or constraints mentioned",
      "confidence": 0.85,
      "evidence_quote": "relevant sentence from the paper",
      "new_concept_description": null
    }
  ],
  "architecture": [
    {
      "component_slot": "architecture component category",
      "variant_name": "specific variant introduced",
      "replaces": "what it replaces or generalizes (null if novel)",
      "key_properties": "key properties or advantages",
      "confidence": 0.9
    }
  ],
  "agentic": [
    {
      "pattern_name": "name of the agent pattern",
      "structure": "high-level structure description",
      "use_case": "primary use case",
      "components": ["list", "of", "components"],
      "confidence": 0.8
    }
  ]
}"""
```

Replace `_TRADEOFFS_SECTION` with a function that builds it dynamically:

```python
def _build_tradeoffs_section(
    vocabulary: list[dict[str, str]] | None = None,
) -> str:
    """Build the tradeoffs section, optionally with guided vocabulary."""
    base = (
        "### 1. Tradeoffs (TradeoffExtraction)\n"
        "Identify engineering tradeoffs: when improving one aspect worsens another.\n"
    )

    if vocabulary:
        params = [v["name"] for v in vocabulary if v["kind"] == "parameter"]
        principles = [v["name"] for v in vocabulary if v["kind"] == "principle"]
        base += (
            "\nUse EXACT names from the vocabulary below for improves, worsens, "
            "and technique fields.\n"
            "\nParameters:\n"
            + "\n".join(f"- {p}" for p in params)
            + "\n\nPrinciples:\n"
            + "\n".join(f"- {p}" for p in principles)
            + "\n\nIf a concept genuinely does not match any entry above, prefix it "
            "with NEW: (e.g., \"NEW: Energy Efficiency\") and set "
            "new_concept_description to a one-line definition.\n"
        )

    base += (
        '\n- "improves": what the technique/method improves (use a Parameter name)\n'
        '- "worsens": what gets worse as a consequence (use a Parameter name)\n'
        '- "technique": the specific technique or method (use a Principle name)\n'
        '- "context": conditions, benchmarks, or constraints mentioned\n'
        '- "confidence": your confidence score (see scale below)\n'
        '- "evidence_quote": a relevant sentence from the paper\n'
        '- "new_concept_description": one-line definition if using NEW: prefix, '
        "else null"
    )
    return base
```

Update `build_extraction_prompt` to accept vocabulary:

```python
def build_extraction_prompt(
    title: str,
    abstract: str,
    full_text: str | None = None,
    vocabulary: list[dict[str, str]] | None = None,
) -> str:
    """Build the extraction prompt for a single paper."""
    paper_content = f"Title: {title}\n\nAbstract: {abstract}"
    if full_text:
        paper_content += f"\n\nFull text:\n{full_text}"

    intro = (
        "You are an expert in LLM research. Analyze the following paper and extract"
        " structured information."
    )
    response_format = (
        "## Response Format\n"
        "Return ONLY valid JSON matching this schema:\n"
        f"{EXTRACTION_RESPONSE_SCHEMA}\n\n"
        "Do not include any text outside the JSON object."
    )

    sections = [
        intro,
        f"## Paper\n{paper_content}",
        _TASK_SECTION,
        _build_tradeoffs_section(vocabulary),
        _ARCHITECTURE_SECTION,
        _AGENTIC_SECTION,
        _CONFIDENCE_SECTION,
        response_format,
    ]
    return "\n\n".join(sections)
```

Remove the old `_TRADEOFFS_SECTION` constant.

- [ ] **Step 4: Update extractor.py to pass vocabulary to prompt builder**

In `src/lens/extract/extractor.py`, update `extract_paper`:

```python
async def extract_paper(
    paper_id: str,
    title: str,
    abstract: str,
    llm_client: LLMClient,
    full_text: str | None = None,
    vocabulary: list[dict[str, str]] | None = None,
) -> ExtractionTuple | None:
    prompt = build_extraction_prompt(title, abstract, full_text=full_text, vocabulary=vocabulary)
    # ... rest unchanged
```

Update `extract_papers` to load and pass vocabulary:

```python
async def extract_papers(
    store: LensStore,
    llm_client: LLMClient,
    concurrency: int = 5,
    paper_id: str | None = None,
) -> int:
    # ... existing paper loading ...

    # Load vocabulary for guided extraction
    vocab_rows = store.query("vocabulary")
    vocabulary = [{"name": r["name"], "kind": r["kind"]} for r in vocab_rows] if vocab_rows else None

    # ... in extract_one:
    async def extract_one(row: dict) -> tuple[str, ExtractionTuple | None]:
        async with semaphore:
            pid = row["paper_id"]
            result = await extract_paper(
                paper_id=pid,
                title=row["title"],
                abstract=row["abstract"],
                llm_client=llm_client,
                vocabulary=vocabulary,
            )
            return pid, result

    # ... rest unchanged
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_extraction.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run existing extraction tests to verify no regressions**

Run: `uv run pytest tests/ -k "extract" -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/lens/extract/prompts.py src/lens/extract/extractor.py tests/test_extraction.py
git commit -m "feat: inject vocabulary into extraction prompt for guided tradeoffs"
```

---

### Task 5: Rewrite matrix.py to use vocabulary lookups

**Files:**
- Modify: `src/lens/knowledge/matrix.py`
- Modify: `tests/test_matrix.py`

- [ ] **Step 1: Write failing test for vocabulary-based matrix building**

Add to `tests/test_matrix.py` (or replace existing test depending on imports):

```python
def test_build_matrix_with_vocabulary(tmp_path):
    """Matrix build uses vocabulary name lookup instead of raw_strings."""
    from lens.store.store import LensStore
    from lens.knowledge.matrix import build_matrix
    from lens.taxonomy.vocabulary import load_seed_vocabulary

    store = LensStore(tmp_path / "test.db")
    load_seed_vocabulary(store)

    store.add_rows("tradeoff_extractions", [
        {
            "paper_id": "p1",
            "improves": "Inference Latency",
            "worsens": "Model Accuracy",
            "technique": "Quantization",
            "context": "test",
            "confidence": 0.9,
            "evidence_quote": "quote",
            "new_concept_description": None,
        },
        {
            "paper_id": "p2",
            "improves": "Inference Latency",
            "worsens": "Model Accuracy",
            "technique": "Quantization",
            "context": "test2",
            "confidence": 0.7,
            "evidence_quote": "quote2",
            "new_concept_description": None,
        },
    ])

    build_matrix(store)

    cells = store.query("matrix_cells")
    assert len(cells) == 1
    assert cells[0]["improving_param_id"] == "inference-latency"
    assert cells[0]["worsening_param_id"] == "model-accuracy"
    assert cells[0]["principle_id"] == "quantization"
    assert cells[0]["count"] == 2
    assert abs(cells[0]["avg_confidence"] - 0.8) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_matrix.py::test_build_matrix_with_vocabulary -v`
Expected: FAIL — current matrix uses integer IDs from parameters/principles tables

- [ ] **Step 3: Rewrite matrix.py to use vocabulary**

Replace the contents of `src/lens/knowledge/matrix.py`:

```python
"""Contradiction matrix — aggregates tradeoff extractions via vocabulary."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def _build_vocab_name_map(store: LensStore) -> dict[str, str]:
    """Build a map from vocabulary display name to ID."""
    rows = store.query("vocabulary")
    return {r["name"]: r["id"] for r in rows}


def build_matrix(store: LensStore) -> None:
    """Build the contradiction matrix from extractions + vocabulary.

    Full rebuild — deletes all existing cells first.
    Filters extractions to confidence >= 0.5.
    """
    with contextlib.suppress(OSError, ValueError):
        store.delete("matrix_cells", "1 = 1", ())

    vocab_map = _build_vocab_name_map(store)
    if not vocab_map:
        logger.info("No vocabulary entries — skipping matrix build")
        return

    extractions = store.query("tradeoff_extractions")
    if not extractions:
        logger.info("No extractions — skipping matrix build")
        return

    extractions = [e for e in extractions if e.get("confidence", 0) >= 0.5]

    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in extractions:
        # Strip NEW: prefix if present (already accepted into vocabulary)
        improves = row["improves"]
        if improves.startswith("NEW: "):
            improves = improves[5:].strip()
        worsens = row["worsens"]
        if worsens.startswith("NEW: "):
            worsens = worsens[5:].strip()
        technique = row["technique"]
        if technique.startswith("NEW: "):
            technique = technique[5:].strip()

        imp_id = vocab_map.get(improves)
        wors_id = vocab_map.get(worsens)
        tech_id = vocab_map.get(technique)

        if imp_id is None or wors_id is None or tech_id is None:
            continue

        key = (imp_id, wors_id, tech_id)
        cells.setdefault(key, []).append(row)

    cell_rows = []
    for (imp_id, wors_id, princ_id), matches in cells.items():
        count = len(matches)
        avg_conf = sum(m["confidence"] for m in matches) / count
        paper_ids = list({m["paper_id"] for m in matches})
        cell_rows.append({
            "improving_param_id": imp_id,
            "worsening_param_id": wors_id,
            "principle_id": princ_id,
            "count": count,
            "avg_confidence": round(avg_conf, 4),
            "paper_ids": paper_ids,
            "taxonomy_version": 0,  # Not version-dependent anymore
        })

    if cell_rows:
        store.add_rows("matrix_cells", cell_rows)
        logger.info("Built matrix with %d cells", len(cell_rows))


def get_ranked_matrix(
    store: LensStore,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Get the contradiction matrix with top-k principles per cell pair.

    Returns a list of dicts ranked by score (count * avg_confidence)
    within each (improving, worsening) pair, limited to top_k.
    """
    cells = store.query("matrix_cells")
    if not cells:
        return []

    for c in cells:
        c["score"] = c["count"] * c["avg_confidence"]

    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cells:
        key = (c["improving_param_id"], c["worsening_param_id"])
        groups[key].append(c)

    result: list[dict[str, Any]] = []
    for _key, group in sorted(groups.items()):
        group.sort(key=lambda x: x["score"], reverse=True)
        result.extend(group[:top_k])

    result.sort(key=lambda x: (x["improving_param_id"], x["worsening_param_id"]))
    return result
```

- [ ] **Step 4: Update matrix_cells schema for text IDs**

In `src/lens/store/store.py`, update the `matrix_cells` DDL:

```python
    """CREATE TABLE IF NOT EXISTS matrix_cells (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        improving_param_id TEXT NOT NULL,
        worsening_param_id TEXT NOT NULL,
        principle_id TEXT NOT NULL,
        count INTEGER NOT NULL,
        avg_confidence REAL NOT NULL,
        paper_ids TEXT NOT NULL,
        taxonomy_version INTEGER NOT NULL
    )""",
```

In `src/lens/store/models.py`, update `MatrixCell`:

```python
class MatrixCell(BaseModel):
    """One cell in the tradeoff matrix (improving × worsening × principle)."""

    improving_param_id: str
    worsening_param_id: str
    principle_id: str
    count: int
    avg_confidence: float
    paper_ids: list[str]
    taxonomy_version: int
```

- [ ] **Step 5: Update existing matrix tests**

Update `tests/test_matrix.py` — existing tests that used integer IDs from parameters/principles tables need to be rewritten to use vocabulary. Remove tests that depended on the old `_build_string_to_id_map` and the taxonomy_version parameter on `build_matrix`. Keep the confidence filtering and empty-database tests, adapting them to the new API.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_matrix.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/lens/knowledge/matrix.py src/lens/store/store.py src/lens/store/models.py tests/test_matrix.py
git commit -m "feat: rewrite matrix to use vocabulary name lookup"
```

---

### Task 6: Split build_taxonomy into three focused functions

**Files:**
- Modify: `src/lens/taxonomy/__init__.py`
- Modify: `src/lens/taxonomy/vocabulary.py`
- Modify: `tests/test_taxonomy.py`

- [ ] **Step 1: Write failing test for build_tradeoff_taxonomy**

Add to `tests/test_taxonomy.py`:

```python
from lens.taxonomy.vocabulary import load_seed_vocabulary
from lens.taxonomy import build_tradeoff_taxonomy


def test_build_tradeoff_taxonomy(tmp_path):
    store = LensStore(tmp_path / "test.db")
    load_seed_vocabulary(store)

    store.add_rows("tradeoff_extractions", [
        {
            "paper_id": "p1",
            "improves": "Inference Latency",
            "worsens": "Model Accuracy",
            "technique": "NEW: Pruning",
            "context": "test",
            "confidence": 0.85,
            "evidence_quote": "quote",
            "new_concept_description": "Removing unnecessary model weights",
        },
    ])

    stats = build_tradeoff_taxonomy(store)
    assert stats["new_entries"] == 1

    # Pruning should be in vocabulary now
    rows = store.query("vocabulary", "id = ?", ("pruning",))
    assert len(rows) == 1
    assert rows[0]["source"] == "extracted"
    assert rows[0]["paper_count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_taxonomy.py::test_build_tradeoff_taxonomy -v`
Expected: FAIL — cannot import `build_tradeoff_taxonomy`

- [ ] **Step 3: Implement build_tradeoff_taxonomy**

Add to `src/lens/taxonomy/vocabulary.py`:

```python
from lens.taxonomy.embedder import embed_strings


def build_tradeoff_taxonomy(
    store: LensStore,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
) -> dict[str, int]:
    """Build the tradeoff taxonomy: process new concepts, update stats, embed vocabulary.

    Returns dict with keys: new_entries, updated_entries.
    """
    stats = process_new_concepts(store)

    # Embed all vocabulary entries that lack embeddings
    vocab_rows = store.query("vocabulary")
    to_embed = [r for r in vocab_rows if not r.get("embedding")]

    if to_embed:
        texts = [f"{r['name']}: {r['description']}" for r in to_embed]
        embeddings = embed_strings(
            texts,
            provider=embedding_provider,
            model_name=embedding_model,
            api_base=embedding_api_base,
            api_key=embedding_api_key,
        )
        for row, emb in zip(to_embed, embeddings):
            store.upsert_embedding("vocabulary", row["id"], emb.tolist())

    logger.info(
        "Tradeoff taxonomy: %d new, %d updated, %d embedded",
        stats["new_entries"],
        stats["updated_entries"],
        len(to_embed),
    )
    return stats
```

Then in `src/lens/taxonomy/__init__.py`, add the import and a thin wrapper:

```python
from lens.taxonomy.vocabulary import build_tradeoff_taxonomy  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_taxonomy.py::test_build_tradeoff_taxonomy -v`
Expected: PASS

- [ ] **Step 5: Split architecture and agentic builders out of build_taxonomy**

In `src/lens/taxonomy/__init__.py`, extract the architecture section of `build_taxonomy` into `build_architecture_taxonomy()` and the agentic section into `build_agentic_taxonomy()`. Both keep their existing logic unchanged. The original `build_taxonomy` function is removed and replaced by calling all three in sequence.

Export all three from `__init__.py`:

```python
__all__ = [
    "build_tradeoff_taxonomy",
    "build_architecture_taxonomy",
    "build_agentic_taxonomy",
]
```

The signatures should be:

```python
async def build_architecture_taxonomy(
    store: LensStore,
    llm_client: LLMClient,
    min_cluster_size: int = 3,
    target_arch_variants: int = 20,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build architecture taxonomy (slots + variants). Returns (slot_entries, variant_entries)."""
    # ... existing architecture logic from build_taxonomy ...


async def build_agentic_taxonomy(
    store: LensStore,
    llm_client: LLMClient,
    min_cluster_size: int = 3,
    target_agentic_patterns: int = 15,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
) -> list[dict]:
    """Build agentic pattern taxonomy. Returns pattern_entries."""
    # ... existing agentic logic from build_taxonomy ...
```

- [ ] **Step 6: Update existing taxonomy tests**

Update `tests/test_taxonomy.py` — replace calls to `build_taxonomy()` with the appropriate split functions. The `test_build_taxonomy` test should call `build_tradeoff_taxonomy` + `build_architecture_taxonomy` + `build_agentic_taxonomy` in sequence.

- [ ] **Step 7: Run all taxonomy tests**

Run: `uv run pytest tests/test_taxonomy.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/lens/taxonomy/__init__.py src/lens/taxonomy/vocabulary.py tests/test_taxonomy.py
git commit -m "refactor: split build_taxonomy into three focused functions"
```

---

### Task 7: Migrate serve layer to vocabulary

**Files:**
- Modify: `src/lens/serve/explorer.py`
- Modify: `src/lens/serve/analyzer.py`
- Modify: `src/lens/serve/explainer.py`
- Modify: `tests/test_explorer.py`
- Modify: `tests/test_analyzer.py`
- Modify: `tests/test_explainer.py`

- [ ] **Step 1: Migrate explorer.py**

Replace `list_parameters` and `list_principles`:

```python
def list_parameters(store: LensStore) -> list[dict[str, Any]]:
    """List all vocabulary entries of kind 'parameter'."""
    rows = store.query("vocabulary", "kind = ?", ("parameter",))
    for r in rows:
        r.pop("embedding", None)
    return rows


def list_principles(store: LensStore) -> list[dict[str, Any]]:
    """List all vocabulary entries of kind 'principle'."""
    rows = store.query("vocabulary", "kind = ?", ("principle",))
    for r in rows:
        r.pop("embedding", None)
    return rows
```

Update `get_matrix_cell` and `list_matrix_overview` — change `improving_param_id` / `worsening_param_id` from int to str (text IDs). Remove `taxonomy_version` parameter from functions that no longer need it (vocabulary is not versioned).

- [ ] **Step 2: Migrate analyzer.py**

Update the `analyze` function: replace `store.query("parameters", ...)` with `store.query("vocabulary", "kind = ?", ("parameter",))` and `store.query("principles", ...)` with `store.query("vocabulary", "kind = ?", ("principle",))`. Use `r["id"]` (text) instead of `r["id"]` (int) for ID mapping.

- [ ] **Step 3: Migrate explainer.py**

Update `resolve_concept`: replace the loop over `("parameters", "parameter"), ("principles", "principle")` with a single search on `"vocabulary"`. The vector search on `vocabulary_vec` returns entries with a `kind` field to distinguish type.

Update `graph_walk`: replace `store.query("parameters", ...)` and `store.query("principles", ...)` with `store.query("vocabulary", ...)`. Build lookup maps from vocabulary entries:

```python
vocab = store.query("vocabulary")
param_entries = [v for v in vocab if v["kind"] == "parameter"]
princ_entries = [v for v in vocab if v["kind"] == "principle"]
id_to_name = {v["id"]: v["name"] for v in vocab}
```

- [ ] **Step 4: Update serve layer tests**

Update `tests/test_explorer.py`, `tests/test_analyzer.py`, `tests/test_explainer.py` — replace test setup that inserts into `parameters` and `principles` tables with `load_seed_vocabulary(store)` calls. Update assertions for text IDs instead of integer IDs.

- [ ] **Step 5: Run all serve layer tests**

Run: `uv run pytest tests/test_explorer.py tests/test_analyzer.py tests/test_explainer.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/lens/serve/explorer.py src/lens/serve/analyzer.py src/lens/serve/explainer.py tests/test_explorer.py tests/test_analyzer.py tests/test_explainer.py
git commit -m "refactor: migrate serve layer from parameters/principles to vocabulary"
```

---

### Task 8: Migrate ideation to vocabulary

**Files:**
- Modify: `src/lens/monitor/ideation.py`
- Modify: `tests/test_ideation.py`

- [ ] **Step 1: Migrate find_sparse_cells**

Replace `store.query("parameters", ...)` with `store.query("vocabulary", "kind = ?", ("parameter",))`. Use `r["id"]` (text) instead of integer IDs.

- [ ] **Step 2: Migrate find_cross_pollination**

Replace the direct SQL query on `parameters_vec`:

```python
vec_rows = store.conn.execute(
    "SELECT pv.id, pv.embedding FROM parameters_vec pv "
    "INNER JOIN parameters p ON pv.id = p.id "
    "WHERE p.taxonomy_version = ?",
    (taxonomy_version,),
).fetchall()
```

With:

```python
vec_rows = store.conn.execute(
    "SELECT vv.id, vv.embedding FROM vocabulary_vec vv "
    "INNER JOIN vocabulary v ON vv.id = v.id "
    "WHERE v.kind = 'parameter'",
).fetchall()
```

Update ID types from int to str throughout the function.

- [ ] **Step 3: Migrate run_ideation**

Replace parameter/principle name lookups:

```python
params = store.query("parameters", "taxonomy_version = ?", (taxonomy_version,))
princs = store.query("principles", "taxonomy_version = ?", (taxonomy_version,))
param_id_to_name = {p["id"]: p["name"] for p in params}
princ_id_to_name = {p["id"]: p["name"] for p in princs}
```

With:

```python
vocab = store.query("vocabulary")
param_id_to_name = {v["id"]: v["name"] for v in vocab if v["kind"] == "parameter"}
princ_id_to_name = {v["id"]: v["name"] for v in vocab if v["kind"] == "principle"}
```

- [ ] **Step 4: Update ideation tests**

Update `tests/test_ideation.py` — replace setup that inserts into `parameters`/`principles` with `load_seed_vocabulary(store)`. Update ID assertions for text IDs.

- [ ] **Step 5: Run ideation tests**

Run: `uv run pytest tests/test_ideation.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/lens/monitor/ideation.py tests/test_ideation.py
git commit -m "refactor: migrate ideation from parameters/principles to vocabulary"
```

---

### Task 9: Update CLI and config

**Files:**
- Modify: `src/lens/cli.py`
- Modify: `src/lens/config.py`

- [ ] **Step 1: Add vocab command group to CLI**

In `src/lens/cli.py`:

```python
vocab_app = typer.Typer(help="Manage the canonical vocabulary.")
app.add_typer(vocab_app, name="vocab")


@vocab_app.command(name="init")
def vocab_init() -> None:
    """Load seed vocabulary into the database."""
    from lens.taxonomy.vocabulary import load_seed_vocabulary

    store = _get_store()
    count = load_seed_vocabulary(store)
    if count:
        typer.echo(f"Loaded {count} seed vocabulary entries.")
    else:
        typer.echo("Vocabulary already initialized — no new entries.")


@vocab_app.command(name="list")
def vocab_list(
    kind: str | None = typer.Option(None, help="Filter by kind: parameter or principle"),
) -> None:
    """List vocabulary entries with evidence stats."""
    store = _get_store()
    if kind:
        rows = store.query("vocabulary", "kind = ?", (kind,))
    else:
        rows = store.query("vocabulary")

    if not rows:
        typer.echo("No vocabulary entries found.")
        return

    for r in rows:
        marker = "★" if r["source"] == "seed" else "◆"
        typer.echo(
            f"  {marker} {r['name']} ({r['kind']}) — "
            f"papers={r['paper_count']}, conf={r['avg_confidence']:.2f}"
        )


@vocab_app.command(name="show")
def vocab_show(
    entry_id: str = typer.Argument(..., help="Vocabulary entry ID (slug)"),
) -> None:
    """Show details for a vocabulary entry."""
    store = _get_store()
    rows = store.query("vocabulary", "id = ?", (entry_id,))
    if not rows:
        typer.echo(f"No vocabulary entry with ID '{entry_id}'")
        raise typer.Exit(1)

    r = rows[0]
    typer.echo(f"Name:        {r['name']}")
    typer.echo(f"Kind:        {r['kind']}")
    typer.echo(f"Description: {r['description']}")
    typer.echo(f"Source:      {r['source']}")
    typer.echo(f"First seen:  {r['first_seen']}")
    typer.echo(f"Papers:      {r['paper_count']}")
    typer.echo(f"Avg conf:    {r['avg_confidence']:.4f}")
```

- [ ] **Step 2: Update build commands**

Update `build_app` commands to use the split taxonomy functions:

```python
@build_app.command()
def taxonomy() -> None:
    """Build taxonomy from current extractions."""
    import asyncio
    from lens.taxonomy import build_tradeoff_taxonomy, build_architecture_taxonomy, build_agentic_taxonomy

    store = _get_store()
    config = load_config()
    emb_kwargs = _embedding_kwargs(config)
    client = LLMClient(**_llm_kwargs(config, key="label_model"))
    tax_cfg = config.get("taxonomy", {})

    # Tradeoff taxonomy (vocabulary-based)
    build_tradeoff_taxonomy(store, **emb_kwargs)

    # Architecture + agentic (clustering-based)
    asyncio.run(build_architecture_taxonomy(
        store, client,
        min_cluster_size=tax_cfg.get("min_cluster_size", 3),
        target_arch_variants=tax_cfg.get("target_arch_variants", 20),
        **emb_kwargs,
    ))
    asyncio.run(build_agentic_taxonomy(
        store, client,
        min_cluster_size=tax_cfg.get("min_cluster_size", 3),
        target_agentic_patterns=tax_cfg.get("target_agentic_patterns", 15),
        **emb_kwargs,
    ))
    typer.echo("Taxonomy built.")
```

Update the `build_all` and `build_matrix_cmd` commands similarly — remove `taxonomy_version` parameter from `build_matrix` calls.

- [ ] **Step 3: Update explore parameters/principles commands**

Update `explore parameters` and `explore principles` to call the updated `list_parameters(store)` and `list_principles(store)` (no taxonomy_version argument).

- [ ] **Step 4: Remove target_parameters and target_principles from config**

In `src/lens/config.py`, update `DEFAULT_CONFIG`:

```python
    "taxonomy": {
        "target_arch_variants": 20,
        "target_agentic_patterns": 15,
        "min_cluster_size": 3,
    },
```

- [ ] **Step 5: Run CLI smoke tests**

Run: `uv run lens vocab init && uv run lens vocab list`
Expected: Seed vocabulary loads and lists correctly

- [ ] **Step 6: Commit**

```bash
git add src/lens/cli.py src/lens/config.py
git commit -m "feat: add vocab CLI commands, update build/explore to use vocabulary"
```

---

### Task 10: Remove old parameters/principles tables and clean up

**Files:**
- Modify: `src/lens/store/store.py`
- Modify: `src/lens/store/models.py`

- [ ] **Step 1: Remove parameters and principles from VEC_TABLES**

In `src/lens/store/store.py`, update `VEC_TABLES`:

```python
VEC_TABLES: dict[str, tuple[str, str]] = {
    "papers": ("paper_id", "TEXT"),
    "vocabulary": ("id", "TEXT"),
    "architecture_variants": ("id", "INTEGER"),
    "agentic_patterns": ("id", "INTEGER"),
}
```

- [ ] **Step 2: Remove parameters and principles table DDL**

Remove these two entries from `_TABLE_DDL`:

```python
    """CREATE TABLE IF NOT EXISTS parameters (
        ...
    )""",
    """CREATE TABLE IF NOT EXISTS principles (
        ...
    )""",
```

- [ ] **Step 3: Remove parameters and principles from JSON_FIELDS**

In `src/lens/store/store.py`, remove the `"parameters"` and `"principles"` entries from `JSON_FIELDS`.

- [ ] **Step 4: Remove Parameter and Principle models**

In `src/lens/store/models.py`, remove the `Parameter` and `Principle` classes. Also update `ExplanationResult` if it references integer IDs — change `resolved_id: int` to `resolved_id: str`.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS — no remaining references to removed tables

- [ ] **Step 6: If any tests fail, fix remaining references**

Search for remaining references:

Run: `uv run pytest -v 2>&1 | head -50` — check for failures

Fix any remaining references to `"parameters"` or `"principles"` table queries.

- [ ] **Step 7: Commit**

```bash
git add src/lens/store/store.py src/lens/store/models.py
git commit -m "chore: remove parameters/principles tables, replaced by vocabulary"
```

---

### Task 11: Full integration test and final cleanup

**Files:**
- Test: `tests/test_vocabulary.py` (append integration test)

- [ ] **Step 1: Write end-to-end integration test**

Append to `tests/test_vocabulary.py`:

```python
def test_end_to_end_guided_extraction_pipeline(tmp_path):
    """Integration test: seed vocab -> extract -> process -> matrix."""
    store = LensStore(tmp_path / "test.db")

    # 1. Seed vocabulary
    count = load_seed_vocabulary(store)
    assert count == 24

    # 2. Simulate guided extraction results
    store.add_rows("tradeoff_extractions", [
        {
            "paper_id": "paper-a",
            "improves": "Inference Latency",
            "worsens": "Model Accuracy",
            "technique": "Quantization",
            "context": "4-bit quantization on 7B models",
            "confidence": 0.9,
            "evidence_quote": "We observe 2x speedup with 4-bit.",
            "new_concept_description": None,
        },
        {
            "paper_id": "paper-b",
            "improves": "Inference Latency",
            "worsens": "Model Accuracy",
            "technique": "Knowledge Distillation",
            "context": "GPT-4 to 1B student",
            "confidence": 0.85,
            "evidence_quote": "Student achieves 95% of teacher.",
            "new_concept_description": None,
        },
        {
            "paper_id": "paper-c",
            "improves": "NEW: Energy Efficiency",
            "worsens": "Training Cost",
            "technique": "Quantization",
            "context": "inference on edge devices",
            "confidence": 0.75,
            "evidence_quote": "40% less power at 4-bit.",
            "new_concept_description": "Power consumption relative to compute throughput",
        },
    ])

    # 3. Process new concepts
    stats = process_new_concepts(store)
    assert stats["new_entries"] == 1

    energy = store.query("vocabulary", "id = ?", ("energy-efficiency",))
    assert len(energy) == 1
    assert energy[0]["source"] == "extracted"

    # 4. Build matrix
    from lens.knowledge.matrix import build_matrix

    build_matrix(store)

    cells = store.query("matrix_cells")
    assert len(cells) >= 2  # At least the two distinct tradeoff patterns

    # 5. Verify matrix cell content
    il_ma = [
        c for c in cells
        if c["improving_param_id"] == "inference-latency"
        and c["worsening_param_id"] == "model-accuracy"
    ]
    assert len(il_ma) == 2  # Quantization and Knowledge Distillation

    ee_tc = [
        c for c in cells
        if c["improving_param_id"] == "energy-efficiency"
        and c["worsening_param_id"] == "training-cost"
    ]
    assert len(ee_tc) == 1
    assert ee_tc[0]["principle_id"] == "quantization"
```

- [ ] **Step 2: Run integration test**

Run: `uv run pytest tests/test_vocabulary.py::test_end_to_end_guided_extraction_pipeline -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_vocabulary.py
git commit -m "test: add end-to-end integration test for guided extraction pipeline"
```

---

## Execution Order & Dependencies

```
Task 1 (models + table schema)
  └─> Task 2 (seed vocabulary)
      └─> Task 3 (process_new_concepts)
          ├─> Task 4 (guided prompt)
          ├─> Task 5 (matrix rewrite)
          └─> Task 6 (taxonomy split)
              ├─> Task 7 (serve migration)
              ├─> Task 8 (ideation migration)
              └─> Task 9 (CLI + config)
                  └─> Task 10 (remove old tables)
                      └─> Task 11 (integration test)
```

Tasks 4, 5, and 6 can run in parallel after Task 3.
Tasks 7, 8, and 9 can run in parallel after Task 6.

# Unified Vocabulary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend guided vocabulary extraction to architecture and agentic patterns, eliminating HDBSCAN/KMeans clustering entirely.

**Architecture:** Expand the `vocabulary` table with `arch_slot` and `agentic_category` kind values. Update extraction prompts to inject these vocabularies. Replace clustering-based taxonomy builders with vocabulary-based `process_new_concepts()`. Remove clusterer, labeler, and HDBSCAN dependency.

**Tech Stack:** Python, SQLite + sqlite-vec, Pydantic, Typer, openai SDK

**Spec:** `docs/superpowers/specs/2026-03-29-unified-vocabulary-design.md`

---

### Task 1: Expand seed vocabulary with arch_slot and agentic_category

**Files:**
- Modify: `src/lens/store/models.py` — expand `VocabularyEntry.kind` validator
- Modify: `src/lens/taxonomy/vocabulary.py` — add seed entries
- Modify: `tests/test_vocabulary.py` — update tests

- [ ] **Step 1: Write failing test for new kinds**

Append to `tests/test_vocabulary.py`:

```python
def test_vocabulary_entry_arch_slot():
    entry = VocabularyEntry(
        id="attention-mechanism",
        name="Attention Mechanism",
        kind="arch_slot",
        description="How the model attends to different parts of the input",
        source="seed",
        first_seen="2026-03-29",
        paper_count=0,
        avg_confidence=0.0,
    )
    assert entry.kind == "arch_slot"


def test_vocabulary_entry_agentic_category():
    entry = VocabularyEntry(
        id="reasoning",
        name="Reasoning",
        kind="agentic_category",
        description="Patterns for multi-step logical inference and problem solving",
        source="seed",
        first_seen="2026-03-29",
        paper_count=0,
        avg_confidence=0.0,
    )
    assert entry.kind == "agentic_category"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vocabulary.py::test_vocabulary_entry_arch_slot -v`
Expected: FAIL — ValidationError, "arch_slot" not in allowed kinds

- [ ] **Step 3: Expand kind validator**

In `src/lens/store/models.py`, update `VocabularyEntry._check_kind`:

```python
    @field_validator("kind")
    @classmethod
    def _check_kind(cls, v: str) -> str:
        allowed = ("parameter", "principle", "arch_slot", "agentic_category")
        if v not in allowed:
            raise ValueError(f"kind must be one of {allowed}, got '{v}'")
        return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vocabulary.py::test_vocabulary_entry_arch_slot tests/test_vocabulary.py::test_vocabulary_entry_agentic_category -v`
Expected: PASS

- [ ] **Step 5: Write failing test for expanded seed vocabulary**

Append to `tests/test_vocabulary.py`:

```python
def test_seed_vocabulary_includes_all_kinds():
    kinds = {e["kind"] for e in SEED_VOCABULARY}
    assert kinds == {"parameter", "principle", "arch_slot", "agentic_category"}


def test_seed_vocabulary_has_expected_counts():
    by_kind = {}
    for e in SEED_VOCABULARY:
        by_kind.setdefault(e["kind"], []).append(e)
    assert len(by_kind["parameter"]) == 12
    assert len(by_kind["principle"]) == 12
    assert len(by_kind["arch_slot"]) == 10
    assert len(by_kind["agentic_category"]) == 6
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_vocabulary.py::test_seed_vocabulary_includes_all_kinds -v`
Expected: FAIL — only parameter and principle in seed data

- [ ] **Step 7: Add seed entries to SEED_VOCABULARY**

In `src/lens/taxonomy/vocabulary.py`, append to `SEED_VOCABULARY` after the principles:

```python
    # Architecture Slots
    {"name": "Attention Mechanism", "kind": "arch_slot",
     "description": "How the model attends to different parts of the input"},
    {"name": "Positional Encoding", "kind": "arch_slot",
     "description": "Methods for representing token position in sequences"},
    {"name": "FFN", "kind": "arch_slot",
     "description": "Feed-forward network layers within transformer blocks"},
    {"name": "Normalization", "kind": "arch_slot",
     "description": "Layer or batch normalization techniques"},
    {"name": "Activation Function", "kind": "arch_slot",
     "description": "Non-linear activation functions in the network"},
    {"name": "MoE Routing", "kind": "arch_slot",
     "description": "Routing strategies for mixture-of-experts architectures"},
    {"name": "Optimizer", "kind": "arch_slot",
     "description": "Training optimization algorithms and strategies"},
    {"name": "Loss Function", "kind": "arch_slot",
     "description": "Objective functions used during training"},
    {"name": "Quantization Method", "kind": "arch_slot",
     "description": "Techniques for reducing numerical precision"},
    {"name": "Retrieval Mechanism", "kind": "arch_slot",
     "description": "Methods for retrieving external knowledge"},
    # Agentic Categories
    {"name": "Reasoning", "kind": "agentic_category",
     "description": "Patterns for multi-step logical inference and problem solving"},
    {"name": "Planning", "kind": "agentic_category",
     "description": "Patterns for decomposing goals into executable steps"},
    {"name": "Tool Use", "kind": "agentic_category",
     "description": "Patterns for LLM interaction with external tools and APIs"},
    {"name": "Multi-Agent Collaboration", "kind": "agentic_category",
     "description": "Patterns involving multiple coordinating agents"},
    {"name": "Self-Reflection", "kind": "agentic_category",
     "description": "Patterns for self-evaluation and iterative improvement"},
    {"name": "Code Generation", "kind": "agentic_category",
     "description": "Patterns for generating, testing, and debugging code"},
```

Also update `test_seed_vocabulary_has_expected_entries` — change the count assertions:

```python
def test_seed_vocabulary_has_expected_entries():
    params = [e for e in SEED_VOCABULARY if e["kind"] == "parameter"]
    principles = [e for e in SEED_VOCABULARY if e["kind"] == "principle"]
    assert len(params) == 12
    assert len(principles) == 12
```

- [ ] **Step 8: Update load_seed_vocabulary test for new count**

Update `test_load_seed_vocabulary` to expect 40 entries (12+12+10+6) and `test_load_seed_vocabulary_is_idempotent` similarly.

- [ ] **Step 9: Run all vocabulary tests**

Run: `uv run pytest tests/test_vocabulary.py -v`
Expected: ALL PASS

- [ ] **Step 10: Commit**

```bash
git add src/lens/store/models.py src/lens/taxonomy/vocabulary.py tests/test_vocabulary.py
git commit -m "feat: expand vocabulary with arch_slot and agentic_category seed entries"
```

---

### Task 2: Update extraction schemas and prompts for architecture + agentic

**Files:**
- Modify: `src/lens/store/models.py` — add fields to ArchitectureExtraction, AgenticExtraction
- Modify: `src/lens/store/store.py` — add columns to DDL
- Modify: `src/lens/extract/prompts.py` — inject vocabulary into arch/agentic sections
- Modify: `src/lens/extract/extractor.py` — pass vocabulary to new sections
- Modify: `tests/test_extraction.py` — add tests

- [ ] **Step 1: Write failing test for architecture vocab in prompt**

Append to `tests/test_extraction.py`:

```python
def test_prompt_includes_architecture_vocabulary():
    vocabulary = [
        {"name": "Inference Latency", "kind": "parameter"},
        {"name": "Quantization", "kind": "principle"},
        {"name": "Attention Mechanism", "kind": "arch_slot"},
        {"name": "FFN", "kind": "arch_slot"},
        {"name": "Reasoning", "kind": "agentic_category"},
    ]
    prompt = build_extraction_prompt(
        title="Test Paper",
        abstract="Test abstract",
        vocabulary=vocabulary,
    )
    assert "Architecture Slots:" in prompt
    assert "Attention Mechanism" in prompt
    assert "FFN" in prompt
    assert "Agentic Categories:" in prompt
    assert "Reasoning" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_extraction.py::test_prompt_includes_architecture_vocabulary -v`
Expected: FAIL — "Architecture Slots:" not in prompt

- [ ] **Step 3: Update extraction models**

In `src/lens/store/models.py`, update `ArchitectureExtraction`:

```python
class ArchitectureExtraction(BaseModel):
    """An architecture component extracted from a paper."""

    paper_id: str
    component_slot: str
    variant_name: str
    replaces: str | None = None
    key_properties: str
    confidence: float
    new_concept_description: str | None = None
```

Update `AgenticExtraction`:

```python
class AgenticExtraction(BaseModel):
    """An agentic pattern extracted from a paper."""

    paper_id: str
    pattern_name: str
    category: str
    structure: str
    use_case: str
    components: list[str]
    confidence: float
    new_concept_description: str | None = None
```

- [ ] **Step 4: Update DDL in store.py**

Update `architecture_extractions` DDL — add `new_concept_description TEXT` after `confidence`:

```python
    """CREATE TABLE IF NOT EXISTS architecture_extractions (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id TEXT NOT NULL,
        component_slot TEXT NOT NULL,
        variant_name TEXT NOT NULL,
        replaces TEXT,
        key_properties TEXT NOT NULL,
        confidence REAL NOT NULL,
        new_concept_description TEXT
    )""",
```

Update `agentic_extractions` DDL — add `category TEXT NOT NULL` and `new_concept_description TEXT`:

```python
    """CREATE TABLE IF NOT EXISTS agentic_extractions (
        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
        paper_id TEXT NOT NULL,
        pattern_name TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT '',
        structure TEXT NOT NULL,
        use_case TEXT NOT NULL,
        components TEXT NOT NULL,
        confidence REAL NOT NULL,
        new_concept_description TEXT
    )""",
```

- [ ] **Step 5: Update prompt builder**

In `src/lens/extract/prompts.py`, replace the static `_ARCHITECTURE_SECTION` with a function:

```python
def _build_architecture_section(
    vocabulary: list[dict[str, str]] | None = None,
) -> str:
    """Build the architecture section, optionally with guided vocabulary."""
    base = (
        "### 2. Architecture Contributions (ArchitectureExtraction)\n"
        "Identify novel or notable architecture components.\n"
    )

    if vocabulary:
        slots = [v["name"] for v in vocabulary if v["kind"] == "arch_slot"]
        if slots:
            base += (
                "\nUse EXACT names from the Architecture Slots below for "
                "component_slot.\n"
                "\nArchitecture Slots:\n"
                + "\n".join(f"- {s}" for s in slots)
                + "\n\nIf a slot genuinely does not match any entry above, prefix "
                "with NEW: and set new_concept_description to a one-line "
                "definition.\n"
            )

    base += (
        '\n- "component_slot": the category (use an Architecture Slot name)\n'
        '- "variant_name": the specific variant name (free text)\n'
        '- "replaces": what it replaces/generalizes (null if entirely novel)\n'
        '- "key_properties": key properties or advantages\n'
        '- "confidence": your confidence score\n'
        '- "new_concept_description": one-line definition if using NEW: prefix, '
        "else null"
    )
    return base
```

Replace the static `_AGENTIC_SECTION` with a function:

```python
def _build_agentic_section(
    vocabulary: list[dict[str, str]] | None = None,
) -> str:
    """Build the agentic section, optionally with guided vocabulary."""
    base = (
        "### 3. Agentic Patterns (AgenticExtraction)\n"
        "Identify LLM agent design patterns.\n"
    )

    if vocabulary:
        categories = [v["name"] for v in vocabulary if v["kind"] == "agentic_category"]
        if categories:
            base += (
                "\nUse EXACT names from the Agentic Categories below for "
                "category.\n"
                "\nAgentic Categories:\n"
                + "\n".join(f"- {c}" for c in categories)
                + "\n\nIf a category genuinely does not match any entry above, "
                "prefix with NEW: and set new_concept_description to a one-line "
                "definition.\n"
            )

    base += (
        '\n- "pattern_name": name of the pattern (free text)\n'
        '- "category": the category (use an Agentic Category name)\n'
        '- "structure": high-level description of the agent structure\n'
        '- "use_case": primary use case or application\n'
        '- "components": list of key components\n'
        '- "confidence": your confidence score\n'
        '- "new_concept_description": one-line definition if using NEW: prefix, '
        "else null"
    )
    return base
```

Update `build_extraction_prompt` to use the new functions:

```python
    sections = [
        intro,
        f"## Paper\n{paper_content}",
        _TASK_SECTION,
        _build_tradeoffs_section(vocabulary),
        _build_architecture_section(vocabulary),
        _build_agentic_section(vocabulary),
        _CONFIDENCE_SECTION,
        response_format,
    ]
```

Also update `EXTRACTION_RESPONSE_SCHEMA` — add `new_concept_description` to architecture and add `category` + `new_concept_description` to agentic:

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
      "confidence": 0.9,
      "new_concept_description": null
    }
  ],
  "agentic": [
    {
      "pattern_name": "name of the agent pattern",
      "category": "agentic category",
      "structure": "high-level structure description",
      "use_case": "primary use case",
      "components": ["list", "of", "components"],
      "confidence": 0.8,
      "new_concept_description": null
    }
  ]
}"""
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_extraction.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/lens/store/models.py src/lens/store/store.py src/lens/extract/prompts.py src/lens/extract/extractor.py tests/test_extraction.py
git commit -m "feat: inject vocabulary into architecture and agentic extraction prompts"
```

---

### Task 3: Expand process_new_concepts for all extraction types

**Files:**
- Modify: `src/lens/taxonomy/vocabulary.py` — expand `process_new_concepts` and rename `build_tradeoff_taxonomy` to `build_vocabulary`
- Modify: `tests/test_vocabulary.py` — add tests

- [ ] **Step 1: Write failing test for architecture concept processing**

Append to `tests/test_vocabulary.py`:

```python
def test_process_new_concepts_handles_architecture(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    load_seed_vocabulary(store)

    store.add_rows("architecture_extractions", [
        {
            "paper_id": "p1",
            "component_slot": "Attention Mechanism",
            "variant_name": "FlashAttention-2",
            "replaces": "FlashAttention",
            "key_properties": "better parallelism",
            "confidence": 0.9,
            "new_concept_description": None,
        },
        {
            "paper_id": "p2",
            "component_slot": "NEW: Embedding Layer",
            "variant_name": "Rotary Embeddings",
            "replaces": None,
            "key_properties": "relative position info",
            "confidence": 0.85,
            "new_concept_description": "Token embedding and projection layer",
        },
    ])

    stats = process_new_concepts(store)
    assert stats["new_entries"] == 1

    rows = store.query("vocabulary", "id = ?", ("embedding-layer",))
    assert len(rows) == 1
    assert rows[0]["kind"] == "arch_slot"
    assert rows[0]["source"] == "extracted"

    attn = store.query("vocabulary", "id = ?", ("attention-mechanism",))
    assert attn[0]["paper_count"] == 1


def test_process_new_concepts_handles_agentic(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
    load_seed_vocabulary(store)

    store.add_rows("agentic_extractions", [
        {
            "paper_id": "p1",
            "pattern_name": "ReAct",
            "category": "Reasoning",
            "structure": "interleaves reasoning and acting",
            "use_case": "multi-step QA",
            "components": ["LLM", "tools"],
            "confidence": 0.9,
            "new_concept_description": None,
        },
        {
            "paper_id": "p2",
            "pattern_name": "LATS",
            "category": "NEW: Search",
            "structure": "tree search over reasoning paths",
            "use_case": "complex reasoning",
            "components": ["LLM", "MCTS"],
            "confidence": 0.8,
            "new_concept_description": "Patterns using systematic search over solution spaces",
        },
    ])

    stats = process_new_concepts(store)
    assert stats["new_entries"] == 1

    rows = store.query("vocabulary", "id = ?", ("search",))
    assert len(rows) == 1
    assert rows[0]["kind"] == "agentic_category"

    reasoning = store.query("vocabulary", "id = ?", ("reasoning",))
    assert reasoning[0]["paper_count"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vocabulary.py::test_process_new_concepts_handles_architecture -v`
Expected: FAIL — `process_new_concepts` only scans `tradeoff_extractions`

- [ ] **Step 3: Expand process_new_concepts**

In `src/lens/taxonomy/vocabulary.py`, rewrite `process_new_concepts`:

```python
def process_new_concepts(store: LensStore) -> dict[str, int]:
    """Scan all extraction tables for NEW: concepts, accept them, update stats.

    Scans tradeoff_extractions (improves/worsens/technique),
    architecture_extractions (component_slot), and
    agentic_extractions (category).

    Returns dict with keys: new_entries, updated_entries.
    """
    existing = store.query("vocabulary")
    existing_by_name: dict[str, dict[str, Any]] = {r["name"]: r for r in existing}
    existing_ids: set[str] = {r["id"] for r in existing}
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    references: dict[str, list[tuple[str, float, str]]] = {}
    new_concepts: dict[str, dict[str, str]] = {}

    # Scan tradeoff extractions
    for ext in store.query("tradeoff_extractions"):
        for field, kind in [
            ("improves", "parameter"),
            ("worsens", "parameter"),
            ("technique", "principle"),
        ]:
            _collect_concept_ref(ext, field, kind, references, new_concepts)

    # Scan architecture extractions
    for ext in store.query("architecture_extractions"):
        _collect_concept_ref(ext, "component_slot", "arch_slot", references, new_concepts)

    # Scan agentic extractions
    for ext in store.query("agentic_extractions"):
        _collect_concept_ref(ext, "category", "agentic_category", references, new_concepts)

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

    # Update paper_count and avg_confidence
    updated = 0
    for name, refs in references.items():
        if name not in existing_by_name:
            continue
        entry = existing_by_name[name]
        entry_id = entry["id"]
        unique_papers = {r[0] for r in refs}
        avg_conf = sum(r[1] for r in refs) / len(refs)
        store.update(
            "vocabulary",
            "paper_count = ?, avg_confidence = ?",
            "id = ?",
            (len(unique_papers), round(avg_conf, 4), entry_id),
        )
        updated += 1

    return {"new_entries": len(new_rows), "updated_entries": updated}


def _collect_concept_ref(
    ext: dict[str, Any],
    field: str,
    kind: str,
    references: dict[str, list[tuple[str, float, str]]],
    new_concepts: dict[str, dict[str, str]],
) -> None:
    """Extract a concept reference from an extraction row."""
    raw_value = ext.get(field, "")
    if not raw_value:
        return
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
```

- [ ] **Step 4: Rename build_tradeoff_taxonomy to build_vocabulary**

In `src/lens/taxonomy/vocabulary.py`, rename:

```python
def build_vocabulary(
    store: LensStore,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
) -> dict[str, int]:
    """Process new concepts from all extraction types, update stats, embed vocabulary.

    Returns dict with keys: new_entries, updated_entries.
    """
    stats = process_new_concepts(store)

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
        for row, emb in zip(to_embed, embeddings, strict=True):
            store.upsert_embedding("vocabulary", row["id"], emb.tolist())

    logger.info(
        "Vocabulary: %d new, %d updated, %d embedded",
        stats["new_entries"],
        stats["updated_entries"],
        len(to_embed),
    )
    return stats
```

- [ ] **Step 5: Update taxonomy/__init__.py**

Replace the `build_tradeoff_taxonomy` import with `build_vocabulary`:

```python
from lens.taxonomy.vocabulary import build_vocabulary  # noqa: F401
```

Update `__all__`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_vocabulary.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/lens/taxonomy/vocabulary.py src/lens/taxonomy/__init__.py tests/test_vocabulary.py
git commit -m "feat: expand process_new_concepts for arch/agentic, rename to build_vocabulary"
```

---

### Task 4: Remove clustering pipeline and HDBSCAN dependency

**Files:**
- Remove: `src/lens/taxonomy/clusterer.py`
- Remove: `src/lens/taxonomy/labeler.py`
- Modify: `src/lens/taxonomy/__init__.py` — remove all clustering code
- Modify: `pyproject.toml` — remove `hdbscan` dependency
- Modify: `tests/test_taxonomy.py` — remove clustering tests

- [ ] **Step 1: Remove clusterer.py and labeler.py**

```bash
rm src/lens/taxonomy/clusterer.py src/lens/taxonomy/labeler.py
```

- [ ] **Step 2: Gut taxonomy/__init__.py**

Replace `src/lens/taxonomy/__init__.py` with:

```python
"""LENS taxonomy pipeline — vocabulary-based guided extraction."""

from __future__ import annotations

from lens.taxonomy.versioning import get_next_version, record_version  # noqa: F401
from lens.taxonomy.vocabulary import build_vocabulary  # noqa: F401

__all__ = [
    "build_vocabulary",
    "get_next_version",
    "record_version",
]
```

- [ ] **Step 3: Remove hdbscan from pyproject.toml**

Remove this line from `[project] dependencies`:

```
"hdbscan>=0.8.41",
```

- [ ] **Step 4: Update taxonomy tests**

Remove from `tests/test_taxonomy.py` all tests that reference clustering, labeling, `build_architecture_taxonomy`, `build_agentic_taxonomy`, `cluster_embeddings`, `label_clusters`, `normalize_slots`, `_next_id`, or any removed helpers.

Keep tests for `build_vocabulary` (renamed from `build_tradeoff_taxonomy`), embedding, and versioning.

Add a test for `build_vocabulary`:

```python
from lens.taxonomy.vocabulary import load_seed_vocabulary, build_vocabulary


def test_build_vocabulary(tmp_path):
    store = LensStore(str(tmp_path / "test.db"))
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
    store.add_rows("architecture_extractions", [
        {
            "paper_id": "p1",
            "component_slot": "Attention Mechanism",
            "variant_name": "GQA",
            "replaces": None,
            "key_properties": "fewer KV heads",
            "confidence": 0.9,
            "new_concept_description": None,
        },
    ])
    store.add_rows("agentic_extractions", [
        {
            "paper_id": "p1",
            "pattern_name": "ReAct",
            "category": "Reasoning",
            "structure": "interleave",
            "use_case": "QA",
            "components": ["LLM"],
            "confidence": 0.8,
            "new_concept_description": None,
        },
    ])

    stats = build_vocabulary(store)
    assert stats["new_entries"] == 1  # Pruning

    vocab = store.query("vocabulary")
    pruning = [v for v in vocab if v["id"] == "pruning"]
    assert len(pruning) == 1

    attn = [v for v in vocab if v["id"] == "attention-mechanism"]
    assert attn[0]["paper_count"] == 1

    reasoning = [v for v in vocab if v["id"] == "reasoning"]
    assert reasoning[0]["paper_count"] == 1
```

- [ ] **Step 5: Run uv sync and tests**

```bash
uv sync
uv run pytest tests/test_taxonomy.py tests/test_vocabulary.py -v
```

Expected: ALL PASS (clustering tests removed, vocabulary tests pass)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove HDBSCAN clustering pipeline, hdbscan dependency"
```

---

### Task 5: Remove old architecture/agentic tables and models

**Files:**
- Modify: `src/lens/store/store.py` — remove table DDL, VEC_TABLES, JSON_FIELDS entries
- Modify: `src/lens/store/models.py` — remove ArchitectureSlot, ArchitectureVariant, AgenticPattern
- Modify: `tests/test_store.py` — update expected tables
- Modify: `tests/test_models.py` — remove model tests

- [ ] **Step 1: Remove table DDL from store.py**

Remove these CREATE TABLE statements from `_TABLE_DDL`:
- `architecture_slots`
- `architecture_variants`
- `agentic_patterns`

Remove from `VEC_TABLES`:
- `"architecture_variants": ("id", "INTEGER")`
- `"agentic_patterns": ("id", "INTEGER")`

Remove from `JSON_FIELDS`:
- `"architecture_variants": {"replaces", "paper_ids"}`
- `"agentic_patterns": {"components", "use_cases", "paper_ids"}`

Add to `JSON_FIELDS` if not present:
- `"agentic_extractions": {"components"}`

- [ ] **Step 2: Remove models**

From `src/lens/store/models.py`, remove:
- `ArchitectureSlot` class
- `ArchitectureVariant` class
- `AgenticPattern` class

- [ ] **Step 3: Update store comment**

Update the `_TABLE_DDL` comment to reflect the correct table count (should be 9: papers, tradeoff_extractions, architecture_extractions, agentic_extractions, vocabulary, matrix_cells, taxonomy_versions, ideation_reports, ideation_gaps).

- [ ] **Step 4: Update tests**

In `tests/test_store.py`, update expected tables set — remove `architecture_slots`, `architecture_variants`, `agentic_patterns`. Update vec tables test to only expect `papers_vec`, `vocabulary_vec`.

In `tests/test_models.py`, remove tests for `ArchitectureSlot`, `ArchitectureVariant`, `AgenticPattern`.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Note: some serve layer / CLI tests will now fail — that's expected, they'll be fixed in Tasks 6-7.

- [ ] **Step 6: Commit**

```bash
git add src/lens/store/store.py src/lens/store/models.py tests/test_store.py tests/test_models.py
git commit -m "chore: remove architecture_slots, architecture_variants, agentic_patterns tables"
```

---

### Task 6: Migrate serve layer to use extractions + vocabulary

**Files:**
- Modify: `src/lens/serve/explorer.py`
- Modify: `src/lens/serve/analyzer.py`
- Modify: `tests/test_explorer.py`
- Modify: `tests/test_analyzer.py`

- [ ] **Step 1: Rewrite explorer architecture functions**

Replace `list_architecture_slots`:

```python
def list_architecture_slots(store: LensStore) -> list[dict[str, Any]]:
    """List all architecture slots from vocabulary."""
    rows = store.query("vocabulary", "kind = ?", ("arch_slot",))
    for r in rows:
        r.pop("embedding", None)
    # Count variants per slot from extractions
    extractions = store.query("architecture_extractions")
    for r in rows:
        r["variant_count"] = len(set(
            e["variant_name"] for e in extractions
            if e["component_slot"] == r["name"]
        ))
    return rows
```

Replace `list_architecture_variants`:

```python
def list_architecture_variants(
    store: LensStore, slot_name: str
) -> list[dict[str, Any]]:
    """List architecture variants for a given slot name."""
    extractions = store.query("architecture_extractions")
    # Also match NEW: prefixed versions
    variants_raw = [
        e for e in extractions
        if e["component_slot"] == slot_name
        or e["component_slot"] == f"NEW: {slot_name}"
    ]
    # Deduplicate by variant_name, aggregate properties
    by_name: dict[str, dict[str, Any]] = {}
    for v in variants_raw:
        name = v["variant_name"]
        if name not in by_name:
            by_name[name] = {
                "variant_name": name,
                "slot": slot_name,
                "replaces": v.get("replaces"),
                "key_properties": v.get("key_properties", ""),
                "paper_ids": [],
                "confidence": v["confidence"],
            }
        by_name[name]["paper_ids"].append(v["paper_id"])
    return list(by_name.values())
```

Replace `list_agentic_patterns`:

```python
def list_agentic_patterns(
    store: LensStore, category: str | None = None
) -> list[dict[str, Any]]:
    """List agentic patterns from extractions, optionally filtered by category."""
    extractions = store.query("agentic_extractions")
    if category:
        extractions = [
            e for e in extractions
            if e.get("category") == category
            or e.get("category") == f"NEW: {category}"
        ]
    # Deduplicate by pattern_name
    by_name: dict[str, dict[str, Any]] = {}
    for e in extractions:
        name = e["pattern_name"]
        if name not in by_name:
            by_name[name] = {
                "pattern_name": name,
                "category": e.get("category", ""),
                "structure": e.get("structure", ""),
                "use_case": e.get("use_case", ""),
                "components": e.get("components", []),
                "paper_ids": [],
            }
        by_name[name]["paper_ids"].append(e["paper_id"])
    return list(by_name.values())
```

Replace `get_architecture_timeline`:

```python
def get_architecture_timeline(
    store: LensStore, slot_name: str
) -> list[dict[str, Any]]:
    """List variants for a slot ordered by earliest paper date."""
    variants = list_architecture_variants(store, slot_name)
    if not variants:
        return []
    papers = store.query("papers")
    paper_date_map = {p["paper_id"]: p["date"] for p in papers}
    for v in variants:
        dates = [paper_date_map[pid] for pid in v.get("paper_ids", []) if pid in paper_date_map]
        v["earliest_date"] = min(dates) if dates else None
    variants.sort(key=lambda x: x.get("earliest_date") or "9999-99-99")
    return variants
```

- [ ] **Step 2: Rewrite analyzer architecture/agentic functions**

Replace `analyze_architecture` — use LLM to identify slot from vocabulary, then filter extractions:

```python
async def analyze_architecture(
    query: str,
    store: LensStore,
    llm_client: LLMClient,
    embedding_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a query about transformer architecture and return matching variants."""
    slots = store.query("vocabulary", "kind = ?", ("arch_slot",))
    slot_names = [s["name"] for s in slots]

    identified_slot: str | None = None
    if slot_names:
        prompt = _build_slot_identify_prompt(query, slot_names)
        try:
            response = await llm_client.complete([{"role": "user", "content": prompt}])
            text = strip_code_fences(response.strip())
            classification = json.loads(text)
            identified_slot = classification.get("slot")
        except Exception:
            logger.warning("Failed to identify architecture slot for query: %s", query)

    # Filter extractions by identified slot
    extractions = store.query("architecture_extractions")
    if identified_slot:
        extractions = [e for e in extractions if e["component_slot"] == identified_slot]

    variants = []
    seen: set[str] = set()
    for row in extractions:
        name = row["variant_name"]
        if name in seen:
            continue
        seen.add(name)
        variants.append({
            "variant_name": name,
            "slot": row["component_slot"],
            "properties": row.get("key_properties", ""),
            "paper_ids": [row["paper_id"]],
        })

    return {
        "query": query,
        "slot": identified_slot,
        "variants": variants,
    }
```

Replace `analyze_agentic` — use LLM to identify category, filter extractions:

```python
async def analyze_agentic(
    query: str,
    store: LensStore,
    llm_client: LLMClient,
    embedding_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a query about agentic patterns and return matching patterns."""
    categories = store.query("vocabulary", "kind = ?", ("agentic_category",))
    cat_names = [c["name"] for c in categories]

    identified_category: str | None = None
    if cat_names:
        prompt = _build_category_identify_prompt(query, cat_names)
        try:
            response = await llm_client.complete([{"role": "user", "content": prompt}])
            text = strip_code_fences(response.strip())
            classification = json.loads(text)
            identified_category = classification.get("category")
        except Exception:
            logger.warning("Failed to identify agentic category for query: %s", query)

    extractions = store.query("agentic_extractions")
    if identified_category:
        extractions = [e for e in extractions if e.get("category") == identified_category]

    patterns = []
    seen: set[str] = set()
    for row in extractions:
        name = row["pattern_name"]
        if name in seen:
            continue
        seen.add(name)
        patterns.append({
            "pattern_name": name,
            "category": row.get("category", ""),
            "structure": row.get("structure", ""),
            "use_case": row.get("use_case", ""),
            "components": row.get("components", []),
            "paper_ids": [row["paper_id"]],
        })

    return {
        "query": query,
        "patterns": patterns,
    }
```

Add the `_build_category_identify_prompt` helper:

```python
def _build_category_identify_prompt(query: str, category_names: list[str]) -> str:
    cats_list = "\n".join(f"- {c}" for c in category_names)
    return (
        "You are an LLM agent design expert. A user is asking about "
        "agentic design patterns.\n\n"
        f"User query: {query}\n\n"
        "Available categories:\n"
        f"{cats_list}\n\n"
        "Identify the most relevant category.\n"
        "Respond with JSON only:\n"
        '{"category": "Category Name"}'
    )
```

Remove `taxonomy_version` parameters from both functions. Remove the `embed_strings` import if no longer used by analyzer.

- [ ] **Step 3: Update tests**

Rewrite `tests/test_explorer.py` architecture/agentic tests to use vocabulary + extractions instead of old tables.

Rewrite `tests/test_analyzer.py` architecture/agentic tests similarly.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_explorer.py tests/test_analyzer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/lens/serve/explorer.py src/lens/serve/analyzer.py tests/test_explorer.py tests/test_analyzer.py
git commit -m "refactor: migrate serve layer to use extractions + vocabulary for arch/agentic"
```

---

### Task 7: Update CLI and config, remove taxonomy config section

**Files:**
- Modify: `src/lens/cli.py` — simplify build commands, update explore commands
- Modify: `src/lens/config.py` — remove taxonomy section

- [ ] **Step 1: Simplify build taxonomy command**

Replace the `taxonomy` command body — remove `build_architecture_taxonomy`, `build_agentic_taxonomy`, `LLMClient`. Just call `build_vocabulary`:

```python
@build_app.command()
def taxonomy() -> None:
    """Build taxonomy from current extractions."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir / "lens.db"))
    store.init_tables()

    from lens.taxonomy import build_vocabulary, get_next_version, record_version

    emb_kwargs = _embedding_kwargs(config)
    version_id = get_next_version(store)

    build_vocabulary(store, **emb_kwargs)

    paper_count = len(store.query("papers"))
    vocab = store.query("vocabulary")
    record_version(
        store, version_id,
        paper_count=paper_count,
        param_count=len([v for v in vocab if v["kind"] == "parameter"]),
        principle_count=len([v for v in vocab if v["kind"] == "principle"]),
        slot_count=len([v for v in vocab if v["kind"] == "arch_slot"]),
        variant_count=0,
        pattern_count=0,
    )
    rprint(f"[green]Taxonomy v{version_id} built.[/green]")
```

Simplify `build_all` similarly.

- [ ] **Step 2: Update explore architecture/agents commands**

Remove `taxonomy_version` from `explore architecture`, `explore agents`, `explore evolution`. Update function calls to match new signatures (no taxonomy_version).

- [ ] **Step 3: Update analyze command**

Remove `taxonomy_version` from `analyze_architecture` and `analyze_agentic` calls.

- [ ] **Step 4: Remove taxonomy config section**

In `src/lens/config.py`, remove the entire `"taxonomy"` key from `DEFAULT_CONFIG`.

- [ ] **Step 5: Update config tests**

In `tests/test_config.py`, remove any assertions referencing `taxonomy` config keys.

- [ ] **Step 6: Run tests**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/lens/cli.py src/lens/config.py tests/test_config.py
git commit -m "feat: simplify CLI build commands, remove taxonomy config section"
```

---

### Task 8: Integration test and version bump

**Files:**
- Modify: `tests/test_vocabulary.py` — update integration test
- Modify: `pyproject.toml` — bump version
- Modify: `CHANGELOG.md` — add entry
- Modify: `CLAUDE.md` — update architecture
- Modify: `README.md` — update architecture section

- [ ] **Step 1: Update integration test**

Replace `test_end_to_end_guided_extraction_pipeline` in `tests/test_vocabulary.py` to cover all three extraction types:

```python
def test_end_to_end_all_extraction_types(tmp_path):
    """Integration: seed vocab -> extract all types -> process -> matrix."""
    from lens.knowledge.matrix import build_matrix

    store = LensStore(str(tmp_path / "test.db"))
    count = load_seed_vocabulary(store)
    assert count == 40

    # Tradeoff extraction
    store.add_rows("tradeoff_extractions", [
        {
            "paper_id": "p1",
            "improves": "Inference Latency",
            "worsens": "Model Accuracy",
            "technique": "Quantization",
            "context": "4-bit on 7B models",
            "confidence": 0.9,
            "evidence_quote": "2x speedup with 4-bit.",
            "new_concept_description": None,
        },
    ])

    # Architecture extraction
    store.add_rows("architecture_extractions", [
        {
            "paper_id": "p1",
            "component_slot": "Attention Mechanism",
            "variant_name": "FlashAttention-2",
            "replaces": "FlashAttention",
            "key_properties": "better parallelism",
            "confidence": 0.9,
            "new_concept_description": None,
        },
        {
            "paper_id": "p2",
            "component_slot": "NEW: Tokenizer",
            "variant_name": "BPE-dropout",
            "replaces": None,
            "key_properties": "regularization via subword sampling",
            "confidence": 0.8,
            "new_concept_description": "Text tokenization and subword segmentation methods",
        },
    ])

    # Agentic extraction
    store.add_rows("agentic_extractions", [
        {
            "paper_id": "p1",
            "pattern_name": "ReAct",
            "category": "Reasoning",
            "structure": "interleaves reasoning and acting",
            "use_case": "multi-step QA",
            "components": ["LLM", "tools"],
            "confidence": 0.85,
            "new_concept_description": None,
        },
    ])

    # Process all
    stats = process_new_concepts(store)
    assert stats["new_entries"] == 1  # Tokenizer

    vocab = store.query("vocabulary")
    assert any(v["id"] == "tokenizer" and v["kind"] == "arch_slot" for v in vocab)
    assert any(v["id"] == "attention-mechanism" and v["paper_count"] == 1 for v in vocab)
    assert any(v["id"] == "reasoning" and v["paper_count"] == 1 for v in vocab)

    # Matrix (tradeoffs only)
    build_matrix(store)
    cells = store.query("matrix_cells")
    assert len(cells) == 1
    assert cells[0]["improving_param_id"] == "inference-latency"
```

- [ ] **Step 2: Bump version to 0.5.0**

In `pyproject.toml`, change `version = "0.4.0"` to `version = "0.5.0"`.

- [ ] **Step 3: Update CHANGELOG.md**

Add a 0.5.0 section at the top documenting the unified vocabulary changes.

- [ ] **Step 4: Update CLAUDE.md**

Remove references to HDBSCAN clustering. Update the taxonomy split description.

- [ ] **Step 5: Update README.md**

Remove HDBSCAN from the architecture section. Update the data flow diagram.

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: bump to v0.5.0, update docs for unified vocabulary"
```

---

## Execution Order & Dependencies

```
Task 1 (vocabulary expansion)
  └─> Task 2 (extraction schemas + prompts)
      └─> Task 3 (process_new_concepts expansion + rename)
          ├─> Task 4 (remove clustering pipeline)
          │   └─> Task 5 (remove old tables)
          │       └─> Task 6 (serve layer migration)
          │           └─> Task 7 (CLI + config)
          │               └─> Task 8 (integration test + docs)
          └─> (Tasks 4-8 are sequential)
```

All tasks are sequential in this plan — each builds on the previous.

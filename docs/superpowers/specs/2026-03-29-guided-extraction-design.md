# Guided Extraction Redesign

**Date:** 2026-03-29
**Status:** Approved

## Problem

The current tradeoff pipeline extracts free-text strings for parameters and principles,
then runs them through embedding, HDBSCAN clustering (with KMeans fallback), and LLM
labeling to build a taxonomy. At 50 papers this produced only 4 parameters, 6 principles,
and 7 matrix cells. HDBSCAN found 0 clusters and fell back to KMeans.

The root cause: clustering is solving a string normalization problem that the LLM can
do better in a single step at extraction time.

## Solution

Replace the embed-cluster-label pipeline for tradeoffs with **guided extraction** — the
LLM extracts tradeoffs using a predefined canonical vocabulary of parameters and
principles, with `NEW:` prefix for novel concepts not in the vocabulary.

Validated on 5 papers (including 2 unseen March 2026 papers) with excellent results.

## Design

### 1. Canonical Vocabulary

A new `vocabulary` table stores parameters and principles as the shared reference:

```
vocabulary:
  id (text, PK)           -- slugified: "inference-latency"
  name (text)             -- display: "Inference Latency"
  kind (text)             -- "parameter" or "principle"
  description (text)      -- one-line definition
  source (text)           -- "seed" or "extracted"
  first_seen (text)       -- ISO date
  paper_count (int)       -- papers referencing this concept
  avg_confidence (float)  -- mean extraction confidence across references
  embedding (blob)        -- vector of name + description, for similarity search
```

Embeddings are `embed(name + ": " + description)` for each entry. Needed for:
- Ideation cross-pollination (cosine similarity between parameters)
- Serve layer concept resolution (vector search)
- Future near-duplicate detection for `NEW:` proposals

A companion `vocabulary_vec` virtual table (sqlite-vec) enables vector search.

**Seed vocabulary** (~12 parameters, ~15 principles) is defined in code and loaded via
`lens vocab init`. New concepts from `NEW:` proposals get `source: "extracted"`.

#### Seed Parameters

| Name | Description |
|------|-------------|
| Inference Latency | Time required to generate output from input at deployment |
| Model Accuracy | Quality of model predictions on target tasks |
| Training Cost | Compute, time, and financial cost to train or fine-tune |
| Model Size | Number of parameters in the model |
| Memory Usage | RAM and VRAM required during inference or training |
| Context Length | Maximum input sequence length the model can process |
| Safety/Alignment | Degree to which model outputs align with human values and intent |
| Reasoning Capability | Ability to perform multi-step logical or mathematical reasoning |
| Data Efficiency | Amount of training data needed to reach target performance |
| Generalization | Ability to perform well on unseen tasks or domains |
| Interpretability | Degree to which model decisions can be understood by humans |
| Robustness | Resilience to adversarial inputs, noise, and distribution shift |

#### Seed Principles

| Name | Description |
|------|-------------|
| Knowledge Distillation | Training a smaller model to mimic a larger teacher model |
| Quantization | Reducing numerical precision of model weights and activations |
| Sparse Attention/MoE | Activating only a subset of parameters or attention heads per input |
| RAG | Augmenting generation with retrieved external knowledge |
| Chain-of-Thought | Prompting or training models to produce intermediate reasoning steps |
| Preference Optimization (RLHF/DPO) | Aligning model outputs to human preferences via reward signals |
| Parameter-Efficient Fine-Tuning (LoRA/QLoRA) | Adapting models by training a small number of added parameters |
| Speculative Decoding | Using a fast draft model to propose tokens verified by a larger model |
| Flash Attention | Memory-efficient attention computation via tiling and recomputation |
| Positional Encoding Innovation | Novel methods for representing token position in sequences |
| Scaling | Increasing model size, data, or compute to improve performance |
| Multi-Agent Collaboration | Multiple LLM agents coordinating to solve complex tasks |

### 2. Guided Extraction Prompt

The tradeoff section of the extraction prompt changes to include the full vocabulary
and instruct the LLM to use canonical names. Architecture and agentic sections are
unchanged.

**Tradeoff extraction schema** (values are now canonical, plus optional description
for new concepts):

```json
{
  "improves": "Inference Latency",
  "worsens": "Model Accuracy",
  "technique": "Knowledge Distillation",
  "context": "conditions/constraints where this tradeoff applies",
  "confidence": 0.85,
  "evidence_quote": "relevant sentence from the paper",
  "new_concept_description": null
}
```

The `new_concept_description` field is optional (null when using existing vocabulary).
When the LLM uses `NEW:` prefix, it must also provide a one-line description:

```json
{
  "improves": "NEW: Energy Efficiency",
  "technique": "Quantization",
  "new_concept_description": "Power consumption relative to compute throughput",
  ...
}
```

Rules injected into the prompt:
1. Use exact names from the provided vocabulary for `improves`, `worsens`, and `technique`
2. If a concept genuinely doesn't match any vocabulary entry, prefix with `NEW:` and
   provide a short name (e.g., `"NEW: Energy Efficiency"`), and set
   `new_concept_description` to a one-line definition
3. `improves`/`worsens` must be parameters; `technique` must be a principle

The vocabulary list is injected dynamically, so future extractions see newly accepted
concepts automatically.

### 3. NEW: Concept Handling

All `NEW:` concepts are **auto-accepted immediately** into the vocabulary with
`source: "extracted"` and `paper_count: 1`. Evidence accumulates over time as more
papers reference the same concept.

No staging table or manual review workflow. Concepts with low evidence are not
unreliable — they may be novel. The scoring system (Section 5) handles this distinction.

### 4. Pipeline Flow

**Before:**
```
extract_papers()  ->  free-text tradeoffs
build_taxonomy()  ->  embed -> HDBSCAN/KMeans -> LLM label -> Parameters + Principles
build_matrix()    ->  string-match extractions -> taxonomy IDs -> aggregate
```

**After:**
```
extract_papers()        ->  vocabulary-guided tradeoffs (canonical names or NEW:)
process_new_concepts()  ->  auto-accept NEW: terms, update vocabulary stats
build_matrix()          ->  direct lookup by canonical name -> aggregate
```

#### build_taxonomy() Split

The monolithic `build_taxonomy()` (454 lines) is split into three focused functions:

- **`build_tradeoff_taxonomy()`** — runs `process_new_concepts()` + updates vocabulary
  `paper_count`, `avg_confidence`, and embeddings for new entries. No clustering of
  raw extraction strings (the key simplification vs. the old pipeline).
- **`build_architecture_taxonomy()`** — existing logic: normalize slots, cluster
  variants, label via LLM. Unchanged behavior.
- **`build_agentic_taxonomy()`** — existing logic: cluster patterns, label with
  category. Unchanged behavior.

### 5. Novelty & Evidence Scoring

Two orthogonal signals tracked separately, giving downstream consumers full flexibility:

**Evidence strength** (stored on vocabulary entry):
- Derived from `paper_count` and `avg_confidence`
- High value = well-established concept across multiple papers

**Novelty** (computed at query time, not stored):
- Derived from `first_seen` date relative to query time + low `paper_count`
- A concept in 1 paper last week is novel; same concept in 10 papers over months is
  established

**On matrix cells:**
- Each cell has `count` and `avg_confidence` as before
- The serve/ideation layer joins with vocabulary to get evidence and novelty for
  involved concepts
- Example queries:
  - "Well-established tradeoffs" -> filter by high evidence
  - "Emerging techniques" -> filter by recent `first_seen` + low `paper_count`
  - "What's new for Inference Latency?" -> filter matrix, sort principles by novelty

No scoring formula baked into the matrix layer — raw data for consumers to slice.

### 6. Data Model Changes

**New tables:**
- `vocabulary` (schema in Section 1)
- `vocabulary_vec` (sqlite-vec virtual table for vector search)

**Removed tables** (replaced by vocabulary + vocabulary_vec):
- `parameters`
- `principles`
- `parameters_vec`
- `principles_vec`

**Modified tables:**
- `matrix_cells` — `improving_param_id` / `worsening_param_id` / `principle_id` now
  reference `vocabulary.id` instead of old parameter/principle IDs

**Modified tables (minor):**
- `tradeoff_extractions` — add optional `new_concept_description` column (TEXT, nullable)

**Unchanged tables:**
- `papers`, `architecture_extractions`, `agentic_extractions`
- `architecture_slots`, `architecture_variants`, `architecture_variants_vec`
- `agentic_patterns`, `agentic_patterns_vec`
- `taxonomy_versions`, `ideation_reports`, `ideation_gaps`

**`taxonomy_versions`:**
- `param_count` and `principle_count` now count vocabulary entries by kind

**Store infrastructure:**
- `VEC_TABLES` dict in `store.py`: remove `"parameters"` and `"principles"` entries,
  add `"vocabulary"`

### 7. CLI Changes

**New commands:**
- `lens vocab init` — load seed vocabulary (idempotent, skips existing)
- `lens vocab list` — show vocabulary entries with evidence/novelty stats
- `lens vocab show <id>` — details for a concept including referencing papers

**Existing commands** (unchanged interface):
- `lens extract` — uses guided prompt internally
- `lens build` — calls three split taxonomy builders + matrix builder

### 8. Migration (One-Time)

Re-extract all 50 papers with the new guided prompt. Clean slate.

1. `lens vocab init` — seed the vocabulary
2. Reset all papers to `extraction_status: "pending"`
3. `lens extract` — re-extract with guided prompt
4. `lens build` — build taxonomy + matrix from fresh extractions

Cost: ~50 LLM calls with Gemini Flash, negligible.

### 9. Downstream Migrations

The `parameters`, `principles`, `parameters_vec`, and `principles_vec` tables are
referenced by several downstream modules that must be migrated to use `vocabulary`
and `vocabulary_vec` instead.

**Serve layer:**
- `serve/explainer.py` — `resolve_concept()` does vector search on parameters and
  principles tables; `graph_walk()` queries both. Migrate to search `vocabulary_vec`
  with a `kind` filter.
- `serve/analyzer.py` — queries parameters for name-to-ID maps, principles to resolve
  results. Migrate to query `vocabulary` with `kind` filter.
- `serve/explorer.py` — `list_parameters()` and `list_principles()` query those tables
  directly. Migrate to query `vocabulary` filtered by kind.

**Ideation / monitor:**
- `monitor/ideation.py` — `find_sparse_cells()` queries parameters table for IDs;
  `find_cross_pollination()` joins `parameters_vec` for embedding similarity;
  `run_ideation()` builds name maps from both tables. All migrate to `vocabulary` /
  `vocabulary_vec` with `kind` filter.

**CLI:**
- `lens explore parameters` and `lens explore principles` — migrate to query
  `vocabulary` by kind. Could also be unified into `lens vocab list --kind parameter`.
- `lens analyze` — underlying `analyzer.py` migration handles this.

All migrations are mechanical — replace table name with `vocabulary` and add
`WHERE kind = ?` filter. No logic changes needed.

### 10. File Changes

| Area | File | Change |
|------|------|--------|
| New | `taxonomy/vocabulary.py` | Seed data, `process_new_concepts()`, vocab stats, embeddings |
| New | `vocabulary` + `vocabulary_vec` in `store.py` | Schema definitions |
| Modified | `extract/prompts.py` | Inject vocabulary into tradeoff prompt section |
| Modified | `extract/extractor.py` | Pass vocabulary to prompt builder |
| Modified | `taxonomy/__init__.py` | Split into three builders |
| Modified | `knowledge/matrix.py` | Lookup vocabulary IDs instead of param/principle IDs |
| Modified | `store/models.py` | Add `VocabularyEntry` model, add `new_concept_description` to `TradeoffExtraction`, remove Parameter/Principle |
| Modified | `store/store.py` | Add vocabulary table, remove parameters/principles, update `VEC_TABLES` |
| Modified | `cli.py` | Add `vocab` command group, update explore commands |
| Modified | `config.py` | Remove `target_parameters`/`target_principles` |
| Modified | `serve/explainer.py` | Migrate parameter/principle queries to vocabulary |
| Modified | `serve/analyzer.py` | Migrate parameter/principle queries to vocabulary |
| Modified | `serve/explorer.py` | Migrate list functions to vocabulary |
| Modified | `monitor/ideation.py` | Migrate to vocabulary + vocabulary_vec |
| Unchanged | `taxonomy/embedder.py` | Used by architecture/agentic + vocabulary embedding |
| Unchanged | `taxonomy/clusterer.py` | Used by architecture/agentic paths |
| Unchanged | `taxonomy/labeler.py` | Used by architecture/agentic paths |
| Unchanged | `acquire/` | No changes |

### 11. Tests

- New: vocabulary CRUD, `process_new_concepts()`, guided prompt generation
- New: `NEW:` concept auto-acceptance and stats update
- New: vocabulary embedding generation
- Updated: matrix tests to use vocabulary IDs
- Updated: taxonomy build tests for three-way split
- Updated: serve layer tests (explainer, analyzer, explorer) for vocabulary queries
- Updated: ideation tests for vocabulary-based cross-pollination
- All tests use real SQLite instances (no mocking, per project convention)

## What This Does NOT Change

- Architecture catalog (already uses LLM normalization via `normalize_slots`)
- Agentic pattern catalog (already uses LLM categorization)
- SQLite store infrastructure, sqlite-vec (tables change, engine does not)
- Paper acquisition and ingestion
- Extraction of architecture and agentic patterns (same free-text approach)

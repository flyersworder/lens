# Unified Vocabulary — Extend Guided Extraction to Architecture & Agentic

**Date:** 2026-03-29
**Status:** Approved
**Depends on:** v0.4.0 guided extraction (merged)

## Problem

Architecture and agentic taxonomy paths still use HDBSCAN/KMeans clustering to
normalize free-text extraction strings — the same approach that failed for tradeoffs.
At 50 papers, architecture extractions produce 25 "unique" slots with obvious
duplicates ("positional encoding" vs "Positional Encoding", "attention mechanism"
vs "Attention") that clustering must resolve.

The guided vocabulary approach validated in v0.4.0 for tradeoffs eliminates this
entirely. Extending it to architecture and agentic makes the pipeline consistent,
simpler, and removes HDBSCAN/scikit-learn as dependencies.

## Solution

Extend the vocabulary table with two new `kind` values (`arch_slot`,
`agentic_category`) and apply the same guided extraction pattern: the LLM uses
canonical names from the vocabulary during extraction, with `NEW:` prefix for
novel concepts.

Variant names (architecture) and pattern names (agentic) stay free-text — they
are the novel discoveries, not the organizational axis.

## Design

### 1. Vocabulary Extension

Two new `kind` values added to the vocabulary table:

| kind | Purpose | Organizational axis |
|------|---------|-------------------|
| `parameter` | Tradeoff parameters (existing) | What improves/worsens |
| `principle` | Tradeoff techniques (existing) | How the tradeoff is resolved |
| `arch_slot` | Architecture component categories | Where in the architecture |
| `agentic_category` | Agentic pattern categories | What kind of agent behavior |

The `VocabularyEntry` model's `kind` validator expands to accept all four values.

#### Seed Architecture Slots

| Name | Description |
|------|-------------|
| Attention Mechanism | How the model attends to different parts of the input |
| Positional Encoding | Methods for representing token position in sequences |
| FFN | Feed-forward network layers within transformer blocks |
| Normalization | Layer or batch normalization techniques |
| Activation Function | Non-linear activation functions in the network |
| MoE Routing | Routing strategies for mixture-of-experts architectures |
| Optimizer | Training optimization algorithms and strategies |
| Loss Function | Objective functions used during training |
| Quantization Method | Techniques for reducing numerical precision |
| Retrieval Mechanism | Methods for retrieving external knowledge |

#### Seed Agentic Categories

| Name | Description |
|------|-------------|
| Reasoning | Patterns for multi-step logical inference and problem solving |
| Planning | Patterns for decomposing goals into executable steps |
| Tool Use | Patterns for LLM interaction with external tools and APIs |
| Multi-Agent Collaboration | Patterns involving multiple coordinating agents |
| Self-Reflection | Patterns for self-evaluation and iterative improvement |
| Code Generation | Patterns for generating, testing, and debugging code |

### 2. Extraction Prompt Changes

Both architecture and agentic prompt sections are updated to inject vocabulary,
same pattern as tradeoffs.

**Architecture extraction schema** (values now canonical for `component_slot`):

```json
{
  "component_slot": "Attention Mechanism",
  "variant_name": "FlashAttention-2",
  "replaces": "FlashAttention",
  "key_properties": "better work partitioning, 2x speedup",
  "confidence": 0.9,
  "new_concept_description": null
}
```

Rules:
1. `component_slot` must use an exact name from the Architecture Slots vocabulary
2. If a slot genuinely doesn't match, prefix with `NEW:` and provide
   `new_concept_description`
3. `variant_name` stays free-text (the novel discovery)

**Agentic extraction schema** (adds `category` field):

```json
{
  "pattern_name": "ReAct",
  "category": "Reasoning",
  "structure": "interleaves reasoning and acting steps",
  "use_case": "multi-step question answering",
  "components": ["LLM", "tool executor", "observation parser"],
  "confidence": 0.85,
  "new_concept_description": null
}
```

Rules:
1. `category` must use an exact name from the Agentic Categories vocabulary
2. If a category genuinely doesn't match, prefix with `NEW:` and provide
   `new_concept_description`
3. `pattern_name` stays free-text

### 3. Schema Changes

**Modified extraction tables:**
- `architecture_extractions` — add `new_concept_description TEXT` column
- `agentic_extractions` — add `category TEXT NOT NULL` and
  `new_concept_description TEXT` columns

**Modified extraction models:**
- `ArchitectureExtraction` — add `new_concept_description: str | None = None`
- `AgenticExtraction` — add `category: str` and
  `new_concept_description: str | None = None`

**Removed tables:**
- `architecture_slots` — replaced by vocabulary `kind="arch_slot"`
- `architecture_variants` + `architecture_variants_vec` — variants stay in
  `architecture_extractions`
- `agentic_patterns` + `agentic_patterns_vec` — patterns stay in
  `agentic_extractions`

**Removed models:**
- `ArchitectureSlot`
- `ArchitectureVariant`
- `AgenticPattern`

### 4. Pipeline Simplification

**Before (v0.4.0):**
```
build_tradeoff_taxonomy()       ->  vocabulary-based (no clustering)
build_architecture_taxonomy()   ->  normalize -> HDBSCAN -> label (async)
build_agentic_taxonomy()        ->  HDBSCAN -> label with category (async)
build_matrix()                  ->  vocabulary lookup
```

**After:**
```
build_vocabulary()              ->  process all NEW: concepts, update stats, embed
build_matrix()                  ->  vocabulary lookup (unchanged)
```

One function replaces three. `build_tradeoff_taxonomy()` is renamed to
`build_vocabulary()` since it now handles all extraction types.

**`process_new_concepts()` expansion:**

Currently scans `tradeoff_extractions` for `improves`/`worsens`/`technique`.
Expands to also scan:
- `architecture_extractions` for `component_slot` (kind: `arch_slot`)
- `agentic_extractions` for `category` (kind: `agentic_category`)

Same logic: `NEW:` prefixed values are auto-accepted, stats updated for all
referenced concepts.

### 5. Removed Code

**Files removed entirely:**
- `taxonomy/clusterer.py` — HDBSCAN/KMeans clustering
- `taxonomy/labeler.py` — LLM-based cluster labeling (label_clusters,
  label_clusters_with_category, normalize_slots, summarize_variant_properties)

**Dependencies removed from `pyproject.toml`:**
- `hdbscan` — density-based clustering
- `sentence-transformers` stays (used by embedder for vocabulary embeddings)

Note: `scikit-learn` is a transitive dependency of `sentence-transformers`, so
it stays even though we no longer use KMeans directly.

**`taxonomy/__init__.py`** becomes thin:
- Re-exports `build_vocabulary` from `vocabulary.py`
- Keeps `get_next_version`, `record_version` versioning helpers
- Removes all clustering/labeling imports and helpers

**Config removed:**
- `taxonomy.target_arch_variants`
- `taxonomy.target_agentic_patterns`
- `taxonomy.min_cluster_size`
- The entire `taxonomy` config section is removed.

### 6. Downstream Migrations

**Serve layer:**
- `explorer.py`:
  - `list_architecture_slots()` queries vocabulary `kind="arch_slot"`
  - `list_architecture_variants()` queries `architecture_extractions` filtered
    by canonical slot name
  - `list_agentic_patterns()` queries `agentic_extractions`, category comes
    from the extraction itself
  - `get_architecture_timeline()` queries `architecture_extractions` for a
    slot, ordered by paper date
- `analyzer.py`:
  - `analyze_architecture()` — the old approach used vector search on
    `architecture_variants` embeddings. With variants now in raw extractions
    (no embeddings), replace with: LLM identifies the slot (already done),
    then filter `architecture_extractions` by that canonical slot name, and
    return matching variants. No vector search needed — the slot is the
    primary filter, and there are few enough variants per slot to return all.
  - `analyze_agentic()` — same pattern: LLM identifies the category from
    vocabulary, then filter `agentic_extractions` by that category. No vector
    search needed.
- `explainer.py`:
  - `resolve_concept()` already searches vocabulary — new `arch_slot` and
    `agentic_category` entries are automatically searchable

**CLI:**
- `explore architecture` — queries vocabulary for slots, then
  `architecture_extractions` for variants
- `explore agents` — queries `agentic_extractions` with category field
- `explore evolution` — queries `architecture_extractions` for a slot
- `build taxonomy` — calls `build_vocabulary()` (synchronous, no async needed)
  + `record_version`
- `build all` — calls `build_vocabulary()` + `build_matrix()`

**Store infrastructure:**
- `VEC_TABLES`: remove `architecture_variants` and `agentic_patterns` entries
- `JSON_FIELDS`: remove `architecture_variants` and `agentic_patterns` entries,
  add `agentic_extractions: {"components"}` if not present

### 7. File Changes

| Area | File | Change |
|------|------|--------|
| Modified | `taxonomy/vocabulary.py` | Expand seed data, `process_new_concepts`, rename to `build_vocabulary` |
| Modified | `taxonomy/__init__.py` | Remove arch/agentic builders, re-export `build_vocabulary` |
| Modified | `extract/prompts.py` | Inject vocab into architecture + agentic sections |
| Modified | `store/models.py` | Update validators, add fields, remove old models |
| Modified | `store/store.py` | Add columns, remove tables, update VEC_TABLES/JSON_FIELDS |
| Modified | `knowledge/matrix.py` | No change (only uses tradeoff path) |
| Modified | `serve/explorer.py` | Migrate to vocabulary + extractions |
| Modified | `serve/analyzer.py` | Migrate to vocabulary + extractions |
| Modified | `cli.py` | Update build commands, remove taxonomy config refs |
| Modified | `config.py` | Remove taxonomy config section |
| Removed | `taxonomy/clusterer.py` | No longer needed |
| Removed | `taxonomy/labeler.py` | No longer needed |
| Unchanged | `taxonomy/embedder.py` | Still used for vocabulary + serve layer |
| Unchanged | `serve/explainer.py` | Already uses vocabulary |
| Unchanged | `monitor/ideation.py` | Only uses parameter vocabulary |

### 8. Migration (One-Time)

Same as v0.4.0 — re-extract all papers with the updated guided prompt.

1. `lens vocab init` — loads expanded seed vocabulary (existing entries skipped)
2. Reset papers to `extraction_status: "pending"`
3. `lens extract` — re-extract with full guided prompt
4. `lens build all` — build vocabulary + matrix

### 9. Tests

- Updated: vocabulary tests for new kinds and expanded `process_new_concepts`
- Updated: extraction prompt tests for architecture/agentic vocabulary injection
- Updated: serve layer tests (explorer, analyzer) for new query patterns
- Updated: CLI build/explore tests
- Removed: taxonomy clustering/labeling tests
- Removed: architecture_variants and agentic_patterns model tests

## What This Does NOT Change

- Tradeoff extraction and matrix (already vocabulary-based)
- Vocabulary table schema (just new kind values)
- Ideation pipeline (only uses parameters)
- Paper acquisition and ingestion
- Embedder module (still needed)

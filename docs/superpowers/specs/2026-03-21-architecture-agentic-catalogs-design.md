# Architecture & Agentic Catalog Processing — Design Spec

**Date**: 2026-03-21
**Status**: Implemented

## Problem

LENS extracts three tuple types from papers: tradeoffs, architecture contributions, and agentic patterns. The tradeoff data flows through the full pipeline (cluster → taxonomy → matrix → serve). Architecture and agentic data is extracted and stored but never processed into taxonomy entries — it sits unused in the database.

## Goal

Process architecture and agentic extraction data into browsable, queryable catalogs optimized for **engineering decision support** — answering "what should I use for X?" rather than "how did X evolve?".

## Design Decisions

### Property-based comparison over evolution trees

Architecture variants are organized by slot with property-based comparison as the primary query model. The `replaces` field from extractions is preserved as optional context but is not the organizing principle. This optimizes for engineering questions ("I need sub-quadratic attention with bounded KV cache") over research history questions ("what replaced MHA?").

### Emergent categories over fixed enums

Agentic pattern categories are discovered from data via LLM normalization, not predefined. This is consistent with how LENS discovers parameters and principles, and avoids the problems of a fixed `{single-agent, multi-agent, orchestration}` enum that can't capture tool-use patterns, memory patterns, planning patterns, etc.

### Auto-increment IDs over offset-based allocation

All entity types use `max(id) + 1` from their respective tables instead of hardcoded ID offsets (`version * 100000 + offset`). This eliminates silent collision risk and scales without coordination between entity types. Existing parameter/principle ID generation is refactored to match.

**Migration note**: Existing stored data (if any) will have offset-based IDs (e.g., `100001`). The `_next_id` function starts from `max(id) + 1`, so new entries will continue from where the old scheme left off. No data migration is needed — old and new IDs coexist safely since they are unique within each table. The only difference is that new IDs won't encode version/type information, which was never relied upon by any query logic.

## Architecture Catalog Processing

### Data flow

```
architecture_extractions table
  |
  +-- component_slot strings --> LLM normalize --> ArchitectureSlot entries
  |                                                  (5-10 canonical slots)
  |
  +-- variant_name strings ---> group by slot ---> per-slot embed+cluster+label
                                                         |
                                                         v
                                                 ArchitectureVariant entries
                                                 (with properties, replaces)
```

### Step 1: Slot normalization

Collect all unique `component_slot` strings from `architecture_extractions` (filtered by `confidence >= 0.5`). Send to LLM in a single call with a prompt like:

> "Here are raw architecture component slot names extracted from LLM research papers. Normalize them into canonical slot names. Group synonyms together. Return JSON: `{"raw_string": "Canonical Slot Name", ...}`"

Example: `{"attention mechanism": "Attention", "self-attention": "Attention", "positional encoding": "Positional Encoding", "pos embedding": "Positional Encoding"}`.

Create one `ArchitectureSlot` entry per canonical name. ID assigned via auto-increment from `max(id)` in the `architecture_slots` table.

**Note**: `ArchitectureSlot` has no `embedding` or `raw_strings` field — this is intentional. Slots come from LLM normalization (a small, finite set), not clustering, so they don't need embeddings for vector search.

### Step 2: Variant clustering per slot

For each slot:
1. Collect all `variant_name` strings from extractions whose `component_slot` maps to that slot.
2. If fewer than 2 strings, create one variant directly (no clustering needed).
3. Otherwise: embed strings → cluster (HDBSCAN with KMeans fallback) → LLM-label each cluster.
4. Each cluster becomes one `ArchitectureVariant` entry.

### Step 3: Aggregate variant metadata

For each variant cluster:
- `properties`: Concatenate unique `key_properties` strings from all source extractions in the cluster, then LLM-summarize into a concise property description.
- `replaces`: Collect `replaces` fields from source extractions. Resolve to variant IDs via string matching against known variant names. Unresolved references are dropped (best-effort). The model field `replaces: list[int]` stores only resolved IDs.
- `paper_ids`: Union of all source extraction paper_ids.
- `embedding`: Centroid of member embeddings, padded/truncated to 768d.

### Step 4: Store

Write `ArchitectureSlot` and `ArchitectureVariant` entries to their respective LanceDB tables with `taxonomy_version`.

## Agentic Pattern Catalog Processing

### Data flow

```
agentic_extractions table
  |
  +-- pattern_name strings ---> embed+cluster+label ---> AgenticPattern entries
  |
  +-- structure, use_case, components ---> aggregated as metadata per pattern
  |
  +-- LLM assigns category from emergent set (not a fixed enum)
```

### Step 1: Cluster patterns

Collect `pattern_name` strings from `agentic_extractions` (filtered by `confidence >= 0.5`). Embed → cluster → LLM-label. Same pipeline as principles.

### Step 2: Category assignment and labeling

After clustering, send each cluster's member strings plus aggregated `structure` descriptions to the LLM in a single call per cluster:

> "Given these agentic pattern names and their structures, assign a category that describes what type of pattern this is. Use short, descriptive category names (e.g., 'Reasoning', 'Reflection', 'Multi-Agent Collaboration', 'Tool Integration', 'Memory & Retrieval', 'Planning'). Return JSON: `{"name": "...", "description": "...", "category": "..."}`"

This replaces the separate `label_clusters` + category-assignment steps with a single LLM call per cluster. The LLM returns `name`, `description`, and `category` together. Requires a new labeling prompt function.

### Step 3: Aggregate metadata

For each pattern cluster:
- `name`: From LLM labeling response (Step 2).
- `description`: From LLM labeling response (Step 2).
- `category`: From LLM labeling response (Step 2).
- `components`: Union of all `components` lists from source extractions.
- `use_cases`: Union of all `use_case` strings from source extractions.
- `paper_ids`: Union of source paper_ids.
- `embedding`: Centroid, 768d.

### Step 4: Store

Write `AgenticPattern` entries to the `agentic_patterns` table with `taxonomy_version`.

## Integration with build_taxonomy

Extend the existing `build_taxonomy` function with two more stages after principles:

```python
async def build_taxonomy(store, llm_client, ...,
                         target_arch_variants=20,
                         target_agentic_patterns=15):
    # Stage 1: Parameters (existing)
    # Stage 2: Principles (existing)
    # Stage 3: Architecture slots + variants (new)
    # Stage 4: Agentic patterns (new)
    # Record version (updated with new counts)
```

One `lens build taxonomy` call builds everything. One `taxonomy_version` covers all entity types.

### ID generation refactor

Replace all offset-based ID generation with auto-increment:

```python
def _next_id(store: LensStore, table_name: str) -> int:
    """Return max(id) + 1 for the given table, or 1 if empty."""
    df = store.get_table(table_name).to_polars()
    if len(df) == 0:
        return 1
    return int(df["id"].max()) + 1
```

Each entity writes IDs starting from `_next_id()`, incrementing per entry. Applied to parameters, principles, slots, variants, and patterns.

### Config additions

```yaml
taxonomy:
  target_parameters: 25       # existing
  target_principles: 35       # existing
  min_cluster_size: 3         # existing
  target_arch_variants: 20    # new: soft target per slot
  target_agentic_patterns: 15 # new: soft target
```

### TaxonomyVersion model update

Add fields to `TaxonomyVersion` in `models.py`:

```python
class TaxonomyVersion(LanceModel):
    version_id: int
    created_at: datetime
    paper_count: int
    param_count: int
    principle_count: int
    slot_count: int = 0        # new
    variant_count: int = 0     # new
    pattern_count: int = 0     # new
```

Default values of 0 ensure backwards compatibility with existing taxonomy versions.

### record_version update

Update `record_version` in `versioning.py` to accept and pass the new count fields:

```python
def record_version(store, version_id, paper_count, param_count, principle_count,
                   slot_count=0, variant_count=0, pattern_count=0):
```

## CLI Commands

### `explore architecture [slot]`

- **No args**: List all slots with variant counts.
- **With slot name**: List all variants in that slot with `properties`, `replaces`, paper count. Property-based comparison is implicit — seeing properties side by side.

### `explore agents [category]`

- **No args**: List all patterns grouped by category with component/use-case summaries.
- **With category**: List patterns in that category with full `components`, `use_cases`.

### `explore evolution <slot>`

Repurposed as a **timeline view**: list variants in a slot ordered by earliest paper date, showing `replaces` links where they exist. Not a tree traversal, just a chronological list with lineage annotations.

**Implementation note**: Determining earliest paper date requires joining `ArchitectureVariant.paper_ids` back to the `papers` table to look up `date`. The explorer function loads the papers table, builds a `paper_id → date` map, then finds `min(date)` for each variant's paper_ids.

### `analyze --type architecture "<query>"`

1. LLM decomposes query into slot + constraints (see "Architecture query decomposition prompt" below).
2. Embed query → vector search against `ArchitectureVariant` embeddings.
3. Optionally filter by slot if the LLM identified one.
4. Return matching variants ranked by similarity, with properties.

### `analyze --type agentic "<query>"`

1. Embed query → vector search against `AgenticPattern` embeddings.
2. Return matching patterns ranked by similarity, with components and use cases.

## New LLM Prompts Needed

1. **Slot normalization prompt**: Raw component_slot strings → canonical slot mapping (JSON).
2. **Variant properties summarization prompt**: Aggregated key_properties strings → concise property description.
3. **Agentic labeling+category prompt**: Extended labeler that returns name, description, AND category for each pattern cluster.
4. **Architecture query decomposition prompt**: User query → identified slot (optional) + technical constraints. Used by `analyze --type architecture`.

## Testing Strategy

- **Unit tests**: Slot normalization parsing, category assignment parsing, auto-increment ID generation, variant metadata aggregation.
- **Integration tests**: Full build_taxonomy with architecture + agentic data in test fixtures, verify slots/variants/patterns stored correctly.
- **CLI tests**: Verify explore architecture/agents/evolution produce expected output.
- **Existing tests**: Must continue passing — the parameter/principle pipeline behavior should not change despite the ID refactor.

## Files to Modify

- `src/lens/taxonomy/__init__.py` — add architecture + agentic stages, refactor ID generation
- `src/lens/taxonomy/labeler.py` — add slot normalization prompt, variant properties prompt, agentic category prompt
- `src/lens/taxonomy/versioning.py` — update `record_version` to accept new count fields
- `src/lens/store/models.py` — update TaxonomyVersion with new count fields
- `src/lens/serve/explorer.py` — add architecture/agentic browse functions, evolution timeline
- `src/lens/serve/analyzer.py` — add --type architecture/agentic query paths + new prompt
- `src/lens/cli.py` — wire up explore architecture/agents/evolution, analyze --type, pass new config to build_taxonomy and build_all
- `src/lens/config.py` — add target_arch_variants, target_agentic_patterns
- `tests/` — new test files + update existing taxonomy tests

## Out of Scope

- Multimodal support (Phase 2)
- Trend detection / BERTrend
- Stalled slot detection (depends on architecture catalog — can be added after)
- REST API / web UI

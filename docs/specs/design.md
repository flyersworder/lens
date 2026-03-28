# LENS — LLM Engineering Navigation System

Design spec for a system that automatically discovers recurring solution patterns, contradiction resolutions, architecture innovations, and agentic design patterns from LLM research papers (arxiv), inspired by TRIZ methodology.

**Status**: Implemented (Contradiction Matrix + Architecture Catalog + Agentic Catalog + Monitor/Ideation)
**Date**: 2026-03-21

> **Note**: This design spec was written when LENS used LanceDB + Polars. The implementation has since migrated to **SQLite + sqlite-vec + plain Python**. The tech stack table below is current; references to LanceDB, Polars, Arrow, and LanceModel elsewhere in this document are historical. See CLAUDE.md and README.md for the current architecture.

---

## Motivation

TRIZ (Theory of Inventive Problem Solving) was built by analyzing ~200,000 patents over decades to discover 39 engineering parameters, 40 inventive principles, and a contradiction matrix mapping tradeoffs to solutions. The methodology is empirical — reverse-engineered from data, not theorized.

LLM research has analogous structure: recurring tradeoffs (accuracy vs. latency, safety vs. helpfulness), recurring solution patterns (distillation, RLHF, MoE, RAG), and rapid architecture evolution. But this knowledge is scattered across thousands of papers with no systematic organization.

LENS applies the TRIZ methodology to LLM research papers, with three key differences:

1. **Richer knowledge structures** — beyond contradictions, LENS captures architecture evolution and agentic design patterns
2. **Fully automated discovery** — parameters and principles emerge from data via clustering, not manual identification
3. **Continuous learning** — the system monitors arxiv and updates its knowledge as the field evolves

## Scope

**In scope**: Papers on Large Language Models (arxiv cs.CL, cs.LG, cs.AI). Covers model architecture, training, alignment, efficiency, deployment, capabilities, agentic systems, and safety.

**Out of scope**: Broader ML/AI (computer vision, classical NLP, reinforcement learning outside LLM context). May expand later.

## Core Knowledge Structures

LENS organizes knowledge into three first-class structures, each serving a different aspect of LLM engineering:

### 1. Contradiction Matrix

Maps LLM tradeoffs to resolution techniques. Directly analogous to the TRIZ contradiction matrix.

- **Parameters** (~25-30): The recurring dimensions of LLM design that trade off against each other. Discovered automatically from papers. Expected examples: inference latency, model accuracy, training compute cost, model size, context window length, factual accuracy, safety/alignment, reasoning capability, data efficiency, generalization, etc.
- **Principles** (~30-40): The abstract solution patterns that resolve tradeoffs. Discovered automatically. Expected examples: knowledge distillation, quantization, sparse activation/MoE, retrieval augmentation, chain-of-thought decomposition, preference optimization, parameter-efficient fine-tuning, speculative decoding, etc.
- **Matrix cells**: `(improving_param, worsening_param) -> [ranked principle IDs]`. Asymmetric — improving A while worsening B may use different principles than improving B while worsening A.

### 2. Architecture Catalog

Tracks the evolution of LLM architecture components as a taxonomy of slots and variants.

- **Slots** (~8-10): Replaceable component positions in an LLM architecture. Expected: attention mechanism, positional encoding, normalization, FFN/MLP, activation function, architecture class, mixture-of-experts routing, context extension.
- **Variants**: Concrete implementations within each slot, with `replaces`/`generalizes` edges forming evolution trees. Example: Multi-Head Attention -> Multi-Query Attention -> Grouped-Query Attention -> Multi-Latent Attention.
- **Properties**: Each variant carries key properties, paper references, and conditions under which it is preferred.

### 3. Agentic Pattern Catalog

Catalogs recurring patterns for building and orchestrating LLM-based agents.

- **Categories**: Single-agent patterns (ReAct, Reflexion, Tree of Thoughts), multi-agent patterns (debate, delegation, ensemble, pipeline), orchestration concerns (memory, tool routing, error recovery, cost budgets).
- **Each pattern**: Name, structure description, components, use cases, known tradeoffs, paper references.

## Architecture

### Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Language | Python 3.12+ | Ecosystem, LLM libraries |
| Package manager | uv | Fast, reliable (same as triz-ai) |
| CLI framework | Typer | Clean CLI + library dual use |
| LLM abstraction | openai SDK (core) + litellm (optional) | Gateway-compatible; litellm adds multi-provider routing |
| Database | SQLite + sqlite-vec | Embedded, vector search (cosine), battle-tested, parameterized queries |
| Clustering | HDBSCAN + KMeans fallback | Density-based clustering with degenerate-case handling |
| Embeddings | sentence-transformers (local) or openai/litellm (cloud) | Configurable: local for offline/free, cloud for scalability |
| Data validation | Pydantic | Structured LLM output validation |
| Paper sources | arxiv API, OpenAlex, Semantic Scholar | Complementary metadata and embeddings |

### Data Model

Four layers: Raw Extraction -> Taxonomy -> Knowledge Structures -> Query Interface.

#### Layer 0: Papers

```
Paper:
  paper_id: str              # arxiv ID, e.g. "2401.12345"
  title: str
  abstract: str
  authors: list[str]
  venue: str | None          # nullable — many arxiv papers are preprint-only
  date: str                  # publication date, e.g. "2024-01-15"
  arxiv_id: str
  citations: int             # from OpenAlex enrichment
  quality_score: float       # computed from citations, venue tier, recency
  extraction_status: str     # "pending" | "complete" | "incomplete" | "failed"
  embedding: Vector(768)     # SPECTER2 embedding for similarity search
```

#### Layer 1: Raw Extractions (per-paper)

```
TradeoffExtraction:
  paper_id: str
  improves: str            # raw text, e.g. "mathematical reasoning accuracy"
  worsens: str             # raw text, e.g. "inference time per token"
  technique: str           # raw text, e.g. "chain-of-thought with self-consistency"
  context: str             # conditions/constraints mentioned
  confidence: float
  evidence_quote: str      # relevant sentence from the paper

ArchitectureExtraction:
  paper_id: str
  component_slot: str      # e.g. "attention mechanism"
  variant_name: str        # e.g. "grouped-query attention"
  replaces: str | None     # e.g. "multi-head attention"
  key_properties: str      # e.g. "reduces KV cache by sharing keys/values"
  confidence: float

AgenticExtraction:
  paper_id: str
  pattern_name: str        # e.g. "reflexion"
  structure: str           # e.g. "single agent with self-critique loop and memory"
  use_case: str            # e.g. "code generation with iterative debugging"
  components: list[str]    # e.g. ["actor", "evaluator", "memory"]
  confidence: float
```

#### Layer 2: Taxonomy (emergent, versioned)

```
Parameter:
  id: int
  name: str                # e.g. "Inference Latency"
  description: str
  raw_strings: list[str]   # extracted strings that mapped here
  paper_ids: list[str]     # papers whose extractions mapped to this parameter
  taxonomy_version: int
  embedding: Vector(768)   # SPECTER2 embedding of name+description for concept resolution

Principle:
  id: int
  name: str                # e.g. "Knowledge Distillation"
  description: str
  sub_techniques: list[str]
  raw_strings: list[str]
  paper_ids: list[str]     # papers whose extractions mapped to this principle
  taxonomy_version: int
  embedding: Vector(768)   # SPECTER2 embedding for concept resolution

ArchitectureSlot:
  id: int
  name: str                # e.g. "Attention Mechanism"
  description: str
  taxonomy_version: int
  # Variants are stored in separate `architecture_variants` table, linked by slot_id

ArchitectureVariant:
  id: int
  slot_id: int
  name: str                # e.g. "Grouped-Query Attention"
  replaces: list[int]      # variant IDs this generalizes/replaces
  properties: str
  paper_ids: list[str]
  taxonomy_version: int
  embedding: Vector(768)   # SPECTER2 embedding for concept resolution

AgenticPattern:
  id: int
  name: str                # e.g. "Reflexion"
  category: str            # "single-agent" | "multi-agent" | "orchestration"
  description: str
  components: list[str]
  use_cases: list[str]
  paper_ids: list[str]
  taxonomy_version: int
  embedding: Vector(768)   # SPECTER2 embedding for concept resolution
```

#### Layer 3: Knowledge Structures

Stored models (LanceDB tables):

```
MatrixCell:
  improving_param_id: int
  worsening_param_id: int
  principle_id: int
  count: int                # number of extractions supporting this cell
  avg_confidence: float     # mean confidence across supporting extractions
  paper_ids: list[str]      # papers that contributed to this cell
  taxonomy_version: int
```

Computed models (in-memory, not stored — assembled from tables via Polars):

```
ContradictionMatrix:
  # Built from `matrix_cells` table: group by (improving, worsening), rank principles by count * avg_confidence, keep top-4
  cells: dict[(improving_param_id, worsening_param_id), list[principle_id]]

ArchitectureCatalog:
  # Built from `architecture_slots` + `architecture_variants` tables: join on slot_id, order variants chronologically
  slots: list[ArchitectureSlot with variants]

AgenticPatternCatalog:
  # Built from `agentic_patterns` table: group by category
  patterns: list[AgenticPattern]
```

Query response models (returned to caller, not stored):

```
ExplanationResult:
  resolved_type: str        # "parameter" | "principle" | "architecture_variant" | "agentic_pattern"
  resolved_id: int          # taxonomy entry ID
  resolved_name: str        # taxonomy entry name
  narrative: str            # LLM-synthesized explanation
  evolution: list[str]      # ordered predecessor/successor chain (if applicable)
  tradeoffs: list[dict]     # matrix cells referencing this concept
  connections: list[str]    # related concepts from shared matrix cells or papers
  paper_refs: list[str]     # key paper references
  alternatives: list[dict]  # other candidate matches (type, id, name, score) for disambiguation

IdeationGap:
  id: int
  report_id: int
  gap_type: str           # "sparse_cell" | "cross_pollination" | "stalled_slot" | "trend_gap"
  description: str        # structured description of the gap
  related_params: list[int]
  related_principles: list[int]
  related_slots: list[int]
  score: float            # relevance/novelty score
  llm_hypothesis: str | None   # filled if LLM enrichment enabled
  created_at: datetime
  taxonomy_version: int

IdeationReport:
  id: int
  created_at: datetime
  taxonomy_version: int
  paper_batch_size: int
  gap_count: int
```

#### Storage

All data is stored in a single LanceDB instance. Each data type is a Lance table with a Pydantic `LanceModel` schema. Data models defined above map directly to table schemas.

LanceDB tables:
- `papers` — paper metadata + SPECTER2 embedding (Layer 0)
- `tradeoff_extractions`, `architecture_extractions`, `agentic_extractions` — raw extraction records (Layer 1)
- `parameters`, `principles` — taxonomy entries + SPECTER2 embeddings for concept resolution (Layer 2)
- `architecture_slots`, `architecture_variants` — architecture taxonomy, variants carry SPECTER2 embeddings (Layer 2)
- `agentic_patterns` — agentic pattern taxonomy + SPECTER2 embeddings (Layer 2)
- `matrix_cells` — one row per (improving_param, worsening_param, principle) with counts and confidence (Layer 3)
- `taxonomy_versions` — taxonomy version metadata
- `ideation_reports`, `ideation_gaps` — ideation pipeline output (Layer 3)
- `paper_figures` (Phase 2: figure images stored as Lance blobs + extracted structured data)

### Database Design: LanceDB + Polars

Single embedded database. LanceDB stores all data (structured, vectors, future multimodal). Polars handles analytical operations (groupby, joins, aggregations) in memory via zero-copy Arrow interchange.

```python
class LensStore:
    def __init__(self, data_dir: str):
        self.db = lancedb.connect(f"{data_dir}/lens.lance")
```

**Why single-database**: At LENS's scale (~200-5000 papers, ~30 parameters, ~35 principles), all data fits comfortably in memory. SQL is unnecessary — every analytical operation (matrix construction, gap detection, taxonomy stats) is a Polars one-liner on Arrow tables loaded from Lance.

**Write path**: Python → Pydantic models → LanceDB `table.add()`. Accepts Pydantic objects, dicts, pandas/polars DataFrames.

**Read path (queries)**: LanceDB `table.search()` for vector similarity, `table.search().where()` for filtered retrieval, `table.to_arrow()` → `polars.from_arrow()` for analytical operations.

**Read path (analytics)**: Load full tables as Arrow → Polars DataFrames for groupby/join/agg. Example: matrix construction is `extractions.group_by(["improving_param_id", "worsening_param_id", "principle_id"]).agg(pl.count(), pl.mean("confidence"))`.

**Versioning**: Taxonomy versioning is application-level — each taxonomy entry carries a `taxonomy_version` field, and `taxonomy_versions` table tracks version metadata. A rebuild writes new entries with a new version ID; old versions are retained for diffing. Filtering by `taxonomy_version` selects the active taxonomy. Lance's built-in MVCC provides an additional safety net: if a rebuild goes wrong, the entire dataset can be restored to a previous Lance snapshot.

**Persistence**: Data stored as Lance columnar files in a directory. Backup = copy the directory. No server process required.

## Pipeline

Five stages, each independently re-runnable:

```
ACQUIRE -> EXTRACT -> TAXONOMIZE -> STRUCTURE -> SERVE
                                                   |
    <---- continuous monitoring loop ---------------+
```

### Stage 1: Acquire

Fetch papers from multiple sources:

- **Curated seed list** (~200 landmark papers): YAML manifest in repo. Bootstraps the system with high-quality, diverse examples covering all major LLM subfields.
- **arxiv API**: Query by categories (cs.CL, cs.LG, cs.AI) + LLM keywords. Bulk historical and continuous.
- **OpenAlex API**: Enrichment — citation counts, venue, institution, topic tags.
- **Semantic Scholar API**: SPECTER2 embeddings for each paper.

Quality scoring for extraction prioritization:
```python
def quality_score(paper: Paper) -> float:
    """Combine: citation count (OpenAlex), venue tier, recency.

    Uses only signals available from APIs without additional fetches.
    venue is nullable — many arxiv papers are preprint-only.
    """
```

Rate limiting: arxiv (1 req/3s), Semantic Scholar (1 req/3s with API key, 100 req/5min), OpenAlex (polite pool, ~10 req/s with mailto). All clients implement backoff and respect rate limits.

### Stage 2: Extract

For each paper, a single LLM call extracts all three tuple types:

```python
async def extract_paper(paper: Paper, llm: LLMClient) -> ExtractionResult:
    """Returns tradeoffs, architecture contributions, and agentic patterns."""
```

**Input**: Title + abstract for most papers. For seed papers and high-quality-score papers, full text is obtained by downloading the arxiv PDF and extracting text via PyMuPDF/Marker. The `lens acquire file` command also accepts local PDFs and extracts full text.

**Output schema**: The Pydantic model from Layer 1 (TradeoffExtraction, ArchitectureExtraction, AgenticExtraction). All lists may be empty — the prompt explicitly instructs the LLM to return empty lists rather than fabricate extractions when a paper does not contain relevant information.

**Confidence scores**: LLM self-assessed on a 0-1 scale, calibrated by the prompt with anchor examples (0.9+ = explicit tradeoff stated in text, 0.7-0.9 = strongly implied, 0.5-0.7 = inferred, <0.5 = speculative). Scores below 0.5 are stored but excluded from taxonomy clustering.

**Extraction prompt approach**: Adapted from Trapp & Warschat (2024) single-prompt contradiction extraction (F1=0.93 on patents). The prompt provides the paper text, defines the three extraction types with examples, and requests structured JSON output. Detailed prompt design is deferred to implementation.

Design decisions:
- Single prompt per paper (cheaper, maintains cross-type context)
- Structured output via Pydantic validation (1 retry with stricter prompt on malformed, then store partial results for successfully parsed types)
- Concurrent processing with configurable `--concurrency` limit (default 5)
- Idempotent (re-extraction overwrites previous)
- Configurable model: cheaper for bulk, expensive for seed papers

### Stage 3: Taxonomize

Periodically cluster raw extraction strings into abstract categories:

1. Collect all raw `improves` + `worsens` strings -> embed with SPECTER2 -> HDBSCAN cluster -> BERTopic LLM labeling -> **Parameters**
2. Collect all raw `technique` strings -> same pipeline -> **Principles**
3. Collect all raw `component_slot` + `variant` strings -> same pipeline -> **ArchitectureSlots + Variants**
4. Collect all raw `pattern_name` + `structure` strings -> same pipeline -> **AgenticPatterns**

**Taxonomy versioning**: Each `build taxonomy` run creates a new version stored in `taxonomy_versions` (version_id, created_at, paper_count, param_count, principle_count). Previous versions are retained for diffing. Running `build taxonomy` is always a full rebuild from all current extractions — not incremental. Downstream structures (matrix, catalogs) must be rebuilt after a new taxonomy version via `build all`.

**Granularity control**: HDBSCAN discovers clusters from density rather than taking a target count. To steer toward the target parameter/principle count: (1) tune `min_cluster_size` — lower values produce more clusters, (2) post-hoc LLM-guided merge of semantically overlapping clusters (e.g., "inference speed" and "inference throughput"), (3) post-hoc LLM-guided split of overly broad clusters. The `target_parameters` and `target_principles` config values are soft targets that guide this post-processing, not hard constraints.

**Bootstrap note**: At ~400-600 raw strings, HDBSCAN may produce noisy results. The bootstrap phase uses a lower `min_cluster_size` (2 instead of 3) and expects manual review of initial clusters before proceeding to matrix construction.

**Embeddings**: Pre-computed SPECTER2 embeddings from Semantic Scholar API are used when available. For papers not in Semantic Scholar's index (rare for arxiv), a local SPECTER2 model (via `transformers`) is used as fallback. The local model is an optional dependency, not required if all papers have pre-computed embeddings.

### Stage 4: Structure

Map raw extractions through the taxonomy to populate knowledge structures.

**Raw-to-taxonomy mapping mechanism**: Each raw extraction string is assigned to a taxonomy entry via cluster membership from Stage 3. During clustering, every raw string is assigned to a cluster (or marked as noise). The cluster-to-taxonomy mapping is stored alongside the taxonomy version. When new extractions arrive between taxonomy rebuilds, they are mapped to existing taxonomy entries using embedding cosine similarity to cluster centroids (nearest-centroid assignment). This provides an approximate mapping without requiring a full re-cluster.

- **Matrix**: For each tradeoff extraction, look up the taxonomy IDs for its `improves`, `worsens`, and `technique` strings. Increment the count for `(improving_param_id, worsening_param_id, principle_id)`. After processing all extractions, keep top-4 principles per cell ranked by weighted count (count * avg_confidence). Asymmetric — (A, B) and (B, A) are independent cells.
- **Architecture catalog**: Build slot -> variant trees with replaces/generalizes edges, ordered chronologically by paper date. Variants within the same slot are linked if their `replaces` fields reference each other.
- **Agentic catalog**: Organize patterns by category, link to principles they employ.

`build matrix` and `build all` are full rebuilds from current extractions + current taxonomy version. They are idempotent — running twice produces the same result.

### Stage 5: Serve

Three user modes:

**Problem-solving** (`lens analyze`): User describes a problem in natural language. Routing depends on `--type`:
- Default (no type): LLM classifies the tradeoff (identifying improving/worsening parameters) -> contradiction matrix lookup -> return ranked principles with paper references.
- `--type architecture`: LLM identifies the relevant architecture slot and constraints -> search architecture catalog by slot + vector similarity -> return relevant variants with properties and evolution context.
- `--type agentic`: LLM identifies the agent task type and constraints -> search agentic catalog by category + vector similarity -> return relevant patterns with components and use cases.

All three modes enrich results with paper references and evidence quotes from the extraction layer. For `--type architecture` and `--type agentic`, the LLM first reformulates the user's conversational query into a technical description suitable for SPECTER2 embedding (e.g., "efficient attention for long context" → "attention mechanism with sub-quadratic complexity for extended sequence lengths"). This reformulated query is embedded and used for vector search against taxonomy tables.

**Education** (`lens explain`): User asks about any LLM concept. The system resolves the query to taxonomy entries, walks the knowledge graph outward, and synthesizes a coherent explanation with adaptive depth.

1. **Resolve** — Match the input query to taxonomy entries via embedding similarity. The query is embedded with SPECTER2, then searched against the `embedding` column on each taxonomy table (`parameters`, `principles`, `architecture_variants`, `agentic_patterns`). Top results across all tables are ranked by distance. If ambiguous, the CLI shows top-3 matches for the user to pick; the library API returns the top match with alternatives listed in `ExplanationResult.alternatives`.
2. **Graph walk** — From the resolved entry, walk outward through knowledge structures:
   - **Identity**: Name, description, paper references (from taxonomy layer)
   - **Context**: Which slot/category it belongs to (architecture slot, agentic category, or parameter/principle type)
   - **Evolution**: Predecessors and successors (for architecture variants), or sub-techniques (for principles)
   - **Tradeoffs**: Which matrix cells reference this concept — what does it improve, what does it cost?
   - **Connections**: Other concepts that share matrix cells or appear in the same papers
   - **Agentic usage**: Which agentic patterns employ this concept (if applicable)
3. **Synthesize** — An LLM receives the structured graph walk data and produces a coherent narrative explanation. The prompt instructs adaptive depth: broad concepts (mapping to many entries) get an overview with pointers; specific concepts get deep coverage of the node and its neighborhood. Newcomer-friendly when the concept is foundational; expert-oriented when the concept is specialized.

Flags for focused exploration:
- `--related`: Emphasize connected concepts and how they relate
- `--evolution`: Focus on the evolution tree (predecessors, successors, timeline)
- `--tradeoffs`: Focus on tradeoffs the concept resolves or introduces

**Exploration** (`lens explore`): Browse parameters, principles, matrix cells, architecture trees, agentic patterns, evolution trajectories, and ideation reports.

## CLI Interface

```bash
# Setup
lens init                           # Initialize databases
lens init --force                   # Reset everything

# Acquire
lens acquire seed                   # Ingest curated ~200 seed papers
lens acquire arxiv --query "LLM"    # Fetch from arxiv
lens acquire arxiv --since 2024-01  # Fetch recent papers
lens acquire file paper.pdf         # Ingest single paper
lens acquire openalex --enrich      # Enrich with metadata

# Extract
lens extract                        # Extract from all unprocessed papers
lens extract --paper-id 2401.12345  # Re-extract specific paper
lens extract --model claude-sonnet  # Use specific model
lens extract --concurrency 5        # Control concurrent LLM calls

# Taxonomize & Build
lens build taxonomy                 # Run clustering
lens build matrix                   # Populate matrix from taxonomy
lens build all                      # Full rebuild

# Analyze (problem-solving)
lens analyze "reduce hallucination without hurting latency"
lens analyze --type architecture "efficient attention for long context"
lens analyze --type agentic "reliable multi-step code generation"

# Explain (education)
lens explain "grouped-query attention"     # adaptive-depth explanation
lens explain "knowledge distillation"      # works for any concept type
lens explain "ReAct pattern"               # agentic patterns too
lens explain "MoE" --related               # emphasize connected concepts
lens explain "attention" --evolution        # focus on evolution tree
lens explain "RLHF" --tradeoffs            # focus on tradeoffs resolved

# Explore (browsing)
lens explore parameters
lens explore principles
lens explore matrix
lens explore matrix 12 8
lens explore architecture
lens explore architecture attention
lens explore agents
lens explore agents multi-agent
lens explore paper 2401.12345
lens explore evolution attention
lens explore ideas                    # browse ideation reports
lens explore ideas --type sparse      # filter by gap type

# Monitor (continuous)
lens monitor --interval weekly
lens monitor --trending               # includes ideation gaps

# Config
lens config set llm.default_model openrouter/anthropic/claude-sonnet
lens config show
```

## Library API

Public API is synchronous (async internals are wrapped with `asyncio.run()` as needed, matching triz-ai's pattern). Typer CLI is a thin wrapper over these functions.

```python
from lens import LensStore, analyze, explain, explore, acquire_arxiv, extract, build_taxonomy

# Problem-solving
result = analyze("reduce hallucination without increasing latency")
print(result.principles)
print(result.architecture_suggestions)
print(result.agentic_suggestions)

# Education
explanation = explain("grouped-query attention")
print(explanation.narrative)
print(explanation.evolution)
print(explanation.tradeoffs)

# Exploration
params = explore.parameters()
cell = explore.matrix(12, 8)
tree = explore.architecture("attention")
ideas = explore.ideas(gap_type="sparse_cell")

# Pipeline (sync wrappers over async internals)
papers = acquire_arxiv(query="LLM reasoning", since="2025-01")
extract(papers)
taxonomy = build_taxonomy()
```

## Project Structure

```
lens/
├── pyproject.toml
├── CLAUDE.md
├── src/
│   └── lens/
│       ├── __init__.py
│       ├── cli.py                  # Typer CLI (sync wrapper over async internals)
│       ├── config.py               # YAML config management
│       ├── acquire/
│       │   ├── arxiv.py            # arXiv API client with retry
│       │   ├── openalex.py         # OpenAlex enrichment (citations, venue)
│       │   ├── semantic_scholar.py # Semantic Scholar SPECTER2 embeddings
│       │   ├── seed.py             # Curated seed paper loader
│       │   ├── pdf.py              # Local PDF ingestion
│       │   └── quality.py          # Quality scoring (citations+venue+recency)
│       ├── extract/
│       │   ├── extractor.py        # LLM extraction orchestrator
│       │   └── prompts.py          # Extraction prompt templates
│       ├── taxonomy/
│       │   ├── __init__.py         # build_taxonomy() entry point
│       │   ├── embedder.py         # Sentence-transformer embeddings
│       │   ├── clusterer.py        # HDBSCAN + KMeans fallback
│       │   ├── labeler.py          # LLM-based cluster naming
│       │   └── versioning.py       # Taxonomy version tracking
│       ├── knowledge/
│       │   └── matrix.py           # Contradiction matrix construction
│       ├── serve/
│       │   ├── analyzer.py         # Tradeoff analysis (query→classify→lookup)
│       │   ├── explainer.py        # Concept explanation with graph walk
│       │   └── explorer.py         # Browse parameters/principles/matrix/papers
│       ├── store/
│       │   ├── store.py            # LensStore: LanceDB wrapper + table mgmt
│       │   └── models.py           # Pydantic LanceModel schemas (all tables)
│       ├── llm/
│       │   └── client.py           # LLMClient (litellm async wrapper)
│       ├── monitor/
│       │   ├── watcher.py          # Monitor cycle: acquire→extract→ideate
│       │   └── ideation.py         # Gap analysis: sparse cells + cross-pollination
│       └── data/
│           └── seed_papers.yaml
└── tests/
    ├── conftest.py                 # Shared fixtures (tmp_path LanceDB instances)
    ├── test_acquire_arxiv.py
    ├── test_acquire_openalex.py
    ├── test_acquire_pdf.py
    ├── test_acquire_seed.py
    ├── test_acquire_semantic.py
    ├── test_analyzer.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_explainer.py
    ├── test_explorer.py
    ├── test_extract.py
    ├── test_ideation.py
    ├── test_llm_client.py
    ├── test_matrix.py
    ├── test_models.py
    ├── test_monitor.py
    ├── test_quality.py
    ├── test_store.py
    └── test_taxonomy.py
```

## Configuration

```yaml
# ~/.lens/config.yaml
llm:
  default_model: openrouter/anthropic/claude-sonnet-4-6
  extract_model: openrouter/google/gemini-2.5-flash
  label_model: openrouter/anthropic/claude-sonnet-4-6
  api_base: ""                    # OpenAI-compatible endpoint (gateway mode)
  api_key: ""                     # API key for the endpoint

acquire:
  arxiv_categories: ["cs.CL", "cs.LG", "cs.AI"]
  openalex_mailto: ""             # set a real email for OpenAlex polite pool
  quality_min_citations: 0
  quality_venue_tiers:
    tier1: ["ICML", "NeurIPS", "ICLR", "ACL", "EMNLP", "COLM"]
    tier2: ["AAAI", "NAACL", "EACL", "COLING"]

taxonomy:
  target_parameters: 25
  target_principles: 35
  target_arch_variants: 20
  target_agentic_patterns: 15
  min_cluster_size: 3
  embedding_provider: local     # "local" (sentence-transformers) or "cloud" (litellm)
  embedding_model: specter2     # local: model name; cloud: litellm model string
  embedding_dim: 768            # vector dimension (change requires lens init --force)

monitor:
  ideate: true              # enable ideation in monitor (default: true)
  ideate_llm: false         # enable LLM enrichment layer (default: false)
  ideate_top_n: 10          # how many gaps to surface per cycle
  ideate_min_gap_score: 0.5 # minimum relevance score to surface

storage:
  data_dir: ~/.lens/data
```

## Bootstrapping Plan

### Seed Paper Coverage (~200 papers)

Organized by primary contribution area:
- Foundational architecture (~10): Transformer, GPT-3, LLaMA, etc.
- Architecture innovations (~30): Attention variants, positional encodings, SSMs, MoE, normalization, FFN
- Training & alignment (~25): RLHF, DPO, KTO, GRPO, Constitutional AI, scaling laws, SFT, data mixing
- Efficiency & deployment (~25): LoRA, QLoRA, quantization, speculative decoding, serving
- Capabilities & reasoning (~25): CoT, self-consistency, ToT, RAG, tool use, code generation
- Agentic systems (~25): ReAct, Reflexion, MetaGPT, AutoGen, orchestration, planning, memory
- Safety & alignment (~15): Red teaming, representation engineering, guardrails
- Evaluation (~10): LLM-as-judge, benchmarks, arenas

### Bootstrap Sequence

1. **Phase 0 — Setup** (~1 day dev): Initialize LanceDB, define Pydantic/LanceModel schemas, implement seed loader
2. **Phase 1 — Acquire** (~minutes runtime): Fetch seed papers, enrich metadata, get SPECTER2 embeddings
3. **Phase 2 — Extract** (~1-2 hours LLM time): Extract tuples from all 200 papers. Expect ~400-600 tradeoffs, ~150-200 architecture tuples, ~100-150 agentic tuples
4. **Phase 3 — Taxonomize** (~minutes compute): Cluster into parameters, principles, slots, patterns. Human spot-check: do emergent categories match intuition?
5. **Phase 4 — Structure** (~seconds): Populate matrix and catalogs
6. **Phase 5 — Validate** (manual): Test queries against expected results

### Validation Criteria

The bootstrap succeeds if:
- Emergent parameters include recognizable dimensions (accuracy, latency, model size, safety, etc.)
- Emergent principles include recognizable techniques (distillation, quantization, MoE, RAG, CoT, etc.)
- Architecture slots include recognizable components (attention, positional encoding, FFN, normalization)
- `lens analyze "reduce hallucination without increasing latency"` returns sensible suggestions (RAG, constrained decoding, etc.)
- `lens explain "grouped-query attention"` returns a coherent narrative covering what it is, its evolution from MHA, and what tradeoffs it resolves
- `lens explain "attention"` returns a broad overview with pointers to variants rather than deep-diving a single node
- Ideation gap analysis on the seed corpus produces at least 5 sparse cells and 3 cross-pollination candidates

## Phase 2: Multimodal (Future)

Schema-ready from v1 but implemented later:

- Figure extraction from PDFs (using Marker/Nougat for detection, multimodal LLM for structured extraction)
- Figure types: architecture diagrams, scaling curves, agent workflow diagrams, ablation plots
- Architecture diagrams -> structured component graphs (enriches Architecture Catalog)
- Scaling curves -> data points + relationship types (future Scaling Atlas)
- Agent workflow diagrams -> structured flow descriptions (enriches Agentic Catalog)
- Stored in LanceDB (native multimodal column support)

## Key Methods & Tools Referenced

Methods selected based on research survey of 2023-2026 literature:

- **BERTopic** + HDBSCAN: Neural topic modeling for taxonomy discovery. Pre-trained arxiv model exists.
- **BERTrend**: Online/streaming extension of BERTopic for emerging trend detection.
- **TopicGPT**: Prompt-based topic discovery with controllable abstraction level.
- **ClusterLLM**: LLM feedback for determining right clustering granularity.
- **LiSA** (ACL 2025): State-of-the-art embed-cluster-label framework.
- **Trapp & Warschat (2024)**: LLM-based contradiction extraction from patents (F1=0.93). Methodology transfers to papers.
- **PaperQA2** (FutureHouse): RAG agent with superhuman literature synthesis, contradiction detection capability.
- **SPECTER2**: Scientific document embeddings trained on 6M triplets across 23 fields.
- **OpenAlex**: 269M papers, free API, keyword-tagged. Data acquisition backbone.

## Monitoring (Observatory Mode)

`lens monitor` implements continuous arxiv monitoring:

1. **Scheduled acquisition**: Query arxiv for new papers in configured categories + LLM keywords. Configurable interval (daily, weekly). Stores new papers, enriches via OpenAlex, fetches SPECTER2 embeddings.
2. **Auto-extraction**: Newly acquired papers are extracted automatically (using `extract_model` for cost efficiency).
3. **Taxonomy drift detection**: After every N new papers (configurable, default 100), compare new extractions against current taxonomy centroids. If >20% of new strings are noise (no close centroid match), flag for taxonomy rebuild.
4. **Trend detection**: BERTrend analysis on extraction timestamps surfaces emerging techniques (weak signals) and established trends (strong signals). Available via `lens monitor --trending`.
5. **Ideation** (automatic, runs after extraction): Analyzes the current knowledge structures for gaps and emerging opportunities. Two layers:
   - **Layer 1 — Gap analysis** (deterministic, always runs): Detects four types of gaps:
     - *Sparse matrix cells*: Tradeoff pairs with fewer than N known principles (default 2) — known problems without well-established solutions.
     - *Cross-pollination candidates*: A principle resolves (A, B) but not (A, B') where B and B' are semantically similar parameters (cosine similarity of their `embedding` vectors in the `parameters` table, computed via NumPy on vectors loaded from Lance, threshold configurable, default 0.75) — untested transfers of known techniques.
     - *Stalled architecture slots*: Slots where the most recent variant is older than a threshold (e.g., 18 months) relative to corpus recency — areas ripe for innovation.
     - *Trend-gap intersections*: BERTrend detects an emerging technique that doesn't yet appear in matrix cells or catalogs — early signals not yet well-characterized. Trend data from BERTrend is ephemeral (recomputed each cycle); only gaps produced from trend-gap intersections are persisted as `IdeationGap` entries.
   - **Layer 2 — LLM enrichment** (optional, controlled by `monitor.ideate_llm` config): Takes the top-N gaps and asks an LLM to articulate each as a human-readable research hypothesis, suggest why the gap exists, propose directions, and rate novelty/feasibility. Uses `default_model` for quality (runs on only top-N gaps so cost is minimal — see Cost Estimation).
   - Ideation results are stored in LanceDB (`ideation_reports`, `ideation_gaps` tables) and surfaced alongside trend data in `lens monitor --trending`.
6. **Taxonomy rebuild is NOT automatic** — `lens build taxonomy` must be run manually or scripted. This prevents taxonomy churn from small batches of noisy papers.

## Error Handling

- **LLM extraction failures**: 1 retry with stricter prompt. On second failure, store partial results (successfully parsed tuple types) and mark paper as `extraction_incomplete`. These papers are retried on next `lens extract` run.
- **API failures** (arxiv, OpenAlex, Semantic Scholar): Exponential backoff with jitter. After 3 retries, skip the paper and log a warning. Acquisition is resumable — papers already stored are not re-fetched.
- **Missing embeddings**: If Semantic Scholar does not have SPECTER2 embeddings for a paper (uncommon for arxiv), fall back to local SPECTER2 model if installed, otherwise skip embedding and mark paper as `embedding_missing`. These papers are excluded from vector search but still participate in extraction and taxonomy.
- **LanceDB consistency**: LanceDB writes are append-only with MVCC versioning, so partial writes do not corrupt existing data. If a crash occurs mid-write, re-running the pipeline stage repairs inconsistencies (all stages are idempotent).
- **Clustering failures**: If HDBSCAN produces degenerate results (all noise, or a single cluster), fall back to KMeans with `target_parameters`/`target_principles` as k, and log a warning.

## Cost Estimation

**Bootstrap** (~200 seed papers):
- Extraction with Gemini 2.5 Flash: ~$1-3 (abstracts only) or ~$5-15 (full text)
- Extraction with Claude Sonnet: ~$5-15 (abstracts) or ~$20-60 (full text)
- Taxonomy clustering: negligible (local compute)
- API calls: free (arxiv, OpenAlex, Semantic Scholar within free tiers)

**Ongoing monitoring** (~50 papers/week):
- Extraction: ~$0.50-3/week with Gemini Flash
- Ideation Layer 1 (gap analysis): zero LLM cost — deterministic Polars operations on Lance tables
- Ideation Layer 2 (LLM enrichment, optional): ~$0.05-0.20 per monitor cycle with default model (top-N gaps only)
- API calls: well within free tier limits
- Taxonomy rebuild (~monthly): negligible

**Education queries** (`lens explain`):
- One LLM call per query (synthesis). Uses `default_model`. Negligible per-query cost.

**Recommendation**: Use `extract_model` (Gemini 2.5 Flash or similar cheap model) for bulk extraction. Use `default_model` (Claude Sonnet) only for seed papers and taxonomy labeling where quality matters most.

## Testing Strategy

- **Unit tests**: Pydantic model validation, taxonomy versioning logic, matrix construction, quality scoring. No LLM calls.
- **Fixture-based integration tests**: Recorded LLM responses (saved as JSON fixtures) for extraction and classification. Tests verify that the full pipeline produces expected outputs from known inputs without hitting live APIs.
- **Snapshot tests for taxonomy**: Store a reference taxonomy built from test fixtures. Verify that pipeline changes don't unexpectedly alter cluster assignments.
- **Education tests**: Verify graph walk produces expected structure for known concepts (e.g., "GQA" should include evolution chain, matrix cells, related concepts). Uses fixture data, no LLM calls for the walk; LLM synthesis tested via recorded responses.
- **Ideation tests**: Verify gap detection logic against a fixture matrix with known sparse cells, known similar parameter pairs, and known stale slots. Layer 2 tested via recorded LLM responses.
- **Live integration tests** (optional, CI-skippable): Hit real arxiv/OpenAlex APIs with a small query to verify client code. Marked with `@pytest.mark.integration`.

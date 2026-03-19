# LENS — LLM Engineering Navigation System

Design spec for a system that automatically discovers recurring solution patterns, contradiction resolutions, architecture innovations, and agentic design patterns from LLM research papers (arxiv), inspired by TRIZ methodology.

**Status**: Draft
**Date**: 2026-03-19

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
| LLM abstraction | litellm | Multi-provider, model-agnostic |
| Structured data DB | DuckDB | Embedded, columnar, excellent SQL + JSON |
| Vector + multimodal DB | LanceDB | Embedded, native vector search, multimodal-ready |
| Topic modeling | BERTopic + HDBSCAN | State-of-the-art neural topic modeling |
| Scientific embeddings | SPECTER2 | Purpose-built for scientific documents |
| Data validation | Pydantic | Structured LLM output validation |
| Paper sources | arxiv API, OpenAlex, Semantic Scholar | Complementary metadata and embeddings |

### Data Model

Four layers: Raw Extraction -> Taxonomy -> Knowledge Structures -> Query Interface.

#### Layer 1: Raw Extractions (per-paper, stored in DuckDB)

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

Principle:
  id: int
  name: str                # e.g. "Knowledge Distillation"
  description: str
  sub_techniques: list[str]
  raw_strings: list[str]

ArchitectureSlot:
  id: int
  name: str                # e.g. "Attention Mechanism"
  variants: list[ArchitectureVariant]

ArchitectureVariant:
  id: int
  slot_id: int
  name: str                # e.g. "Grouped-Query Attention"
  replaces: list[int]      # variant IDs this generalizes/replaces
  properties: str
  paper_ids: list[str]

AgenticPattern:
  id: int
  name: str                # e.g. "Reflexion"
  category: str            # "single-agent" | "multi-agent" | "orchestration"
  description: str
  components: list[str]
  use_cases: list[str]
  paper_ids: list[str]
```

#### Layer 3: Knowledge Structures

```
ContradictionMatrix:
  cells: dict[(improving_param_id, worsening_param_id), list[principle_id]]

ArchitectureCatalog:
  slots: list[ArchitectureSlot]   # each with variant evolution trees

AgenticPatternCatalog:
  patterns: list[AgenticPattern]  # organized by category
```

#### Storage

DuckDB (structured data):
- `papers` (id, title, abstract, authors, venue, date, arxiv_id, citations, quality_score)
- `tradeoff_extractions`, `architecture_extractions`, `agentic_extractions`
- `parameters`, `principles`, `architecture_slots`, `architecture_variants`, `agentic_patterns`
- `matrix_cells`
- `taxonomy_versions`

LanceDB (vectors + future multimodal):
- `paper_embeddings` (SPECTER2 vectors + paper metadata)
- `paper_figures` (Phase 2: figure images + extracted structured data)

### Database Design: DuckDB + LanceDB

Two embedded databases, each handling what it does best:

```python
class LensStore:
    def __init__(self, data_dir: str):
        self.db = duckdb.connect(f"{data_dir}/lens.duckdb")
        self.vectors = lancedb.connect(f"{data_dir}/lens.lance")
```

- **DuckDB**: All structured/relational data. Powerful SQL with native JSON, list, and struct types. Excellent for analytical queries (aggregations, pattern frequency counting).
- **LanceDB**: Vector search (paper similarity, clustering input) and future multimodal storage (figures/diagrams). Native ANN search, built on Apache Arrow.

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

Two user modes:

**Problem-solving** (`lens analyze`): User describes a problem in natural language. Routing depends on `--type`:
- Default (no type): LLM classifies the tradeoff (identifying improving/worsening parameters) -> contradiction matrix lookup -> return ranked principles with paper references.
- `--type architecture`: LLM identifies the relevant architecture slot and constraints -> search architecture catalog by slot + vector similarity -> return relevant variants with properties and evolution context.
- `--type agentic`: LLM identifies the agent task type and constraints -> search agentic catalog by category + vector similarity -> return relevant patterns with components and use cases.

All three modes enrich results with paper references and evidence quotes from the extraction layer.

**Exploration** (`lens explore`): Browse parameters, principles, matrix cells, architecture trees, agentic patterns, evolution trajectories.

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

# Monitor (continuous)
lens monitor --interval weekly
lens monitor --trending

# Config
lens config set llm.default_model openrouter/anthropic/claude-sonnet
lens config show
```

## Library API

Public API is synchronous (async internals are wrapped with `asyncio.run()` as needed, matching triz-ai's pattern). Typer CLI is a thin wrapper over these functions.

```python
from lens import LensStore, analyze, explore, acquire_arxiv, extract, build_taxonomy

# Problem-solving
result = analyze("reduce hallucination without increasing latency")
print(result.principles)
print(result.architecture_suggestions)
print(result.agentic_suggestions)

# Exploration
params = explore.parameters()
cell = explore.matrix(12, 8)
tree = explore.architecture("attention")

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
│       ├── cli.py
│       ├── acquire/
│       │   ├── arxiv.py
│       │   ├── openalex.py
│       │   ├── semantic_scholar.py
│       │   └── seed.py
│       ├── extract/
│       │   ├── extractor.py
│       │   └── prompts.py
│       ├── taxonomy/
│       │   ├── clusterer.py
│       │   ├── labeler.py
│       │   └── versioning.py
│       ├── knowledge/
│       │   ├── matrix.py
│       │   ├── architecture.py
│       │   └── agentic.py
│       ├── serve/
│       │   ├── analyzer.py
│       │   └── explorer.py
│       ├── store/
│       │   ├── duckdb_store.py
│       │   ├── lance_store.py
│       │   └── models.py
│       ├── llm/
│       │   └── client.py
│       ├── monitor/
│       │   └── watcher.py
│       └── data/
│           └── seed_papers.yaml
└── tests/
    ├── test_extract.py
    ├── test_taxonomy.py
    ├── test_matrix.py
    ├── test_architecture.py
    ├── test_agentic.py
    └── fixtures/
```

## Configuration

```yaml
# ~/.lens/config.yaml
llm:
  default_model: openrouter/anthropic/claude-sonnet-4-6
  extract_model: openrouter/google/gemini-2.5-flash
  label_model: openrouter/anthropic/claude-sonnet-4-6

acquire:
  arxiv_categories: ["cs.CL", "cs.LG", "cs.AI"]
  quality_min_citations: 0
  quality_venue_tiers:
    tier1: ["ICML", "NeurIPS", "ICLR", "ACL", "EMNLP", "COLM"]
    tier2: ["AAAI", "NAACL", "EACL", "COLING"]

taxonomy:
  target_parameters: 25
  target_principles: 35
  min_cluster_size: 3
  embedding_model: specter2

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

1. **Phase 0 — Setup** (~1 day dev): Initialize databases, define models, implement seed loader
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
5. **Taxonomy rebuild is NOT automatic** — `lens build taxonomy` must be run manually or scripted. This prevents taxonomy churn from small batches of noisy papers.

## Error Handling

- **LLM extraction failures**: 1 retry with stricter prompt. On second failure, store partial results (successfully parsed tuple types) and mark paper as `extraction_incomplete`. These papers are retried on next `lens extract` run.
- **API failures** (arxiv, OpenAlex, Semantic Scholar): Exponential backoff with jitter. After 3 retries, skip the paper and log a warning. Acquisition is resumable — papers already stored are not re-fetched.
- **Missing embeddings**: If Semantic Scholar does not have SPECTER2 embeddings for a paper (uncommon for arxiv), fall back to local SPECTER2 model if installed, otherwise skip embedding and mark paper as `embedding_missing`. These papers are excluded from vector search but still participate in extraction and taxonomy.
- **DuckDB/LanceDB consistency**: Both databases are embedded and not transactionally linked. If a crash occurs between writes, re-running the pipeline stage repairs inconsistencies (all stages are idempotent). LanceDB writes are append-only with versioning, so partial writes do not corrupt existing data.
- **Clustering failures**: If HDBSCAN produces degenerate results (all noise, or a single cluster), fall back to KMeans with `target_parameters`/`target_principles` as k, and log a warning.

## Cost Estimation

**Bootstrap** (~200 seed papers):
- Extraction with Gemini 2.5 Flash: ~$1-3 (abstracts only) or ~$5-15 (full text)
- Extraction with Claude Sonnet: ~$5-15 (abstracts) or ~$20-60 (full text)
- Taxonomy clustering: negligible (local compute)
- API calls: free (arxiv, OpenAlex, Semantic Scholar within free tiers)

**Ongoing monitoring** (~50 papers/week):
- Extraction: ~$0.50-3/week with Gemini Flash
- API calls: well within free tier limits
- Taxonomy rebuild (~monthly): negligible

**Recommendation**: Use `extract_model` (Gemini 2.5 Flash or similar cheap model) for bulk extraction. Use `default_model` (Claude Sonnet) only for seed papers and taxonomy labeling where quality matters most.

## Testing Strategy

- **Unit tests**: Pydantic model validation, taxonomy versioning logic, matrix construction, quality scoring. No LLM calls.
- **Fixture-based integration tests**: Recorded LLM responses (saved as JSON fixtures) for extraction and classification. Tests verify that the full pipeline produces expected outputs from known inputs without hitting live APIs.
- **Snapshot tests for taxonomy**: Store a reference taxonomy built from test fixtures. Verify that pipeline changes don't unexpectedly alter cluster assignments.
- **Live integration tests** (optional, CI-skippable): Hit real arxiv/OpenAlex APIs with a small query to verify client code. Marked with `@pytest.mark.integration`.

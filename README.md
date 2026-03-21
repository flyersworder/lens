# LENS — LLM Engineering Navigation System

Automatically discovers recurring solution patterns, contradiction resolutions, architecture innovations, and agentic design patterns from LLM research papers (arxiv).

Inspired by [TRIZ](https://en.wikipedia.org/wiki/TRIZ) methodology — but with richer knowledge structures, fully automated discovery, and continuous learning from the evolving LLM research landscape.

## Core Knowledge Structures

1. **Contradiction Matrix** — Maps LLM tradeoffs (e.g., accuracy vs. latency) to resolution techniques (e.g., distillation, speculative decoding). Parameters and principles are discovered automatically from papers.

2. **Architecture Catalog** — Tracks the evolution of LLM architecture components (attention, positional encoding, FFN, etc.) as taxonomy trees of variants with replaces/generalizes relationships. *(scaffolded, not yet implemented)*

3. **Agentic Pattern Catalog** — Catalogs recurring patterns for building and orchestrating LLM-based agents (ReAct, Reflexion, multi-agent debate, etc.). *(scaffolded, not yet implemented)*

## Status

**Phase 1 implemented.** The contradiction matrix pipeline (acquire → extract → taxonomy → matrix → serve) and the monitor/ideation system are fully functional. Architecture and agentic catalogs are scaffolded in the data model but not yet populated.

See [docs/specs/design.md](docs/specs/design.md) for the full design spec.

## Quick Start

```bash
# Install
uv sync

# Initialize the database and config
uv run lens init

# Acquire seed papers (~200 landmark LLM papers)
uv run lens acquire seed

# Extract tradeoffs, architecture, and agentic patterns from papers
uv run lens extract

# Build taxonomy (cluster raw extractions into parameters and principles)
uv run lens build taxonomy

# Build contradiction matrix
uv run lens build matrix

# Or do both at once
uv run lens build all
```

## Usage

```bash
# Analyze a tradeoff — suggests resolution techniques from the matrix
uv run lens analyze "reduce hallucination without hurting latency"

# Explain any LLM concept with adaptive depth
uv run lens explain "grouped-query attention"
uv run lens explain "knowledge distillation" --tradeoffs
uv run lens explain "MoE" --related

# Browse the knowledge base
uv run lens explore parameters
uv run lens explore principles
uv run lens explore matrix
uv run lens explore matrix 12 8        # specific parameter pair
uv run lens explore paper 2401.12345

# Acquire more papers from arxiv
uv run lens acquire arxiv --query "LLM" --since 2025-01
uv run lens acquire file paper.pdf     # ingest a local PDF

# Run a monitoring cycle (acquire → extract → ideate)
uv run lens monitor
uv run lens monitor --trending         # show ideation gaps

# Browse research opportunities
uv run lens explore ideas
uv run lens explore ideas --type sparse_cell

# Configuration
uv run lens config show
uv run lens config set llm.default_model openrouter/anthropic/claude-sonnet-4-6
```

## Architecture

- **Python 3.12+** with `uv` package manager
- **LanceDB** — embedded vector database with Pydantic schema definitions
- **Polars** — zero-copy Arrow-native analytics for matrix construction
- **litellm** — multi-provider LLM abstraction (async)
- **HDBSCAN + KMeans** — density-based clustering with fallback
- **sentence-transformers** — SPECTER2 / MiniLM embeddings
- **Typer** — CLI framework

Data flows through four layers:

```
Layer 0: Papers (arxiv, PDF, OpenAlex enrichment)
    ↓
Layer 1: Raw Extractions (LLM-extracted tradeoffs, architecture, agentic patterns)
    ↓
Layer 2: Taxonomy (clustered + labeled parameters, principles, versioned)
    ↓
Layer 3: Knowledge Structures (contradiction matrix, ideation gaps)
```

Public API is synchronous; async internals are wrapped with `asyncio.run()`.

## Testing

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_store.py -v   # run specific test file
uv run pytest -m "not integration"     # skip live API tests
```

Tests use `tmp_path` fixtures for isolated LanceDB instances. No mocking of LanceDB — real embedded instances are used in all tests.

## License

MIT

# LENS — LLM Engineering Navigation System

[![PyPI version](https://img.shields.io/pypi/v/lens-research)](https://pypi.org/project/lens-research/)
[![Python](https://img.shields.io/pypi/pyversions/lens-research)](https://pypi.org/project/lens-research/)
[![CI](https://github.com/flyersworder/lens/actions/workflows/ci-and-publish.yml/badge.svg)](https://github.com/flyersworder/lens/actions/workflows/ci-and-publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Automatically discovers recurring solution patterns, contradiction resolutions, architecture innovations, and agentic design patterns from LLM research papers (arxiv).

Inspired by [TRIZ](https://en.wikipedia.org/wiki/TRIZ) methodology — but with richer knowledge structures, fully automated discovery, and continuous learning from the evolving LLM research landscape.

## Core Knowledge Structures

1. **Contradiction Matrix** — Maps LLM tradeoffs (e.g., accuracy vs. latency) to resolution techniques (e.g., distillation, speculative decoding). Uses a canonical vocabulary of parameters and principles, extensible via LLM-proposed new concepts.

2. **Architecture Catalog** — Organizes LLM architecture components (attention, positional encoding, FFN, etc.) by slot with property-based comparison across variants. Answers "what are my options for component X?"

3. **Agentic Pattern Catalog** — Catalogs recurring patterns for building LLM-based agents (ReAct, Reflexion, multi-agent debate, etc.) with emergent categories discovered from data.

## Status

**Core pipeline implemented.** All three knowledge structures are functional: contradiction matrix, architecture catalog (property-based comparison), and agentic pattern catalog (emergent categories). Monitor/ideation pipeline operational.

See [docs/architecture.md](docs/architecture.md) for the full architecture doc.

## Quick Start

```bash
# Install dependencies
uv sync

# Install pre-commit hooks (uses prek, a fast Rust-based pre-commit runner)
uv tool install prek
prek install

# Initialize the database and config
uv run lens init

# Acquire seed papers (10 landmark LLM papers)
uv run lens acquire seed

# Initialize canonical vocabulary (12 parameters + 12 principles)
uv run lens vocab init

# Extract tradeoffs, architecture, and agentic patterns from papers
uv run lens extract

# Build taxonomy and contradiction matrix
uv run lens build all
```

## Usage

```bash
# Analyze a tradeoff — suggests resolution techniques from the matrix
uv run lens analyze "reduce hallucination without hurting latency"

# Analyze architecture — find matching variants by property
uv run lens analyze --type architecture "efficient attention for long context"

# Analyze agentic — find matching patterns
uv run lens analyze --type agentic "reliable multi-step code generation"

# Explain any LLM concept with adaptive depth
uv run lens explain "grouped-query attention"
uv run lens explain "knowledge distillation" --tradeoffs
uv run lens explain "MoE" --related

# Browse the knowledge base
uv run lens vocab list                      # list vocabulary (parameters + principles)
uv run lens vocab list --kind parameter     # filter by kind
uv run lens vocab show inference-latency    # details for a concept
uv run lens explore matrix
uv run lens explore paper 2401.12345

# Browse architecture catalog
uv run lens explore architecture            # list all slots with variant counts
uv run lens explore architecture Attention  # compare variants with properties
uv run lens explore evolution Attention     # timeline view by paper date

# Browse agentic patterns
uv run lens explore agents                  # list patterns by category
uv run lens explore agents Reasoning        # filter by category

# Acquire more papers from arxiv
uv run lens acquire arxiv --query "LLM" --since 2025-01
uv run lens acquire file paper.pdf          # ingest a local PDF

# Run a monitoring cycle (acquire → extract → ideate)
uv run lens monitor
uv run lens monitor --trending              # show ideation gaps

# Browse research opportunities
uv run lens explore ideas
uv run lens explore ideas --type sparse_cell

# Health-check the knowledge base
uv run lens lint                               # report issues across 6 categories
uv run lens lint --fix                         # auto-fix safe issues
uv run lens lint --check orphans,stale         # run specific checks only

# View the event log (audit trail of all mutations)
uv run lens log                                # last 20 events
uv run lens log --kind extract                 # filter by event kind
uv run lens log --since 2026-04-01 --limit 50  # date range + limit

# Configuration
uv run lens config show
uv run lens config set llm.default_model openrouter/anthropic/claude-sonnet-4-6
```

## LLM Backend

LENS needs an LLM for extraction, taxonomy labeling, and analysis. For production deployment, see [docs/deployment.md](docs/deployment.md). Two options:

**Gateway mode (recommended for production)** — Point to any OpenAI-compatible endpoint (litellm gateway, vLLM, Ollama). No litellm dependency needed. Keeps API keys out of application pods.

```yaml
# ~/.lens/config.yaml
llm:
  api_base: "http://litellm-gateway:4000/v1"
  api_key: "your-gateway-key"
  default_model: "gpt-4"
```

**Direct mode** — Install litellm for multi-provider routing (OpenRouter, OpenAI, Anthropic, etc.):

```bash
uv add lens[litellm]
```

## Embeddings

Two embedding providers, configurable via `~/.lens/config.yaml`:

**Local (default)** — sentence-transformers (SPECTER2 / MiniLM fallback). Free, works offline, but requires ~400MB model download on first use.

**Cloud** — Any embedding API via litellm or OpenAI-compatible endpoint. Fast, scalable, no local model needed.

```bash
# Switch to cloud embeddings
uv run lens config set embeddings.provider cloud
uv run lens config set embeddings.model text-embedding-3-small
```

## Architecture

- **Python 3.12+** with `uv` package manager
- **SQLite + sqlite-vec** — embedded database with vector search (cosine distance)
- **openai SDK** — LLM and embedding client (works with any OpenAI-compatible endpoint)
- **litellm** (optional) — multi-provider routing for direct API access
- **Guided extraction** — canonical vocabulary for all extraction types (tradeoffs, architecture, agentic)
- **sentence-transformers** or **cloud embeddings** — configurable provider
- **Typer** — CLI framework

Data flows through four layers:

```
Layer 0: Papers (arxiv, PDF, OpenAlex enrichment)
    ↓
Layer 1: Raw Extractions (LLM-extracted tradeoffs, architecture, agentic patterns)
    ↓
Layer 2: Taxonomy (unified vocabulary: parameters, principles, arch slots, agentic categories)
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

Tests use `tmp_path` fixtures for isolated SQLite instances. No mocking — real embedded database instances are used in all tests.

## License

MIT

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

**Core pipeline implemented.** All three knowledge structures are functional: contradiction matrix, architecture catalog (property-based comparison), and agentic pattern catalog (emergent categories). Full monitor pipeline: acquire → enrich → extract → build → ideate.

**Live demo:** [lens-fawn.vercel.app](https://lens-fawn.vercel.app/) — public read-only site backed by the same Python services (search, analyze, explain) that power the CLI.

See [docs/architecture.md](docs/architecture.md) for the full architecture doc.

See [docs/web-deployment.md](docs/web-deployment.md) for the public-web deployment architecture (Turso + Vercel + GitHub Actions).

## Quick Start

```bash
# Install from PyPI
pip install lens-research
# or
uv add lens-research

# Initialize the database and config
lens init

# Acquire seed papers (10 landmark LLM papers)
lens acquire seed

# Initialize canonical vocabulary (12 parameters + 12 principles)
lens vocab init

# Extract tradeoffs, architecture, and agentic patterns from papers
lens extract

# Build taxonomy and contradiction matrix
lens build all
```

### Optional extras

```bash
# Multi-provider LLM routing (OpenRouter, Anthropic, etc.)
pip install "lens-research[litellm]"   # or: uv add "lens-research[litellm]"

# Agent-optimized paper search via DeepXiv
pip install "lens-research[deepxiv]"   # or: uv add "lens-research[deepxiv]"
```

## Usage

```bash
# Analyze a tradeoff — suggests resolution techniques from the matrix
lens analyze "reduce hallucination without hurting latency"

# Analyze architecture — find matching variants by property
lens analyze --type architecture "efficient attention for long context"

# Analyze agentic — find matching patterns
lens analyze --type agentic "reliable multi-step code generation"

# Explain any LLM concept with adaptive depth
lens explain "grouped-query attention"
lens explain "knowledge distillation" --tradeoffs
lens explain "MoE" --related

# Emit a YAML provenance sidecar linking claims to papers + vocabulary
lens analyze "reduce latency" --provenance analyze.yaml
lens explain "MoE" --provenance moe.yaml

# Search papers
lens search "attention mechanisms"          # hybrid keyword + semantic
lens search --author "Vaswani"              # filter by author
lens search "efficiency" --after 2024-01-01 # combine search + filters
lens search --venue "NeurIPS" --limit 5     # filter by venue

# Browse the knowledge base
lens vocab list                      # list vocabulary (parameters + principles)
lens vocab list --kind parameter     # filter by kind
lens vocab show inference-latency    # details for a concept
lens explore matrix
lens explore paper 2401.12345

# Browse architecture catalog
lens explore architecture            # list all slots with variant counts
lens explore architecture Attention  # compare variants with properties
lens explore evolution Attention     # timeline view by paper date

# Browse agentic patterns
lens explore agents                  # list patterns by category
lens explore agents Reasoning        # filter by category

# Acquire more papers from arxiv
lens acquire arxiv --query "LLM" --since 2025-01
lens acquire file paper.pdf          # ingest a local PDF

# Acquire via DeepXiv (requires: pip install "lens-research[deepxiv]")
lens acquire deepxiv "LLM agent architecture" --max-results 10
lens acquire deepxiv --paper 2507.01701  # single paper with rich metadata

# Fetch SPECTER2 embeddings from Semantic Scholar
lens acquire semantic                    # all papers missing embeddings
lens acquire semantic --paper-id 2401.12345  # specific paper

# Knowledge base overview
lens status                          # paper counts, vocab, matrix, issues

# Run a monitoring cycle (acquire → enrich → extract → build → ideate)
lens monitor
lens monitor --skip-enrich           # skip OpenAlex enrichment
lens monitor --skip-build            # skip taxonomy/matrix rebuild
lens monitor --trending              # show ideation gaps

# Browse research opportunities
lens explore ideas
lens explore ideas --type sparse_cell

# Health-check the knowledge base
lens lint                               # report issues across 7 categories
lens lint --fix                         # auto-fix safe issues
lens lint --check orphans,stale         # run specific checks only

# View the event log (audit trail of all mutations)
lens log                                # last 20 events
lens log --kind extract                 # filter by event kind
lens log --since 2026-04-01 --limit 50  # date range + limit

# Configuration
lens config show
lens config set llm.default_model openrouter/anthropic/claude-sonnet-4-6

# Verbose logging (-v=INFO, -vv=DEBUG)
lens -v extract
lens -vv monitor
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
pip install "lens-research[litellm]"   # or: uv add "lens-research[litellm]"
```

## Embeddings

Two embedding providers, configurable via `~/.lens/config.yaml`:

**Local (default)** — sentence-transformers (SPECTER2 / MiniLM fallback). Free, works offline, but requires ~400MB model download on first use.

**Cloud** — Any embedding API via litellm or OpenAI-compatible endpoint. Fast, scalable, no local model needed.

```bash
# Switch to cloud embeddings
lens config set embeddings.provider cloud
lens config set embeddings.model text-embedding-3-small
```

## Architecture

- **Python 3.12+** with `uv` package manager
- **SQLite + sqlite-vec** — embedded database with vector search (cosine distance)
- **openai SDK** — LLM and embedding client (works with any OpenAI-compatible endpoint)
- **litellm** (optional) — multi-provider routing for direct API access
- **deepxiv-sdk** (optional) — agent-optimized paper search with hybrid retrieval and progressive reading
- **Guided extraction** — canonical vocabulary for all extraction types (tradeoffs, architecture, agentic)
- **sentence-transformers** or **cloud embeddings** — configurable provider
- **Typer** — CLI framework

Data flows through four layers:

```
Layer 0: Papers (arxiv, DeepXiv, PDF, OpenAlex enrichment)
    ↓
Layer 1: Raw Extractions (LLM-extracted tradeoffs, architecture, agentic patterns)
    ↓
Layer 2: Taxonomy (unified vocabulary: parameters, principles, arch slots, agentic categories)
    ↓
Layer 3: Knowledge Structures (contradiction matrix, ideation gaps)
```

Public API is synchronous; async internals are wrapped with `asyncio.run()`.

## Development

```bash
# Clone and set up for development
git clone https://github.com/flyersworder/lens.git
cd lens
uv sync

# Install pre-commit hooks (uses prek, a fast Rust-based pre-commit runner)
uv tool install prek
prek install

# Run the CLI from source
uv run lens init
```

## Testing

```bash
uv run pytest                          # run all tests
uv run pytest tests/test_store.py -v   # run specific test file
uv run pytest -m "not integration"     # skip live API tests
```

Tests use `tmp_path` fixtures for isolated SQLite instances. No mocking — real embedded database instances are used in all tests.

## License

MIT

# LENS — LLM Engineering Navigation System

Automatically discovers recurring solution patterns, contradiction resolutions, architecture innovations, and agentic design patterns from LLM research papers (arxiv).

Inspired by [TRIZ](https://en.wikipedia.org/wiki/TRIZ) methodology — but with richer knowledge structures, fully automated discovery, and continuous learning from the evolving LLM research landscape.

## Core Knowledge Structures

1. **Contradiction Matrix** — Maps LLM tradeoffs (e.g., accuracy vs. latency) to resolution techniques (e.g., distillation, speculative decoding). Parameters and principles are discovered automatically from papers.

2. **Architecture Catalog** — Tracks the evolution of LLM architecture components (attention, positional encoding, FFN, etc.) as taxonomy trees of variants with replaces/generalizes relationships.

3. **Agentic Pattern Catalog** — Catalogs recurring patterns for building and orchestrating LLM-based agents (ReAct, Reflexion, multi-agent debate, etc.).

## Status

**Design phase.** See [docs/specs/design.md](docs/specs/design.md) for the full design spec.

## License

MIT

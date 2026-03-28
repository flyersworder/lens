# Changelog

## 0.3.0 (2026-03-28)

### Breaking Changes
- **SQLite replaces LanceDB** — Database migrated from LanceDB to SQLite + sqlite-vec. Existing `.lance` databases are not compatible; run `lens init --force` to create a new database.
- **Polars removed** — All analytics now use SQL queries and plain Python. Polars is no longer a dependency.
- **Config restructured** — Embedding config moved from `taxonomy.embedding_provider/model/dim` to dedicated `embeddings.provider/model/dimensions/api_base/api_key` section.

### Added
- **Architecture Catalog** — Property-based comparison of LLM architecture variants organized by slot (attention, positional encoding, FFN, etc.). Browse with `lens explore architecture`.
- **Agentic Pattern Catalog** — Patterns organized by emergent categories discovered from data (not a fixed enum). Browse with `lens explore agents`.
- **Architecture timeline** — Chronological view of variants by paper date via `lens explore evolution`.
- **Architecture/agentic analysis** — `lens analyze --type architecture` and `--type agentic` for vector search against variants and patterns.
- **Cloud embedding support** — Configurable embedding provider (`local` or `cloud`) with independent API base/key from LLM endpoint.
- **Gateway mode** — LLM backend works with any OpenAI-compatible endpoint via `llm.api_base` config. No litellm required.
- **Cosine distance** for all vector search (sqlite-vec `distance_metric=cosine`).
- **Parameterized SQL queries** everywhere — eliminates SQL injection risk (replaces manual `escape_sql_string`).
- **Dependabot** enabled for weekly dependency vulnerability scanning.

### Changed
- **litellm is now optional** — Install with `uv add lens[litellm]` for multi-provider routing. Core dependency is the lightweight `openai` SDK.
- **Models are Pydantic BaseModel** — No longer coupled to LanceDB's LanceModel. Validation only.
- **Auto-increment IDs** — Replaced fragile offset-based ID allocation with `MAX(id) + 1` pattern.
- **Pre-commit hooks** — Switched from `pre-commit` to `prek` (Rust-based, faster, no Python runtime needed).
- **ty type checker** — All false positives resolved properly (12 of 19 fixed at source, 7 genuinely unfixable from upstream).

### Fixed
- OpenAlex enrichment now persists data back to database (was silently a no-op).
- LLM client raises `ValueError` on null content instead of crashing callers.
- XML ParseError handled in arxiv parser (rate-limit pages no longer crash pipeline).
- `find_sparse_cells` now surfaces zero-evidence gaps (most interesting research gaps).
- Seed/semantic scholar retry on HTTP 429 (rate limit).
- PDF ingestion uses file modification time instead of hardcoded date.
- Narrowed exception handling from bare `Exception` to specific types.
- ArXiv API switched from HTTP to HTTPS.

### Removed
- `lancedb` dependency
- `polars` dependency
- `pandas` dependency
- `_TableWrapper`, `_DatabaseWrapper` wrapper classes
- `escape_sql_string` utility (replaced by parameterized queries)
- Offset-based ID allocation (`version * 100000 + offset`)

## 0.2.0 (2026-03-28)

### Added
- Cloud embedding provider support via litellm.
- Configurable `EMBEDDING_DIM` constant (was hardcoded 768).
- Gateway mode: `llm.api_base` config for OpenAI-compatible endpoints.
- litellm moved to optional dependency.

## 0.1.0 (2026-03-21)

### Added
- Initial implementation of contradiction matrix pipeline.
- Acquire pipeline: arxiv, OpenAlex, Semantic Scholar, seed papers, PDF ingestion.
- Extract pipeline: LLM-based tradeoff/architecture/agentic extraction.
- Taxonomy pipeline: HDBSCAN clustering + LLM labeling.
- Knowledge structures: contradiction matrix construction.
- Serve pipeline: analyze (tradeoff resolution), explain (concept education), explore (browsing).
- Monitor pipeline: acquire-extract-ideate cycle with gap analysis.
- CLI via Typer with all subcommands.
- 10 curated seed papers.

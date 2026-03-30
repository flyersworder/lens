# Changelog

## 0.5.0 (2026-03-29)

### Breaking Changes
- **Unified vocabulary replaces all taxonomy tables** — `architecture_slots`,
  `architecture_variants`, `architecture_variants_vec`, `agentic_patterns`,
  `agentic_patterns_vec` tables removed. Architecture data stays in
  `architecture_extractions` with canonical slot names. Agentic data stays
  in `agentic_extractions` with a `category` field.
- **HDBSCAN/KMeans removed** — No more clustering. All taxonomy building
  uses vocabulary-based guided extraction.
- **`hdbscan` dependency removed** from pyproject.toml.
- **`taxonomy` config section removed** — `target_arch_variants`,
  `target_agentic_patterns`, `min_cluster_size` no longer exist.

### Added
- **Architecture slot vocabulary** — 10 seed slots (Attention Mechanism,
  FFN, Positional Encoding, etc.) as `kind="arch_slot"` in vocabulary.
- **Agentic category vocabulary** — 6 seed categories (Reasoning, Planning,
  Tool Use, etc.) as `kind="agentic_category"` in vocabulary.
- **`category` field** on `agentic_extractions` — LLM assigns category
  during extraction using guided vocabulary.
- **`new_concept_description` field** on `architecture_extractions` and
  `agentic_extractions` for `NEW:` concept proposals.

### Changed
- **`build_taxonomy` simplified** — single `build_vocabulary()` call replaces
  three separate builders (tradeoff + architecture + agentic).
- **Serve layer** — architecture/agentic queries use extraction tables directly
  instead of taxonomy tables. LLM-based slot/category identification replaces
  vector search.

### Removed
- `taxonomy/clusterer.py` (HDBSCAN/KMeans clustering)
- `taxonomy/labeler.py` (LLM cluster labeling)
- `ArchitectureSlot`, `ArchitectureVariant`, `AgenticPattern` models
- `hdbscan` dependency

## 0.4.0 (2026-03-29)

### Breaking Changes
- **Vocabulary replaces parameters/principles tables** — The `parameters`, `principles`, `parameters_vec`, and `principles_vec` tables are removed. All tradeoff concepts are now stored in a unified `vocabulary` table with text IDs (slugs). Existing databases are not compatible; run `lens vocab init` then re-extract.
- **Matrix cells use text IDs** — `improving_param_id`, `worsening_param_id`, and `principle_id` in `matrix_cells` changed from INTEGER to TEXT (vocabulary slugs).
- **API signatures changed** — `build_matrix()`, `get_ranked_matrix()`, serve layer functions, and ideation functions no longer take `taxonomy_version` parameter. `list_parameters()`/`list_principles()` no longer take `taxonomy_version`.
- **Config keys removed** — `taxonomy.target_parameters` and `taxonomy.target_principles` no longer exist (no clustering for tradeoffs).

### Added
- **Guided extraction** — The extraction prompt now includes a canonical vocabulary of parameters and principles. The LLM uses exact names from the vocabulary instead of free-text, eliminating the need for clustering to normalize tradeoff concepts.
- **Canonical vocabulary** — New `vocabulary` table with 12 seed parameters and 12 seed principles, each with descriptions and embeddings. Managed via `lens vocab init`, `lens vocab list`, `lens vocab show`.
- **NEW: concept auto-acceptance** — When the LLM encounters a concept not in the vocabulary during extraction, it prefixes with `NEW:` and provides a description. These are auto-accepted into the vocabulary with `source: "extracted"`.
- **Evidence & novelty scoring** — Each vocabulary entry tracks `paper_count`, `avg_confidence`, and `first_seen` date. Evidence strength and novelty are orthogonal signals for downstream consumers.
- **`new_concept_description` field** — Added to `tradeoff_extractions` for LLM-proposed concept descriptions.

### Changed
- **`build_taxonomy()` split into three functions** — `build_tradeoff_taxonomy()` (vocabulary-based, no clustering), `build_architecture_taxonomy()` (unchanged clustering), `build_agentic_taxonomy()` (unchanged clustering).
- **Matrix uses direct vocabulary lookup** — No more `raw_strings` mapping. Extraction values are canonical names, looked up directly in the vocabulary.
- **Serve layer queries vocabulary** — `explorer.py`, `analyzer.py`, `explainer.py` all query `vocabulary` table filtered by `kind` instead of separate `parameters`/`principles` tables.
- **Ideation uses vocabulary_vec** — Cross-pollination computes parameter similarity from `vocabulary_vec` embeddings.

### Removed
- `parameters` and `principles` tables (replaced by `vocabulary`)
- `parameters_vec` and `principles_vec` virtual tables (replaced by `vocabulary_vec`)
- `Parameter` and `Principle` Pydantic models (replaced by `VocabularyEntry`)
- HDBSCAN/KMeans clustering for tradeoff taxonomy (kept for architecture/agentic)
- `_build_string_to_id_map()` in matrix.py

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

# Changelog

## 0.9.1 (2026-04-21)

### Fixed
- **DeepXiv retrieve API migration** — adapt `search_deepxiv()` to the new
  `/arxiv/?type=retrieve` backend that DeepXiv rolled out on 2026-04-20.
  Response top-level key is now `result` (was `results`) and per-item citation
  field is `citation_count` (was `citation`). Old SDKs returned empty lists
  against the new backend, which broke the `test_deepxiv_search_live`
  integration test.
- **Deprecated `search_mode` kwarg removed** — the unified retrieve endpoint
  ignores `search_mode` / `bm25_weight` / `vector_weight` and warns on each
  call; stop passing them.
- **Live integration tests degrade gracefully** — `test_deepxiv_search_live`
  and `test_deepxiv_fetch_paper_live` now `pytest.skip` when the upstream
  returns an empty or stub response, not just when it raises
  `ServerError` / `RateLimitError`. Prevents unrelated PRs from being blocked
  by upstream hiccups.
- **`test_cli_skip_flags_short` ANSI-safe on CI** — strip ANSI color codes
  before asserting on CLI output so the test passes under different terminal
  widths.

### Security
- **`requests` pinned to `>=2.33.0`** via `[tool.uv] override-dependencies`
  to pick up the fix for CVE-2026-25645 (arxiv's `~=2.32.0` pin was holding
  us back on the vulnerable line).

### Changed
- **`deepxiv-sdk` minimum bumped to `>=0.2.5`** — 0.2.5 is the first release
  compatible with the new retrieve endpoint. Install with
  `pip install "lens-research[deepxiv]"` or `uv sync --extra deepxiv`.
- **`astral-sh/setup-uv` in CI bumped to 8.1.0** — upstream GitHub Action
  version bump; no functional change for LENS.
- **`pytest` dev-dep bumped to `9.0.3`**.

### Documentation
- **README rewritten for end-user installation** — focuses on `pip` / `uv`
  install flow for users consuming LENS from PyPI, rather than the
  contributor-oriented workflow.
- **Backfilled 0.9.0 entry below** — was previously only documented in the
  GitHub release notes.

## 0.9.0 (2026-04-12)

### Added
- **`lens status`** — quick overview of the knowledge base: paper counts by
  extraction status, vocabulary breakdown by kind, matrix density, top
  parameters, taxonomy version, last event, and cheap lint checks.
- **`lens acquire semantic`** — fetch SPECTER2 embeddings from Semantic
  Scholar for papers with zero-vector or missing embeddings.
- **Monitor pipeline, 5 configurable stages** — acquire (arxiv) → enrich
  (OpenAlex + quality scores) → extract (LLM) → build (taxonomy + matrix) →
  ideate (gap analysis, optional LLM). Flags `--skip-enrich`, `--skip-build`.
  Config `monitor.ideate_llm` for LLM-enriched ideation.
- **Quality scoring** — `acquire/quality.py` computes 0-1 scores (citations +
  venue tier + recency). Auto-computed after seed acquisition and OpenAlex
  enrichment.
- **`--verbose / -v` flag on all commands** — `-v` = INFO, `-vv` = DEBUG.
- **API key validation** — `_require_llm_config()` checks for LLM API key
  (or gateway config) before commands that need it (`extract`, `analyze`,
  `explain`, `monitor`) and prints a clear setup message.

### Fixed
- **`lens explore paper`** displayed nothing for date (was reading a
  nonexistent `year` field).
- **Removed misleading `--interval` no-op flag from `monitor`**.

### Documentation
- README and CLAUDE.md updated with the new commands, flags, and pipeline.

## 0.8.0 (2026-04-12)

### Added
- **Paper search** — New top-level `lens search` command for finding papers
  via hybrid search (FTS5 keyword + sqlite-vec semantic) with Reciprocal Rank
  Fusion scoring.
- **Metadata filters** — `--author`, `--venue`, `--after`, `--before` flags
  for filtering by author name, venue, date range. Works standalone or combined
  with text search.
- **`papers_fts` FTS5 table** — Full-text index on paper titles and abstracts,
  created automatically and kept in sync via `add_papers()`.
- **Embedding fallback** — Search gracefully degrades to keyword-only when
  no embedding provider is configured.
- **Existing data migration** — `init_tables()` rebuilds the FTS index on
  startup, ensuring papers from older databases are searchable.

## 0.7.0 (2026-04-08)

### Added
- **DeepXiv integration** — New `lens acquire deepxiv` command for paper search
  and retrieval via [deepxiv-sdk](https://github.com/DeepXiv/deepxiv_sdk).
  Supports hybrid search (BM25 + vector), single paper fetch with rich metadata,
  date filtering, and category filtering.
- **`keywords` field** on papers — JSON list of keywords populated by DeepXiv's
  `brief()` API. Feeds into vocabulary and aids search.
- **`github_url` field** on papers — Link to associated code repository, populated
  by DeepXiv. Useful for reproducibility and code availability tracking.
- **`deepxiv` optional dependency** — Install with `uv sync --extra deepxiv`.
  When not installed, all existing functionality works unchanged.

### Changed
- **Schema migrations** — Two new column migrations (`keywords`, `github_url`)
  applied automatically on database upgrade via `init_tables()`.

## 0.6.0 (2026-04-05)

### Added
- **Event log** — unified `event_log` table records all mutations (ingest,
  extract, build, lint, fix) with timestamps and session IDs. Query with
  `lens log` CLI command (filters: `--kind`, `--since`, `--limit`, `--session`).
- **Knowledge base linter** — `lens lint` health-checks the knowledge base
  across 6 categories: orphan vocabulary, contradictions, weak evidence,
  missing embeddings, stale extractions, and near-duplicates.
- **Auto-fix mode** — `lens lint --fix` applies safe repairs: deletes orphan
  entries, generates missing embeddings, requeues stale extractions, and
  merges near-duplicate vocabulary entries (rewrites extraction references).
- **`EventLog` model** and `LintReport` model in `models.py`.
- **`log_event()` helper** in `knowledge/events.py` — called explicitly at
  each instrumentation site (no decorator magic).
- **Session ID threading** — each CLI invocation generates a session ID that
  groups all events from that run.

### Changed
- **Extraction pipeline** now emits `extract.extraction.completed` and
  `extract.extraction.failed` events.
- **Vocabulary pipeline** now emits `extract.vocabulary.created`,
  `extract.vocabulary.updated`, and `build.taxonomy.built` events.
- **Matrix builder** now emits `build.matrix.built` events.
- **Acquire commands** (seed, arxiv, file, openalex) now emit `ingest.*` events.
- **`extract_papers()`**, **`build_vocabulary()`**, **`build_matrix()`**, and
  **`record_version()`** accept optional `session_id` parameter.

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
- **`new_concepts` field** (JSON dict) on all extraction tables — maps each
  `NEW:` concept name to a one-line description. Replaces the old shared
  `new_concept_description` string field.
- **Hybrid search** — FTS5 keyword + sqlite-vec vector search combined via
  Reciprocal Rank Fusion (RRF) for concept resolution in `explain`.
- **LLM candidate selection** — `explain` presents top 3 hybrid search
  candidates to the LLM, which picks the best match for the user's query.
- **Kind-specific explain** — `explain` produces context-appropriate
  explanations for all 4 vocabulary kinds (tradeoff graph for parameters/
  principles, variant listing for arch slots, pattern listing for categories).
- **Schema migrations** — `_COLUMN_MIGRATIONS` in `store.py` adds missing
  columns to existing tables on upgrade via `ALTER TABLE`.
- **`json_repair`** dependency for robust parsing of malformed LLM JSON.

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

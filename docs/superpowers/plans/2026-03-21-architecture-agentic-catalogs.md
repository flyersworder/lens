# Architecture & Agentic Catalogs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process architecture and agentic extraction data into browsable, queryable catalogs with property-based comparison and emergent categories.

**Architecture:** Extend `build_taxonomy` with two new stages (architecture slots+variants, agentic patterns). LLM normalizes raw slot names and assigns pattern categories. Auto-increment IDs replace offset-based allocation across all entity types.

**Tech Stack:** Python 3.12+, LanceDB, Pydantic, Polars, Typer, litellm, HDBSCAN, sentence-transformers

**Spec:** `docs/superpowers/specs/2026-03-21-architecture-agentic-catalogs-design.md`

---

### Task 1: Update models and config (foundation)

**Files:**
- Modify: `src/lens/store/models.py:175-183` (TaxonomyVersion)
- Modify: `src/lens/config.py:30-35` (DEFAULT_CONFIG taxonomy section)
- Modify: `src/lens/taxonomy/versioning.py:24-43` (record_version)
- Test: `tests/test_models.py`, `tests/test_config.py`

- [ ] **Step 1: Write tests for TaxonomyVersion new fields**

In `tests/test_models.py`, add:

```python
def test_taxonomy_version_with_catalog_counts():
    from lens.store.models import TaxonomyVersion
    tv = TaxonomyVersion(
        version_id=1, created_at=datetime.now(), paper_count=10,
        param_count=5, principle_count=10,
        slot_count=3, variant_count=12, pattern_count=8,
    )
    assert tv.slot_count == 3
    assert tv.variant_count == 12
    assert tv.pattern_count == 8


def test_taxonomy_version_defaults_backward_compat():
    from lens.store.models import TaxonomyVersion
    tv = TaxonomyVersion(
        version_id=1, created_at=datetime.now(), paper_count=10,
        param_count=5, principle_count=10,
    )
    assert tv.slot_count == 0
    assert tv.variant_count == 0
    assert tv.pattern_count == 0
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/test_models.py::test_taxonomy_version_with_catalog_counts -v`
Expected: FAIL — `TaxonomyVersion` doesn't have `slot_count` yet.

- [ ] **Step 3: Add new fields to TaxonomyVersion**

In `src/lens/store/models.py`, update `TaxonomyVersion`:

```python
class TaxonomyVersion(LanceModel):
    version_id: int
    created_at: datetime
    paper_count: int
    param_count: int
    principle_count: int
    slot_count: int = 0
    variant_count: int = 0
    pattern_count: int = 0
```

- [ ] **Step 4: Update record_version signature**

In `src/lens/taxonomy/versioning.py`, update:

```python
def record_version(
    store: LensStore,
    version_id: int,
    paper_count: int,
    param_count: int,
    principle_count: int,
    slot_count: int = 0,
    variant_count: int = 0,
    pattern_count: int = 0,
) -> None:
    """Record a taxonomy version in the store."""
    store.add_rows(
        "taxonomy_versions",
        [
            {
                "version_id": version_id,
                "created_at": datetime.now(UTC),
                "paper_count": paper_count,
                "param_count": param_count,
                "principle_count": principle_count,
                "slot_count": slot_count,
                "variant_count": variant_count,
                "pattern_count": pattern_count,
            }
        ],
    )
```

- [ ] **Step 5: Add config keys**

In `src/lens/config.py`, update the taxonomy section of `DEFAULT_CONFIG`:

```python
    "taxonomy": {
        "target_parameters": 25,
        "target_principles": 35,
        "min_cluster_size": 3,
        "embedding_model": "specter2",
        "target_arch_variants": 20,
        "target_agentic_patterns": 15,
    },
```

- [ ] **Step 6: Run all tests to verify no regressions**

Run: `uv run pytest -x -q`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/lens/store/models.py src/lens/taxonomy/versioning.py src/lens/config.py tests/test_models.py
git commit -m "feat: add TaxonomyVersion catalog fields, config keys, record_version update"
```

---

### Task 2: Refactor ID generation to auto-increment

**Files:**
- Modify: `src/lens/taxonomy/__init__.py:72-121` (_build_taxonomy_entries, build_taxonomy)
- Test: `tests/test_taxonomy.py`

- [ ] **Step 1: Write test for _next_id helper**

In `tests/test_taxonomy.py`, add:

```python
def test_next_id_empty_table(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy import _next_id

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()
    assert _next_id(store, "parameters") == 1


def test_next_id_with_existing_data(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy import _next_id

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()
    store.add_rows("parameters", [{
        "id": 42, "name": "Test", "description": "d",
        "raw_strings": ["t"], "paper_ids": ["p1"],
        "taxonomy_version": 1, "embedding": [0.0] * 768,
    }])
    assert _next_id(store, "parameters") == 43
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/test_taxonomy.py::test_next_id_empty_table -v`
Expected: FAIL — `_next_id` doesn't exist yet.

- [ ] **Step 3: Add _next_id and refactor _build_taxonomy_entries**

In `src/lens/taxonomy/__init__.py`, add the helper and refactor:

```python
def _next_id(store: LensStore, table_name: str) -> int:
    """Return max(id) + 1 for the given table, or 1 if empty."""
    df = store.get_table(table_name).to_polars()
    if len(df) == 0:
        return 1
    return int(df["id"].max()) + 1
```

Update `_build_taxonomy_entries` to accept `start_id: int` instead of `id_offset: int`:

```python
def _build_taxonomy_entries(
    cluster_label_info: dict[int, dict[str, str]],
    clusters: dict[int, list[str]],
    strings: list[str],
    embeddings: np.ndarray,
    version_id: int,
    paper_ids_by_string: dict[str, list[str]],
    start_id: int = 1,
) -> list[dict[str, Any]]:
```

Replace the ID line: `entry_id = version_id * 100000 + id_offset + cluster_id` with:
```python
        entry_id = start_id + len(entries)
```

Update both call sites in `build_taxonomy` to use `start_id=_next_id(store, "parameters")` and `start_id=_next_id(store, "principles")`.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -x -q`
Expected: All tests pass (existing test_build_taxonomy still works).

- [ ] **Step 5: Commit**

```bash
git add src/lens/taxonomy/__init__.py tests/test_taxonomy.py
git commit -m "refactor: replace offset-based IDs with auto-increment _next_id"
```

---

### Task 3: Add LLM prompts for slot normalization and agentic labeling

**Files:**
- Modify: `src/lens/taxonomy/labeler.py` (add new prompt functions)
- Test: `tests/test_taxonomy.py`

- [ ] **Step 1: Write tests for normalize_slots**

```python
@pytest.mark.asyncio
async def test_normalize_slots():
    from lens.taxonomy.labeler import normalize_slots

    raw_strings = ["attention mechanism", "self-attention", "positional encoding", "pos embedding"]
    mock_client = AsyncMock()
    mock_client.complete.return_value = json.dumps({
        "attention mechanism": "Attention",
        "self-attention": "Attention",
        "positional encoding": "Positional Encoding",
        "pos embedding": "Positional Encoding",
    })

    mapping = await normalize_slots(raw_strings, mock_client)
    assert mapping["attention mechanism"] == "Attention"
    assert mapping["self-attention"] == "Attention"
    assert mapping["positional encoding"] == "Positional Encoding"


@pytest.mark.asyncio
async def test_normalize_slots_malformed_fallback():
    from lens.taxonomy.labeler import normalize_slots

    raw_strings = ["attention mechanism"]
    mock_client = AsyncMock()
    mock_client.complete.return_value = "not json"

    mapping = await normalize_slots(raw_strings, mock_client)
    # Fallback: each string maps to its title-cased self
    assert mapping["attention mechanism"] == "Attention Mechanism"


@pytest.mark.asyncio
async def test_label_clusters_with_category():
    from lens.taxonomy.labeler import label_clusters_with_category

    clusters = {0: ["ReAct", "react pattern", "reasoning and acting"]}
    structures = {0: ["LLM agent with tool use and reasoning loop"]}
    mock_client = AsyncMock()
    mock_client.complete.return_value = json.dumps({
        "name": "ReAct",
        "description": "Reasoning and acting pattern for tool-using agents",
        "category": "Reasoning",
    })

    labels = await label_clusters_with_category(clusters, structures, mock_client)
    assert labels[0]["name"] == "ReAct"
    assert labels[0]["category"] == "Reasoning"
    assert "description" in labels[0]
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/test_taxonomy.py::test_normalize_slots -v`
Expected: FAIL — function doesn't exist.

- [ ] **Step 3: Implement normalize_slots**

In `src/lens/taxonomy/labeler.py`, add:

```python
def _build_slot_normalization_prompt(raw_strings: list[str]) -> str:
    strings_text = "\n".join(f"- {s}" for s in raw_strings)
    return (
        "These are raw architecture component slot names extracted from LLM "
        "research papers. Normalize them into canonical slot names by grouping "
        "synonyms together.\n\n"
        f"Raw names:\n{strings_text}\n\n"
        "Return a JSON object mapping each raw name to its canonical slot name:\n"
        '{"raw name": "Canonical Slot Name", ...}'
    )


async def normalize_slots(
    raw_strings: list[str],
    llm_client: LLMClient,
) -> dict[str, str]:
    """Normalize raw component_slot strings into canonical slot names via LLM."""
    if not raw_strings:
        return {}
    prompt = _build_slot_normalization_prompt(raw_strings)
    try:
        response = await llm_client.complete([{"role": "user", "content": prompt}])
        text = strip_code_fences(response.strip())
        mapping = json.loads(text)
        if isinstance(mapping, dict):
            return {k: str(v) for k, v in mapping.items()}
    except Exception:
        logger.warning("Slot normalization failed, using title-case fallback")
    return {s: s.title() for s in raw_strings}
```

- [ ] **Step 4: Implement summarize_variant_properties**

In `src/lens/taxonomy/labeler.py`, add a function to LLM-summarize raw key_properties strings into a concise description (per the spec: "Concatenate unique key_properties strings... then LLM-summarize into a concise property description"):

```python
async def summarize_variant_properties(
    raw_properties: list[str],
    variant_name: str,
    llm_client: LLMClient,
) -> str:
    """Summarize raw key_properties strings into a concise description via LLM."""
    if not raw_properties:
        return ""
    if len(raw_properties) == 1:
        return raw_properties[0]
    props_text = "\n".join(f"- {p}" for p in set(raw_properties))
    prompt = (
        f"These are properties of the architecture variant '{variant_name}' "
        f"extracted from multiple papers:\n\n{props_text}\n\n"
        "Summarize into a single concise sentence describing the key properties. "
        "Respond with just the summary text, no JSON."
    )
    try:
        response = await llm_client.complete([{"role": "user", "content": prompt}])
        return response.strip()
    except Exception:
        return "; ".join(set(raw_properties))
```

Note: This is called from the architecture stage in Task 4 for clustered variants. For single-variant or small clusters where LLM cost isn't worth it, the raw concatenation fallback (`"; ".join(set(...))`) is used directly in the build_taxonomy code. The LLM summarization is an optional enhancement that can be wired in for clustered variants with many properties.

- [ ] **Step 5: Implement label_clusters_with_category**

In `src/lens/taxonomy/labeler.py`, add:

```python
def _build_category_label_prompt(
    cluster_strings: list[str], structures: list[str]
) -> str:
    sample = cluster_strings[:20]
    strings_text = "\n".join(f"- {s}" for s in sample)
    struct_text = "\n".join(f"- {s}" for s in structures[:10])
    return (
        "These are related agentic pattern names extracted from LLM research papers, "
        "along with their structural descriptions.\n\n"
        f"Pattern names:\n{strings_text}\n\n"
        f"Structures:\n{struct_text}\n\n"
        "Provide a concise name, one-sentence description, and a category for the "
        "pattern type. Use short category names (e.g., 'Reasoning', 'Reflection', "
        "'Multi-Agent Collaboration', 'Tool Integration', 'Memory & Retrieval', "
        "'Planning'). Respond with JSON only:\n"
        '{"name": "Pattern Name", "description": "One sentence.", "category": "Category"}'
    )


async def label_clusters_with_category(
    clusters: dict[int, list[str]],
    structures: dict[int, list[str]],
    llm_client: LLMClient,
) -> dict[int, dict[str, str]]:
    """Label each cluster with name, description, and category via LLM."""
    labels: dict[int, dict[str, str]] = {}

    for cluster_id, strings in clusters.items():
        structs = structures.get(cluster_id, [])
        prompt = _build_category_label_prompt(strings, structs)
        try:
            response = await llm_client.complete([{"role": "user", "content": prompt}])
            text = strip_code_fences(response.strip())
            data = json.loads(text)
            labels[cluster_id] = {
                "name": data.get("name", strings[0]),
                "description": data.get("description", ""),
                "category": data.get("category", "Uncategorized"),
            }
        except Exception:
            most_common = Counter(strings).most_common(1)[0][0]
            labels[cluster_id] = {
                "name": most_common.title(),
                "description": f"Cluster of {len(strings)} related patterns",
                "category": "Uncategorized",
            }
            logger.warning(
                "LLM labeling+category failed for cluster %d, using fallback",
                cluster_id,
            )

    return labels
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_taxonomy.py -v -k "normalize_slots or label_clusters_with_category"`
Expected: All 3 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/lens/taxonomy/labeler.py tests/test_taxonomy.py
git commit -m "feat: add normalize_slots and label_clusters_with_category prompts"
```

---

### Task 4: Add architecture + agentic stages to build_taxonomy

**Files:**
- Modify: `src/lens/taxonomy/__init__.py` (add stages 3 and 4)
- Test: `tests/test_taxonomy.py`

- [ ] **Step 1: Write integration test for architecture stage**

```python
@pytest.mark.asyncio
async def test_build_taxonomy_with_architecture(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy import build_taxonomy

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    # Add minimal tradeoff extractions (required for parameters/principles)
    store.add_rows("tradeoff_extractions", [
        {"paper_id": "p1", "improves": "speed", "worsens": "size",
         "technique": "quantization", "context": "t", "confidence": 0.8,
         "evidence_quote": "q"},
    ] * 5)

    # Add architecture extractions
    store.add_rows("architecture_extractions", [
        {"paper_id": "p1", "component_slot": "attention mechanism",
         "variant_name": "multi-head attention", "replaces": None,
         "key_properties": "parallel heads", "confidence": 0.9},
        {"paper_id": "p2", "component_slot": "attention mechanism",
         "variant_name": "grouped-query attention", "replaces": "multi-head attention",
         "key_properties": "shared KV cache", "confidence": 0.85},
        {"paper_id": "p3", "component_slot": "positional encoding",
         "variant_name": "RoPE", "replaces": None,
         "key_properties": "relative position", "confidence": 0.9},
    ])

    mock_client = AsyncMock()
    mock_client.complete.return_value = '{"name": "Test", "description": "test"}'

    version = await build_taxonomy(store, mock_client, min_cluster_size=2)

    slots = store.get_table("architecture_slots").to_polars()
    slots = slots.filter(slots["taxonomy_version"] == version)
    assert len(slots) >= 1

    variants = store.get_table("architecture_variants").to_polars()
    variants = variants.filter(variants["taxonomy_version"] == version)
    assert len(variants) >= 1
```

- [ ] **Step 2: Write integration test for agentic stage**

```python
@pytest.mark.asyncio
async def test_build_taxonomy_with_agentic(tmp_path):
    from lens.store.store import LensStore
    from lens.taxonomy import build_taxonomy

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    store.add_rows("tradeoff_extractions", [
        {"paper_id": "p1", "improves": "speed", "worsens": "size",
         "technique": "quantization", "context": "t", "confidence": 0.8,
         "evidence_quote": "q"},
    ] * 5)

    store.add_rows("agentic_extractions", [
        {"paper_id": "p1", "pattern_name": "ReAct",
         "structure": "reasoning and acting loop", "use_case": "tool use",
         "components": ["LLM", "tools", "memory"], "confidence": 0.9},
        {"paper_id": "p2", "pattern_name": "Reflexion",
         "structure": "self-critique loop", "use_case": "code generation",
         "components": ["actor", "evaluator", "memory"], "confidence": 0.85},
    ])

    mock_client = AsyncMock()
    mock_client.complete.return_value = (
        '{"name": "Test Pattern", "description": "test", "category": "Reasoning"}'
    )

    version = await build_taxonomy(store, mock_client, min_cluster_size=2)

    patterns = store.get_table("agentic_patterns").to_polars()
    patterns = patterns.filter(patterns["taxonomy_version"] == version)
    assert len(patterns) >= 1
    assert "category" in patterns.columns
```

- [ ] **Step 3: Run tests — expect FAIL**

Run: `uv run pytest tests/test_taxonomy.py::test_build_taxonomy_with_architecture -v`
Expected: FAIL — architecture stage not implemented yet.

- [ ] **Step 4: Implement architecture stage in build_taxonomy**

In `src/lens/taxonomy/__init__.py`, add after the principles stage:

```python
    # --- Architecture slots + variants ---
    arch_strings = _collect_strings_from_table(
        store, "architecture_extractions", ["component_slot"]
    )
    slot_entries: list[dict[str, Any]] = []
    variant_entries: list[dict[str, Any]] = []

    if arch_strings:
        from lens.taxonomy.labeler import normalize_slots

        slot_mapping = await normalize_slots(arch_strings, llm_client)

        # Create slot entries
        canonical_slots = sorted(set(slot_mapping.values()))
        next_slot_id = _next_id(store, "architecture_slots")
        slot_name_to_id: dict[str, int] = {}
        for i, slot_name in enumerate(canonical_slots):
            sid = next_slot_id + i
            slot_name_to_id[slot_name] = sid
            slot_entries.append({
                "id": sid,
                "name": slot_name,
                "description": f"Architecture component: {slot_name}",
                "taxonomy_version": version_id,
            })
        if slot_entries:
            store.add_rows("architecture_slots", slot_entries)

        # Cluster variants per slot
        # Use a running counter to avoid duplicate IDs across slots
        # (calling _next_id per slot would return the same value since
        # nothing is written to the table until after all slots are processed)
        arch_df = store.get_table("architecture_extractions").to_polars()
        arch_df = arch_df.filter(pl.col("confidence") >= 0.5)
        next_var_id = _next_id(store, "architecture_variants")

        for slot_name, slot_id in slot_name_to_id.items():
            # Collect variant_name strings for this slot
            raw_slots_for_this = [k for k, v in slot_mapping.items() if v == slot_name]
            slot_variants_df = arch_df.filter(
                pl.col("component_slot").is_in(raw_slots_for_this)
            )
            if len(slot_variants_df) == 0:
                continue

            var_strings = list(set(
                s for s in slot_variants_df["variant_name"].to_list() if s
            ))
            var_paper_ids: dict[str, list[str]] = {}
            var_properties: dict[str, list[str]] = {}
            for row in slot_variants_df.to_dicts():
                vn = row.get("variant_name", "")
                if vn:
                    var_paper_ids.setdefault(vn, []).append(row.get("paper_id", ""))
                    kp = row.get("key_properties", "")
                    if kp:
                        var_properties.setdefault(vn, []).append(kp)

            if len(var_strings) < 2:
                # Single variant — no clustering needed
                for vn in var_strings:
                    props = "; ".join(set(var_properties.get(vn, [])))
                    variant_entries.append({
                        "id": next_var_id,
                        "slot_id": slot_id,
                        "name": vn.title(),
                        "replaces": [],
                        "properties": props,
                        "paper_ids": list(set(var_paper_ids.get(vn, []))),
                        "taxonomy_version": version_id,
                        "embedding": [0.0] * 768,
                    })
                    next_var_id += 1
            else:
                var_emb = embed_strings(var_strings)
                var_labels = cluster_embeddings(
                    var_emb,
                    min_cluster_size=min(min_cluster_size, max(2, len(var_strings) // 3)),
                    target_clusters=target_arch_variants,
                )
                var_clusters = _group_by_cluster(var_strings, var_labels)
                var_names = await label_clusters(var_clusters, llm_client)
                raw_variant_entries = _build_taxonomy_entries(
                    var_names, var_clusters, var_strings, var_emb,
                    version_id, var_paper_ids,
                    start_id=next_var_id,
                )
                # _build_taxonomy_entries returns entries with "raw_strings" which
                # ArchitectureVariant doesn't have — convert to variant-specific fields
                for entry in raw_variant_entries:
                    entry["slot_id"] = slot_id
                    # Aggregate properties from cluster members
                    member_props: list[str] = []
                    for s in entry.get("raw_strings", []):
                        member_props.extend(var_properties.get(s, []))
                    entry["properties"] = "; ".join(set(member_props)) if member_props else ""
                    entry["replaces"] = []
                    del entry["raw_strings"]
                next_var_id += len(raw_variant_entries)
                variant_entries.extend(raw_variant_entries)

        if variant_entries:
            store.add_rows("architecture_variants", variant_entries)
```

Also update the `build_taxonomy` signature to accept new parameters:

```python
async def build_taxonomy(
    store: LensStore,
    llm_client: LLMClient,
    min_cluster_size: int = 3,
    target_parameters: int = 25,
    target_principles: int = 35,
    target_arch_variants: int = 20,
    target_agentic_patterns: int = 15,
) -> int:
```

- [ ] **Step 5: Implement agentic stage in build_taxonomy**

Add after the architecture stage:

```python
    # --- Agentic patterns ---
    agentic_strings = _collect_strings_from_table(
        store, "agentic_extractions", ["pattern_name"]
    )
    agentic_paper_ids = _build_paper_id_map(store, "agentic_extractions", ["pattern_name"])
    pattern_entries: list[dict[str, Any]] = []

    if agentic_strings:
        from lens.taxonomy.labeler import label_clusters_with_category

        ag_emb = embed_strings(agentic_strings)
        ag_labels = cluster_embeddings(
            ag_emb,
            min_cluster_size=min_cluster_size,
            target_clusters=target_agentic_patterns,
        )
        ag_clusters = _group_by_cluster(agentic_strings, ag_labels)

        # Build structure map for category assignment
        ag_df = store.get_table("agentic_extractions").to_polars()
        if "confidence" in ag_df.columns:
            ag_df = ag_df.filter(pl.col("confidence") >= 0.5)
        structure_by_name: dict[str, list[str]] = {}
        components_by_name: dict[str, list[str]] = {}
        use_cases_by_name: dict[str, list[str]] = {}
        for row in ag_df.to_dicts():
            pn = row.get("pattern_name", "")
            if pn:
                s = row.get("structure", "")
                if s:
                    structure_by_name.setdefault(pn, []).append(s)
                uc = row.get("use_case", "")
                if uc:
                    use_cases_by_name.setdefault(pn, []).append(uc)
                comps = row.get("components", [])
                if comps:
                    components_by_name.setdefault(pn, []).extend(comps)

        # Aggregate structures per cluster
        cluster_structures: dict[int, list[str]] = {}
        for cid, members in ag_clusters.items():
            structs: list[str] = []
            for m in members:
                structs.extend(structure_by_name.get(m, []))
            cluster_structures[cid] = list(set(structs))

        ag_names = await label_clusters_with_category(
            ag_clusters, cluster_structures, llm_client
        )
        raw_pattern_entries = _build_taxonomy_entries(
            ag_names, ag_clusters, agentic_strings, ag_emb,
            version_id, agentic_paper_ids,
            start_id=_next_id(store, "agentic_patterns"),
        )
        for entry in raw_pattern_entries:
            label_info = ag_names.get(
                next(
                    cid for cid, members in ag_clusters.items()
                    if any(s in entry.get("raw_strings", []) for s in members)
                ),
                {},
            )
            entry["category"] = label_info.get("category", "Uncategorized")
            # Aggregate components and use_cases from cluster members
            all_components: list[str] = []
            all_use_cases: list[str] = []
            for s in entry.get("raw_strings", []):
                all_components.extend(components_by_name.get(s, []))
                all_use_cases.extend(use_cases_by_name.get(s, []))
            entry["components"] = list(set(all_components))
            entry["use_cases"] = list(set(all_use_cases))
            del entry["raw_strings"]
        pattern_entries = raw_pattern_entries
        if pattern_entries:
            store.add_rows("agentic_patterns", pattern_entries)
```

- [ ] **Step 6: Update record_version call at the end of build_taxonomy**

```python
    record_version(
        store,
        version_id,
        paper_count=paper_count,
        param_count=len(param_entries),
        principle_count=len(principle_entries),
        slot_count=len(slot_entries),
        variant_count=len(variant_entries),
        pattern_count=len(pattern_entries),
    )

    logger.info(
        "Taxonomy v%d: %d params, %d principles, %d slots, %d variants, %d patterns",
        version_id,
        len(param_entries),
        len(principle_entries),
        len(slot_entries),
        len(variant_entries),
        len(pattern_entries),
    )
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_taxonomy.py -v`
Expected: All tests pass including new architecture/agentic tests.

- [ ] **Step 8: Commit**

```bash
git add src/lens/taxonomy/__init__.py tests/test_taxonomy.py
git commit -m "feat: add architecture and agentic stages to build_taxonomy"
```

---

### Task 5: Add explorer functions for architecture and agentic browsing

**Files:**
- Modify: `src/lens/serve/explorer.py`
- Test: `tests/test_explorer.py`

- [ ] **Step 1: Write tests**

In `tests/test_explorer.py`, add a fixture and tests:

```python
@pytest.fixture
def arch_store(tmp_path):
    from lens.store.store import LensStore
    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()
    store.add_rows("architecture_slots", [
        {"id": 1, "name": "Attention", "description": "Attention mechanism", "taxonomy_version": 1},
        {"id": 2, "name": "Positional Encoding", "description": "Position info", "taxonomy_version": 1},
    ])
    store.add_rows("architecture_variants", [
        {"id": 10, "slot_id": 1, "name": "Multi-Head Attention", "replaces": [],
         "properties": "parallel heads", "paper_ids": ["p1"],
         "taxonomy_version": 1, "embedding": [0.0] * 768},
        {"id": 11, "slot_id": 1, "name": "Grouped-Query Attention", "replaces": [10],
         "properties": "shared KV cache", "paper_ids": ["p2"],
         "taxonomy_version": 1, "embedding": [0.0] * 768},
        {"id": 12, "slot_id": 2, "name": "RoPE", "replaces": [],
         "properties": "relative position", "paper_ids": ["p3"],
         "taxonomy_version": 1, "embedding": [0.0] * 768},
    ])
    store.add_rows("agentic_patterns", [
        {"id": 20, "name": "ReAct", "category": "Reasoning",
         "description": "Reasoning and acting", "components": ["LLM", "tools"],
         "use_cases": ["tool use"], "paper_ids": ["p1"],
         "taxonomy_version": 1, "embedding": [0.0] * 768},
        {"id": 21, "name": "Reflexion", "category": "Reflection",
         "description": "Self-critique loop", "components": ["actor", "evaluator"],
         "use_cases": ["code generation"], "paper_ids": ["p2"],
         "taxonomy_version": 1, "embedding": [0.0] * 768},
    ])
    store.add_rows("taxonomy_versions", [
        {"version_id": 1, "created_at": "2026-03-21T00:00:00", "paper_count": 3,
         "param_count": 0, "principle_count": 0, "slot_count": 2,
         "variant_count": 3, "pattern_count": 2},
    ])
    return store


def test_list_architecture_slots(arch_store):
    from lens.serve.explorer import list_architecture_slots
    slots = list_architecture_slots(arch_store, taxonomy_version=1)
    assert len(slots) == 2
    attn = next(s for s in slots if s["name"] == "Attention")
    assert attn["variant_count"] == 2


def test_list_architecture_variants(arch_store):
    from lens.serve.explorer import list_architecture_variants
    variants = list_architecture_variants(arch_store, slot_name="Attention", taxonomy_version=1)
    assert len(variants) == 2
    names = {v["name"] for v in variants}
    assert "Multi-Head Attention" in names
    assert "Grouped-Query Attention" in names


def test_list_agentic_patterns(arch_store):
    from lens.serve.explorer import list_agentic_patterns
    patterns = list_agentic_patterns(arch_store, taxonomy_version=1)
    assert len(patterns) == 2


def test_list_agentic_patterns_by_category(arch_store):
    from lens.serve.explorer import list_agentic_patterns
    patterns = list_agentic_patterns(arch_store, taxonomy_version=1, category="Reasoning")
    assert len(patterns) == 1
    assert patterns[0]["name"] == "ReAct"
```

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement explorer functions**

In `src/lens/serve/explorer.py`, add:

```python
def list_architecture_slots(store, taxonomy_version):
    """List all architecture slots with variant counts."""

def list_architecture_variants(store, slot_name, taxonomy_version):
    """List all variants for a given slot name."""

def list_agentic_patterns(store, taxonomy_version, category=None):
    """List agentic patterns, optionally filtered by category."""

def get_architecture_timeline(store, slot_name, taxonomy_version):
    """List variants in a slot ordered by earliest paper date."""
```

- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add architecture and agentic explorer functions"
```

---

### Task 6: Add architecture/agentic analysis to analyzer

**Files:**
- Modify: `src/lens/serve/analyzer.py`
- Modify: `src/lens/taxonomy/labeler.py` (add architecture query decomposition prompt)
- Test: `tests/test_analyzer.py`

- [ ] **Step 1: Write test for architecture analysis**

```python
@pytest.mark.asyncio
async def test_analyze_architecture(analyzer_store):
    from lens.serve.analyzer import analyze_architecture
    # analyzer_store needs architecture_variants added to the existing fixture
    mock_client = AsyncMock()
    mock_client.complete.return_value = '{"slot": "Attention", "constraints": "sub-quadratic"}'
    result = await analyze_architecture("efficient attention for long context", analyzer_store, mock_client, taxonomy_version=1)
    assert "variants" in result
```

- [ ] **Step 2: Implement analyze_architecture and analyze_agentic**

In `src/lens/serve/analyzer.py`, add:
- `analyze_architecture(query, store, llm_client, taxonomy_version)` — LLM decomposes query into slot + constraints, then vector search against `ArchitectureVariant` embeddings.
- `analyze_agentic(query, store, llm_client, taxonomy_version)` — vector search against `AgenticPattern` embeddings.

Add the architecture query decomposition prompt to `labeler.py`:

```python
async def decompose_architecture_query(query: str, slot_names: list[str], llm_client: LLMClient) -> dict[str, str]:
    """Decompose a user query into slot + constraints for architecture search."""
```

- [ ] **Step 3: Run tests and commit**

```bash
git commit -m "feat: add architecture and agentic analysis paths"
```

---

### Task 7: Wire up CLI commands

**Files:**
- Modify: `src/lens/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Replace the `explore architecture` stub**

Replace the stub with a functional command that calls `list_architecture_slots` (no args) or `list_architecture_variants` (with slot name). Load taxonomy version, check it exists.

- [ ] **Step 2: Replace the `explore agents` stub**

Replace the stub with a functional command that calls `list_agentic_patterns`.

- [ ] **Step 3: Replace the `explore evolution` stub**

Replace the stub with a functional command that calls `get_architecture_timeline`.

- [ ] **Step 4: Update `analyze` command to route --type**

Replace the yellow warning at `cli.py:84-88` with actual routing:
- `--type architecture` → call `analyze_architecture`
- `--type agentic` → call `analyze_agentic`
- Default (no type) → existing tradeoff analysis

- [ ] **Step 5: Update `build taxonomy` and `build all` to pass new config**

Update both CLI call sites (`taxonomy()` and `build_all()`) to pass `target_arch_variants` and `target_agentic_patterns` from `tax_config`.

- [ ] **Step 6: Add CLI tests**

In `tests/test_cli.py`, add:

```python
def test_explore_architecture_help():
    result = runner.invoke(app, ["explore", "architecture", "--help"])
    assert result.exit_code == 0
    assert "architecture" in result.output.lower()

def test_explore_agents_help():
    result = runner.invoke(app, ["explore", "agents", "--help"])
    assert result.exit_code == 0

def test_explore_evolution_help():
    result = runner.invoke(app, ["explore", "evolution", "--help"])
    assert result.exit_code == 0
```

- [ ] **Step 7: Run all tests**

Run: `uv run pytest -x -q`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git commit -m "feat: wire up architecture and agentic CLI commands"
```

---

### Task 8: Final integration test and cleanup

**Files:**
- All modified files
- Test: full suite

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -x -q`
Expected: All tests pass.

- [ ] **Step 2: Run ruff lint check**

Run: `uv run ruff check src/ tests/`
Expected: No errors.

- [ ] **Step 3: Verify CLI commands work**

Run: `uv run lens explore architecture --help`
Run: `uv run lens explore agents --help`
Run: `uv run lens explore evolution --help`
Expected: Help text displays correctly.

- [ ] **Step 4: Update design spec status**

Change `docs/superpowers/specs/2026-03-21-architecture-agentic-catalogs-design.md` status from "Draft" to "Implemented".

- [ ] **Step 5: Final commit**

```bash
git commit -m "docs: mark architecture & agentic catalogs spec as implemented"
```

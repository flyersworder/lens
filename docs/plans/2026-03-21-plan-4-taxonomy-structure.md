# Plan 4: Taxonomy + Structure — Clustering, Labeling, Matrix Construction

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cluster raw extraction strings into abstract parameters and principles (taxonomy), build the contradiction matrix mapping tradeoffs to solutions, and construct architecture and agentic catalogs.

**Architecture:** Raw extraction strings are embedded with sentence-transformers, clustered with HDBSCAN, and labeled with an LLM. The taxonomy maps each raw string to a cluster. The matrix is built by looking up taxonomy IDs for each tradeoff extraction and counting co-occurrences. All operations use Polars for in-memory analytics.

**Tech Stack:** sentence-transformers (embedding), hdbscan, numpy, existing LensStore + Polars + LLMClient

**Spec:** `docs/specs/design.md` (Stage 3: Taxonomize, lines 331-346; Stage 4: Structure, lines 348-358; Data Model Layer 2-3, lines 122-186)

**Design note:** The spec references BERTopic + HDBSCAN + SPECTER2. For simplicity and to avoid heavyweight dependencies:
- We use `sentence-transformers` with the `allenai-specter2` model for embedding (same model, direct access)
- We use `hdbscan` directly for clustering (BERTopic's core is HDBSCAN + a representation model — we skip the BERTopic wrapper and use LLM labeling directly)
- LLM labeling replaces BERTopic's built-in representation — we send cluster centroids + member strings to the LLM and ask for a name + description

---

## File Structure

```
src/lens/
├── taxonomy/
│   ├── __init__.py           # Public API: build_taxonomy
│   ├── embedder.py           # Embed raw strings with sentence-transformers
│   ├── clusterer.py          # HDBSCAN clustering
│   ├── labeler.py            # LLM-based cluster labeling
│   └── versioning.py         # Taxonomy version management
├── knowledge/
│   ├── __init__.py           # Public API: build_matrix, build_all
│   ├── matrix.py             # Contradiction matrix construction
│   ├── architecture.py       # Architecture catalog construction
│   └── agentic.py            # Agentic pattern catalog construction
tests/
├── test_taxonomy.py          # Taxonomy pipeline tests
├── test_matrix.py            # Matrix construction tests
├── test_architecture.py      # Architecture catalog tests
├── test_agentic.py           # Agentic catalog tests
```

---

### Task 1: String Embedder

**Files:**
- Create: `src/lens/taxonomy/__init__.py`
- Create: `src/lens/taxonomy/embedder.py`
- Create: `tests/test_taxonomy.py`

- [ ] **Step 1: Add sentence-transformers dependency**

```bash
uv add sentence-transformers
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_taxonomy.py
"""Tests for the taxonomy pipeline."""

import numpy as np
import pytest


def test_embed_strings_returns_array():
    from lens.taxonomy.embedder import embed_strings

    embeddings = embed_strings(["inference latency", "model accuracy"])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0  # embedding dimension


def test_embed_strings_empty():
    from lens.taxonomy.embedder import embed_strings

    embeddings = embed_strings([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 0


def test_embed_strings_deterministic():
    from lens.taxonomy.embedder import embed_strings

    e1 = embed_strings(["test string"])
    e2 = embed_strings(["test string"])
    np.testing.assert_array_almost_equal(e1, e2)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_taxonomy.py -v
```

- [ ] **Step 4: Implement embedder**

```python
# src/lens/taxonomy/__init__.py
"""LENS taxonomy pipeline — clustering and labeling."""
```

```python
# src/lens/taxonomy/embedder.py
"""Embed raw extraction strings for clustering.

Uses sentence-transformers with a scientific embedding model.
Falls back to a lightweight model if the scientific model is unavailable.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}

# Primary: scientific embedding model
# Fallback: lightweight general-purpose model
MODELS = [
    "allenai-specter2",
    "all-MiniLM-L6-v2",
]


def _get_model(model_name: str | None = None) -> SentenceTransformer:
    """Load and cache a sentence-transformers model."""
    if model_name and model_name in _model_cache:
        return _model_cache[model_name]

    for name in ([model_name] if model_name else MODELS):
        if name in _model_cache:
            return _model_cache[name]
        try:
            model = SentenceTransformer(name)
            _model_cache[name] = model
            logger.info("Loaded embedding model: %s", name)
            return model
        except Exception:
            logger.warning("Failed to load model %s, trying next", name)
            continue

    raise RuntimeError("No embedding model available")


def embed_strings(
    strings: list[str],
    model_name: str | None = None,
) -> np.ndarray:
    """Embed a list of strings into dense vectors.

    Args:
        strings: Text strings to embed.
        model_name: Optional model override.

    Returns:
        numpy array of shape (len(strings), embedding_dim).
        Returns empty array with shape (0,) if strings is empty.
    """
    if not strings:
        return np.array([])

    model = _get_model(model_name)
    embeddings = model.encode(strings, show_progress_bar=False)
    return np.array(embeddings)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_taxonomy.py -v
```

Note: First run will download the model (~100MB). This may take a minute.

- [ ] **Step 6: Commit**

```bash
git add src/lens/taxonomy/ tests/test_taxonomy.py
git commit -m "feat: add string embedder with sentence-transformers"
```

---

### Task 2: HDBSCAN Clusterer

**Files:**
- Create: `src/lens/taxonomy/clusterer.py`
- Modify: `tests/test_taxonomy.py` — add clustering tests

- [ ] **Step 1: Add hdbscan dependency**

```bash
uv add hdbscan
```

- [ ] **Step 2: Write failing tests** (append to `tests/test_taxonomy.py`)

```python
def test_cluster_embeddings():
    from lens.taxonomy.clusterer import cluster_embeddings

    # Create clearly separable clusters
    rng = np.random.RandomState(42)
    cluster_a = rng.randn(10, 50) + np.array([5.0] + [0.0] * 49)
    cluster_b = rng.randn(10, 50) + np.array([0.0, 5.0] + [0.0] * 48)
    embeddings = np.vstack([cluster_a, cluster_b])

    labels = cluster_embeddings(embeddings, min_cluster_size=3)
    assert len(labels) == 20
    # Should find at least 2 clusters (labels 0 and 1), -1 is noise
    unique = set(labels)
    unique.discard(-1)
    assert len(unique) >= 2


def test_cluster_embeddings_small_dataset():
    from lens.taxonomy.clusterer import cluster_embeddings

    # Very small dataset — should not crash
    embeddings = np.random.randn(5, 50)
    labels = cluster_embeddings(embeddings, min_cluster_size=2)
    assert len(labels) == 5


def test_cluster_embeddings_fallback_to_kmeans():
    from lens.taxonomy.clusterer import cluster_embeddings

    # If all points are identical, HDBSCAN assigns all to noise
    # Should fallback to KMeans
    embeddings = np.ones((20, 50))
    labels = cluster_embeddings(
        embeddings, min_cluster_size=3, target_clusters=3
    )
    assert len(labels) == 20
    unique = set(labels)
    unique.discard(-1)
    assert len(unique) >= 1  # KMeans should find at least 1 cluster
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_taxonomy.py -v -k cluster
```

- [ ] **Step 4: Implement clusterer**

```python
# src/lens/taxonomy/clusterer.py
"""HDBSCAN clustering for taxonomy discovery.

Clusters raw extraction string embeddings into groups that become
taxonomy entries (parameters, principles, etc.).
Falls back to KMeans if HDBSCAN produces degenerate results.
"""

from __future__ import annotations

import logging

import hdbscan
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    target_clusters: int | None = None,
) -> list[int]:
    """Cluster embeddings using HDBSCAN with KMeans fallback.

    Args:
        embeddings: Array of shape (n_samples, n_features).
        min_cluster_size: HDBSCAN min_cluster_size parameter.
        target_clusters: Target number of clusters for KMeans fallback.

    Returns:
        List of cluster labels (-1 for noise).
    """
    if len(embeddings) < min_cluster_size:
        return [-1] * len(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings).tolist()

    # Check for degenerate results: all noise or single cluster
    unique = set(labels)
    unique.discard(-1)
    if len(unique) <= 1:
        logger.warning(
            "HDBSCAN produced %d clusters, falling back to KMeans",
            len(unique),
        )
        k = target_clusters or max(2, len(embeddings) // 5)
        k = min(k, len(embeddings))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings).tolist()

    return labels
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_taxonomy.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/lens/taxonomy/clusterer.py tests/test_taxonomy.py
git commit -m "feat: add HDBSCAN clusterer with KMeans fallback"
```

---

### Task 3: LLM Cluster Labeler

**Files:**
- Create: `src/lens/taxonomy/labeler.py`
- Modify: `tests/test_taxonomy.py` — add labeler tests

- [ ] **Step 1: Write failing tests** (append to `tests/test_taxonomy.py`)

```python
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_label_clusters():
    from lens.taxonomy.labeler import label_clusters

    clusters = {
        0: ["inference latency", "inference speed", "generation time"],
        1: ["model accuracy", "benchmark performance", "task accuracy"],
    }
    mock_client = AsyncMock()
    mock_client.complete.side_effect = [
        '{"name": "Inference Latency", "description": "Speed of generating output tokens"}',
        '{"name": "Model Accuracy", "description": "Performance on evaluation benchmarks"}',
    ]

    labels = await label_clusters(clusters, mock_client)
    assert len(labels) == 2
    assert labels[0]["name"] == "Inference Latency"
    assert labels[1]["name"] == "Model Accuracy"
    assert "description" in labels[0]


@pytest.mark.asyncio
async def test_label_clusters_handles_malformed():
    from lens.taxonomy.labeler import label_clusters

    clusters = {0: ["test string"]}
    mock_client = AsyncMock()
    mock_client.complete.return_value = "not json"

    labels = await label_clusters(clusters, mock_client)
    assert len(labels) == 1
    # Fallback: should use the most common string as name
    assert labels[0]["name"] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_taxonomy.py -v -k label
```

- [ ] **Step 3: Implement labeler**

```python
# src/lens/taxonomy/labeler.py
"""LLM-based cluster labeling for taxonomy entries.

Takes clustered extraction strings and asks an LLM to name and describe
each cluster. Falls back to using the most representative string if
the LLM returns invalid output.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from lens.llm.client import LLMClient

logger = logging.getLogger(__name__)


def _build_label_prompt(cluster_strings: list[str]) -> str:
    """Build a prompt asking the LLM to name and describe a cluster."""
    sample = cluster_strings[:20]  # limit to avoid token overflow
    strings_text = "\n".join(f"- {s}" for s in sample)
    return (
        "These are related terms extracted from LLM research papers. "
        "They all describe the same abstract concept.\n\n"
        f"Terms:\n{strings_text}\n\n"
        "Provide a concise name and one-sentence description for the "
        "concept they share. Respond with JSON only:\n"
        '{"name": "Concept Name", "description": "One sentence description."}'
    )


async def label_clusters(
    clusters: dict[int, list[str]],
    llm_client: LLMClient,
) -> dict[int, dict[str, str]]:
    """Label each cluster with a name and description via LLM.

    Args:
        clusters: Mapping of cluster_id -> list of member strings.
        llm_client: LLM client for completions.

    Returns:
        Mapping of cluster_id -> {"name": ..., "description": ...}.
    """
    labels: dict[int, dict[str, str]] = {}

    for cluster_id, strings in clusters.items():
        prompt = _build_label_prompt(strings)
        try:
            response = await llm_client.complete(
                [{"role": "user", "content": prompt}]
            )
            text = response.strip()
            # Strip code fences
            if text.startswith("```"):
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    text = text[start : end + 1]
            data = json.loads(text)
            labels[cluster_id] = {
                "name": data.get("name", strings[0]),
                "description": data.get("description", ""),
            }
        except (json.JSONDecodeError, Exception):
            # Fallback: use most common string
            most_common = Counter(strings).most_common(1)[0][0]
            labels[cluster_id] = {
                "name": most_common.title(),
                "description": f"Cluster of {len(strings)} related terms",
            }
            logger.warning(
                "LLM labeling failed for cluster %d, using fallback",
                cluster_id,
            )

    return labels
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_taxonomy.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/lens/taxonomy/labeler.py tests/test_taxonomy.py
git commit -m "feat: add LLM cluster labeler with fallback naming"
```

---

### Task 4: Taxonomy Version Management and Build Pipeline

**Files:**
- Create: `src/lens/taxonomy/versioning.py`
- Modify: `src/lens/taxonomy/__init__.py` — add `build_taxonomy` function
- Modify: `tests/test_taxonomy.py` — add build pipeline tests

- [ ] **Step 1: Write failing tests** (append to `tests/test_taxonomy.py`)

```python
@pytest.mark.asyncio
async def test_build_taxonomy(tmp_path):
    from lens.taxonomy import build_taxonomy
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    # Add some tradeoff extractions
    tradeoffs = [
        {
            "paper_id": f"paper_{i}",
            "improves": "inference speed" if i % 2 == 0 else "model accuracy",
            "worsens": "model size" if i % 2 == 0 else "training cost",
            "technique": "quantization" if i % 3 == 0 else "distillation",
            "context": "test",
            "confidence": 0.8,
            "evidence_quote": "test quote",
        }
        for i in range(20)
    ]
    store.add_rows("tradeoff_extractions", tradeoffs)

    mock_client = AsyncMock()
    mock_client.complete.return_value = (
        '{"name": "Test Concept", "description": "A test concept"}'
    )

    version = await build_taxonomy(store, mock_client, min_cluster_size=2)
    assert version >= 1

    # Check taxonomy entries were created
    params = store.get_table("parameters").to_polars()
    assert len(params) >= 1

    principles = store.get_table("principles").to_polars()
    assert len(principles) >= 1

    # Check taxonomy version was recorded
    versions = store.get_table("taxonomy_versions").to_polars()
    assert len(versions) >= 1


def test_get_next_version(tmp_path):
    from lens.taxonomy.versioning import get_next_version
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()
    assert get_next_version(store) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_taxonomy.py -v -k "build_taxonomy or get_next"
```

- [ ] **Step 3: Implement versioning**

```python
# src/lens/taxonomy/versioning.py
"""Taxonomy version management."""

from __future__ import annotations

from datetime import datetime

from lens.store.store import LensStore


def get_next_version(store: LensStore) -> int:
    """Get the next taxonomy version number."""
    df = store.get_table("taxonomy_versions").to_polars()
    if len(df) == 0:
        return 1
    return int(df["version_id"].max()) + 1


def record_version(
    store: LensStore,
    version_id: int,
    paper_count: int,
    param_count: int,
    principle_count: int,
) -> None:
    """Record a taxonomy version in the store."""
    store.add_rows(
        "taxonomy_versions",
        [
            {
                "version_id": version_id,
                "created_at": datetime.now(),
                "paper_count": paper_count,
                "param_count": param_count,
                "principle_count": principle_count,
            }
        ],
    )
```

- [ ] **Step 4: Implement build_taxonomy in `__init__.py`**

```python
# src/lens/taxonomy/__init__.py
"""LENS taxonomy pipeline — clustering and labeling."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

from lens.llm.client import LLMClient
from lens.store.store import LensStore
from lens.taxonomy.clusterer import cluster_embeddings
from lens.taxonomy.embedder import embed_strings
from lens.taxonomy.labeler import label_clusters
from lens.taxonomy.versioning import get_next_version, record_version

logger = logging.getLogger(__name__)


def _collect_strings_from_table(
    store: LensStore,
    table_name: str,
    columns: list[str],
    min_confidence: float = 0.5,
) -> list[str]:
    """Collect unique strings from specified columns of a table.

    Filters to rows with confidence >= min_confidence (spec requirement:
    scores below 0.5 are excluded from taxonomy clustering).
    """
    df = store.get_table(table_name).to_polars()
    if len(df) == 0:
        return []
    if "confidence" in df.columns:
        df = df.filter(pl.col("confidence") >= min_confidence)
    strings = []
    for col in columns:
        if col in df.columns:
            strings.extend(df[col].to_list())
    return list(set(s for s in strings if s))


def _group_by_cluster(
    strings: list[str], labels: list[int]
) -> dict[int, list[str]]:
    """Group strings by their cluster labels, excluding noise (-1)."""
    clusters: dict[int, list[str]] = {}
    for s, label in zip(strings, labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(s)
    return clusters


def _build_taxonomy_entries(
    cluster_labels: dict[int, dict[str, str]],
    clusters: dict[int, list[str]],
    strings: list[str],
    embeddings: np.ndarray,
    labels: list[int],
    version_id: int,
    paper_ids_by_string: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Build taxonomy entry dicts from clusters."""
    entries = []
    # Map strings to their embeddings
    string_to_idx = {s: i for i, s in enumerate(strings)}

    for cluster_id, label_info in cluster_labels.items():
        members = clusters.get(cluster_id, [])
        if not members:
            continue

        # Compute centroid embedding
        member_indices = [
            string_to_idx[s] for s in members if s in string_to_idx
        ]
        if member_indices:
            centroid = embeddings[member_indices].mean(axis=0)
        else:
            centroid = np.zeros(embeddings.shape[1])

        # Pad/truncate to 768
        if len(centroid) < 768:
            centroid = np.pad(centroid, (0, 768 - len(centroid)))
        elif len(centroid) > 768:
            centroid = centroid[:768]

        # Collect paper_ids
        all_paper_ids = []
        for s in members:
            all_paper_ids.extend(paper_ids_by_string.get(s, []))

        entry_id = version_id * 10000 + cluster_id
        entries.append(
            {
                "id": entry_id,
                "name": label_info["name"],
                "description": label_info["description"],
                "raw_strings": members,
                "paper_ids": list(set(all_paper_ids)),
                "taxonomy_version": version_id,
                "embedding": centroid.tolist(),
            }
        )

    return entries


async def build_taxonomy(
    store: LensStore,
    llm_client: LLMClient,
    min_cluster_size: int = 3,
    target_parameters: int = 25,
    target_principles: int = 35,
) -> int:
    """Build taxonomy from current extractions. Full rebuild.

    Returns the new taxonomy version number.
    """
    version_id = get_next_version(store)
    logger.info("Building taxonomy version %d", version_id)

    # --- Parameters (from improves + worsens strings) ---
    param_strings = _collect_strings_from_table(
        store, "tradeoff_extractions", ["improves", "worsens"]
    )
    param_paper_ids = _build_paper_id_map(
        store, "tradeoff_extractions", ["improves", "worsens"]
    )

    param_entries = []
    if param_strings:
        param_emb = embed_strings(param_strings)
        param_labels = cluster_embeddings(
            param_emb,
            min_cluster_size=min_cluster_size,
            target_clusters=target_parameters,
        )
        param_clusters = _group_by_cluster(param_strings, param_labels)
        param_names = await label_clusters(param_clusters, llm_client)
        param_entries = _build_taxonomy_entries(
            param_names,
            param_clusters,
            param_strings,
            param_emb,
            param_labels,
            version_id,
            param_paper_ids,
        )
        if param_entries:
            store.add_rows("parameters", param_entries)

    # --- Principles (from technique strings) ---
    principle_strings = _collect_strings_from_table(
        store, "tradeoff_extractions", ["technique"]
    )
    principle_paper_ids = _build_paper_id_map(
        store, "tradeoff_extractions", ["technique"]
    )

    principle_entries = []
    if principle_strings:
        princ_emb = embed_strings(principle_strings)
        princ_labels = cluster_embeddings(
            princ_emb,
            min_cluster_size=min_cluster_size,
            target_clusters=target_principles,
        )
        princ_clusters = _group_by_cluster(principle_strings, princ_labels)
        princ_names = await label_clusters(princ_clusters, llm_client)
        # Add sub_techniques field
        princ_entries_raw = _build_taxonomy_entries(
            princ_names,
            princ_clusters,
            principle_strings,
            princ_emb,
            princ_labels,
            version_id,
            principle_paper_ids,
        )
        for entry in princ_entries_raw:
            entry["sub_techniques"] = list(entry.get("raw_strings", []))
        principle_entries = princ_entries_raw
        if principle_entries:
            store.add_rows("principles", principle_entries)

    # Record version
    paper_count = len(store.get_table("papers").to_polars())
    record_version(
        store,
        version_id,
        paper_count=paper_count,
        param_count=len(param_entries),
        principle_count=len(principle_entries),
    )

    logger.info(
        "Taxonomy v%d: %d parameters, %d principles",
        version_id,
        len(param_entries),
        len(principle_entries),
    )
    return version_id


def _build_paper_id_map(
    store: LensStore, table_name: str, columns: list[str]
) -> dict[str, list[str]]:
    """Build a mapping from raw strings to paper_ids."""
    df = store.get_table(table_name).to_polars()
    if len(df) == 0:
        return {}
    result: dict[str, list[str]] = {}
    for row in df.to_dicts():
        pid = row.get("paper_id", "")
        for col in columns:
            s = row.get(col, "")
            if s:
                result.setdefault(s, []).append(pid)
    return result
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_taxonomy.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/lens/taxonomy/ tests/test_taxonomy.py
git commit -m "feat: add taxonomy build pipeline with versioning"
```

---

### Task 5: Contradiction Matrix Construction

**Files:**
- Create: `src/lens/knowledge/__init__.py`
- Create: `src/lens/knowledge/matrix.py`
- Create: `tests/test_matrix.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_matrix.py
"""Tests for contradiction matrix construction."""

import polars as pl
import pytest


def test_build_matrix(tmp_path):
    from lens.knowledge.matrix import build_matrix
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()

    # Add parameters
    store.add_rows(
        "parameters",
        [
            {
                "id": 1,
                "name": "Latency",
                "description": "Inference speed",
                "raw_strings": ["latency", "speed"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * 768,
            },
            {
                "id": 2,
                "name": "Accuracy",
                "description": "Model accuracy",
                "raw_strings": ["accuracy", "performance"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * 768,
            },
        ],
    )

    # Add principles
    store.add_rows(
        "principles",
        [
            {
                "id": 1,
                "name": "Quantization",
                "description": "Reduce precision",
                "sub_techniques": ["int8", "int4"],
                "raw_strings": ["quantization"],
                "paper_ids": [],
                "taxonomy_version": 1,
                "embedding": [0.0] * 768,
            },
        ],
    )

    # Add tradeoff extractions with matching raw strings
    store.add_rows(
        "tradeoff_extractions",
        [
            {
                "paper_id": "p1",
                "improves": "latency",
                "worsens": "accuracy",
                "technique": "quantization",
                "context": "",
                "confidence": 0.9,
                "evidence_quote": "quote",
            },
            {
                "paper_id": "p2",
                "improves": "speed",
                "worsens": "performance",
                "technique": "quantization",
                "context": "",
                "confidence": 0.8,
                "evidence_quote": "quote2",
            },
        ],
    )

    build_matrix(store, taxonomy_version=1)

    cells = store.get_table("matrix_cells").to_polars()
    assert len(cells) >= 1
    # Should have a cell for (Latency, Accuracy) -> Quantization
    cell = cells.filter(
        (pl.col("improving_param_id") == 1)
        & (pl.col("worsening_param_id") == 2)
        & (pl.col("principle_id") == 1)
    )
    assert len(cell) == 1
    assert cell["count"][0] == 2
    assert cell["avg_confidence"][0] == pytest.approx(0.85)


def test_build_matrix_empty(tmp_path):
    from lens.knowledge.matrix import build_matrix
    from lens.store.store import LensStore

    store = LensStore(str(tmp_path / "test.lance"))
    store.init_tables()
    build_matrix(store, taxonomy_version=1)
    cells = store.get_table("matrix_cells").to_polars()
    assert len(cells) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_matrix.py -v
```

- [ ] **Step 3: Implement matrix construction**

```python
# src/lens/knowledge/__init__.py
"""LENS knowledge structure construction."""

from lens.knowledge.matrix import build_matrix, get_ranked_matrix

__all__ = ["build_matrix", "get_ranked_matrix"]
```

```python
# src/lens/knowledge/matrix.py
"""Contradiction matrix construction.

Maps raw extraction strings through the taxonomy to build matrix cells.
Each cell: (improving_param_id, worsening_param_id) -> ranked principles.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def _build_string_to_id_map(
    store: LensStore, table_name: str, version: int
) -> dict[str, int]:
    """Build a map from raw_strings to taxonomy entry IDs."""
    df = store.get_table(table_name).to_polars()
    if len(df) == 0:
        return {}
    df = df.filter(pl.col("taxonomy_version") == version)
    result: dict[str, int] = {}
    for row in df.to_dicts():
        entry_id = row["id"]
        for s in row.get("raw_strings", []):
            result[s] = entry_id
    return result


def get_ranked_matrix(
    store: LensStore,
    taxonomy_version: int,
    top_k: int = 4,
) -> pl.DataFrame:
    """Get the contradiction matrix with top-k principles per cell pair.

    Returns a DataFrame with columns: improving_param_id, worsening_param_id,
    principle_id, count, avg_confidence, score (count * avg_confidence),
    ranked by score within each (improving, worsening) pair.
    """
    cells = store.get_table("matrix_cells").to_polars()
    if len(cells) == 0:
        return cells
    cells = cells.filter(pl.col("taxonomy_version") == taxonomy_version)
    if len(cells) == 0:
        return cells

    return (
        cells.with_columns(
            (pl.col("count") * pl.col("avg_confidence")).alias("score")
        )
        .sort("score", descending=True)
        .group_by(["improving_param_id", "worsening_param_id"])
        .head(top_k)
    )


def build_matrix(
    store: LensStore,
    taxonomy_version: int,
) -> None:
    """Build the contradiction matrix from extractions + taxonomy.

    Full rebuild — deletes existing cells for this version and recreates.
    """
    # Delete old cells for this version (idempotent rebuild)
    import contextlib

    with contextlib.suppress(OSError, ValueError):
        store.get_table("matrix_cells").delete(
            f"taxonomy_version = {taxonomy_version}"
        )

    # Load taxonomy mappings
    param_map = _build_string_to_id_map(store, "parameters", taxonomy_version)
    principle_map = _build_string_to_id_map(
        store, "principles", taxonomy_version
    )

    if not param_map or not principle_map:
        logger.info("No taxonomy entries — skipping matrix build")
        return

    # Load tradeoff extractions
    extractions = store.get_table("tradeoff_extractions").to_polars()
    if len(extractions) == 0:
        logger.info("No extractions — skipping matrix build")
        return

    # Filter to confidence >= 0.5 (spec requirement)
    extractions = extractions.filter(pl.col("confidence") >= 0.5)

    # Map raw strings to taxonomy IDs
    cells: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in extractions.to_dicts():
        imp_id = param_map.get(row["improves"])
        wors_id = param_map.get(row["worsens"])
        tech_id = principle_map.get(row["technique"])

        if imp_id is None or wors_id is None or tech_id is None:
            continue

        key = (imp_id, wors_id, tech_id)
        cells.setdefault(key, []).append(row)

    # Build matrix cell rows
    cell_rows = []
    for (imp_id, wors_id, princ_id), matches in cells.items():
        count = len(matches)
        avg_conf = sum(m["confidence"] for m in matches) / count
        paper_ids = list({m["paper_id"] for m in matches})
        cell_rows.append(
            {
                "improving_param_id": imp_id,
                "worsening_param_id": wors_id,
                "principle_id": princ_id,
                "count": count,
                "avg_confidence": avg_conf,
                "paper_ids": paper_ids,
                "taxonomy_version": taxonomy_version,
            }
        )

    if cell_rows:
        store.add_rows("matrix_cells", cell_rows)
        logger.info("Built matrix with %d cells", len(cell_rows))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_matrix.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/lens/knowledge/ tests/test_matrix.py
git commit -m "feat: add contradiction matrix construction from taxonomy"
```

---

### Task 6: Wire Up CLI Build Commands

**Files:**
- Modify: `src/lens/cli.py` — replace build stubs

- [ ] **Step 1: Replace the build stubs in cli.py**

Replace `taxonomy`, `build_matrix`, and `build_all` stubs:

```python
@build_app.command()
def taxonomy() -> None:
    """Build taxonomy by clustering extraction strings."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.llm.client import LLMClient
    from lens.taxonomy import build_taxonomy

    llm_model = config["llm"]["label_model"]
    client = LLMClient(model=llm_model)
    tax_config = config["taxonomy"]
    version = asyncio.run(
        build_taxonomy(
            store,
            client,
            min_cluster_size=tax_config["min_cluster_size"],
            target_parameters=tax_config["target_parameters"],
            target_principles=tax_config["target_principles"],
        )
    )
    rprint(f"[green]Built taxonomy version {version}[/green]")


@build_app.command(name="matrix")
def build_matrix_cmd() -> None:
    """Build contradiction matrix from taxonomy."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.knowledge.matrix import build_matrix
    from lens.taxonomy.versioning import get_next_version

    version = get_next_version(store) - 1
    if version < 1:
        rprint("[red]No taxonomy built yet. Run 'lens build taxonomy' first.[/red]")
        raise typer.Exit(code=1)
    build_matrix(store, taxonomy_version=version)
    rprint(f"[green]Built matrix for taxonomy v{version}[/green]")


@build_app.command(name="all")
def build_all() -> None:
    """Full rebuild: taxonomy + matrix."""
    config = load_config(_get_config_path())
    data_dir = _get_data_dir(config)
    store = LensStore(str(data_dir))
    store.init_tables()

    from lens.knowledge.matrix import build_matrix
    from lens.llm.client import LLMClient
    from lens.taxonomy import build_taxonomy

    llm_model = config["llm"]["label_model"]
    client = LLMClient(model=llm_model)
    tax_config = config["taxonomy"]
    version = asyncio.run(
        build_taxonomy(
            store,
            client,
            min_cluster_size=tax_config["min_cluster_size"],
            target_parameters=tax_config["target_parameters"],
            target_principles=tax_config["target_principles"],
        )
    )
    build_matrix(store, taxonomy_version=version)
    rprint(f"[green]Built taxonomy v{version} + matrix[/green]")
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest -v
```

- [ ] **Step 3: Verify CLI**

```bash
uv run lens build --help
uv run lens build taxonomy --help
uv run lens build matrix --help
uv run lens build all --help
```

- [ ] **Step 4: Commit**

```bash
git add src/lens/cli.py
git commit -m "feat: wire up build CLI commands (taxonomy, matrix, all)"
```

---

## Summary

After completing this plan, LENS can:
- **Embed raw strings** with sentence-transformers for clustering
- **Cluster extractions** into taxonomy entries using HDBSCAN (with KMeans fallback)
- **Label clusters** with LLM-generated names and descriptions
- **Version taxonomies** with full rebuild semantics
- **Build the contradiction matrix** by mapping extractions through the taxonomy
- **Track taxonomy versions** for reproducibility

CLI commands functional: `lens build taxonomy`, `lens build matrix`, `lens build all`.

**Note:** Architecture and agentic catalog construction is deferred — the taxonomy and matrix are the core knowledge structures needed for `lens analyze` and `lens explore` (Plan 5). Architecture/agentic catalogs can be added incrementally in a later plan.

**Next:** Plan 5 (Serve) — `lens analyze`, `lens explore`, `lens explain` commands.

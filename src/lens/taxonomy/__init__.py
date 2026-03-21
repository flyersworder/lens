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

__all__ = ["build_taxonomy"]


def _collect_strings_from_table(
    store: LensStore,
    table_name: str,
    columns: list[str],
    min_confidence: float = 0.5,
) -> list[str]:
    """Collect unique strings from specified columns of a table.

    Filters to rows with confidence >= min_confidence.
    """
    df = store.get_table(table_name).to_polars()
    if len(df) == 0:
        return []
    if "confidence" in df.columns:
        df = df.filter(pl.col("confidence") >= min_confidence)
    strings: list[str] = []
    for col in columns:
        if col in df.columns:
            strings.extend(df[col].to_list())
    return list(set(s for s in strings if s))


def _group_by_cluster(strings: list[str], cluster_labels: list[int]) -> dict[int, list[str]]:
    """Group strings by their cluster labels, excluding noise (-1)."""
    clusters: dict[int, list[str]] = {}
    for s, label in zip(strings, cluster_labels, strict=True):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(s)
    return clusters


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


def _build_taxonomy_entries(
    cluster_label_info: dict[int, dict[str, str]],
    clusters: dict[int, list[str]],
    strings: list[str],
    embeddings: np.ndarray,
    version_id: int,
    paper_ids_by_string: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Build taxonomy entry dicts from clusters."""
    entries: list[dict[str, Any]] = []
    string_to_idx = {s: i for i, s in enumerate(strings)}

    for cluster_id, label_info in cluster_label_info.items():
        members = clusters.get(cluster_id, [])
        if not members:
            continue

        # Compute centroid embedding
        member_indices = [string_to_idx[s] for s in members if s in string_to_idx]
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
        all_paper_ids: list[str] = []
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
    param_paper_ids = _build_paper_id_map(store, "tradeoff_extractions", ["improves", "worsens"])

    param_entries: list[dict[str, Any]] = []
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
            version_id,
            param_paper_ids,
        )
        if param_entries:
            store.add_rows("parameters", param_entries)

    # --- Principles (from technique strings) ---
    principle_strings = _collect_strings_from_table(store, "tradeoff_extractions", ["technique"])
    principle_paper_ids = _build_paper_id_map(store, "tradeoff_extractions", ["technique"])

    principle_entries: list[dict[str, Any]] = []
    if principle_strings:
        princ_emb = embed_strings(principle_strings)
        princ_labels = cluster_embeddings(
            princ_emb,
            min_cluster_size=min_cluster_size,
            target_clusters=target_principles,
        )
        princ_clusters = _group_by_cluster(principle_strings, princ_labels)
        princ_names = await label_clusters(princ_clusters, llm_client)
        princ_entries_raw = _build_taxonomy_entries(
            princ_names,
            princ_clusters,
            principle_strings,
            princ_emb,
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

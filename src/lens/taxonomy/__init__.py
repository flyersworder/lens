"""LENS taxonomy pipeline — clustering and labeling."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lens.llm.client import LLMClient
from lens.store.models import EMBEDDING_DIM
from lens.store.store import LensStore
from lens.taxonomy.clusterer import cluster_embeddings
from lens.taxonomy.embedder import embed_strings
from lens.taxonomy.labeler import (
    label_clusters,
    label_clusters_with_category,
    normalize_slots,
)
from lens.taxonomy.versioning import get_next_version, record_version  # noqa: F401
from lens.taxonomy.vocabulary import build_tradeoff_taxonomy  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "build_tradeoff_taxonomy",
    "build_architecture_taxonomy",
    "build_agentic_taxonomy",
    "_next_id",
]


def _next_id(store: LensStore, table_name: str) -> int:
    """Return max(id) + 1 from the table, or 1 if the table is empty."""
    rows = store.query_sql(f"SELECT MAX(id) AS max_id FROM {table_name}")
    max_id = rows[0]["max_id"] if rows else None
    if max_id is None:
        return 1
    return int(max_id) + 1


def _collect_strings_from_table(
    store: LensStore,
    table_name: str,
    columns: list[str],
    min_confidence: float = 0.5,
) -> list[str]:
    """Collect unique strings from specified columns of a table.

    Filters to rows with confidence >= min_confidence.
    """
    rows = store.query(table_name)
    if not rows:
        return []
    # Check if table has a confidence column
    has_confidence = "confidence" in rows[0]
    if has_confidence:
        rows = [r for r in rows if r.get("confidence", 0) >= min_confidence]
    strings: list[str] = []
    for row in rows:
        for col in columns:
            val = row.get(col)
            if val:
                strings.append(val)
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
    rows = store.query(table_name)
    if not rows:
        return {}
    result: dict[str, list[str]] = {}
    for row in rows:
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
    start_id: int = 1,
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

        # Pad/truncate to EMBEDDING_DIM
        if len(centroid) < EMBEDDING_DIM:
            centroid = np.pad(centroid, (0, EMBEDDING_DIM - len(centroid)))
        elif len(centroid) > EMBEDDING_DIM:
            centroid = centroid[:EMBEDDING_DIM]

        # Collect paper_ids
        all_paper_ids: list[str] = []
        for s in members:
            all_paper_ids.extend(paper_ids_by_string.get(s, []))

        entry_id = start_id + len(entries)
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


async def build_architecture_taxonomy(
    store: LensStore,
    llm_client: LLMClient,
    min_cluster_size: int = 3,
    target_arch_variants: int = 20,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
    version_id: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build architecture taxonomy (slots + variants) from extractions.

    Returns dict with keys: slot_entries, variant_entries.
    """

    def _embed(strings: list[str]) -> np.ndarray:
        return embed_strings(
            strings,
            provider=embedding_provider,
            model_name=embedding_model,
            api_base=embedding_api_base,
            api_key=embedding_api_key,
        )

    if version_id is None:
        version_id = 0

    arch_strings = _collect_strings_from_table(
        store, "architecture_extractions", ["component_slot"]
    )
    slot_entries: list[dict[str, Any]] = []
    variant_entries: list[dict[str, Any]] = []

    if arch_strings:
        # Normalize slot names via LLM
        slot_mapping = await normalize_slots(arch_strings, llm_client)
        canonical_names = sorted(set(slot_mapping.values()))

        # Create ArchitectureSlot entries
        slot_start_id = _next_id(store, "architecture_slots")
        slot_id_by_name: dict[str, int] = {}
        for i, canonical in enumerate(canonical_names):
            sid = slot_start_id + i
            slot_id_by_name[canonical] = sid
            slot_entries.append(
                {
                    "id": sid,
                    "name": canonical,
                    "description": "",
                    "taxonomy_version": version_id,
                }
            )

        # Write slots to DB immediately so they exist
        if slot_entries:
            store.add_rows("architecture_slots", slot_entries)

        # Load architecture extractions for variant aggregation
        arch_rows = store.query("architecture_extractions")
        arch_rows = [r for r in arch_rows if r.get("confidence", 0) >= 0.5]

        # Build paper_id map ONCE (not per-slot) to avoid repeated table scans
        var_paper_ids = _build_paper_id_map(store, "architecture_extractions", ["variant_name"])

        # Initialize running variant ID counter ONCE before the loop
        next_var_id = _next_id(store, "architecture_variants")

        for canonical in canonical_names:
            slot_id = slot_id_by_name[canonical]
            # Find raw slot strings that map to this canonical name
            raw_slots_for_canonical = [
                raw for raw, canon in slot_mapping.items() if canon == canonical
            ]
            # Collect variant_name strings from extractions matching this slot
            matching_rows = [
                r for r in arch_rows if r.get("component_slot") in raw_slots_for_canonical
            ]
            var_strings = list(set(s for r in matching_rows for s in [r["variant_name"]] if s))

            if not var_strings:
                continue

            if len(var_strings) < 2:
                # Create variant directly, no clustering
                vname = var_strings[0]
                # Aggregate properties
                props = [
                    r["key_properties"]
                    for r in matching_rows
                    if r.get("variant_name") == vname and r.get("key_properties")
                ]
                # Aggregate paper_ids
                pids = list(
                    set(r["paper_id"] for r in matching_rows if r.get("variant_name") == vname)
                )
                emb = _embed([vname])
                centroid = emb[0]
                if len(centroid) < EMBEDDING_DIM:
                    centroid = np.pad(centroid, (0, EMBEDDING_DIM - len(centroid)))
                elif len(centroid) > EMBEDDING_DIM:
                    centroid = centroid[:EMBEDDING_DIM]

                variant_entries.append(
                    {
                        "id": next_var_id,
                        "slot_id": slot_id,
                        "name": vname.title(),
                        "replaces": [],
                        "properties": "; ".join(set(props)) if props else "",
                        "paper_ids": pids,
                        "taxonomy_version": version_id,
                        "embedding": centroid.tolist(),
                    }
                )
                next_var_id += 1
            else:
                # Embed -> cluster -> label
                var_emb = _embed(var_strings)
                var_labels = cluster_embeddings(
                    var_emb,
                    min_cluster_size=min_cluster_size,
                    target_clusters=target_arch_variants,
                )
                var_clusters = _group_by_cluster(var_strings, var_labels)
                var_label_info = await label_clusters(var_clusters, llm_client)
                entries = _build_taxonomy_entries(
                    var_label_info,
                    var_clusters,
                    var_strings,
                    var_emb,
                    version_id,
                    var_paper_ids,
                    start_id=next_var_id,
                )
                for entry in entries:
                    entry["slot_id"] = slot_id
                    # Aggregate properties from source extractions
                    raw_members = entry.get("raw_strings", [])
                    props = []
                    for r in matching_rows:
                        if r.get("variant_name") in raw_members and r.get("key_properties"):
                            props.append(r["key_properties"])
                    entry["properties"] = "; ".join(set(props)) if props else ""
                    entry["replaces"] = []
                    del entry["raw_strings"]
                    del entry["description"]
                variant_entries.extend(entries)
                next_var_id += len(entries)

        # Best-effort resolution of replaces links
        if variant_entries:
            variant_name_to_id = {e["name"].lower(): e["id"] for e in variant_entries}
            for entry in variant_entries:
                raw_replaces: set[str] = set()
                for r in arch_rows:
                    vn = (r.get("variant_name") or "").title()
                    if vn.lower() == entry["name"].lower() and r.get("replaces"):
                        raw_replaces.add(r["replaces"].lower())
                resolved = [
                    variant_name_to_id[rr] for rr in raw_replaces if rr in variant_name_to_id
                ]
                entry["replaces"] = resolved

            store.add_rows("architecture_variants", variant_entries)

    logger.info(
        "Architecture taxonomy: %d slots, %d variants",
        len(slot_entries),
        len(variant_entries),
    )
    return {"slot_entries": slot_entries, "variant_entries": variant_entries}


async def build_agentic_taxonomy(
    store: LensStore,
    llm_client: LLMClient,
    min_cluster_size: int = 3,
    target_agentic_patterns: int = 15,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
    version_id: int | None = None,
) -> list[dict[str, Any]]:
    """Build agentic pattern taxonomy from extractions.

    Returns list of pattern entries created.
    """

    def _embed(strings: list[str]) -> np.ndarray:
        return embed_strings(
            strings,
            provider=embedding_provider,
            model_name=embedding_model,
            api_base=embedding_api_base,
            api_key=embedding_api_key,
        )

    if version_id is None:
        version_id = 0

    agentic_strings = _collect_strings_from_table(store, "agentic_extractions", ["pattern_name"])
    pattern_entries: list[dict[str, Any]] = []

    if agentic_strings:
        ag_emb = _embed(agentic_strings)
        ag_labels = cluster_embeddings(
            ag_emb,
            min_cluster_size=min_cluster_size,
            target_clusters=target_agentic_patterns,
        )
        ag_clusters = _group_by_cluster(agentic_strings, ag_labels)
        ag_paper_ids = _build_paper_id_map(store, "agentic_extractions", ["pattern_name"])

        # Build structures context for label_clusters_with_category
        ag_rows = store.query("agentic_extractions")
        ag_rows = [r for r in ag_rows if r.get("confidence", 0) >= 0.5]

        # Map cluster_id -> list of structure strings
        structures: dict[int, list[str]] = {}
        for cluster_id, members in ag_clusters.items():
            member_set = set(members)
            structs = [
                r["structure"]
                for r in ag_rows
                if r.get("pattern_name") in member_set and r.get("structure")
            ]
            structures[cluster_id] = structs

        ag_label_info = await label_clusters_with_category(ag_clusters, structures, llm_client)
        entries = _build_taxonomy_entries(
            ag_label_info,
            ag_clusters,
            agentic_strings,
            ag_emb,
            version_id,
            ag_paper_ids,
            start_id=_next_id(store, "agentic_patterns"),
        )
        for entry in entries:
            raw_members = entry.get("raw_strings", [])
            # Get category from label info
            entry_category = "Uncategorized"
            for _cid, info in ag_label_info.items():
                if info["name"] == entry["name"]:
                    entry_category = info.get("category", "Uncategorized")
                    break
            entry["category"] = entry_category
            # Aggregate components and use_cases from source extractions
            raw_member_set = set(raw_members)
            components: list[str] = []
            use_cases: list[str] = []
            for r in ag_rows:
                if r.get("pattern_name") in raw_member_set:
                    if r.get("components"):
                        components.extend(r["components"])
                    if r.get("use_case"):
                        use_cases.append(r["use_case"])
            entry["components"] = list(set(components))
            entry["use_cases"] = list(set(use_cases))
            del entry["raw_strings"]
        pattern_entries = entries
        if pattern_entries:
            store.add_rows("agentic_patterns", pattern_entries)

    logger.info(
        "Agentic taxonomy: %d patterns",
        len(pattern_entries),
    )
    return pattern_entries

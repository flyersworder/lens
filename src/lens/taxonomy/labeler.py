"""LLM-based cluster labeling for taxonomy entries.

Takes clustered extraction strings and asks an LLM to name and describe
each cluster. Falls back to using the most representative string if
the LLM returns invalid output.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from lens.llm.client import LLMClient
from lens.llm.utils import strip_code_fences

logger = logging.getLogger(__name__)


def _build_label_prompt(cluster_strings: list[str]) -> str:
    """Build a prompt asking the LLM to name and describe a cluster."""
    sample = cluster_strings[:20]
    strings_text = "\n".join(f"- {s}" for s in sample)
    return (
        "These are related terms extracted from LLM research papers. "
        "They all describe the same abstract concept.\n\n"
        f"Terms:\n{strings_text}\n\n"
        "Provide a concise name and one-sentence description for the "
        "concept they share. Respond with JSON only:\n"
        '{"name": "Concept Name", "description": "One sentence description."}'
    )


def _build_normalize_slots_prompt(raw_strings: list[str]) -> str:
    """Build a prompt asking the LLM to normalize raw slot strings into canonical names."""
    strings_text = "\n".join(f"- {s}" for s in raw_strings)
    mapping_example = json.dumps({raw_strings[0]: "Canonical Name"}) if raw_strings else "{}"
    return (
        "These are raw component slot names extracted from research papers. "
        "Normalize them into a small set of canonical slot names.\n\n"
        f"Raw strings:\n{strings_text}\n\n"
        "Respond with a JSON object mapping each raw string to its canonical name. "
        "Use title-cased, concise canonical names that group synonyms together.\n"
        f"Example format: {mapping_example}"
    )


async def normalize_slots(
    raw_strings: list[str],
    llm_client: LLMClient,
) -> dict[str, str]:
    """Normalize raw component_slot strings into canonical names via a single LLM call.

    Returns a dict mapping raw string -> canonical name.
    Falls back to {s: s.title() for s in raw_strings} on LLM failure.
    """
    if not raw_strings:
        return {}

    prompt = _build_normalize_slots_prompt(raw_strings)
    try:
        response = await llm_client.complete([{"role": "user", "content": prompt}])
        text = strip_code_fences(response.strip())
        data = json.loads(text)
        # Ensure all raw strings have a mapping; fill missing with title-case fallback
        result = {s: data.get(s, s.title()) for s in raw_strings}
        return result
    except Exception:
        logger.warning("LLM slot normalization failed, using title-case fallback")
        return {s: s.title() for s in raw_strings}


def _build_summarize_properties_prompt(raw_properties: list[str], variant_name: str) -> str:
    """Build a prompt asking the LLM to summarize variant properties into one sentence."""
    props_text = "\n".join(f"- {p}" for p in raw_properties)
    return (
        f"These are key properties of the '{variant_name}' variant"
        " extracted from research papers:\n\n"
        f"{props_text}\n\n"
        "Summarize these properties into a single concise sentence that captures "
        "the essential characteristics. Respond with the sentence only, no JSON."
    )


async def summarize_variant_properties(
    raw_properties: list[str],
    variant_name: str,
    llm_client: LLMClient,
) -> str:
    """Summarize a list of raw key_properties strings into a concise single sentence.

    Returns the single property directly if only one is provided.
    Falls back to '; '.join(set(raw_properties)) on LLM failure.
    """
    if len(raw_properties) <= 1:
        return raw_properties[0] if raw_properties else ""

    prompt = _build_summarize_properties_prompt(raw_properties, variant_name)
    try:
        response = await llm_client.complete([{"role": "user", "content": prompt}])
        return response.strip()
    except Exception:
        logger.warning(
            "LLM property summarization failed for variant '%s', using fallback",
            variant_name,
        )
        return "; ".join(set(raw_properties))


def _build_label_with_category_prompt(cluster_strings: list[str], structures: list[str]) -> str:
    """Build a prompt asking the LLM to name, describe, and categorize a cluster."""
    sample = cluster_strings[:20]
    strings_text = "\n".join(f"- {s}" for s in sample)
    structures_text = "\n".join(f"- {s}" for s in structures[:10]) if structures else "(none)"
    return (
        "These are related terms extracted from LLM research papers. "
        "They all describe the same abstract concept.\n\n"
        f"Terms:\n{strings_text}\n\n"
        f"Related structures:\n{structures_text}\n\n"
        "Provide a concise name, one-sentence description, and a high-level category "
        "for the concept they share. Respond with JSON only:\n"
        '{"name": "Concept Name", "description": "One sentence description.",'
        ' "category": "Category Name"}'
    )


async def label_clusters_with_category(
    clusters: dict[int, list[str]],
    structures: dict[int, list[str]],
    llm_client: LLMClient,
) -> dict[int, dict[str, str]]:
    """Label each cluster with a name, description, and category via LLM.

    Returns mapping of cluster_id -> {"name": ..., "description": ..., "category": ...}.
    Falls back to most common string for name/description and "Uncategorized" for category.
    """
    labels: dict[int, dict[str, str]] = {}

    for cluster_id, strings in clusters.items():
        cluster_structures = structures.get(cluster_id, [])
        prompt = _build_label_with_category_prompt(strings, cluster_structures)
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
                "description": f"Cluster of {len(strings)} related terms",
                "category": "Uncategorized",
            }
            logger.warning(
                "LLM labeling with category failed for cluster %d, using fallback",
                cluster_id,
            )

    return labels


async def label_clusters(
    clusters: dict[int, list[str]],
    llm_client: LLMClient,
) -> dict[int, dict[str, str]]:
    """Label each cluster with a name and description via LLM.

    Returns mapping of cluster_id -> {"name": ..., "description": ...}.
    Falls back to most common string on LLM failure.
    """
    labels: dict[int, dict[str, str]] = {}

    for cluster_id, strings in clusters.items():
        prompt = _build_label_prompt(strings)
        try:
            response = await llm_client.complete([{"role": "user", "content": prompt}])
            text = strip_code_fences(response.strip())
            data = json.loads(text)
            labels[cluster_id] = {
                "name": data.get("name", strings[0]),
                "description": data.get("description", ""),
            }
        except Exception:
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

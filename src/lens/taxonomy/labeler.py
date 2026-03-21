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
            text = response.strip()
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

"""Problem-solving: classify tradeoff → matrix lookup → ranked principles."""

from __future__ import annotations

import json
import logging
from typing import Any

import polars as pl

from lens.llm.client import LLMClient
from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def _build_classify_prompt(query: str, param_names: list[str]) -> str:
    """Build prompt to classify a query into improving/worsening params."""
    params_list = "\n".join(f"- {p}" for p in param_names)
    return (
        "You are an LLM engineering expert. A user describes a "
        "tradeoff they want to resolve.\n\n"
        f"User query: {query}\n\n"
        "Available parameters (dimensions of LLM design):\n"
        f"{params_list}\n\n"
        "Identify which parameter the user wants to IMPROVE and "
        "which parameter they accept WORSENING.\n"
        "Respond with JSON only:\n"
        '{"improving": "Parameter Name", '
        '"worsening": "Parameter Name"}'
    )


async def analyze(
    query: str,
    store: LensStore,
    llm_client: LLMClient,
    taxonomy_version: int,
) -> dict[str, Any]:
    """Analyze a tradeoff query and return ranked principles."""
    # Load parameters
    params_df = store.get_table("parameters").to_polars()
    params_df = params_df.filter(pl.col("taxonomy_version") == taxonomy_version)
    if len(params_df) == 0:
        return {
            "query": query,
            "improving": None,
            "worsening": None,
            "principles": [],
        }

    param_names = params_df["name"].to_list()
    param_name_to_id = dict(
        zip(
            params_df["name"].to_list(),
            params_df["id"].to_list(),
            strict=False,
        )
    )

    # Classify query via LLM
    prompt = _build_classify_prompt(query, param_names)
    try:
        response = await llm_client.complete([{"role": "user", "content": prompt}])
        text = response.strip()
        if text.startswith("```"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]
        classification = json.loads(text)
    except Exception:
        logger.warning("Failed to classify query: %s", query)
        return {
            "query": query,
            "improving": None,
            "worsening": None,
            "principles": [],
        }

    improving_name = classification.get("improving", "")
    worsening_name = classification.get("worsening", "")
    improving_id = param_name_to_id.get(improving_name)
    worsening_id = param_name_to_id.get(worsening_name)

    if improving_id is None or worsening_id is None:
        return {
            "query": query,
            "improving": improving_name,
            "worsening": worsening_name,
            "principles": [],
        }

    # Look up matrix
    cells_df = store.get_table("matrix_cells").to_polars()
    cells_df = cells_df.filter(
        (pl.col("taxonomy_version") == taxonomy_version)
        & (pl.col("improving_param_id") == improving_id)
        & (pl.col("worsening_param_id") == worsening_id)
    )

    if len(cells_df) == 0:
        return {
            "query": query,
            "improving": improving_name,
            "worsening": worsening_name,
            "principles": [],
        }

    # Rank and enrich
    cells_df = cells_df.with_columns(
        (pl.col("count") * pl.col("avg_confidence")).alias("score")
    ).sort("score", descending=True)

    princs_df = store.get_table("principles").to_polars()
    princs_df = princs_df.filter(pl.col("taxonomy_version") == taxonomy_version)
    princ_id_to_name = dict(
        zip(princs_df["id"].to_list(), princs_df["name"].to_list(), strict=False)
    )

    principles = []
    for row in cells_df.to_dicts():
        principles.append(
            {
                "principle_id": row["principle_id"],
                "name": princ_id_to_name.get(row["principle_id"], "Unknown"),
                "count": row["count"],
                "avg_confidence": row["avg_confidence"],
                "score": row["score"],
                "paper_ids": row["paper_ids"],
            }
        )

    return {
        "query": query,
        "improving": improving_name,
        "worsening": worsening_name,
        "principles": principles,
    }

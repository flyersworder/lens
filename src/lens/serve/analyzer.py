"""Problem-solving: classify tradeoff -> matrix lookup -> ranked principles."""

from __future__ import annotations

import json
import logging
from typing import Any

from lens.llm.client import LLMClient
from lens.llm.utils import strip_code_fences
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
) -> dict[str, Any]:
    """Analyze a tradeoff query and return ranked principles."""
    # Load parameters from vocabulary
    params = store.query("vocabulary", "kind = ?", ("parameter",))
    if not params:
        return {
            "query": query,
            "improving": None,
            "worsening": None,
            "principles": [],
        }

    param_names = [p["name"] for p in params]
    param_name_to_id = {p["name"]: p["id"] for p in params}

    # Classify query via LLM
    prompt = _build_classify_prompt(query, param_names)
    try:
        response = await llm_client.complete([{"role": "user", "content": prompt}])
        text = strip_code_fences(response.strip())
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
    cells = store.query(
        "matrix_cells",
        "improving_param_id = ? AND worsening_param_id = ?",
        (improving_id, worsening_id),
    )

    if not cells:
        return {
            "query": query,
            "improving": improving_name,
            "worsening": worsening_name,
            "principles": [],
        }

    # Add score and sort
    for c in cells:
        c["score"] = c["count"] * c["avg_confidence"]
    cells.sort(key=lambda x: x["score"], reverse=True)

    princs = store.query("vocabulary", "kind = ?", ("principle",))
    princ_id_to_name = {p["id"]: p["name"] for p in princs}

    principles = []
    for row in cells:
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


def _build_slot_identify_prompt(query: str, slot_names: list[str]) -> str:
    """Build prompt to identify the relevant architecture slot and constraints."""
    slots_list = "\n".join(f"- {s}" for s in slot_names)
    return (
        "You are an LLM architecture expert. A user is asking about a specific "
        "component or aspect of transformer architecture.\n\n"
        f"User query: {query}\n\n"
        "Available architecture slots:\n"
        f"{slots_list}\n\n"
        "Identify which slot is most relevant and extract any technical constraints "
        "from the query (e.g., 'sub-quadratic', 'bounded KV cache', 'long context').\n"
        "Respond with JSON only:\n"
        '{"slot": "Slot Name", "constraints": "extracted technical constraints"}'
    )


def _build_category_identify_prompt(query: str, category_names: list[str]) -> str:
    """Build prompt to identify the relevant agentic category."""
    cats_list = "\n".join(f"- {c}" for c in category_names)
    return (
        "You are an LLM agent design expert. A user is asking about agentic design patterns.\n\n"
        f"User query: {query}\n\n"
        "Available categories:\n" + cats_list + "\n\n"
        "Identify the most relevant category.\nRespond with JSON only:\n"
        '{"category": "Category Name"}'
    )


async def analyze_architecture(
    query: str,
    store: LensStore,
    llm_client: LLMClient,
) -> dict[str, Any]:
    """Analyze a query about transformer architecture and return matching variants."""
    slots = store.query("vocabulary", "kind = ?", ("arch_slot",))
    slot_names = [s["name"] for s in slots]

    identified_slot: str | None = None
    if slot_names:
        prompt = _build_slot_identify_prompt(query, slot_names)
        try:
            response = await llm_client.complete([{"role": "user", "content": prompt}])
            text = strip_code_fences(response.strip())
            classification = json.loads(text)
            identified_slot = classification.get("slot")
        except Exception:
            logger.warning("Failed to identify slot for: %s", query)

    extractions = store.query("architecture_extractions")
    if identified_slot:
        extractions = [e for e in extractions if e["component_slot"] == identified_slot]

    by_name: dict[str, dict[str, Any]] = {}
    for row in extractions:
        name = row["variant_name"]
        if name not in by_name:
            by_name[name] = {
                "variant_name": name,
                "slot": row["component_slot"],
                "properties": row.get("key_properties", ""),
                "paper_ids": [],
            }
        by_name[name]["paper_ids"].append(row["paper_id"])

    return {"query": query, "slot": identified_slot, "variants": list(by_name.values())}


async def analyze_agentic(
    query: str,
    store: LensStore,
    llm_client: LLMClient,
) -> dict[str, Any]:
    """Analyze a query about agentic design patterns and return matching patterns."""
    categories = store.query("vocabulary", "kind = ?", ("agentic_category",))
    cat_names = [c["name"] for c in categories]

    identified_category: str | None = None
    if cat_names:
        prompt = _build_category_identify_prompt(query, cat_names)
        try:
            response = await llm_client.complete([{"role": "user", "content": prompt}])
            text = strip_code_fences(response.strip())
            classification = json.loads(text)
            identified_category = classification.get("category")
        except Exception:
            logger.warning("Failed to identify category for: %s", query)

    extractions = store.query("agentic_extractions")
    if identified_category:
        extractions = [e for e in extractions if e.get("category") == identified_category]

    by_name: dict[str, dict[str, Any]] = {}
    for row in extractions:
        name = row["pattern_name"]
        if name not in by_name:
            by_name[name] = {
                "pattern_name": name,
                "category": row.get("category", ""),
                "structure": row.get("structure", ""),
                "use_case": row.get("use_case", ""),
                "components": row.get("components", []),
                "paper_ids": [],
            }
        by_name[name]["paper_ids"].append(row["paper_id"])

    return {
        "query": query,
        "category": identified_category,
        "patterns": list(by_name.values()),
    }

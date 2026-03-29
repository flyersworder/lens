"""Problem-solving: classify tradeoff -> matrix lookup -> ranked principles."""

from __future__ import annotations

import json
import logging
from typing import Any

from lens.llm.client import LLMClient
from lens.llm.utils import strip_code_fences
from lens.store.store import LensStore
from lens.taxonomy.embedder import embed_strings

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
    params = store.query("parameters", "taxonomy_version = ?", (taxonomy_version,))
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
        "taxonomy_version = ? AND improving_param_id = ? AND worsening_param_id = ?",
        (taxonomy_version, improving_id, worsening_id),
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

    princs = store.query("principles", "taxonomy_version = ?", (taxonomy_version,))
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


async def analyze_architecture(
    query: str,
    store: LensStore,
    llm_client: LLMClient,
    taxonomy_version: int,
    embedding_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a query about transformer architecture and return matching variants."""
    taxonomy_version = int(taxonomy_version)  # defense-in-depth: ensure int for SQL filter
    # Load architecture slots for the version
    slots = store.query("architecture_slots", "taxonomy_version = ?", (taxonomy_version,))

    slot_names = [s["name"] for s in slots] if slots else []
    slot_name_to_id: dict[str, int] = {s["name"]: s["id"] for s in slots} if slots else {}

    # Ask LLM to identify the relevant slot
    identified_slot: str | None = None
    identified_slot_id: int | None = None
    if slot_names:
        prompt = _build_slot_identify_prompt(query, slot_names)
        try:
            response = await llm_client.complete([{"role": "user", "content": prompt}])
            text = strip_code_fences(response.strip())
            classification = json.loads(text)
            identified_slot = classification.get("slot")
            identified_slot_id = slot_name_to_id.get(identified_slot or "")
        except Exception:
            logger.warning("Failed to identify architecture slot for query: %s", query)
            identified_slot = None
            identified_slot_id = None

    # Embed query for vector search
    query_embeddings = embed_strings([query], **(embedding_kwargs or {}))
    if len(query_embeddings) == 0:
        return {
            "query": query,
            "slot": identified_slot,
            "variants": [],
        }
    query_embedding = query_embeddings[0].tolist()

    # Vector search on architecture_variants
    try:
        raw_results = store.vector_search(
            "architecture_variants",
            query_embedding,
            limit=5,
            where="taxonomy_version = ?",
            params=(taxonomy_version,),
        )
    except Exception:
        logger.warning("Vector search failed for architecture_variants")
        raw_results = []

    # Optionally filter by identified slot
    if identified_slot_id is not None:
        raw_results = [r for r in raw_results if r.get("slot_id") == identified_slot_id]

    variants = []
    for row in raw_results:
        variants.append(
            {
                "id": row.get("id"),
                "slot_id": row.get("slot_id"),
                "name": row.get("name"),
                "properties": row.get("properties"),
                "paper_ids": row.get("paper_ids", []),
            }
        )

    return {
        "query": query,
        "slot": identified_slot,
        "variants": variants,
    }


async def analyze_agentic(
    query: str,
    store: LensStore,
    llm_client: LLMClient,
    taxonomy_version: int,
    embedding_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a query about agentic design patterns and return matching patterns."""
    taxonomy_version = int(taxonomy_version)  # defense-in-depth: ensure int for SQL filter
    # Embed query for vector search
    query_embeddings = embed_strings([query], **(embedding_kwargs or {}))
    if len(query_embeddings) == 0:
        return {
            "query": query,
            "patterns": [],
        }
    query_embedding = query_embeddings[0].tolist()

    # Vector search on agentic_patterns
    try:
        raw_results = store.vector_search(
            "agentic_patterns",
            query_embedding,
            limit=5,
            where="taxonomy_version = ?",
            params=(taxonomy_version,),
        )
    except Exception:
        logger.warning("Vector search failed for agentic_patterns")
        raw_results = []

    patterns = []
    for row in raw_results:
        patterns.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "category": row.get("category"),
                "description": row.get("description"),
                "components": row.get("components", []),
                "use_cases": row.get("use_cases", []),
                "paper_ids": row.get("paper_ids", []),
            }
        )

    return {
        "query": query,
        "patterns": patterns,
    }

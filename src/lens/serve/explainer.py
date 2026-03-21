"""Education: resolve → graph walk → LLM synthesis."""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from lens.llm.client import LLMClient
from lens.store.models import ExplanationResult
from lens.store.store import LensStore
from lens.taxonomy.embedder import embed_strings

logger = logging.getLogger(__name__)


def resolve_concept(
    query: str,
    store: LensStore,
    taxonomy_version: int,
    top_k: int = 3,
) -> dict[str, Any] | None:
    """Resolve a query to the best matching taxonomy entry."""
    query_embedding = embed_strings([query])[0].tolist()

    best_match: dict[str, Any] | None = None
    best_distance = float("inf")
    alternatives: list[dict[str, Any]] = []

    for table_name, entry_type in [
        ("parameters", "parameter"),
        ("principles", "principle"),
    ]:
        try:
            results = (
                store.get_table(table_name)
                .search(query_embedding)
                .where(f"taxonomy_version = {taxonomy_version}")
                .limit(top_k)
                .to_list()
            )
            if not results:
                continue
            for row in results:
                dist = row.get("_distance", float("inf"))
                entry: dict[str, Any] = {
                    "resolved_type": entry_type,
                    "resolved_id": int(row["id"]),
                    "resolved_name": row["name"],
                    "distance": float(dist),
                }
                alternatives.append(entry)
                if dist < best_distance:
                    best_distance = dist
                    best_match = entry
        except Exception:
            logger.warning("Vector search failed on %s", table_name)
            continue

    if best_match is None:
        return None

    alternatives.sort(key=lambda x: x["distance"])
    best_match["alternatives"] = alternatives[1:top_k]
    return best_match


def graph_walk(
    resolved_type: str,
    resolved_id: int,
    store: LensStore,
    taxonomy_version: int,
) -> dict[str, Any]:
    """Walk the knowledge graph outward from a resolved concept."""
    walk: dict[str, Any] = {}

    # Pre-load taxonomy tables (filtered by version)
    params_df = store.get_table("parameters").to_polars()
    params_df = params_df.filter(pl.col("taxonomy_version") == taxonomy_version)
    princs_df = store.get_table("principles").to_polars()
    princs_df = princs_df.filter(pl.col("taxonomy_version") == taxonomy_version)

    # Identity
    if resolved_type == "parameter":
        entry = params_df.filter(pl.col("id") == resolved_id)
        if len(entry) > 0:
            row = entry.to_dicts()[0]
            walk["identity"] = {
                "name": row["name"],
                "description": row["description"],
                "type": "parameter",
                "paper_ids": row.get("paper_ids", []),
            }
    elif resolved_type == "principle":
        entry = princs_df.filter(pl.col("id") == resolved_id)
        if len(entry) > 0:
            row = entry.to_dicts()[0]
            walk["identity"] = {
                "name": row["name"],
                "description": row["description"],
                "type": "principle",
                "sub_techniques": row.get("sub_techniques", []),
                "paper_ids": row.get("paper_ids", []),
            }

    if "identity" not in walk:
        walk["identity"] = {
            "name": "Unknown",
            "description": "",
            "type": resolved_type,
        }

    # Tradeoffs from matrix cells
    cells = store.get_table("matrix_cells").to_polars()
    cells = cells.filter(pl.col("taxonomy_version") == taxonomy_version)

    tradeoff_cells: list[dict[str, Any]] = []
    if resolved_type == "parameter":
        improving = cells.filter(pl.col("improving_param_id") == resolved_id)
        worsening = cells.filter(pl.col("worsening_param_id") == resolved_id)
        tradeoff_cells = improving.to_dicts() + worsening.to_dicts()
    elif resolved_type == "principle":
        matching = cells.filter(pl.col("principle_id") == resolved_id)
        tradeoff_cells = matching.to_dicts()
    walk["tradeoffs"] = tradeoff_cells

    # Connections from shared matrix cells
    connected_ids: set[int] = set()
    for cell in tradeoff_cells:
        connected_ids.add(cell["improving_param_id"])
        connected_ids.add(cell["worsening_param_id"])
        connected_ids.add(cell["principle_id"])
    connected_ids.discard(resolved_id)

    connections: list[str] = []
    for pid in connected_ids:
        match = params_df.filter(pl.col("id") == pid)
        if len(match) > 0:
            connections.append(match["name"][0])
            continue
        match = princs_df.filter(pl.col("id") == pid)
        if len(match) > 0:
            connections.append(match["name"][0])
    walk["connections"] = connections

    # ID-to-name map for synthesis prompt
    id_map: list[dict[str, Any]] = []
    for row in params_df.to_dicts():
        id_map.append({"id": row["id"], "name": row["name"]})
    for row in princs_df.to_dicts():
        id_map.append({"id": row["id"], "name": row["name"]})
    walk["_id_map"] = id_map

    return walk


def _build_synthesis_prompt(
    walk: dict[str, Any],
    focus: str | None = None,
) -> str:
    """Build synthesis prompt using names (not raw IDs)."""
    identity = walk.get("identity", {})
    name = identity.get("name", "Unknown")
    desc = identity.get("description", "")
    concept_type = identity.get("type", "concept")

    # Build ID lookup
    id_to_name: dict[int, str] = {}
    for item in walk.get("_id_map", []):
        id_to_name[item["id"]] = item["name"]

    tradeoffs_text = ""
    if walk.get("tradeoffs"):
        tradeoffs_text = f"\n\nTradeoffs involving {name}:\n"
        for t in walk["tradeoffs"][:10]:
            imp = id_to_name.get(
                t["improving_param_id"],
                f"param#{t['improving_param_id']}",
            )
            wors = id_to_name.get(
                t["worsening_param_id"],
                f"param#{t['worsening_param_id']}",
            )
            princ = id_to_name.get(
                t["principle_id"],
                f"principle#{t['principle_id']}",
            )
            tradeoffs_text += (
                f"- Improving {imp} worsens {wors} (via {princ}, evidence={t['count']})\n"
            )

    connections_text = ""
    if walk.get("connections"):
        connections_text = "\n\nRelated concepts: " + ", ".join(walk["connections"])

    sub_tech = ""
    if identity.get("sub_techniques"):
        sub_tech = "\nSub-techniques: " + ", ".join(identity["sub_techniques"])

    focus_instruction = ""
    if focus == "tradeoffs":
        focus_instruction = " Focus on tradeoffs this concept is involved in."
    elif focus == "related":
        focus_instruction = " Focus on connections to related concepts."
    elif focus == "evolution":
        focus_instruction = " Focus on how this concept has evolved."

    return (
        f"You are an LLM research expert. Explain '{name}' "
        f"({concept_type}) to someone studying LLM engineering."
        f"\n\nDescription: {desc}{sub_tech}"
        f"{tradeoffs_text}{connections_text}\n\n"
        f"Write a clear, educational explanation."
        f"{focus_instruction} "
        f"Adapt depth: broad concepts get an overview; "
        f"specific concepts get deep detail."
    )


async def explain(
    query: str,
    store: LensStore,
    llm_client: LLMClient,
    taxonomy_version: int,
    focus: str | None = None,
) -> ExplanationResult | None:
    """Explain a concept: resolve -> graph walk -> synthesize."""
    resolved = resolve_concept(query, store, taxonomy_version=taxonomy_version)
    if resolved is None:
        return None

    walk = graph_walk(
        resolved_type=resolved["resolved_type"],
        resolved_id=resolved["resolved_id"],
        store=store,
        taxonomy_version=taxonomy_version,
    )

    prompt = _build_synthesis_prompt(walk, focus=focus)
    try:
        narrative = await llm_client.complete([{"role": "user", "content": prompt}])
    except Exception:
        logger.warning("LLM synthesis failed for %s", query)
        narrative = walk["identity"].get("description", "")

    return ExplanationResult(
        resolved_type=resolved["resolved_type"],
        resolved_id=resolved["resolved_id"],
        resolved_name=resolved["resolved_name"],
        narrative=narrative,
        evolution=walk["identity"].get("sub_techniques", []),
        tradeoffs=walk.get("tradeoffs", []),
        connections=walk.get("connections", []),
        paper_refs=walk["identity"].get("paper_ids", []),
        alternatives=resolved.get("alternatives", []),
    )

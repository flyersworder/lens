"""Education: resolve -> graph walk -> LLM synthesis.

Handles all four vocabulary kinds (parameter, principle, arch_slot,
agentic_category) with kind-specific graph walks and synthesis prompts.
"""

from __future__ import annotations

import logging
from typing import Any

from lens.llm.client import LLMClient
from lens.serve.explorer import (
    _matches_canonical,
    list_architecture_variants,
)
from lens.store.models import ExplanationResult
from lens.store.store import LensStore
from lens.taxonomy.embedder import embed_strings

logger = logging.getLogger(__name__)


def _compute_richness(entry_id: str, kind: str, store: LensStore) -> int:
    """Score how much data is available for explaining this concept.

    Higher = more informative explanation possible.
    """
    if kind == "parameter":
        cells = store.query("matrix_cells")
        return sum(
            1
            for c in cells
            if c["improving_param_id"] == entry_id or c["worsening_param_id"] == entry_id
        )
    elif kind == "principle":
        cells = store.query("matrix_cells")
        return sum(1 for c in cells if c["principle_id"] == entry_id)
    elif kind == "arch_slot":
        vocab = store.query("vocabulary", "id = ?", (entry_id,))
        if not vocab:
            return 0
        name = vocab[0]["name"]
        extractions = store.query("architecture_extractions")
        return sum(1 for e in extractions if _matches_canonical(e["component_slot"], name))
    elif kind == "agentic_category":
        vocab = store.query("vocabulary", "id = ?", (entry_id,))
        if not vocab:
            return 0
        name = vocab[0]["name"]
        extractions = store.query("agentic_extractions")
        return sum(1 for e in extractions if _matches_canonical(e.get("category", ""), name))
    return 0


def resolve_concept(
    query: str,
    store: LensStore,
    top_k: int = 3,
    embedding_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve a query to the best matching vocabulary entry.

    Uses vector search for semantic matching, then re-ranks by data richness
    so the concept with the most useful explanation wins.
    """
    query_embedding = embed_strings([query], **(embedding_kwargs or {}))[0].tolist()

    candidates: list[dict[str, Any]] = []
    try:
        results = store.vector_search(
            "vocabulary",
            query_embedding,
            limit=top_k * 3,
        )
        for row in results:
            dist = row.get("_distance", float("inf"))
            candidates.append(
                {
                    "resolved_type": row.get("kind", "unknown"),
                    "resolved_id": row["id"],
                    "resolved_name": row["name"],
                    "distance": float(dist),
                }
            )
    except Exception:
        logger.warning("Vector search failed on vocabulary")

    if not candidates:
        return None

    # Re-rank: among the top semantic matches, pick the one with most data.
    # Only consider candidates within 2x the best distance (semantically close).
    best_dist = candidates[0]["distance"]
    close_candidates = [c for c in candidates if c["distance"] <= best_dist * 2]

    query_lower = query.lower().strip()
    for c in close_candidates:
        c["richness"] = _compute_richness(c["resolved_id"], c["resolved_type"], store)
        # Bonus for exact name match — strong signal the user means this concept
        if c["resolved_name"].lower() == query_lower:
            c["richness"] += 100

    # Sort by richness descending, then distance ascending as tiebreaker
    close_candidates.sort(key=lambda x: (-x["richness"], x["distance"]))
    best_match = close_candidates[0]

    # Build alternatives from all candidates (excluding the best)
    alternatives = [c for c in candidates if c["resolved_id"] != best_match["resolved_id"]]
    alternatives.sort(key=lambda x: x["distance"])
    best_match["alternatives"] = alternatives[: top_k - 1]

    return best_match


def graph_walk(
    resolved_type: str,
    resolved_id: str,
    store: LensStore,
) -> dict[str, Any]:
    """Walk the knowledge graph outward from a resolved concept.

    Routes to kind-specific walks: tradeoff graph for parameters/principles,
    architecture variants for arch_slots, pattern listing for agentic_categories.
    """
    if resolved_type in ("parameter", "principle"):
        return _walk_tradeoff(resolved_type, resolved_id, store)
    elif resolved_type == "arch_slot":
        return _walk_architecture(resolved_id, store)
    elif resolved_type == "agentic_category":
        return _walk_agentic(resolved_id, store)
    return {"identity": {"name": "Unknown", "description": "", "type": resolved_type}}


def _walk_tradeoff(resolved_type: str, resolved_id: str, store: LensStore) -> dict[str, Any]:
    """Walk tradeoff graph for parameter or principle."""
    walk: dict[str, Any] = {}
    vocab = store.query("vocabulary")

    entry = next((v for v in vocab if v["id"] == resolved_id), None)
    if entry:
        walk["identity"] = {
            "name": entry["name"],
            "description": entry["description"],
            "type": resolved_type,
            "paper_count": entry.get("paper_count", 0),
        }
    else:
        walk["identity"] = {"name": "Unknown", "description": "", "type": resolved_type}

    cells = store.query("matrix_cells")
    if resolved_type == "parameter":
        tradeoff_cells = [
            c
            for c in cells
            if c["improving_param_id"] == resolved_id or c["worsening_param_id"] == resolved_id
        ]
    else:
        tradeoff_cells = [c for c in cells if c["principle_id"] == resolved_id]
    walk["tradeoffs"] = tradeoff_cells

    connected_ids: set[str] = set()
    for cell in tradeoff_cells:
        connected_ids.add(cell["improving_param_id"])
        connected_ids.add(cell["worsening_param_id"])
        connected_ids.add(cell["principle_id"])
    connected_ids.discard(resolved_id)

    id_to_name = {v["id"]: v["name"] for v in vocab}
    walk["connections"] = [id_to_name[vid] for vid in connected_ids if vid in id_to_name]
    walk["_id_map"] = [{"id": v["id"], "name": v["name"]} for v in vocab]

    return walk


def _walk_architecture(resolved_id: str, store: LensStore) -> dict[str, Any]:
    """Walk architecture data for an arch_slot."""
    walk: dict[str, Any] = {}
    entry = store.query("vocabulary", "id = ?", (resolved_id,))
    if not entry:
        walk["identity"] = {"name": "Unknown", "description": "", "type": "arch_slot"}
        return walk

    slot_name = entry[0]["name"]
    walk["identity"] = {
        "name": slot_name,
        "description": entry[0]["description"],
        "type": "arch_slot",
    }

    variants = list_architecture_variants(store, slot_name)
    walk["variants"] = variants

    # Build paper timeline
    papers = store.query("papers")
    paper_map = {p["paper_id"]: p for p in papers}
    for v in variants:
        pids = v.get("paper_ids", [])
        dates = [paper_map[pid]["date"] for pid in pids if pid in paper_map]
        v["earliest_date"] = min(dates) if dates else None
    variants.sort(key=lambda x: x.get("earliest_date") or "9999-99-99")

    return walk


def _walk_agentic(resolved_id: str, store: LensStore) -> dict[str, Any]:
    """Walk agentic data for an agentic_category."""
    walk: dict[str, Any] = {}
    entry = store.query("vocabulary", "id = ?", (resolved_id,))
    if not entry:
        walk["identity"] = {"name": "Unknown", "description": "", "type": "agentic_category"}
        return walk

    cat_name = entry[0]["name"]
    walk["identity"] = {
        "name": cat_name,
        "description": entry[0]["description"],
        "type": "agentic_category",
    }

    extractions = store.query("agentic_extractions")
    patterns = [e for e in extractions if _matches_canonical(e.get("category", ""), cat_name)]
    # Deduplicate by pattern name
    by_name: dict[str, dict[str, Any]] = {}
    for p in patterns:
        name = p["pattern_name"]
        if name not in by_name:
            by_name[name] = {
                "pattern_name": name,
                "structure": p.get("structure", ""),
                "use_case": p.get("use_case", ""),
                "components": p.get("components", []),
            }
    walk["patterns"] = list(by_name.values())

    return walk


def _build_synthesis_prompt(
    walk: dict[str, Any],
    focus: str | None = None,
) -> str:
    """Build synthesis prompt with kind-specific context."""
    identity = walk.get("identity", {})
    name = identity.get("name", "Unknown")
    desc = identity.get("description", "")
    concept_type = identity.get("type", "concept")

    focus_instruction = ""
    if focus == "tradeoffs":
        focus_instruction = " Focus on tradeoffs this concept is involved in."
    elif focus == "related":
        focus_instruction = " Focus on connections to related concepts."
    elif focus == "evolution":
        focus_instruction = " Focus on how this concept has evolved over time."

    if concept_type in ("parameter", "principle"):
        return _build_tradeoff_prompt(walk, name, desc, concept_type, focus_instruction)
    elif concept_type == "arch_slot":
        return _build_architecture_prompt(walk, name, desc, focus_instruction)
    elif concept_type == "agentic_category":
        return _build_agentic_prompt(walk, name, desc, focus_instruction)

    return (
        f"You are an LLM research expert. Explain '{name}' ({concept_type}) "
        f"to someone studying LLM engineering.\n\nDescription: {desc}\n\n"
        f"Write a clear, educational explanation.{focus_instruction}"
    )


def _build_tradeoff_prompt(
    walk: dict[str, Any],
    name: str,
    desc: str,
    concept_type: str,
    focus_instruction: str,
) -> str:
    """Build prompt for parameter or principle."""
    id_to_name: dict[str, str] = {}
    for item in walk.get("_id_map", []):
        id_to_name[item["id"]] = item["name"]

    tradeoffs_text = ""
    if walk.get("tradeoffs"):
        tradeoffs_text = f"\n\nTradeoffs involving {name}:\n"
        for t in walk["tradeoffs"][:10]:
            imp = id_to_name.get(t["improving_param_id"], t["improving_param_id"])
            wors = id_to_name.get(t["worsening_param_id"], t["worsening_param_id"])
            princ = id_to_name.get(t["principle_id"], t["principle_id"])
            tradeoffs_text += (
                f"- Improving {imp} worsens {wors} (via {princ}, evidence={t['count']})\n"
            )

    connections_text = ""
    if walk.get("connections"):
        connections_text = "\n\nRelated concepts: " + ", ".join(walk["connections"])

    return (
        f"You are an LLM research expert. Explain '{name}' "
        f"({concept_type}) to someone studying LLM engineering."
        f"\n\nDescription: {desc}"
        f"{tradeoffs_text}{connections_text}\n\n"
        f"Write a clear, educational explanation.{focus_instruction} "
        f"Adapt depth: broad concepts get an overview; "
        f"specific concepts get deep detail."
    )


def _build_architecture_prompt(
    walk: dict[str, Any],
    name: str,
    desc: str,
    focus_instruction: str,
) -> str:
    """Build prompt for arch_slot."""
    variants_text = ""
    variants = walk.get("variants", [])
    if variants:
        variants_text = f"\n\nKnown variants of {name}:\n"
        for v in variants:
            date = v.get("earliest_date", "?")
            props = v.get("key_properties", "")
            replaces = v.get("replaces") or ""
            line = f"- {v['variant_name']} ({date})"
            if props:
                line += f": {props}"
            if replaces:
                line += f" [replaces: {replaces}]"
            variants_text += line + "\n"

    return (
        f"You are an LLM architecture expert. Explain the '{name}' "
        f"component slot to someone studying transformer architecture."
        f"\n\nDescription: {desc}"
        f"\n\n{len(variants)} variants found across the research literature."
        f"{variants_text}\n\n"
        f"Write a clear, educational explanation covering what this component does, "
        f"how it has evolved, and the key design tradeoffs between variants."
        f"{focus_instruction}"
    )


def _build_agentic_prompt(
    walk: dict[str, Any],
    name: str,
    desc: str,
    focus_instruction: str,
) -> str:
    """Build prompt for agentic_category."""
    patterns_text = ""
    patterns = walk.get("patterns", [])
    if patterns:
        patterns_text = f"\n\nKnown patterns in {name}:\n"
        for p in patterns:
            line = f"- {p['pattern_name']}"
            if p.get("use_case"):
                line += f": {p['use_case']}"
            if p.get("structure"):
                line += f" ({p['structure']})"
            patterns_text += line + "\n"

    return (
        f"You are an LLM agent design expert. Explain the '{name}' "
        f"category of agentic patterns to someone studying LLM agents."
        f"\n\nDescription: {desc}"
        f"\n\n{len(patterns)} patterns found across the research literature."
        f"{patterns_text}\n\n"
        f"Write a clear, educational explanation covering what this category "
        f"encompasses, the key patterns, and when to use them."
        f"{focus_instruction}"
    )


async def explain(
    query: str,
    store: LensStore,
    llm_client: LLMClient,
    focus: str | None = None,
    embedding_kwargs: dict[str, Any] | None = None,
) -> ExplanationResult | None:
    """Explain a concept: resolve -> graph walk -> synthesize."""
    resolved = resolve_concept(query, store, embedding_kwargs=embedding_kwargs)
    if resolved is None:
        return None

    walk = graph_walk(
        resolved_type=resolved["resolved_type"],
        resolved_id=resolved["resolved_id"],
        store=store,
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
        evolution=[],
        tradeoffs=walk.get("tradeoffs", []),
        connections=walk.get("connections", []),
        paper_refs=[],
        alternatives=resolved.get("alternatives", []),
    )

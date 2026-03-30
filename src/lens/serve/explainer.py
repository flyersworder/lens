"""Education: resolve -> graph walk -> LLM synthesis.

Handles all four vocabulary kinds (parameter, principle, arch_slot,
agentic_category) with kind-specific graph walks and synthesis prompts.
Uses hybrid search (FTS5 + vector) to find candidates, then lets the
LLM pick the best interpretation given the user's query.
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


def find_candidates(
    query: str,
    store: LensStore,
    top_k: int = 3,
    embedding_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Find top vocabulary candidates for a query using hybrid search.

    Returns up to top_k deduplicated candidates ranked by RRF score
    (combined FTS5 keyword + vector semantic relevance).
    """
    query_embedding = embed_strings([query], **(embedding_kwargs or {}))[0].tolist()

    candidates: list[dict[str, Any]] = []
    try:
        results = store.hybrid_search(
            query=query,
            embedding=query_embedding,
            limit=top_k * 2,
        )
        for row in results:
            candidates.append(
                {
                    "kind": row.get("kind", "unknown"),
                    "id": row["id"],
                    "name": row["name"],
                    "description": row.get("description", ""),
                }
            )
    except Exception:
        logger.warning("Hybrid search failed, falling back to vector search")
        try:
            results = store.vector_search("vocabulary", query_embedding, limit=top_k * 2)
            for row in results:
                candidates.append(
                    {
                        "kind": row.get("kind", "unknown"),
                        "id": row["id"],
                        "name": row["name"],
                        "description": row.get("description", ""),
                    }
                )
        except Exception:
            logger.warning("Vector search also failed")

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for c in candidates:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)

    return unique[:top_k]


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


def _summarize_walk(walk: dict[str, Any]) -> str:
    """Create a brief summary of a graph walk for LLM candidate selection."""
    identity = walk.get("identity", {})
    name = identity.get("name", "Unknown")
    desc = identity.get("description", "")
    kind = identity.get("type", "unknown")

    summary = f"{name} ({kind}): {desc}"

    if kind in ("parameter", "principle"):
        n_tradeoffs = len(walk.get("tradeoffs", []))
        n_connections = len(walk.get("connections", []))
        summary += f" [{n_tradeoffs} tradeoffs, {n_connections} connections]"
    elif kind == "arch_slot":
        n_variants = len(walk.get("variants", []))
        summary += f" [{n_variants} variants]"
    elif kind == "agentic_category":
        n_patterns = len(walk.get("patterns", []))
        summary += f" [{n_patterns} patterns]"

    return summary


def _build_selection_prompt(
    query: str,
    candidate_summaries: list[str],
) -> str:
    """Ask the LLM to pick the best candidate for the user's query."""
    options = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(candidate_summaries))
    return (
        "A user wants to learn about a concept in LLM engineering. "
        "Multiple matching concepts were found in our knowledge base.\n\n"
        f'User query: "{query}"\n\n'
        f"Candidates:\n{options}\n\n"
        "Which candidate best matches the user's intent? "
        "Respond with ONLY the number (e.g., 1)."
    )


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
    """Explain a concept: find candidates -> LLM selects -> graph walk -> synthesize."""
    candidates = find_candidates(query, store, top_k=3, embedding_kwargs=embedding_kwargs)
    if not candidates:
        return None

    # Graph walk each candidate to gather context
    walks: list[dict[str, Any]] = []
    for c in candidates:
        walk = graph_walk(resolved_type=c["kind"], resolved_id=c["id"], store=store)
        walks.append(walk)

    # If only one candidate or one clearly has all the data, skip LLM selection
    if len(candidates) == 1:
        selected_idx = 0
    else:
        # Let the LLM pick the best match
        summaries = [_summarize_walk(w) for w in walks]
        selection_prompt = _build_selection_prompt(query, summaries)
        try:
            response = await llm_client.complete([{"role": "user", "content": selection_prompt}])
            choice = response.strip().strip(".")
            selected_idx = int(choice) - 1
            if selected_idx < 0 or selected_idx >= len(candidates):
                selected_idx = 0
        except Exception:
            logger.warning("LLM selection failed, using first candidate")
            selected_idx = 0

    selected = candidates[selected_idx]
    walk = walks[selected_idx]

    # Synthesize explanation
    prompt = _build_synthesis_prompt(walk, focus=focus)
    try:
        narrative = await llm_client.complete([{"role": "user", "content": prompt}])
    except Exception:
        logger.warning("LLM synthesis failed for %s", query)
        narrative = walk["identity"].get("description", "")

    alternatives = [
        {
            "resolved_type": c["kind"],
            "resolved_id": c["id"],
            "resolved_name": c["name"],
        }
        for c in candidates
        if c["id"] != selected["id"]
    ]

    return ExplanationResult(
        resolved_type=selected["kind"],
        resolved_id=selected["id"],
        resolved_name=selected["name"],
        narrative=narrative,
        evolution=[],
        tradeoffs=walk.get("tradeoffs", []),
        connections=walk.get("connections", []),
        paper_refs=[],
        alternatives=alternatives,
    )

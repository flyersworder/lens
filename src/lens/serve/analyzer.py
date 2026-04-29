"""Problem-solving: classify tradeoff -> matrix lookup -> ranked principles."""

from __future__ import annotations

import json
import logging
from typing import Any

from lens.llm.client import LLMClient
from lens.llm.utils import strip_code_fences
from lens.store.protocols import ReadableStore

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
    store: ReadableStore,
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


def _build_vocab_links_index(
    store: ReadableStore,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Index for cross-linking variants/patterns/components to vocabulary.

    Returns (name_index, princ_tradeoff_counts) where:

    * ``name_index`` maps lowercased vocab name → its row, used for
      fuzzy matching of free-text strings (e.g. "FlashAttention" → the
      "Flash Attention" principle).
    * ``princ_tradeoff_counts`` maps principle_id → number of matrix
      cells citing it, used to surface "involved in N tradeoffs" badges.
    """
    rows = store.query("vocabulary")
    name_index = {r["name"].lower(): r for r in rows}
    cells = store.query("matrix_cells")
    counts: dict[str, int] = {}
    for c in cells:
        pid = c["principle_id"]
        counts[pid] = counts.get(pid, 0) + 1
    return name_index, counts


def _find_vocab_link(name: str, name_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Best-effort match free-text ``name`` to a vocabulary entry.

    Tries exact (case-insensitive) match first, then falls back to a
    longest-substring match in either direction. The 5-char minimum
    avoids matching trivial shared substrings like "the" or "tool".
    """
    norm = name.lower().strip()
    if norm in name_index:
        return name_index[norm]
    best: tuple[int, dict[str, Any]] | None = None
    for key, row in name_index.items():
        if len(key) < 5:
            continue
        if key in norm or norm in key:
            length = min(len(key), len(norm))
            if best is None or length > best[0]:
                best = (length, row)
    return best[1] if best else None


def _earliest_date(paper_ids: list[str], paper_dates: dict[str, str]) -> str | None:
    """Return the earliest non-epoch date among ``paper_ids``."""
    dates = [
        paper_dates[p]
        for p in paper_ids
        if p in paper_dates and paper_dates[p] and paper_dates[p] != "1970-01-01"
    ]
    return min(dates) if dates else None


async def analyze_architecture(
    query: str,
    store: ReadableStore,
    llm_client: LLMClient,
) -> dict[str, Any]:
    """Analyze a query about transformer architecture and return matching variants.

    Variants are date-sorted (oldest first → genealogy view) and
    cross-linked to vocabulary entries when their name matches a known
    principle/parameter/arch_slot — surfacing "explain →" + tradeoff
    counts in the UI.
    """
    slots = store.query("vocabulary", "kind = ?", ("arch_slot",))
    slot_names = [s["name"] for s in slots]
    slot_id_by_name = {s["name"].lower(): s["id"] for s in slots}

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

    name_index, princ_counts = _build_vocab_links_index(store)
    paper_dates = {p["paper_id"]: p["date"] for p in store.query("papers")}

    by_name: dict[str, dict[str, Any]] = {}
    for row in extractions:
        name = row["variant_name"]
        if name not in by_name:
            by_name[name] = {
                "variant_name": name,
                "slot": row["component_slot"],
                "properties": row.get("key_properties", ""),
                "replaces": row.get("replaces") or None,
                "paper_ids": [],
            }
        by_name[name]["paper_ids"].append(row["paper_id"])
        # Keep the first non-empty `replaces` value across rows.
        if not by_name[name]["replaces"] and row.get("replaces"):
            by_name[name]["replaces"] = row["replaces"]

    variants: list[dict[str, Any]] = []
    for v in by_name.values():
        v["earliest_date"] = _earliest_date(v["paper_ids"], paper_dates)
        link = _find_vocab_link(v["variant_name"], name_index)
        if link:
            v["vocab_id"] = link["id"]
            v["vocab_kind"] = link["kind"]
            if link["kind"] == "principle":
                v["tradeoff_count"] = princ_counts.get(link["id"], 0)
        variants.append(v)

    # Sort by date ascending — None dates sink to the bottom.
    variants.sort(key=lambda x: x.get("earliest_date") or "9999-99-99")

    slot_vocab_id = slot_id_by_name.get(identified_slot.lower()) if identified_slot else None

    return {
        "query": query,
        "slot": identified_slot,
        "slot_id": slot_vocab_id,
        "variants": variants,
    }


async def analyze_agentic(
    query: str,
    store: ReadableStore,
    llm_client: LLMClient,
) -> dict[str, Any]:
    """Analyze a query about agentic design patterns.

    Patterns are date-sorted and components are resolved against the
    full vocabulary — when a component string matches a known
    parameter/principle/arch_slot, the UI renders it as a clickable
    link to the explain page for that concept.
    """
    categories = store.query("vocabulary", "kind = ?", ("agentic_category",))
    cat_names = [c["name"] for c in categories]
    cat_id_by_name = {c["name"].lower(): c["id"] for c in categories}

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

    name_index, _ = _build_vocab_links_index(store)
    paper_dates = {p["paper_id"]: p["date"] for p in store.query("papers")}

    by_name: dict[str, dict[str, Any]] = {}
    for row in extractions:
        name = row["pattern_name"]
        if name not in by_name:
            by_name[name] = {
                "pattern_name": name,
                "category": row.get("category", ""),
                "structure": row.get("structure", ""),
                "use_case": row.get("use_case", ""),
                "components": list(row.get("components") or []),
                "paper_ids": [],
            }
        by_name[name]["paper_ids"].append(row["paper_id"])

    patterns: list[dict[str, Any]] = []
    for p in by_name.values():
        p["earliest_date"] = _earliest_date(p["paper_ids"], paper_dates)
        # Resolve each component string to a vocab entry where possible.
        # Plain strings are kept as `{name, vocab_id: null, vocab_kind: null}`
        # so the frontend has a uniform shape.
        resolved_components: list[dict[str, Any]] = []
        for comp in p["components"]:
            if not isinstance(comp, str):
                continue
            link = _find_vocab_link(comp, name_index)
            resolved_components.append(
                {
                    "name": comp,
                    "vocab_id": link["id"] if link else None,
                    "vocab_kind": link["kind"] if link else None,
                }
            )
        p["components"] = resolved_components
        patterns.append(p)

    patterns.sort(key=lambda x: x.get("earliest_date") or "9999-99-99")

    category_vocab_id = (
        cat_id_by_name.get(identified_category.lower()) if identified_category else None
    )

    return {
        "query": query,
        "category": identified_category,
        "category_id": category_vocab_id,
        "patterns": patterns,
    }

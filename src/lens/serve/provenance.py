"""Provenance sidecars for analyze/explain outputs.

Each sidecar captures the exact paper/vocabulary evidence behind a given
CLI invocation so a reader can trace any claim back to the stored data.
Inspired by the Feynman project's ``<slug>.provenance.md`` pattern.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from lens.store.models import ExplanationResult


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def build_analyze_provenance(
    *,
    query: str,
    type_: str | None,
    result: dict[str, Any],
    session_id: str,
    taxonomy_version: int | None,
) -> dict[str, Any]:
    """Construct a provenance dict for a `lens analyze` invocation."""
    base: dict[str, Any] = {
        "command": "analyze",
        "query": query,
        "type": type_ or "tradeoff",
        "session_id": session_id,
        "taxonomy_version": taxonomy_version,
        "generated_at": _now_iso(),
    }

    if type_ == "architecture":
        variants = result.get("variants", []) or []
        paper_ids: list[str] = []
        claims = []
        for v in variants:
            pids = list(v.get("paper_ids", []) or [])
            paper_ids.extend(pids)
            claims.append(
                {
                    "variant_name": v.get("variant_name"),
                    "slot": v.get("slot"),
                    "paper_ids": pids,
                }
            )
        base["resolved"] = {"slot": result.get("slot")}
        base["claims"] = claims
        base["paper_ids"] = sorted(set(paper_ids))
        return base

    if type_ == "agentic":
        patterns = result.get("patterns", []) or []
        paper_ids = []
        claims = []
        for p in patterns:
            pids = list(p.get("paper_ids", []) or [])
            paper_ids.extend(pids)
            claims.append(
                {
                    "pattern_name": p.get("pattern_name"),
                    "category": p.get("category"),
                    "paper_ids": pids,
                }
            )
        base["resolved"] = {"category": result.get("category")}
        base["claims"] = claims
        base["paper_ids"] = sorted(set(paper_ids))
        return base

    # Default: tradeoff analysis.
    principles = result.get("principles", []) or []
    paper_ids = []
    claims = []
    for p in principles:
        pids = list(p.get("paper_ids", []) or [])
        paper_ids.extend(pids)
        claims.append(
            {
                "principle_id": p.get("principle_id"),
                "name": p.get("name"),
                "evidence_count": p.get("count"),
                "avg_confidence": p.get("avg_confidence"),
                "paper_ids": pids,
            }
        )
    base["resolved"] = {
        "improving": result.get("improving"),
        "worsening": result.get("worsening"),
    }
    base["claims"] = claims
    base["paper_ids"] = sorted(set(paper_ids))
    return base


def build_explain_provenance(
    *,
    query: str,
    focus: str | None,
    result: ExplanationResult,
    session_id: str,
    taxonomy_version: int | None,
) -> dict[str, Any]:
    """Construct a provenance dict for a `lens explain` invocation."""
    # Each matrix-cell tradeoff row carries an explicit paper_ids list.
    paper_ids: list[str] = []
    vocab_ids: set[str] = {result.resolved_id}
    for t in result.tradeoffs or []:
        paper_ids.extend(t.get("paper_ids", []) or [])
        for key in ("improving_param_id", "worsening_param_id", "principle_id"):
            if key in t:
                vocab_ids.add(t[key])

    return {
        "command": "explain",
        "query": query,
        "focus": focus,
        "session_id": session_id,
        "taxonomy_version": taxonomy_version,
        "generated_at": _now_iso(),
        "resolved": {
            "type": result.resolved_type,
            "id": result.resolved_id,
            "name": result.resolved_name,
        },
        "paper_ids": sorted(set(paper_ids)),
        "vocab_ids": sorted(vocab_ids),
        "connections": list(result.connections),
        "alternatives": [
            {"type": a["resolved_type"], "id": a["resolved_id"], "name": a["resolved_name"]}
            for a in (result.alternatives or [])
        ],
        "narrative_chars": len(result.narrative or ""),
    }


_HEADER = (
    "# LENS provenance sidecar. `paper_ids` list the papers backing each claim;\n"
    "# raw extraction rows can be recovered via SELECT from *_extractions\n"
    "# WHERE paper_id IN (...) against the SQLite DB at ~/.lens/data/lens.db.\n"
)


def write_provenance(data: dict[str, Any], path: str | Path) -> Path:
    """Write a provenance dict as YAML to *path*. Returns the resolved path."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(_HEADER)
        yaml.safe_dump(data, f, sort_keys=False)
    return p

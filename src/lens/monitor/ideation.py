"""Ideation gap analysis pipeline — sparse cells, cross-pollination, LLM enrichment."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np

from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def find_sparse_cells(
    store: LensStore,
    taxonomy_version: int,
    min_principles: int = 2,
) -> list[dict[str, Any]]:
    """Find parameter pairs that have fewer than *min_principles* resolving them.

    Returns a list of dicts with keys:
        improving_param_id, worsening_param_id, existing_principle_ids, count
    """
    params_df = store.get_table("parameters").to_polars()
    params_df = params_df.filter(params_df["taxonomy_version"] == taxonomy_version)
    param_ids = params_df["id"].to_list()

    cells_df = store.get_table("matrix_cells").to_polars()
    cells_df = cells_df.filter(cells_df["taxonomy_version"] == taxonomy_version)
    cells_list = cells_df.to_dicts()

    # Build a mapping: (improving, worsening) -> list of principle_ids
    pair_principles: dict[tuple[int, int], list[int]] = {}
    for cell in cells_list:
        key = (cell["improving_param_id"], cell["worsening_param_id"])
        pair_principles.setdefault(key, []).append(cell["principle_id"])

    gaps: list[dict[str, Any]] = []
    for imp_id in param_ids:
        for wors_id in param_ids:
            if imp_id == wors_id:
                continue
            key = (imp_id, wors_id)
            existing = pair_principles.get(key, [])
            if len(existing) < min_principles:
                gaps.append(
                    {
                        "improving_param_id": imp_id,
                        "worsening_param_id": wors_id,
                        "existing_principle_ids": existing,
                        "count": len(existing),
                    }
                )
    return gaps


def find_cross_pollination(
    store: LensStore,
    taxonomy_version: int,
    similarity_threshold: float = 0.75,
) -> list[dict[str, Any]]:
    """Find principles that could transfer to similar parameter pairs.

    If principle P resolves (A, B) and A' is similar to A, but P does not
    resolve (A', B), then (A', B, P) is a cross-pollination candidate.
    Checks BOTH improving and worsening roles.
    """
    params_df = store.get_table("parameters").to_polars()
    params_df = params_df.filter(params_df["taxonomy_version"] == taxonomy_version)
    if len(params_df) == 0:
        return []

    param_ids = params_df["id"].to_list()
    embeddings = np.array(params_df["embedding"].to_list())

    # Compute cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    similarity = normalized @ normalized.T

    cells_df = store.get_table("matrix_cells").to_polars()
    cells_df = cells_df.filter(cells_df["taxonomy_version"] == taxonomy_version)
    cells_list = cells_df.to_dicts()

    # Build set of existing (improving, worsening, principle) triples
    existing_triples: set[tuple[int, int, int]] = set()
    for cell in cells_list:
        existing_triples.add(
            (
                cell["improving_param_id"],
                cell["worsening_param_id"],
                cell["principle_id"],
            )
        )

    # Build index from param_id -> position in param_ids list
    id_to_idx = {pid: i for i, pid in enumerate(param_ids)}

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()

    for cell in cells_list:
        imp_id = cell["improving_param_id"]
        wors_id = cell["worsening_param_id"]
        principle_id = cell["principle_id"]

        imp_idx = id_to_idx.get(imp_id)
        wors_idx = id_to_idx.get(wors_id)
        if imp_idx is None or wors_idx is None:
            continue

        # Check improving role: find params similar to imp_id
        for j, other_id in enumerate(param_ids):
            if other_id == imp_id:
                continue
            if similarity[imp_idx, j] >= similarity_threshold:
                candidate_key = (other_id, wors_id, principle_id)
                if candidate_key not in existing_triples and candidate_key not in seen:
                    seen.add(candidate_key)
                    candidates.append(
                        {
                            "improving_param_id": other_id,
                            "worsening_param_id": wors_id,
                            "principle_id": principle_id,
                            "source_improving_param_id": imp_id,
                            "source_worsening_param_id": wors_id,
                            "similarity": float(similarity[imp_idx, j]),
                            "role": "improving",
                        }
                    )

        # Check worsening role: find params similar to wors_id
        for j, other_id in enumerate(param_ids):
            if other_id == wors_id:
                continue
            if similarity[wors_idx, j] >= similarity_threshold:
                candidate_key = (imp_id, other_id, principle_id)
                if candidate_key not in existing_triples and candidate_key not in seen:
                    seen.add(candidate_key)
                    candidates.append(
                        {
                            "improving_param_id": imp_id,
                            "worsening_param_id": other_id,
                            "principle_id": principle_id,
                            "source_improving_param_id": imp_id,
                            "source_worsening_param_id": wors_id,
                            "similarity": float(similarity[wors_idx, j]),
                            "role": "worsening",
                        }
                    )

    return candidates


def run_ideation(
    store: LensStore,
    taxonomy_version: int,
    min_principles: int = 2,
    similarity_threshold: float = 0.75,
) -> dict[str, Any]:
    """Run Layer 1 ideation: sparse cells + cross-pollination.

    Persists gaps and report to LanceDB and returns a summary dict.
    """
    now = datetime.now(UTC)

    sparse = find_sparse_cells(store, taxonomy_version, min_principles=min_principles)
    cross = find_cross_pollination(
        store, taxonomy_version, similarity_threshold=similarity_threshold
    )

    # Fetch param names for descriptions
    params_df = store.get_table("parameters").to_polars()
    params_df = params_df.filter(params_df["taxonomy_version"] == taxonomy_version)
    param_names = {row["id"]: row["name"] for row in params_df.to_dicts()}

    # Fetch principle names for descriptions
    principles_df = store.get_table("principles").to_polars()
    principles_df = principles_df.filter(principles_df["taxonomy_version"] == taxonomy_version)
    principle_names = {row["id"]: row["name"] for row in principles_df.to_dicts()}

    # Determine next report_id
    reports_df = store.get_table("ideation_reports").to_polars()
    report_id = int(reports_df["id"].max()) + 1 if len(reports_df) > 0 else 1  # type: ignore[arg-type]

    # Determine next gap_id
    gaps_df = store.get_table("ideation_gaps").to_polars()
    next_gap_id = int(gaps_df["id"].max()) + 1 if len(gaps_df) > 0 else 1  # type: ignore[arg-type]

    all_gaps: list[dict[str, Any]] = []

    for gap in sparse:
        imp_name = param_names.get(gap["improving_param_id"], "?")
        wors_name = param_names.get(gap["worsening_param_id"], "?")
        count = gap["count"]
        desc = (
            f"Sparse cell: {imp_name} vs {wors_name} has no principles"
            if count == 0
            else f"Sparse cell: {imp_name} vs {wors_name} has only {count} principle(s)"
        )
        gap_record = {
            "id": next_gap_id,
            "report_id": report_id,
            "gap_type": "sparse_cell",
            "description": desc,
            "related_params": [
                gap["improving_param_id"],
                gap["worsening_param_id"],
            ],
            "related_principles": gap["existing_principle_ids"],
            "related_slots": [],
            "score": 1.0 - (gap["count"] / min_principles),
            "llm_hypothesis": None,
            "created_at": now,
            "taxonomy_version": taxonomy_version,
        }
        all_gaps.append(gap_record)
        next_gap_id += 1

    for cand in cross:
        imp_name = param_names.get(cand["improving_param_id"], "?")
        wors_name = param_names.get(cand["worsening_param_id"], "?")
        p_name = principle_names.get(cand["principle_id"], "?")
        desc = (
            f"Cross-pollination: {p_name} could apply to "
            f"{imp_name} vs {wors_name} "
            f"(similarity={cand['similarity']:.2f}, role={cand['role']})"
        )
        gap_record = {
            "id": next_gap_id,
            "report_id": report_id,
            "gap_type": "cross_pollination",
            "description": desc,
            "related_params": [
                cand["improving_param_id"],
                cand["worsening_param_id"],
            ],
            "related_principles": [cand["principle_id"]],
            "related_slots": [],
            "score": cand["similarity"],
            "llm_hypothesis": None,
            "created_at": now,
            "taxonomy_version": taxonomy_version,
        }
        all_gaps.append(gap_record)
        next_gap_id += 1

    # Persist gaps
    if all_gaps:
        store.add_rows("ideation_gaps", all_gaps)

    # Persist report
    report_record = {
        "id": report_id,
        "created_at": now,
        "taxonomy_version": taxonomy_version,
        "paper_batch_size": 0,
        "gap_count": len(all_gaps),
    }
    store.add_rows("ideation_reports", [report_record])

    logger.info(
        "Ideation v%d: %d gaps (%d sparse, %d cross-pollination)",
        taxonomy_version,
        len(all_gaps),
        sum(1 for g in all_gaps if g["gap_type"] == "sparse_cell"),
        sum(1 for g in all_gaps if g["gap_type"] == "cross_pollination"),
    )

    return {
        "report_id": report_id,
        "gap_count": len(all_gaps),
        "gaps": all_gaps,
    }


async def run_ideation_with_llm(
    store: LensStore,
    llm_client: Any,
    taxonomy_version: int,
    min_principles: int = 2,
    similarity_threshold: float = 0.75,
) -> dict[str, Any]:
    """Run ideation then enrich gaps with LLM-generated hypotheses."""
    report = run_ideation(
        store,
        taxonomy_version,
        min_principles=min_principles,
        similarity_threshold=similarity_threshold,
    )

    for gap in report["gaps"]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research analyst. Given a gap in a "
                    "tradeoff matrix, propose a hypothesis for why "
                    "this gap exists and how it might be resolved."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Gap description: {gap['description']}\n"
                    f"Gap type: {gap['gap_type']}\n"
                    f"Score: {gap['score']:.2f}\n\n"
                    "Propose a brief hypothesis."
                ),
            },
        ]
        try:
            hypothesis = await llm_client.complete(messages)
            gap["llm_hypothesis"] = hypothesis

            # Update the gap in LanceDB
            gap_id = int(gap["id"])
            table = store.get_table("ideation_gaps")
            table.update(
                where=f"id = {gap_id}",
                values={"llm_hypothesis": hypothesis},
            )
        except Exception:
            logger.warning(
                "LLM enrichment failed for gap %d",
                gap.get("id", -1),
            )

    return report

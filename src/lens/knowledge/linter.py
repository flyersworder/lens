"""LENS knowledge base linter — health checks with optional auto-fix."""

from __future__ import annotations

import logging

from lens.knowledge.events import log_event
from lens.store.models import LintReport
from lens.store.store import LensStore

logger = logging.getLogger(__name__)


def check_orphan_vocabulary(store: LensStore) -> list[dict]:
    """Find extracted vocabulary entries with zero paper references."""
    return store.query_sql(
        "SELECT id, name, kind, description, source, paper_count "
        "FROM vocabulary WHERE paper_count = 0 AND source != 'seed'"
    )


def fix_orphans(store: LensStore) -> list[str]:
    """Delete orphan vocabulary entries. Returns list of deleted IDs."""
    orphans = check_orphan_vocabulary(store)
    deleted_ids = []
    for orphan in orphans:
        store.delete("vocabulary", "id = ?", (orphan["id"],))
        deleted_ids.append(orphan["id"])
    return deleted_ids


def check_weak_evidence(store: LensStore, confidence_threshold: float = 0.5) -> list[dict]:
    """Find vocabulary entries with thin evidence (1 paper or low confidence).

    Seed entries are excluded — they start with paper_count=0 by design.
    """
    return store.query_sql(
        "SELECT id, name, kind, paper_count, avg_confidence "
        "FROM vocabulary "
        "WHERE source != 'seed' AND (paper_count = 1 OR avg_confidence < ?)",
        (confidence_threshold,),
    )


def check_missing_embeddings(store: LensStore) -> list[dict]:
    """Find vocabulary entries with no corresponding row in vocabulary_vec."""
    return store.query_sql(
        "SELECT v.id, v.name, v.kind "
        "FROM vocabulary v "
        "LEFT JOIN vocabulary_vec vv ON v.id = vv.id "
        "WHERE vv.id IS NULL"
    )


def fix_missing_embeddings(
    store: LensStore,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
) -> list[str]:
    """Generate and store embeddings for entries missing them. Returns fixed IDs."""
    from lens.taxonomy.embedder import embed_strings

    missing = check_missing_embeddings(store)
    if not missing:
        return []

    texts = [f"{r['name']}: {r.get('kind', '')}" for r in missing]
    embeddings = embed_strings(
        texts,
        provider=embedding_provider,
        model_name=embedding_model,
        api_base=embedding_api_base,
        api_key=embedding_api_key,
    )

    fixed_ids = []
    for row, emb in zip(missing, embeddings, strict=True):
        store.upsert_embedding("vocabulary", row["id"], emb.tolist())
        fixed_ids.append(row["id"])
    return fixed_ids


def check_stale_extractions(store: LensStore) -> list[dict]:
    """Find papers with non-complete extraction status."""
    return store.query_sql(
        "SELECT paper_id, title, extraction_status "
        "FROM papers "
        "WHERE extraction_status IN ('pending', 'incomplete', 'failed')"
    )


def fix_stale_extractions(store: LensStore) -> list[str]:
    """Reset stale papers to 'pending' for re-extraction. Returns fixed paper_ids."""
    stale = check_stale_extractions(store)
    fixed_ids = []
    for paper in stale:
        store.update(
            "papers",
            "extraction_status = ?",
            "paper_id = ?",
            ("pending", paper["paper_id"]),
        )
        fixed_ids.append(paper["paper_id"])
    return fixed_ids


def check_near_duplicates(store: LensStore, similarity_threshold: float = 0.92) -> list[dict]:
    """Find vocabulary entries with cosine similarity above threshold within the same kind.

    Uses sqlite-vec's MATCH subquery to compare embeddings directly in SQL.
    Returns deduplicated pairs (A,B not B,A).
    """
    vocab = store.query("vocabulary")
    if not vocab:
        return []

    by_kind: dict[str, list[dict]] = {}
    for entry in vocab:
        by_kind.setdefault(entry["kind"], []).append(entry)

    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    max_distance = 1.0 - similarity_threshold

    for kind, entries in by_kind.items():
        if len(entries) < 2:
            continue

        entry_ids = {e["id"] for e in entries}

        for entry in entries:
            try:
                neighbors = store.query_sql(
                    "SELECT id, distance "
                    "FROM vocabulary_vec "
                    "WHERE embedding MATCH (SELECT embedding FROM vocabulary_vec WHERE id = ?) "
                    "AND k = ? AND id != ?",
                    (entry["id"], min(len(entries), 20), entry["id"]),
                )
            except Exception:
                logger.debug("Near-duplicate search failed for %s", entry["id"], exc_info=True)
                continue

            for neighbor in neighbors:
                if neighbor["distance"] > max_distance:
                    continue
                if neighbor["id"] not in entry_ids:
                    continue

                key = (min(entry["id"], neighbor["id"]), max(entry["id"], neighbor["id"]))
                if key in seen:
                    continue
                seen.add(key)

                neighbor_entry = next(e for e in entries if e["id"] == neighbor["id"])
                pairs.append(
                    {
                        "id_a": entry["id"],
                        "name_a": entry["name"],
                        "id_b": neighbor["id"],
                        "name_b": neighbor_entry["name"],
                        "kind": kind,
                        "distance": neighbor["distance"],
                    }
                )
    return pairs


def lint(
    store: LensStore,
    fix: bool = False,
    session_id: str | None = None,
    checks: list[str] | None = None,
    confidence_threshold: float = 0.5,
    similarity_threshold: float = 0.92,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    embedding_api_base: str | None = None,
    embedding_api_key: str | None = None,
) -> LintReport:
    """Run lint checks and optionally apply fixes. Returns a LintReport."""
    active_checks = (
        set(checks)
        if checks
        else set(
            [
                "orphans",
                "contradictions",
                "weak_evidence",
                "missing_embeddings",
                "stale",
                "near_duplicates",
            ]
        )
    )

    report = LintReport()

    # 1. Orphans
    if "orphans" in active_checks:
        report.orphans = check_orphan_vocabulary(store)
        for finding in report.orphans:
            log_event(
                store,
                "lint",
                "orphan.found",
                target_type="vocabulary",
                target_id=finding["id"],
                detail={"name": finding["name"], "kind": finding["kind"]},
                session_id=session_id,
            )

    # 2. Contradictions
    if "contradictions" in active_checks:
        report.contradictions = check_contradictions(store)
        for finding in report.contradictions:
            log_event(
                store,
                "lint",
                "contradiction.found",
                target_type="matrix",
                detail={"params": finding["params"], "principle_id": finding["principle_id"]},
                session_id=session_id,
            )

    # 3. Weak evidence
    if "weak_evidence" in active_checks:
        report.weak_evidence = check_weak_evidence(store, confidence_threshold)
        for finding in report.weak_evidence:
            log_event(
                store,
                "lint",
                "weak_evidence.found",
                target_type="vocabulary",
                target_id=finding["id"],
                detail={
                    "paper_count": finding["paper_count"],
                    "avg_confidence": finding["avg_confidence"],
                },
                session_id=session_id,
            )

    # 4. Missing embeddings
    if "missing_embeddings" in active_checks:
        report.missing_embeddings = check_missing_embeddings(store)
        for finding in report.missing_embeddings:
            log_event(
                store,
                "lint",
                "missing_embedding.found",
                target_type="vocabulary",
                target_id=finding["id"],
                detail={"name": finding["name"]},
                session_id=session_id,
            )

    # 5. Stale extractions
    if "stale" in active_checks:
        report.stale_extractions = check_stale_extractions(store)
        for finding in report.stale_extractions:
            log_event(
                store,
                "lint",
                "stale_extraction.found",
                target_type="paper",
                target_id=finding["paper_id"],
                detail={"status": finding["extraction_status"]},
                session_id=session_id,
            )

    # 6. Near-duplicates
    if "near_duplicates" in active_checks:
        report.near_duplicates = check_near_duplicates(store, similarity_threshold)
        for finding in report.near_duplicates:
            log_event(
                store,
                "lint",
                "near_duplicate.found",
                target_type="vocabulary",
                detail={
                    "id_a": finding["id_a"],
                    "id_b": finding["id_b"],
                    "kind": finding["kind"],
                },
                session_id=session_id,
            )

    # Apply fixes if requested
    if fix:
        if "orphans" in active_checks and report.orphans:
            deleted = fix_orphans(store)
            for oid in deleted:
                report.fixes_applied.append({"action": "orphan.deleted", "target_id": oid})
                log_event(
                    store,
                    "fix",
                    "orphan.deleted",
                    target_type="vocabulary",
                    target_id=oid,
                    session_id=session_id,
                )

        if "missing_embeddings" in active_checks and report.missing_embeddings:
            fixed = fix_missing_embeddings(
                store,
                embedding_provider,
                embedding_model,
                embedding_api_base,
                embedding_api_key,
            )
            for fid in fixed:
                report.fixes_applied.append({"action": "embedding.repaired", "target_id": fid})
                log_event(
                    store,
                    "fix",
                    "embedding.repaired",
                    target_type="vocabulary",
                    target_id=fid,
                    session_id=session_id,
                )

        if "stale" in active_checks and report.stale_extractions:
            requeued = fix_stale_extractions(store)
            for pid in requeued:
                report.fixes_applied.append({"action": "extraction.requeued", "target_id": pid})
                log_event(
                    store,
                    "fix",
                    "extraction.requeued",
                    target_type="paper",
                    target_id=pid,
                    session_id=session_id,
                )

    return report


def check_contradictions(store: LensStore, min_count: int = 2) -> list[dict]:
    """Find parameter pairs with opposing directionality in the matrix.

    A contradiction exists when both (A improves, B worsens, principle P)
    and (B improves, A worsens, principle P) exist, each with count >= min_count.
    """
    cells = store.query("matrix_cells")
    if not cells:
        return []

    by_principle: dict[str, list[dict]] = {}
    for cell in cells:
        by_principle.setdefault(cell["principle_id"], []).append(cell)

    contradictions = []
    seen: set[tuple[str, str, str]] = set()

    for principle_id, group in by_principle.items():
        for cell in group:
            if cell["count"] < min_count:
                continue
            imp = cell["improving_param_id"]
            wors = cell["worsening_param_id"]

            for other in group:
                if other["count"] < min_count:
                    continue
                if other["improving_param_id"] == wors and other["worsening_param_id"] == imp:
                    key = (min(imp, wors), max(imp, wors), principle_id)
                    if key not in seen:
                        seen.add(key)
                        contradictions.append(
                            {
                                "params": [imp, wors],
                                "principle_id": principle_id,
                                "forward_count": cell["count"],
                                "reverse_count": other["count"],
                            }
                        )
    return contradictions

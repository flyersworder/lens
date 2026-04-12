"""Monitor pipeline: acquire new papers -> enrich -> extract -> build -> ideate.

Runs one monitoring cycle with configurable stages.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from lens.acquire.arxiv import fetch_arxiv_papers
from lens.extract.extractor import extract_papers
from lens.llm.client import LLMClient
from lens.store.models import EMBEDDING_DIM
from lens.store.store import LensStore

logger = logging.getLogger(__name__)


async def run_monitor_cycle(
    store: LensStore,
    llm_client: LLMClient,
    query: str = "LLM",
    categories: list[str] | None = None,
    max_results: int = 50,
    run_enrich: bool = True,
    run_build: bool = True,
    run_ideation_flag: bool = True,
    ideate_with_llm: bool = False,
    openalex_mailto: str = "",
    embedding_kwargs: dict[str, Any] | None = None,
    venue_tiers: dict[str, list[str]] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run one monitoring cycle with configurable stages.

    Stages:
    1. Acquire — fetch new papers from arxiv
    2. Enrich — OpenAlex metadata (if run_enrich and openalex_mailto)
    3. Extract — LLM knowledge extraction
    4. Build — taxonomy + matrix rebuild (if run_build)
    5. Ideate — gap analysis (if run_ideation_flag)
    """
    cats = categories or ["cs.CL", "cs.LG", "cs.AI"]
    session_id = session_id or str(uuid4())[:8]

    # --- Stage 1: Acquire ---
    try:
        papers = await fetch_arxiv_papers(
            query=query,
            categories=cats,
            max_results=max_results,
        )
    except Exception:
        logger.warning("Failed to fetch papers from arxiv")
        papers = []

    existing = store.query("papers")
    existing_ids = {p["paper_id"] for p in existing}
    new_papers = [p for p in papers if p["paper_id"] not in existing_ids]

    for p in new_papers:
        if "embedding" not in p:
            p["embedding"] = [0.0] * EMBEDDING_DIM

    papers_acquired = len(new_papers)
    if new_papers:
        store.add_papers(new_papers)
        logger.info("Acquired %d new papers", papers_acquired)

    # --- Stage 2: Enrich ---
    papers_enriched = 0
    if run_enrich and not openalex_mailto and papers_acquired > 0:
        logger.info("Skipping OpenAlex enrichment: openalex_mailto not configured")
    if run_enrich and openalex_mailto and papers_acquired > 0:
        try:
            from lens.acquire.openalex import enrich_with_openalex
            from lens.acquire.quality import quality_score as compute_quality

            papers_for_enrich = [
                {k: v for k, v in p.items() if k != "embedding"} for p in new_papers
            ]
            enriched = await enrich_with_openalex(papers_for_enrich, mailto=openalex_mailto)

            for paper in enriched:
                pid = paper.get("paper_id", "")
                store.update(
                    "papers",
                    "citations = ?, venue = ?",
                    "paper_id = ?",
                    (paper.get("citations", 0), paper.get("venue"), pid),
                )
                score = compute_quality(
                    citations=paper.get("citations", 0) or 0,
                    venue=paper.get("venue"),
                    paper_date=paper.get("date", "2020-01-01"),
                    venue_tiers=venue_tiers,
                )
                store.update("papers", "quality_score = ?", "paper_id = ?", (score, pid))
                papers_enriched += 1
            logger.info("Enriched %d papers via OpenAlex", papers_enriched)
        except Exception:
            logger.warning(
                "OpenAlex enrichment failed, continuing without enrichment",
                exc_info=True,
            )

    # --- Stage 3: Extract ---
    papers_extracted = 0
    if papers_acquired > 0:
        papers_extracted = await extract_papers(
            store, llm_client, concurrency=3, session_id=session_id
        )

    # --- Stage 4: Build ---
    taxonomy_built = False
    matrix_built = False
    if run_build:
        try:
            from lens.knowledge.matrix import build_matrix
            from lens.taxonomy import get_next_version, record_version
            from lens.taxonomy.vocabulary import build_vocabulary

            emb_kw = embedding_kwargs or {}
            build_vocabulary(
                store,
                embedding_provider=emb_kw.get("provider", "local"),
                embedding_model=emb_kw.get("model_name"),
                embedding_api_base=emb_kw.get("api_base"),
                embedding_api_key=emb_kw.get("api_key"),
                session_id=session_id,
            )
            taxonomy_built = True

            build_matrix(store, session_id=session_id)
            matrix_built = True

            version_id = get_next_version(store)
            paper_count = len(store.query("papers"))
            vocab = store.query("vocabulary")
            record_version(
                store,
                version_id,
                paper_count=paper_count,
                param_count=len([v for v in vocab if v["kind"] == "parameter"]),
                principle_count=len([v for v in vocab if v["kind"] == "principle"]),
                slot_count=len([v for v in vocab if v["kind"] == "arch_slot"]),
                variant_count=0,
                pattern_count=len([v for v in vocab if v["kind"] == "agentic_category"]),
                session_id=session_id,
            )
            logger.info("Built taxonomy + matrix")
        except Exception:
            logger.warning("Taxonomy/matrix build failed", exc_info=True)

    # --- Stage 5: Ideate ---
    ideation_report = None
    if run_ideation_flag:
        try:
            if ideate_with_llm:
                from lens.monitor.ideation import run_ideation_with_llm

                ideation_report = await run_ideation_with_llm(store, llm_client)
            else:
                from lens.monitor.ideation import run_ideation

                ideation_report = run_ideation(store)
        except Exception:
            logger.warning("Ideation failed", exc_info=True)

    return {
        "papers_acquired": papers_acquired,
        "papers_enriched": papers_enriched,
        "papers_extracted": papers_extracted,
        "taxonomy_built": taxonomy_built,
        "matrix_built": matrix_built,
        "ideation_report": ideation_report,
    }

"""Monitor pipeline: acquire new papers → extract → ideate.

Runs one monitoring cycle: fetches new papers from arxiv,
extracts knowledge, and optionally runs ideation gap analysis.
"""

from __future__ import annotations

import logging
from typing import Any

from lens.acquire.arxiv import fetch_arxiv_papers
from lens.extract.extractor import extract_papers
from lens.llm.client import LLMClient
from lens.monitor.ideation import run_ideation
from lens.store.store import LensStore
from lens.taxonomy.versioning import get_latest_version

logger = logging.getLogger(__name__)


async def run_monitor_cycle(
    store: LensStore,
    llm_client: LLMClient,
    query: str = "LLM",
    categories: list[str] | None = None,
    max_results: int = 50,
    run_ideation_flag: bool = True,
) -> dict[str, Any]:
    """Run one monitoring cycle."""
    cats = categories or ["cs.CL", "cs.LG", "cs.AI"]

    # Step 1: Acquire
    try:
        papers = await fetch_arxiv_papers(
            query=query,
            categories=cats,
            max_results=max_results,
        )
    except Exception:
        logger.warning("Failed to fetch papers from arxiv")
        papers = []

    # Filter out already-stored papers
    existing_df = store.get_table("papers").to_polars()
    existing_ids = set(existing_df["paper_id"].to_list()) if len(existing_df) > 0 else set()
    new_papers = [p for p in papers if p["paper_id"] not in existing_ids]

    for p in new_papers:
        if "embedding" not in p:
            p["embedding"] = [0.0] * 768

    papers_acquired = len(new_papers)
    if new_papers:
        store.add_papers(new_papers)
        logger.info("Acquired %d new papers", papers_acquired)

    # Step 2: Extract
    papers_extracted = 0
    if papers_acquired > 0:
        papers_extracted = await extract_papers(store, llm_client, concurrency=3)

    # Step 3: Ideation (if taxonomy exists)
    ideation_report = None
    version = get_latest_version(store)
    if version is not None and run_ideation_flag:
        ideation_report = run_ideation(store, taxonomy_version=version)

    return {
        "papers_acquired": papers_acquired,
        "papers_extracted": papers_extracted,
        "ideation_report": ideation_report,
    }

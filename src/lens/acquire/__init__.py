"""LENS paper acquisition pipeline."""

from lens.acquire.arxiv import fetch_arxiv_papers
from lens.acquire.openalex import enrich_with_openalex
from lens.acquire.pdf import ingest_pdf
from lens.acquire.quality import quality_score
from lens.acquire.seed import acquire_seed, load_seed_manifest
from lens.acquire.semantic_scholar import fetch_embedding, fetch_embeddings_batch

__all__ = [
    "acquire_seed",
    "enrich_with_openalex",
    "ingest_pdf",
    "fetch_arxiv_papers",
    "fetch_embedding",
    "fetch_embeddings_batch",
    "load_seed_manifest",
    "quality_score",
]

try:
    from lens.acquire.deepxiv import (
        HAS_DEEPXIV,
        fetch_deepxiv_paper,
        search_deepxiv,
    )

    __all__ += ["HAS_DEEPXIV", "search_deepxiv", "fetch_deepxiv_paper"]
except ImportError:
    pass

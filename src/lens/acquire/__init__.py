"""LENS paper acquisition pipeline."""

from lens.acquire.arxiv import fetch_arxiv_papers
from lens.acquire.openalex import enrich_with_openalex
from lens.acquire.quality import quality_score
from lens.acquire.seed import acquire_seed, load_seed_manifest
from lens.acquire.semantic_scholar import fetch_embedding, fetch_embeddings_batch
from lens.acquire.pdf import extract_text_from_pdf

__all__ = [
    "acquire_seed",
    "enrich_with_openalex",
    "extract_text_from_pdf",
    "fetch_arxiv_papers",
    "fetch_embedding",
    "fetch_embeddings_batch",
    "load_seed_manifest",
    "quality_score",
]

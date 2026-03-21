"""Local PDF file ingestion.

Stores PDF path and basic metadata. Full text extraction is deferred to the
extract phase (Plan 3), where multimodal LLMs read the PDF directly — this
produces better results since the LLM sees formatting, figures, and tables.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def ingest_pdf(pdf_path: Path | str) -> dict[str, Any]:
    """Create a paper dict from a local PDF file.

    Stores the file path for later multimodal LLM extraction.
    Title is derived from the filename.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Paper dict ready for LensStore.add_papers().

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    paper_id = path.stem
    return {
        "paper_id": paper_id,
        "arxiv_id": paper_id,
        "title": paper_id.replace("-", " ").replace("_", " "),
        "abstract": "",  # will be populated during LLM extraction from PDF
        "authors": [],
        "date": "2024-01-01",
        "venue": None,
        "citations": 0,
        "quality_score": 0.0,
        "extraction_status": "pending",
        "embedding": [0.0] * 768,
    }

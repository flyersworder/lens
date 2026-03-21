"""PDF text extraction using PyMuPDF.

Extracts full text from PDF files for seed papers and local file ingestion.
"""
from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf


def extract_text_from_pdf(pdf_path: Path | str) -> str:
    """Extract all text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(path))
    try:
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        return "\n".join(text_parts).strip()
    finally:
        doc.close()

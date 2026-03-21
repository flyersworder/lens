"""Tests for local PDF file ingestion."""

from pathlib import Path

import pytest


def test_ingest_pdf(tmp_path):
    from lens.acquire.pdf import ingest_pdf

    # Create a dummy PDF file (content doesn't matter — we just store the path)
    pdf = tmp_path / "attention-is-all-you-need.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    paper = ingest_pdf(pdf)
    assert paper["paper_id"] == "attention-is-all-you-need"
    assert paper["title"] == "attention is all you need"
    assert paper["extraction_status"] == "pending"
    assert paper["abstract"] == ""
    assert len(paper["embedding"]) == 768


def test_ingest_pdf_returns_dict(tmp_path):
    from lens.acquire.pdf import ingest_pdf

    pdf = tmp_path / "test-paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 dummy")

    paper = ingest_pdf(pdf)
    assert isinstance(paper, dict)
    assert "paper_id" in paper
    assert "arxiv_id" in paper


def test_ingest_pdf_nonexistent_file():
    from lens.acquire.pdf import ingest_pdf

    with pytest.raises(FileNotFoundError):
        ingest_pdf(Path("/nonexistent/file.pdf"))

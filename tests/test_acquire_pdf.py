"""Tests for PDF text extraction."""
import pytest
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_extract_text_from_pdf():
    from lens.acquire.pdf import extract_text_from_pdf
    text = extract_text_from_pdf(FIXTURE_DIR / "sample.pdf")
    assert "Test Paper Title" in text or len(text) > 0


def test_extract_text_returns_string():
    from lens.acquire.pdf import extract_text_from_pdf
    text = extract_text_from_pdf(FIXTURE_DIR / "sample.pdf")
    assert isinstance(text, str)


def test_extract_text_nonexistent_file():
    from lens.acquire.pdf import extract_text_from_pdf
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(Path("/nonexistent/file.pdf"))

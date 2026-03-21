"""Shared test fixtures."""

import pytest

from lens.store.store import LensStore


@pytest.fixture
def store(tmp_path):
    """Create a LensStore backed by a temporary directory."""
    return LensStore(str(tmp_path / "test.lance"))


@pytest.fixture
def sample_paper_data():
    """Sample paper data as a dict (for table.add())."""
    return {
        "paper_id": "2401.12345",
        "title": "Attention Is All You Need",
        "abstract": "We propose a new architecture...",
        "authors": ["Vaswani", "Shazeer"],
        "venue": "NeurIPS",
        "date": "2017-06-12",
        "arxiv_id": "1706.03762",
        "citations": 100000,
        "quality_score": 0.95,
        "extraction_status": "pending",
        "embedding": [0.1] * 768,
    }

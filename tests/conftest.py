"""Shared test fixtures."""

import pytest

from lens.store.models import EMBEDDING_DIM


@pytest.fixture
def store(tmp_path):
    """Create a LensStore backed by a temporary SQLite database."""
    from lens.store.store import LensStore

    s = LensStore(str(tmp_path / "test.db"))
    s.init_tables()
    return s


@pytest.fixture
def sample_paper_data():
    """Sample paper data as a dict."""
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
        "embedding": [0.1] * EMBEDDING_DIM,
    }

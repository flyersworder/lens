"""Tests for the taxonomy pipeline."""

import numpy as np


def test_embed_strings_returns_array():
    from lens.taxonomy.embedder import embed_strings

    embeddings = embed_strings(["inference latency", "model accuracy"])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0


def test_embed_strings_empty():
    from lens.taxonomy.embedder import embed_strings

    embeddings = embed_strings([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 0


def test_embed_strings_deterministic():
    from lens.taxonomy.embedder import embed_strings

    e1 = embed_strings(["test string"])
    e2 = embed_strings(["test string"])
    np.testing.assert_array_almost_equal(e1, e2)

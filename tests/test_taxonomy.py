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


def test_cluster_embeddings():
    from lens.taxonomy.clusterer import cluster_embeddings

    rng = np.random.RandomState(42)
    cluster_a = rng.randn(10, 50) + np.array([5.0] + [0.0] * 49)
    cluster_b = rng.randn(10, 50) + np.array([0.0, 5.0] + [0.0] * 48)
    embeddings = np.vstack([cluster_a, cluster_b])

    labels = cluster_embeddings(embeddings, min_cluster_size=3)
    assert len(labels) == 20
    unique = set(labels)
    unique.discard(-1)
    assert len(unique) >= 2


def test_cluster_embeddings_small_dataset():
    from lens.taxonomy.clusterer import cluster_embeddings

    embeddings = np.random.randn(5, 50)
    labels = cluster_embeddings(embeddings, min_cluster_size=2)
    assert len(labels) == 5


def test_cluster_embeddings_fallback_to_kmeans():
    from lens.taxonomy.clusterer import cluster_embeddings

    # All identical points — HDBSCAN assigns all to noise, should fallback
    embeddings = np.ones((20, 50))
    labels = cluster_embeddings(embeddings, min_cluster_size=3, target_clusters=3)
    assert len(labels) == 20
    unique = set(labels)
    unique.discard(-1)
    assert len(unique) >= 1

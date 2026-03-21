"""HDBSCAN clustering for taxonomy discovery.

Clusters raw extraction string embeddings into groups.
Falls back to KMeans if HDBSCAN produces degenerate results (all noise
or single cluster), as required by the spec.
"""

from __future__ import annotations

import logging

import hdbscan
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    target_clusters: int | None = None,
) -> list[int]:
    """Cluster embeddings using HDBSCAN with KMeans fallback.

    Args:
        embeddings: Array of shape (n_samples, n_features).
        min_cluster_size: HDBSCAN min_cluster_size parameter.
        target_clusters: Target number of clusters for KMeans fallback.

    Returns:
        List of cluster labels (-1 for noise).
    """
    if len(embeddings) < min_cluster_size:
        return [-1] * len(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings).tolist()

    # Check for degenerate results
    unique = set(labels)
    unique.discard(-1)
    if len(unique) <= 1:
        logger.warning(
            "HDBSCAN produced %d clusters, falling back to KMeans",
            len(unique),
        )
        k = target_clusters or max(2, len(embeddings) // 5)
        k = min(k, len(embeddings))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings).tolist()

    return labels

"""Embed raw extraction strings for clustering.

Uses sentence-transformers. Falls back to lightweight model if
scientific model unavailable.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}

MODELS = [
    "allenai-specter2",  # scientific embeddings (768d), preferred for research papers
    "all-MiniLM-L6-v2",  # lightweight fallback (384d), always available
]


def _get_model(model_name: str | None = None) -> SentenceTransformer:
    """Load and cache a sentence-transformers model.

    When *model_name* is ``None``, tries MODELS in order and falls back to
    the lightweight model.  When a specific *model_name* is given, only that
    model is attempted (no fallback).
    """
    if model_name and model_name in _model_cache:
        return _model_cache[model_name]

    for name in [model_name] if model_name else MODELS:
        if name in _model_cache:
            return _model_cache[name]
        try:
            model = SentenceTransformer(name)
            _model_cache[name] = model
            logger.info("Loaded embedding model: %s", name)
            return model
        except Exception:
            logger.warning("Failed to load model %s, trying next", name)
            continue

    raise RuntimeError("No embedding model available")


def embed_strings(
    strings: list[str],
    model_name: str | None = None,
) -> np.ndarray:
    """Embed a list of strings into dense vectors.

    Returns numpy array of shape (len(strings), embedding_dim).
    Returns empty array with shape (0,) if strings is empty.
    """
    if not strings:
        return np.array([])

    model = _get_model(model_name)
    embeddings = model.encode(strings, show_progress_bar=False)
    return np.array(embeddings)

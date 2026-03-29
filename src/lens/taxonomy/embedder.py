"""Embed raw extraction strings for clustering.

Supports two providers:
- ``local``: sentence-transformers (SPECTER2 / MiniLM fallback). Free, offline.
- ``cloud``: litellm or openai SDK embedding API. Fast, scalable.

Provider is selected via config ``taxonomy.embedding_provider``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lens.store.models import EMBEDDING_DIM

logger = logging.getLogger(__name__)

try:
    import litellm  # ty: ignore[unresolved-import]

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

# ---------------------------------------------------------------------------
# Local provider (sentence-transformers)
# ---------------------------------------------------------------------------

_model_cache: dict[str, Any] = {}

LOCAL_MODELS = [
    "allenai-specter2",  # scientific embeddings (768d), preferred for research papers
    "all-MiniLM-L6-v2",  # lightweight fallback (384d), always available
]


def _get_local_model(model_name: str | None = None) -> Any:
    """Load and cache a sentence-transformers model.

    When *model_name* is ``None``, tries LOCAL_MODELS in order and falls back
    to the lightweight model. When a specific *model_name* is given, only that
    model is attempted (no fallback).
    """
    from sentence_transformers import SentenceTransformer

    if model_name and model_name in _model_cache:
        return _model_cache[model_name]

    for name in [model_name] if model_name else LOCAL_MODELS:
        if name in _model_cache:
            return _model_cache[name]
        try:
            model = SentenceTransformer(name)
            _model_cache[name] = model
            logger.info("Loaded local embedding model: %s", name)
            return model
        except Exception:
            logger.warning("Failed to load model %s, trying next", name)
            continue

    raise RuntimeError("No local embedding model available")


def _embed_local(strings: list[str], model_name: str | None = None) -> np.ndarray:
    """Embed strings using a local sentence-transformers model."""
    model = _get_local_model(model_name)
    embeddings = model.encode(strings, show_progress_bar=False)
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Cloud provider (litellm with openai SDK fallback)
# ---------------------------------------------------------------------------


async def _embed_cloud_async(
    strings: list[str],
    model: str = "openai/text-embedding-3-small",
    dimensions: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> np.ndarray:
    """Embed strings using a cloud embedding API.

    Uses litellm if installed, otherwise falls back to the openai SDK
    (requires api_base pointing to an OpenAI-compatible endpoint).
    """
    batch_size = 512
    all_embeddings: list[list[float]] = []

    if HAS_LITELLM:
        kwargs: dict[str, Any] = {}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        if api_base:
            kwargs["api_base"] = api_base
        if api_key:
            kwargs["api_key"] = api_key

        for i in range(0, len(strings), batch_size):
            batch = strings[i : i + batch_size]
            response = await litellm.aembedding(model=model, input=batch, **kwargs)
            for item in response.data:
                all_embeddings.append(item["embedding"])
    else:
        import openai

        client_kwargs: dict[str, Any] = {}
        if api_base:
            client_kwargs["base_url"] = api_base
        if api_key:
            client_kwargs["api_key"] = api_key
        client = openai.AsyncOpenAI(**client_kwargs)

        oai_kwargs: dict[str, Any] = {}
        if dimensions is not None:
            oai_kwargs["dimensions"] = dimensions

        for i in range(0, len(strings), batch_size):
            batch = strings[i : i + batch_size]
            response = await client.embeddings.create(model=model, input=batch, **oai_kwargs)
            for item in response.data:
                all_embeddings.append(item.embedding)

    return np.array(all_embeddings)


def _embed_cloud(
    strings: list[str],
    model: str = "openai/text-embedding-3-small",
    dimensions: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> np.ndarray:
    """Synchronous cloud embedding using the openai sync client.

    Uses the synchronous OpenAI client to avoid event loop conflicts
    when called from within async code (e.g., build_taxonomy).
    """
    import openai

    client_kwargs: dict[str, Any] = {}
    if api_base:
        client_kwargs["base_url"] = api_base
    if api_key:
        client_kwargs["api_key"] = api_key
    client = openai.OpenAI(**client_kwargs)

    oai_kwargs: dict[str, Any] = {}
    if dimensions is not None:
        oai_kwargs["dimensions"] = dimensions

    batch_size = 512
    all_embeddings: list[list[float]] = []

    for i in range(0, len(strings), batch_size):
        batch = strings[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch, **oai_kwargs)
        for item in response.data:
            all_embeddings.append(item.embedding)

    return np.array(all_embeddings)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_strings(
    strings: list[str],
    model_name: str | None = None,
    provider: str = "local",
    dimensions: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> np.ndarray:
    """Embed a list of strings into dense vectors.

    Args:
        strings: Texts to embed.
        model_name: Model name. For local: sentence-transformers model name.
            For cloud: litellm model string or OpenAI model name.
        provider: ``"local"`` for sentence-transformers, ``"cloud"`` for cloud API.
        dimensions: Target dimensions for cloud models that support it.
        api_base: Optional API base URL for cloud provider (OpenAI-compatible endpoint).
        api_key: Optional API key for cloud provider.

    Returns:
        numpy array of shape ``(len(strings), embedding_dim)``.
        Returns empty array with shape ``(0,)`` if strings is empty.
    """
    if not strings:
        return np.array([])

    if provider == "cloud":
        raw = _embed_cloud(
            strings,
            model=model_name or "openai/text-embedding-3-small",
            dimensions=dimensions,
            api_base=api_base,
            api_key=api_key,
        )
    else:
        raw = _embed_local(strings, model_name=model_name)

    # Normalize dimensions to EMBEDDING_DIM
    if len(raw) > 0 and raw.shape[1] != EMBEDDING_DIM:
        if raw.shape[1] < EMBEDDING_DIM:
            padding = np.zeros((raw.shape[0], EMBEDDING_DIM - raw.shape[1]))
            raw = np.hstack([raw, padding])
        else:
            raw = raw[:, :EMBEDDING_DIM]

    return raw

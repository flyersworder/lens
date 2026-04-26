"""FastAPI handlers for the LENS public web tier.

Single-app, multi-route layout — the shape Vercel auto-detects when a
top-level ``app: FastAPI`` lives at ``api/index.py``. Each handler is
a thin wrapper around the existing ``serve/*`` modules, accepting
either backend through the :class:`ReadableStore` protocol.

Module globals (``_store`` and ``_llm``) are constructed at import
time so a warm Vercel instance pays the connection cost once and
reuses it across requests. Cold starts open a Turso libSQL HTTP
connection (~50–150 ms typical) and a litellm/openai async client.

Endpoints:

* ``POST /api/analyze``  — tradeoff / architecture / agentic resolution
* ``POST /api/explain``  — concept disambiguation + narrative
* ``GET  /api/search``   — paper search (FTS + vector)
* ``GET  /api/health``   — readiness probe (no LLM, no DB read)

Configuration is via environment variables:

* ``TURSO_DATABASE_URL`` + ``TURSO_AUTH_TOKEN`` — production libSQL
* ``OPENROUTER_API_KEY`` — used for both LLM completions and embeddings
* ``LENS_LLM_MODEL``    — default ``deepseek/deepseek-v4-flash``
* ``LENS_EMBED_MODEL``  — default ``openai/text-embedding-3-small``
* ``LENS_LOCAL_DB``     — fallback path used when ``TURSO_*`` is unset
  (defaults to ``~/.lens/data/lens.db``); intended for local dev only

Rate limiting and response caching live in ``api/_middleware.py``
(planned next commit) — this module is intentionally minimal so the
storage + LLM wiring can be reviewed in isolation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from lens.llm.client import LLMClient
from lens.serve.analyzer import (
    analyze,
    analyze_agentic,
    analyze_architecture,
)
from lens.serve.explainer import explain
from lens.serve.explorer import search_papers as serve_search_papers
from lens.store.protocols import ReadableStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-global lazy singletons
# ---------------------------------------------------------------------------


def _build_store() -> ReadableStore:
    """Pick the storage backend based on environment.

    ``TURSO_DATABASE_URL`` set → libSQL TursoStore (production).
    Otherwise → local sqlite-vec LensStore (local dev / docker container
    with a pre-baked DB).
    """
    turso_url = os.environ.get("TURSO_DATABASE_URL")
    turso_token = os.environ.get("TURSO_AUTH_TOKEN")
    if turso_url and turso_token:
        from lens.store.turso_store import TursoStore  # noqa: PLC0415

        logger.info("api: using TursoStore (libSQL)")
        return TursoStore(url=turso_url, auth_token=turso_token)

    local_path = os.environ.get("LENS_LOCAL_DB", str(Path.home() / ".lens" / "data" / "lens.db"))
    if not Path(local_path).exists():
        raise RuntimeError(
            f"No storage backend configured. Set TURSO_DATABASE_URL + "
            f"TURSO_AUTH_TOKEN for production, or place a local DB at "
            f"{local_path} (override via LENS_LOCAL_DB)."
        )
    from lens.store.store import LensStore  # noqa: PLC0415

    logger.info("api: using LensStore (sqlite-vec) at %s", local_path)
    return LensStore(local_path)


def _build_llm() -> LLMClient:
    """Construct the LLM client from env, defaulting to deepseek-v4-flash."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY must be set. Top up credits at "
            "https://openrouter.ai/credits if needed."
        )
    return LLMClient(
        model=os.environ.get("LENS_LLM_MODEL", "deepseek/deepseek-v4-flash"),
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def _build_embed_kwargs() -> dict[str, Any]:
    """kwargs forwarded to ``embed_strings`` for runtime query embedding."""
    return {
        "provider": "cloud",
        "model_name": os.environ.get("LENS_EMBED_MODEL", "openai/text-embedding-3-small"),
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
    }


# Lazy singletons — evaluated on first use rather than at import time so
# a misconfigured environment fails the affected request, not the whole
# function instance. This also keeps the test path importable without
# real credentials.
_store: ReadableStore | None = None
_llm: LLMClient | None = None
_embed_kwargs: dict[str, Any] | None = None


def get_store() -> ReadableStore:
    global _store
    if _store is None:
        _store = _build_store()
    return _store


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


def get_embed_kwargs() -> dict[str, Any]:
    global _embed_kwargs
    if _embed_kwargs is None:
        _embed_kwargs = _build_embed_kwargs()
    return _embed_kwargs


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


AnalysisType = Literal["tradeoff", "architecture", "agentic"]


class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    type: AnalysisType = "tradeoff"


class ExplainRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    focus: str | None = Field(default=None, max_length=64)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(
    title="LENS API",
    description="LLM Engineering Navigation System — public read-only API.",
    version="0.10.1",
)


# Annotated[T, Depends(...)] is the modern FastAPI dependency-injection
# style — equivalent to `T = Depends(...)` but ruff-clean (B008 only
# flags the older positional-default form).
StoreDep = Annotated[ReadableStore, Depends(get_store)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
EmbedKwargsDep = Annotated[dict[str, Any], Depends(get_embed_kwargs)]


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe. Does not touch the DB or the LLM."""
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_endpoint(
    req: AnalyzeRequest,
    store: StoreDep,
    llm: LLMDep,
) -> dict[str, Any]:
    """Analyze a tradeoff / architecture / agentic query.

    Dispatches to the right ``serve/analyzer`` entry point based on
    ``req.type``. All three return a JSON-serializable dict.
    """
    try:
        if req.type == "tradeoff":
            return await analyze(req.query, store, llm)
        if req.type == "architecture":
            return await analyze_architecture(req.query, store, llm)
        # agentic — Literal exhaustiveness guarantees this is the last case.
        return await analyze_agentic(req.query, store, llm)
    except Exception as e:
        logger.exception("analyze failed for query=%r type=%s", req.query, req.type)
        raise HTTPException(status_code=500, detail=f"analyze failed: {e}") from e


@app.post("/api/explain")
async def explain_endpoint(
    req: ExplainRequest,
    store: StoreDep,
    llm: LLMDep,
    embed_kwargs: EmbedKwargsDep,
) -> dict[str, Any]:
    """Disambiguate a concept and synthesize an explanation.

    Returns ``{"result": null}`` when the corpus has no candidate; the
    frontend should render a "no match" state rather than a 404.
    """
    try:
        result = await explain(
            req.query,
            store,
            llm,
            focus=req.focus,
            embedding_kwargs=embed_kwargs,
        )
    except Exception as e:
        logger.exception("explain failed for query=%r", req.query)
        raise HTTPException(status_code=500, detail=f"explain failed: {e}") from e

    return {"result": result.model_dump() if result is not None else None}


@app.get("/api/search")
def search_endpoint(
    store: StoreDep,
    embed_kwargs: EmbedKwargsDep,
    q: str | None = Query(default=None, max_length=200),
    author: str | None = Query(default=None, max_length=100),
    venue: str | None = Query(default=None, max_length=100),
    after: str | None = Query(default=None, max_length=10),
    before: str | None = Query(default=None, max_length=10),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    """Search papers by hybrid FTS + vector + metadata filters.

    The serve layer's ``search_papers`` already formats results for
    display; we wrap the list in a top-level dict so future fields
    (cursor, total_count, etc.) can be added without changing the
    response envelope.
    """
    try:
        results = serve_search_papers(
            store,
            query=q,
            author=author,
            venue=venue,
            after=after,
            before=before,
            limit=limit,
            embedding_kwargs=embed_kwargs,
        )
    except Exception as e:
        logger.exception("search failed for q=%r", q)
        raise HTTPException(status_code=500, detail=f"search failed: {e}") from e
    return {"results": results, "count": len(results)}

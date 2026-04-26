"""FastAPI handlers for the LENS public web tier.

Single-app, multi-route layout — the shape Vercel auto-detects when a
top-level ``app: FastAPI`` lives at ``api/index.py``. Each handler is
a thin wrapper around the existing ``serve/*`` modules, accepting
either backend through the :class:`ReadableStore` protocol.

Module globals (the ``get_store`` / ``get_llm`` / ``get_embed_kwargs``
``lru_cache``-d singletons) are constructed on first request so a
warm Vercel instance pays the connection cost once and reuses it
across requests. Cold starts open a Turso libSQL HTTP connection
(~50–150 ms typical) and a litellm/openai async client. Vercel
function teardown is a hard kill (no graceful shutdown), so the
singletons rely on the underlying libSQL client being stateless
HTTP — there's no connection to drain.

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
* ``LENS_CORS_ORIGINS`` — comma-separated origin allowlist
  (e.g. ``"https://lens.example,https://staging.lens.example"``).
  Defaults to no CORS (same-origin only) if unset.

Rate limiting and response caching live in ``api/_middleware.py``
(planned next commit) — this module is intentionally minimal so the
storage + LLM wiring can be reviewed in isolation.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

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
    """kwargs forwarded to ``embed_strings`` for runtime query embedding.

    Fails loud when ``OPENROUTER_API_KEY`` is missing instead of letting
    the embed call silently degrade — see the equivalent check in
    ``_build_llm``.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set for runtime query embedding.")
    return {
        "provider": "cloud",
        "model_name": os.environ.get("LENS_EMBED_MODEL", "openai/text-embedding-3-small"),
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": api_key,
    }


# Cache-based singletons via lru_cache. `lru_cache(maxsize=1)` is
# atomic under concurrent access — two cold-start requests that race
# the constructor will both get the same instance, no duplicate
# initialization. Keeps the test path importable without credentials
# because the build call only fires on first request.
@lru_cache(maxsize=1)
def get_store() -> ReadableStore:
    return _build_store()


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return _build_llm()


@lru_cache(maxsize=1)
def get_embed_kwargs() -> dict[str, Any]:
    return _build_embed_kwargs()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


AnalysisType = Literal["tradeoff", "architecture", "agentic"]


def _strip_or_reject(v: str) -> str:
    """Strip whitespace; reject if the result is empty.

    Pydantic's ``min_length=1`` only checks raw length, so a query of
    ``"   "`` would pass — this validator catches that.
    """
    stripped = v.strip()
    if not stripped:
        raise ValueError("must contain non-whitespace characters")
    return stripped


class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    type: AnalysisType = "tradeoff"

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        return _strip_or_reject(v)


class ExplainRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    focus: str | None = Field(default=None, max_length=64)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        return _strip_or_reject(v)


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


try:
    _LENS_VERSION = _pkg_version("lens-research")
except PackageNotFoundError:  # pragma: no cover — local source checkout edge case
    _LENS_VERSION = "0.0.0+unknown"

app = FastAPI(
    title="LENS API",
    description="LLM Engineering Navigation System — public read-only API.",
    version=_LENS_VERSION,
)


# CORS allowlist driven by env. Empty / unset → no CORS headers
# (same-origin only). Set ``LENS_CORS_ORIGINS`` to a comma-separated
# list of origins (e.g. ``"https://lens.example,https://staging..."``).
_cors_env = os.environ.get("LENS_CORS_ORIGINS", "").strip()
if _cors_env:
    _cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,  # public read-only API; no cookies/auth
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
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
        # match/case enforces exhaustiveness against the AnalysisType
        # Literal — adding a new variant without a case here is a
        # type-checker error rather than a silent fallthrough.
        match req.type:
            case "tradeoff":
                return await analyze(req.query, store, llm)
            case "architecture":
                return await analyze_architecture(req.query, store, llm)
            case "agentic":
                return await analyze_agentic(req.query, store, llm)
    except Exception as e:
        logger.exception("analyze failed for query=%r type=%s", req.query, req.type)
        # Generic detail to avoid leaking internals (stack frames, model
        # names, even auth-token-bearing URLs in libSQL error messages).
        # The full error is in server logs.
        raise HTTPException(status_code=500, detail="analyze failed") from e


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
        raise HTTPException(status_code=500, detail="explain failed") from e

    return {"result": result.model_dump() if result is not None else None}


def _validate_date_param(name: str, value: str | None) -> None:
    """Reject non-ISO-8601 date inputs at the API layer."""
    if value is not None and not _DATE_RE.match(value):
        raise HTTPException(
            status_code=422,
            detail=f"{name} must be ISO-8601 date (YYYY-MM-DD)",
        )


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
    _validate_date_param("after", after)
    _validate_date_param("before", before)
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
        raise HTTPException(status_code=500, detail="search failed") from e
    return {"results": results, "count": len(results)}

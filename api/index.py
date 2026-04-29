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

import contextlib
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncGenerator
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from lens.llm.client import LLMClient
from lens.serve.analyzer import (
    analyze,
    analyze_agentic,
    analyze_architecture,
)
from lens.serve.explainer import explain_stream
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


TrackEvent = Literal[
    "view_home",
    "view_analyze",
    "view_explain",  # /explain landing page
    "view_explain_concept",  # /explain/[concept] page-load (before resolution)
    "search",
    "analyze",
    "explain",  # /explain/[concept] resolution succeeded
]


class TrackRequest(BaseModel):
    event: TrackEvent
    # Optional 200-char string for the originating query / path slug;
    # we hash it on the server before persisting so we don't store
    # raw user inputs (privacy + small storage footprint).
    query: str | None = Field(default=None, max_length=200)


# ---------------------------------------------------------------------------
# Usage tracking — write path on Turso
# ---------------------------------------------------------------------------
#
# TursoStore is read-only by design. For tracking we punch a small hole:
# a dedicated libsql client just for INSERTs into ``usage_events``. The
# table is created lazily on first track call (idempotent CREATE TABLE
# IF NOT EXISTS), so a fresh ``lens-prod`` doesn't need a manual
# migration step. Local dev (no TURSO_*) gets a no-op tracker — we
# don't want test runs polluting a usage table on disk.


_USAGE_DDL = """CREATE TABLE IF NOT EXISTS usage_events (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    event TEXT NOT NULL,
    query_hash TEXT
)"""


def _track_origin_allowed(request: Request) -> bool:
    """Return True if the request's Origin (or Referer) is acceptable
    for `/api/track`.

    Behaviour:

    * ``LENS_TRACK_ORIGINS`` set (comma-separated)  → strict allowlist;
      reject anything else. Use this in production.
    * ``LENS_CORS_ORIGINS``   set (comma-separated) → reuse it as the
      track allowlist so a single env controls both.
    * Neither set → permissive (dev mode). Without an allowlist there
      is no way to distinguish a legitimate caller from a script, so
      we don't pretend to enforce.

    The check looks at ``Origin`` first (sent by browsers on
    cross-origin requests), then falls back to a prefix match against
    ``Referer`` (some clients elide ``Origin`` for same-origin POSTs).
    Same-origin browser requests typically omit ``Origin`` entirely;
    we accept those as long as the allowlist is non-empty and the
    ``Referer`` matches.
    """
    raw = (
        os.environ.get("LENS_TRACK_ORIGINS") or os.environ.get("LENS_CORS_ORIGINS") or ""
    ).strip()
    if not raw:
        return True
    allow = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    origin = (request.headers.get("origin") or "").rstrip("/")
    if origin and origin in allow:
        return True
    referer = request.headers.get("referer") or ""
    return any(referer.startswith(o + "/") or referer == o for o in allow)


# Hand-rolled "memoize on success only" cell. ``lru_cache`` would memoize
# the failure paths too — we don't want a half-broken client (DDL failed)
# or an early-cold-start ``None`` (env not yet present) pinned for the
# instance's lifetime.
_WRITER_CELL: dict[str, Any] = {}


def _get_writer():  # type: ignore[no-untyped-def]
    """Return a libsql write client, or ``None`` when tracking is unavailable.

    Returns the underlying ``libsql_client.Client`` rather than wrapping
    it — the only operation we need is ``execute(sql, params)``. The DDL
    fires once per successful init and is idempotent.

    Memoization rules:

    * Successful client + DDL → cached for the warm instance lifetime.
    * Missing ``TURSO_*`` env → returns ``None`` *without* caching, so a
      late-arriving env (test harness, dev server reload) recovers on
      the next call.
    * DDL raise → close the client, return ``None``, do *not* cache.
      Next call retries from scratch, matching the inline contract
      ("don't cache a half-broken client").
    """
    cached = _WRITER_CELL.get("client")
    if cached is not None:
        return cached

    turso_url = os.environ.get("TURSO_DATABASE_URL")
    turso_token = os.environ.get("TURSO_AUTH_TOKEN")
    if not (turso_url and turso_token):
        logger.info("track: no TURSO_* env, tracking is a no-op")
        return None

    import libsql_client  # noqa: PLC0415

    url = turso_url
    if url.startswith("libsql://"):
        url = "https://" + url[len("libsql://") :]
    client = libsql_client.create_client_sync(url=url, auth_token=turso_token)
    try:
        client.execute(_USAGE_DDL)
    except Exception:
        logger.exception("track: failed to ensure usage_events table")
        with contextlib.suppress(Exception):  # best-effort cleanup
            client.close()
        return None

    _WRITER_CELL["client"] = client
    return client


def _hash_query(q: str | None) -> str | None:
    """Stable short hash of a query string for usage analytics.

    Truncated SHA-256 — 16 hex chars is plenty for de-duping common
    queries without giving us a way to reconstruct the input.
    """
    if not q:
        return None
    return hashlib.sha256(q.strip().lower().encode("utf-8")).hexdigest()[:16]


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
) -> StreamingResponse:
    """Stream a concept explanation as NDJSON.

    Each line is one JSON object with discriminator ``t``:

    * ``{"t": "meta", ...}`` once, after candidate resolution + graph walk.
    * ``{"t": "token", "v": "..."}`` many, one per LLM delta.
    * ``{"t": "empty"}`` once when the corpus has no candidate (in
      place of "meta"); the page renders a "no match" state.
    * ``{"t": "error", "msg": "..."}`` on internal failure.

    No JSON envelope — clients read line-by-line. ``X-Accel-Buffering:
    no`` keeps Vercel's edge proxy from holding the chunked body.
    """

    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            async for event in explain_stream(
                req.query,
                store,
                llm,
                focus=req.focus,
                embedding_kwargs=embed_kwargs,
            ):
                yield (json.dumps(event) + "\n").encode("utf-8")
        except Exception:
            logger.exception("explain stream failed for query=%r", req.query)
            yield (json.dumps({"t": "error", "msg": "explain failed"}) + "\n").encode("utf-8")

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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


# Hand-rolled TTL cache (one slot) for /api/stats. We can't use
# ``lru_cache`` here because the cached function would have to call
# ``get_store()`` directly, defeating ``app.dependency_overrides`` in
# tests. Instead we key on a 5-minute time bucket and pass the
# DI-resolved store through.
_STATS_TTL_SECONDS = 300
_stats_cache: dict[str, Any] = {"bucket": -1, "value": None}


def _stats_bucket() -> int:
    """Cache bucket key — same value for 5 minutes, then rolls over."""
    return int(time.time() // _STATS_TTL_SECONDS)


@app.get("/api/stats")
def stats_endpoint(store: StoreDep) -> dict[str, Any]:
    """Aggregate corpus stats for the public landing page.

    Cached for 5 minutes per warm instance — counts barely move
    between weekly monitor runs, and a cold start refreshes anyway.
    The cache is *per-store-identity-and-bucket* so a test using
    ``app.dependency_overrides[get_store]`` gets a fresh aggregation
    instead of a stale entry computed against the production store.
    """
    bucket = _stats_bucket()
    cache_key = (bucket, id(store))
    if _stats_cache.get("key") == cache_key and _stats_cache.get("value") is not None:
        return _stats_cache["value"]
    value = _compute_stats(store)
    _stats_cache["key"] = cache_key
    _stats_cache["value"] = value
    return value


def _compute_stats(store: ReadableStore) -> dict[str, Any]:
    """One-shot aggregation across the read-only corpus tables.

    Each ``query_sql`` is independent — failures degrade to ``None``
    so a missing table (e.g. fresh DB) doesn't 500 the whole stats
    response. The frontend treats ``None`` as "—".
    """

    def _count(sql: str) -> int | None:
        try:
            rows = store.query_sql(sql)
            if rows and "n" in rows[0]:
                return int(rows[0]["n"])
        except Exception:
            logger.exception("stats: query failed: %s", sql)
        return None

    def _kind_counts() -> dict[str, int] | None:
        """Return per-kind counts, or ``None`` when the aggregation fails.

        We deliberately distinguish ``None`` (query failed / table
        missing) from ``{}`` (table exists but empty) so the response
        shape is uniformly null-vs-int — see the ``vocabulary`` block
        below where the failure case sets *all* fields to ``None``.
        """
        try:
            rows = store.query_sql("SELECT kind, COUNT(*) AS n FROM vocabulary GROUP BY kind")
            return {str(r["kind"]): int(r["n"]) for r in rows}
        except Exception:
            logger.exception("stats: vocabulary kind aggregation failed")
            return None

    vocab = _kind_counts()
    if vocab is None:
        # Failure → propagate ``None`` to every kind field. Avoids the
        # confusing UI state where ``total: None`` ("—") sits next to
        # ``parameter: 0`` ("0") for the same root cause.
        vocabulary_block: dict[str, int | None] = {
            "total": None,
            "parameter": None,
            "principle": None,
            "arch_slot": None,
            "agentic": None,
        }
    else:
        vocabulary_block = {
            "total": sum(vocab.values()),
            "parameter": vocab.get("parameter", 0),
            "principle": vocab.get("principle", 0),
            "arch_slot": vocab.get("arch_slot", 0),
            "agentic": vocab.get("agentic", 0),
        }
    return {
        "papers": _count("SELECT COUNT(*) AS n FROM papers"),
        "vocabulary": vocabulary_block,
        "matrix_cells": _count("SELECT COUNT(*) AS n FROM matrix_cells"),
        "tradeoffs": _count("SELECT COUNT(*) AS n FROM tradeoff_extractions"),
        "taxonomy_version": _count("SELECT MAX(version_id) AS n FROM taxonomy_versions"),
    }


@app.post("/api/track")
def track_endpoint(req: TrackRequest, request: Request) -> dict[str, str]:
    """Best-effort write of a single usage event.

    Always returns 200 so the client doesn't have to handle errors
    on a fire-and-forget call. Failures land in server logs.

    Cross-origin requests are silently ignored when an allowlist is
    configured (``LENS_TRACK_ORIGINS`` or ``LENS_CORS_ORIGINS``).
    Returning 200 with ``{"status":"ignored"}`` rather than 403 is
    deliberate: a polite no-op is harder to weaponize for probing than
    a distinct error code, and keeps the public response surface flat.
    """
    if not _track_origin_allowed(request):
        return {"status": "ignored"}

    writer = _get_writer()
    if writer is None:
        return {"status": "noop"}
    try:
        writer.execute(
            "INSERT INTO usage_events (ts, event, query_hash) VALUES (?, ?, ?)",
            [int(time.time()), req.event, _hash_query(req.query)],
        )
    except Exception:
        logger.exception("track: insert failed event=%s", req.event)
        return {"status": "error"}
    return {"status": "ok"}


# 60-second TTL cache for usage-summary aggregation. Same pattern as
# /api/stats: cheap but cuts the libsql round-trip for repeat hits.
_USAGE_SUMMARY_TTL_SECONDS = 60
_usage_cache: dict[str, Any] = {"bucket": -1, "value": None}


@app.get("/api/usage-summary")
def usage_summary_endpoint() -> dict[str, Any]:
    """Aggregate counts per event type from ``usage_events``.

    Read path uses the same libsql client as the write path. Returns
    ``{"events": [], "total": 0}`` when tracking isn't configured (no
    ``TURSO_*`` env) or the table is empty, so the public ``/usage``
    page can render uniformly without special-casing those states.
    """
    bucket = int(time.time() // _USAGE_SUMMARY_TTL_SECONDS)
    if _usage_cache.get("bucket") == bucket and _usage_cache.get("value") is not None:
        return _usage_cache["value"]

    writer = _get_writer()
    if writer is None:
        value = {"events": [], "total": 0, "first_seen": None, "last_seen": None}
        _usage_cache["bucket"] = bucket
        _usage_cache["value"] = value
        return value

    try:
        rows = writer.execute(
            "SELECT event, COUNT(*) AS n FROM usage_events GROUP BY event ORDER BY n DESC"
        ).rows
        events = [{"event": str(r[0]), "count": int(r[1])} for r in rows]
        total = sum(e["count"] for e in events)
        bounds = writer.execute(
            "SELECT MIN(ts) AS first_ts, MAX(ts) AS last_ts FROM usage_events"
        ).rows
        first_seen = int(bounds[0][0]) if bounds and bounds[0][0] is not None else None
        last_seen = int(bounds[0][1]) if bounds and bounds[0][1] is not None else None
    except Exception:
        logger.exception("usage-summary: aggregation failed")
        return {"events": [], "total": 0, "first_seen": None, "last_seen": None}

    value = {
        "events": events,
        "total": total,
        "first_seen": first_seen,
        "last_seen": last_seen,
    }
    _usage_cache["bucket"] = bucket
    _usage_cache["value"] = value
    return value

"""Unit tests for the Vercel FastAPI handlers in ``api/index.py``.

Uses FastAPI's ``TestClient`` plus ``app.dependency_overrides`` to
inject fake stores and a fake LLM client. These tests deliberately
do NOT hit OpenRouter or Turso — that's covered separately by:

* ``tests/test_turso_store.py`` (real libSQL queries)
* The Phase 1 publish workflow (full publish + smoke test)
* Manual end-to-end runs against a real Vercel deployment

What's exercised here:

* Route registration and method/path correctness
* Pydantic request validation (rejects empty / oversized inputs)
* Dependency-injection wiring
* Response envelope shape (the dict keys downstream callers will rely on)
* Error path: the handlers wrap exceptions into HTTP 500 with detail
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# `from api.index import ...` works because pyproject.toml's
# [tool.pytest.ini_options] pythonpath includes the repo root.


@pytest.fixture
def client():
    """Importing inside the fixture so module-level globals are fresh."""
    pytest.importorskip("fastapi")
    from api.index import (
        app,
        get_embed_kwargs,
        get_llm,
        get_store,
    )
    from fastapi.testclient import TestClient

    fake_store = MagicMock(name="store")
    fake_llm = MagicMock(name="llm")
    # `complete` is called via `await` in the serve layer; AsyncMock returns
    # an awaitable that resolves to the configured value.
    fake_llm.complete = AsyncMock(return_value='{"improving": "x", "worsening": "y"}')

    app.dependency_overrides[get_store] = lambda: fake_store
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_embed_kwargs] = lambda: {"provider": "cloud"}

    try:
        yield TestClient(app), fake_store, fake_llm
    finally:
        # Belt-and-suspenders cleanup: a test failure between yield and
        # this line would otherwise leave overrides on the module-global
        # `app` and contaminate subsequent tests.
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client):
    c, _, _ = client
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /api/analyze — request validation
# ---------------------------------------------------------------------------


def test_analyze_rejects_empty_query(client):
    c, _, _ = client
    r = c.post("/api/analyze", json={"query": ""})
    assert r.status_code == 422


def test_analyze_rejects_oversized_query(client):
    c, _, _ = client
    r = c.post("/api/analyze", json={"query": "x" * 1000})
    assert r.status_code == 422


def test_analyze_rejects_unknown_type(client):
    c, _, _ = client
    r = c.post("/api/analyze", json={"query": "ok", "type": "bogus"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /api/analyze — happy path (tradeoff, mocked LLM)
# ---------------------------------------------------------------------------


def test_analyze_tradeoff_returns_serve_result(client, monkeypatch):
    c, fake_store, fake_llm = client

    # Mock serve.analyze to avoid the inner LLM/JSON parsing logic.
    async def fake_analyze(query, store, llm):
        assert store is fake_store
        assert llm is fake_llm
        return {"query": query, "improving": "accuracy", "principles": []}

    monkeypatch.setattr("api.index.analyze", fake_analyze)

    r = c.post("/api/analyze", json={"query": "reduce hallucination"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "reduce hallucination"
    assert body["improving"] == "accuracy"


def test_analyze_dispatches_to_architecture(client, monkeypatch):
    c, _, _ = client

    called: dict[str, Any] = {}

    async def fake_arch(query, store, llm):
        called["query"] = query
        return {"slot": "attention", "variants": []}

    monkeypatch.setattr("api.index.analyze_architecture", fake_arch)
    r = c.post("/api/analyze", json={"query": "long context", "type": "architecture"})
    assert r.status_code == 200
    assert called["query"] == "long context"
    assert r.json()["slot"] == "attention"


def test_analyze_dispatches_to_agentic(client, monkeypatch):
    c, _, _ = client

    called: dict[str, Any] = {}

    async def fake_agentic(query, store, llm):
        called["query"] = query
        return {"category": "react", "patterns": []}

    monkeypatch.setattr("api.index.analyze_agentic", fake_agentic)
    r = c.post("/api/analyze", json={"query": "tool use", "type": "agentic"})
    assert r.status_code == 200
    assert called["query"] == "tool use"


def test_analyze_returns_500_on_serve_exception(client, monkeypatch):
    c, _, _ = client

    async def boom(query, store, llm):
        raise RuntimeError("kaboom-internal-token-xyz")

    monkeypatch.setattr("api.index.analyze", boom)
    r = c.post("/api/analyze", json={"query": "x"})
    assert r.status_code == 500
    # Detail must be generic — never leaks the internal exception
    # message to the client (information-disclosure protection).
    assert r.json()["detail"] == "analyze failed"
    assert "kaboom" not in r.json()["detail"]


def test_analyze_rejects_whitespace_only_query(client):
    c, _, _ = client
    r = c.post("/api/analyze", json={"query": "   \t  "})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /api/explain
# ---------------------------------------------------------------------------


def test_explain_returns_null_when_no_candidates(client, monkeypatch):
    c, _, _ = client

    async def fake_explain(query, store, llm, focus=None, embedding_kwargs=None):
        return None

    monkeypatch.setattr("api.index.explain", fake_explain)
    r = c.post("/api/explain", json={"query": "obscure"})
    assert r.status_code == 200
    assert r.json() == {"result": None}


def test_explain_serializes_result_via_pydantic(client, monkeypatch):
    c, _, _ = client

    fake_result = MagicMock()
    fake_result.model_dump.return_value = {
        "resolved_type": "principle",
        "resolved_id": "attention",
        "narrative": "Attention is...",
    }

    async def fake_explain(query, store, llm, focus=None, embedding_kwargs=None):
        return fake_result

    monkeypatch.setattr("api.index.explain", fake_explain)
    r = c.post("/api/explain", json={"query": "attention"})
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["resolved_id"] == "attention"


def test_explain_passes_focus_through(client, monkeypatch):
    c, _, _ = client
    received: dict[str, Any] = {}

    async def fake_explain(query, store, llm, focus=None, embedding_kwargs=None):
        received["focus"] = focus
        return None

    monkeypatch.setattr("api.index.explain", fake_explain)
    c.post("/api/explain", json={"query": "MoE", "focus": "tradeoffs"})
    assert received["focus"] == "tradeoffs"


# ---------------------------------------------------------------------------
# /api/search
# ---------------------------------------------------------------------------


def test_search_with_query(client, monkeypatch):
    c, _, _ = client

    def fake_search(store, query=None, **kwargs):
        return [{"paper_id": "p1", "title": "t1"}]

    monkeypatch.setattr("api.index.serve_search_papers", fake_search)
    r = c.get("/api/search", params={"q": "transformer"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["results"][0]["paper_id"] == "p1"


def test_search_with_filters_only(client, monkeypatch):
    c, _, _ = client
    received: dict[str, Any] = {}

    def fake_search(store, query=None, **kwargs):
        received.update(kwargs)
        return []

    monkeypatch.setattr("api.index.serve_search_papers", fake_search)
    r = c.get(
        "/api/search",
        params={"author": "Vaswani", "venue": "NeurIPS", "limit": 5},
    )
    assert r.status_code == 200
    assert received["author"] == "Vaswani"
    assert received["venue"] == "NeurIPS"
    assert received["limit"] == 5


def test_search_rejects_oversized_limit(client):
    c, _, _ = client
    r = c.get("/api/search", params={"limit": 999})
    assert r.status_code == 422


def test_search_rejects_zero_limit(client):
    c, _, _ = client
    r = c.get("/api/search", params={"limit": 0})
    assert r.status_code == 422


def test_search_rejects_non_iso_after_date(client):
    c, _, _ = client
    r = c.get("/api/search", params={"after": "yesterday"})
    assert r.status_code == 422
    assert "ISO-8601" in r.json()["detail"]


def test_search_rejects_non_iso_before_date(client):
    c, _, _ = client
    r = c.get("/api/search", params={"before": "01-01-2026"})
    assert r.status_code == 422


def test_search_accepts_iso_date(client, monkeypatch):
    c, _, _ = client

    def fake_search(store, **kwargs):
        return []

    monkeypatch.setattr("api.index.serve_search_papers", fake_search)
    r = c.get("/api/search", params={"after": "2024-01-01", "before": "2026-12-31"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------


def _stub_kind_query_sql(rows_by_sql: dict[str, list[dict[str, Any]]]):
    """Return a ``query_sql`` that maps SQL prefixes to canned rows.

    Stats fires several independent SQL strings; we match by *prefix*
    so the test doesn't have to mirror the exact SELECT verbatim.
    """

    def _impl(sql: str, params: tuple | None = None) -> list[dict[str, Any]]:
        for prefix, rows in rows_by_sql.items():
            if sql.startswith(prefix):
                return rows
        return []

    return _impl


def _reset_stats_cache():
    """Stats has a module-level TTL cache. Reset between cases so an
    earlier test doesn't pin a stale value into a later one."""
    from api.index import _stats_cache

    _stats_cache.pop("key", None)
    _stats_cache.pop("value", None)


def test_stats_returns_aggregated_shape(client):
    c, fake_store, _ = client
    _reset_stats_cache()
    fake_store.query_sql = _stub_kind_query_sql(
        {
            "SELECT COUNT(*) AS n FROM papers": [{"n": 77}],
            "SELECT kind, COUNT(*) AS n FROM vocabulary": [
                {"kind": "parameter", "n": 21},
                {"kind": "principle", "n": 29},
                {"kind": "arch_slot", "n": 23},
            ],
            "SELECT COUNT(*) AS n FROM matrix_cells": [{"n": 65}],
            "SELECT COUNT(*) AS n FROM tradeoff_extractions": [{"n": 80}],
            "SELECT MAX(version_id) AS n FROM taxonomy_versions": [{"n": 4}],
        }
    )

    r = c.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["papers"] == 77
    assert body["vocabulary"] == {
        "total": 73,
        "parameter": 21,
        "principle": 29,
        "arch_slot": 23,
        "agentic": 0,  # missing from rows → present-but-empty kind defaults to 0
    }
    assert body["matrix_cells"] == 65
    assert body["tradeoffs"] == 80
    assert body["taxonomy_version"] == 4


def test_stats_propagates_none_when_kind_aggregation_fails(client):
    c, fake_store, _ = client
    _reset_stats_cache()

    def boom_on_kind(sql: str, params: tuple | None = None):
        if sql.startswith("SELECT kind, COUNT(*)"):
            raise RuntimeError("vocabulary table missing")
        if sql.startswith("SELECT COUNT(*) AS n FROM papers"):
            return [{"n": 5}]
        return []

    fake_store.query_sql = boom_on_kind
    r = c.get("/api/stats")
    assert r.status_code == 200
    vocab = r.json()["vocabulary"]
    # Failure must be uniform across all five fields — the whole point
    # of the post-review fix is that "—" / 0 inconsistency is gone.
    assert vocab == {
        "total": None,
        "parameter": None,
        "principle": None,
        "arch_slot": None,
        "agentic": None,
    }


def test_stats_dependency_override_is_respected(client):
    """Regression: previously ``_stats_cached`` called ``get_store()``
    directly inside an ``lru_cache`` body, so ``app.dependency_overrides``
    silently failed for ``/api/stats``. The fix keys the cache on the
    DI-resolved store identity, so a per-test override flows through."""
    c, fake_store, _ = client
    _reset_stats_cache()
    fake_store.query_sql = _stub_kind_query_sql({"SELECT COUNT(*) AS n FROM papers": [{"n": 999}]})
    r = c.get("/api/stats")
    assert r.status_code == 200
    # 999 is a synthetic value only the fake's stub returns. Hitting
    # it through the response proves the DI override flowed all the
    # way to ``_compute_stats`` instead of the cache short-circuiting
    # to a stale or default-store value.
    assert r.json()["papers"] == 999


def test_stats_cache_warm_path(client):
    """Two consecutive calls within the same 5-min bucket reuse the
    cached payload — confirmed by the second call NOT calling
    ``query_sql`` again."""
    c, fake_store, _ = client
    _reset_stats_cache()
    calls = {"n": 0}

    def counting(sql: str, params: tuple | None = None):
        calls["n"] += 1
        if sql.startswith("SELECT COUNT(*) AS n FROM papers"):
            return [{"n": 1}]
        return []

    fake_store.query_sql = counting
    c.get("/api/stats")
    first = calls["n"]
    c.get("/api/stats")
    assert calls["n"] == first  # cache hit, no further SQL


# ---------------------------------------------------------------------------
# /api/track
# ---------------------------------------------------------------------------


def test_track_noop_without_turso_env(client, monkeypatch):
    """Default test environment has no TURSO_* — track must be a
    no-op so test runs don't accidentally mutate a real DB."""
    c, _, _ = client
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    # Reset writer cell — a previous test may have populated it.
    from api.index import _WRITER_CELL

    _WRITER_CELL.pop("client", None)

    r = c.post("/api/track", json={"event": "view_home"})
    assert r.status_code == 200
    assert r.json() == {"status": "noop"}


def test_track_rejects_unknown_event(client):
    c, _, _ = client
    r = c.post("/api/track", json={"event": "drop_table_users"})
    assert r.status_code == 422


def test_track_accepts_view_explain_concept(client):
    """The post-review fix added ``view_explain_concept`` to the
    Literal — guard against accidental removal."""
    c, _, _ = client
    r = c.post("/api/track", json={"event": "view_explain_concept", "query": "gqa"})
    assert r.status_code == 200


def test_track_rejects_oversized_query(client):
    c, _, _ = client
    r = c.post("/api/track", json={"event": "search", "query": "x" * 500})
    assert r.status_code == 422


def test_track_origin_allowlist_blocks_cross_origin(client, monkeypatch):
    """When ``LENS_TRACK_ORIGINS`` is set, a request without a matching
    Origin/Referer is silently ignored (returns ``status: ignored``)."""
    c, _, _ = client
    monkeypatch.setenv("LENS_TRACK_ORIGINS", "https://lens.example")
    r = c.post(
        "/api/track",
        json={"event": "view_home"},
        headers={"origin": "https://evil.example"},
    )
    assert r.status_code == 200
    assert r.json() == {"status": "ignored"}


def test_track_origin_allowlist_admits_listed_origin(client, monkeypatch):
    c, _, _ = client
    monkeypatch.setenv("LENS_TRACK_ORIGINS", "https://lens.example")
    monkeypatch.delenv("TURSO_DATABASE_URL", raising=False)
    monkeypatch.delenv("TURSO_AUTH_TOKEN", raising=False)
    from api.index import _WRITER_CELL

    _WRITER_CELL.pop("client", None)

    r = c.post(
        "/api/track",
        json={"event": "view_home"},
        headers={"origin": "https://lens.example"},
    )
    assert r.status_code == 200
    # Origin OK → fall through to writer; writer is None in tests → noop.
    assert r.json() == {"status": "noop"}


def test_hash_query_is_truncated_sha256():
    """Privacy contract: 16 hex chars, lowercase-and-trim normalized."""
    from api.index import _hash_query

    assert _hash_query(None) is None
    assert _hash_query("") is None
    a = _hash_query("Transformer Attention")
    b = _hash_query("  transformer attention  ")
    assert a == b  # case + whitespace insensitive
    assert a is not None and len(a) == 16
    assert all(ch in "0123456789abcdef" for ch in a)

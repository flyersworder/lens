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

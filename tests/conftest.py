"""Shared test fixtures."""

from typing import Any, cast

import pytest

from lens.store.models import EMBEDDING_DIM


@pytest.fixture
def store(tmp_path):
    """Create a LensStore backed by a temporary SQLite database."""
    from lens.store.store import LensStore

    s = LensStore(str(tmp_path / "test.db"))
    s.init_tables()
    return s


@pytest.fixture
def sample_paper_data():
    """Sample paper data as a dict."""
    return {
        "paper_id": "2401.12345",
        "title": "Attention Is All You Need",
        "abstract": "We propose a new architecture...",
        "authors": ["Vaswani", "Shazeer"],
        "venue": "NeurIPS",
        "date": "2017-06-12",
        "arxiv_id": "1706.03762",
        "citations": 100000,
        "quality_score": 0.95,
        "extraction_status": "pending",
        "embedding": [0.1] * EMBEDDING_DIM,
    }


# ---------------------------------------------------------------------------
# LLM stubs
# ---------------------------------------------------------------------------


class FakeCompletions:
    """Stand-in for ``client.chat.completions`` that serves canned payloads.

    Used with a real :class:`LLMClient` so tests exercise the genuine
    ``complete_structured`` logic — capability probe, fallback and corrective
    retry — instead of asserting against a mock's call count.
    """

    def __init__(self, payloads, *, supports_schema: bool = False):
        self.payloads = list(payloads)
        self.supports_schema = supports_schema
        self.calls: list[bool] = []  # True when response_format was sent

    async def create(self, **kwargs):
        used_schema = "response_format" in kwargs
        self.calls.append(used_schema)
        if used_schema and not self.supports_schema:
            import httpx
            import openai

            req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
            raise openai.BadRequestError(
                "response_format json_schema is not supported by this endpoint",
                response=httpx.Response(400, request=req),
                body=None,
            )
        if not self.payloads:
            raise AssertionError("FakeCompletions ran out of payloads")
        content = self.payloads.pop(0)
        if isinstance(content, Exception):
            raise content
        msg = type("M", (), {"content": content})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


def make_llm_client(payloads, *, supports_schema: bool = False):
    """A real LLMClient whose transport is a :class:`FakeCompletions`."""
    from lens.llm.client import LLMClient

    fake = FakeCompletions(payloads, supports_schema=supports_schema)
    client = LLMClient(
        model="openai/gpt-4o-mini", api_base="https://example.invalid/v1", api_key="k"
    )
    # Injecting a stand-in transport into a typed slot; the cast keeps the shim
    # local to this helper rather than loosening the production type.
    client._openai_client = cast(
        Any, type("X", (), {"chat": type("Y", (), {"completions": fake})()})()
    )
    return client, fake


@pytest.fixture(autouse=True)
def _clear_schema_capability_cache():
    """``_NO_SCHEMA_SUPPORT`` is module-level; leaking it between tests would let
    one test silently suppress another's capability probe."""
    from lens.llm.structured import _NO_SCHEMA_SUPPORT

    _NO_SCHEMA_SUPPORT.clear()
    yield
    _NO_SCHEMA_SUPPORT.clear()

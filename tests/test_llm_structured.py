"""Tests for schema-constrained structured output (llm.schemas / llm.structured)."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import openai
import pytest

from lens.llm.client import LLMClient
from lens.llm.schemas import ExtractionResponse


def test_extraction_response_excludes_caller_owned_fields():
    """The LLM must never be asked to produce paper_id or verification_status.

    ``paper_id`` is assigned by the caller and ``verification_status`` is derived
    by ``compute_verification_status``. Under strict json_schema every property is
    required, so leaking either field into the response model would force the
    model to invent it.
    """
    tradeoff_item = ExtractionResponse.model_fields["tradeoffs"].annotation.__args__[0]
    item_fields = set(tradeoff_item.model_fields)

    assert "improves" in item_fields
    assert "paper_id" not in item_fields
    assert "verification_status" not in item_fields


def test_strict_schema_marks_every_property_required_and_closes_objects():
    """Strict json_schema requires all properties listed and additionalProperties false.

    Pydantic omits defaulted fields from ``required``; providers reject that under
    ``strict: true``. The builder must close every object node, including nested
    ``$defs``.
    """
    from lens.llm.structured import strict_schema

    schema = strict_schema(ExtractionResponse)

    assert schema["additionalProperties"] is False
    # tradeoffs/architecture/agentic all have defaults, but must still be required
    assert set(schema["required"]) == {"tradeoffs", "architecture", "agentic"}

    tradeoff = schema["$defs"]["TradeoffItem"]
    assert tradeoff["additionalProperties"] is False
    # new_concepts has a default_factory, yet strict mode still demands it
    assert "new_concepts" in tradeoff["required"]
    assert set(tradeoff["required"]) == set(tradeoff["properties"])


def test_strict_schema_strips_unsupported_default_keyword():
    """`default` is an unsupported keyword under strict mode and must be removed."""
    from lens.llm.structured import strict_schema

    schema = strict_schema(ExtractionResponse)

    def defaults(node, path="root"):
        found = []
        if isinstance(node, dict):
            if "default" in node:
                found.append(path)
            for k, v in node.items():
                found += defaults(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                found += defaults(v, f"{path}[{i}]")
        return found

    assert defaults(schema) == []


def test_strict_schema_contains_no_free_form_objects():
    """Open-ended maps (dict[str, str]) are prohibited; additionalProperties is always false."""
    from lens.llm.structured import strict_schema

    schema = strict_schema(ExtractionResponse)

    def open_maps(node, path="root"):
        found = []
        if isinstance(node, dict):
            ap = node.get("additionalProperties")
            if isinstance(ap, dict):
                found.append(path)
            for k, v in node.items():
                found += open_maps(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                found += open_maps(v, f"{path}[{i}]")
        return found

    assert open_maps(schema) == []


# ---------------------------------------------------------------------------
# complete_structured: strict path, fallback, and capability caching
# ---------------------------------------------------------------------------

_VALID = {
    "tradeoffs": [
        {
            "improves": "latency",
            "worsens": "accuracy",
            "technique": "speculative decoding",
            "context": "batch size 1",
            "confidence": 0.8,
            "evidence_quote": "we observe a 2x speedup",
            "new_concepts": [],
        }
    ],
    "architecture": [],
    "agentic": [],
}


@pytest.fixture(autouse=True)
def _clear_capability_cache():
    """The negative-capability cache is module-level; leaking it across tests
    would make a later test silently skip the strict attempt."""
    from lens.llm.structured import _NO_SCHEMA_SUPPORT

    _NO_SCHEMA_SUPPORT.clear()
    yield
    _NO_SCHEMA_SUPPORT.clear()


def _bad_request(msg: str) -> openai.BadRequestError:
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


class _FakeCompletions:
    """Stand-in for client.chat.completions that records how it was called."""

    def __init__(self, strict_error: Exception | None, payloads: list[str]):
        self.strict_error = strict_error
        self.payloads = list(payloads)
        self.calls: list[bool] = []  # True when response_format was sent

    async def create(self, **kwargs):
        used_schema = "response_format" in kwargs
        self.calls.append(used_schema)
        if used_schema and self.strict_error is not None:
            raise self.strict_error
        content = self.payloads.pop(0)
        msg = type("M", (), {"content": content})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


def _client_with(fake: _FakeCompletions) -> LLMClient:
    c = LLMClient(model="openai/gpt-4o-mini", api_base="https://example.invalid/v1", api_key="k")
    c._openai_client = cast(Any, type("X", (), {"chat": type("Y", (), {"completions": fake})()})())
    return c


@pytest.mark.asyncio
async def test_complete_structured_falls_back_when_endpoint_rejects_schema():
    """An endpoint without json_schema support must still yield a validated model."""
    fake = _FakeCompletions(
        strict_error=_bad_request("response_format.type json_schema is not supported"),
        payloads=[json.dumps(_VALID)],
    )
    client = _client_with(fake)

    result = await client.complete_structured(
        [{"role": "user", "content": "extract"}], ExtractionResponse
    )

    assert isinstance(result, ExtractionResponse)
    assert result.tradeoffs[0].improves == "latency"
    assert fake.calls == [True, False]  # tried strict, then fell back


@pytest.mark.asyncio
async def test_complete_structured_uses_schema_when_endpoint_supports_it():
    """When the endpoint accepts json_schema, no fallback call is made."""
    fake = _FakeCompletions(strict_error=None, payloads=[json.dumps(_VALID)])
    client = _client_with(fake)

    result = await client.complete_structured(
        [{"role": "user", "content": "extract"}], ExtractionResponse
    )

    assert result.tradeoffs[0].technique == "speculative decoding"
    assert fake.calls == [True]  # one enforced call, no fallback


@pytest.mark.asyncio
async def test_complete_structured_caches_unsupported_endpoint():
    """The rejection round-trip is paid once per model, not per call."""
    fake = _FakeCompletions(
        strict_error=_bad_request("json_schema is not supported by this endpoint"),
        payloads=[json.dumps(_VALID), json.dumps(_VALID)],
    )
    client = _client_with(fake)
    msgs = [{"role": "user", "content": "extract"}]

    await client.complete_structured(msgs, ExtractionResponse)
    await client.complete_structured(msgs, ExtractionResponse)

    # First call probes then falls back; second skips the probe entirely.
    assert fake.calls == [True, False, False]


@pytest.mark.asyncio
async def test_complete_structured_propagates_non_capability_errors():
    """A transient 500 must not be mistaken for 'endpoint lacks support'.

    Downgrading on any error would let one outage permanently disable enforced
    decoding for the process.
    """
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    boom = openai.InternalServerError(
        "upstream exploded", response=httpx.Response(500, request=req), body=None
    )
    fake = _FakeCompletions(strict_error=boom, payloads=[json.dumps(_VALID)])
    client = _client_with(fake)

    with pytest.raises(openai.InternalServerError):
        await client.complete_structured([{"role": "user", "content": "x"}], ExtractionResponse)


def test_validation_rejects_conforming_json_that_violates_schema():
    """The point of the change: 'parses' is not 'conforms'.

    json_repair would happily accept this object; Pydantic must not.
    """
    from pydantic import ValidationError

    from lens.llm.structured import validate

    missing_confidence = json.dumps(
        {"tradeoffs": [{"improves": "a", "worsens": "b", "technique": "c", "context": "d"}]}
    )

    with pytest.raises(ValidationError):
        validate(ExtractionResponse, missing_confidence, repair=True)


# ---------------------------------------------------------------------------
# Bridging the LLM response onto the persistence shape
# ---------------------------------------------------------------------------


def test_response_to_tuple_attaches_caller_owned_fields_and_rebuilds_concept_map():
    """Fields the model must not emit are supplied on our side.

    ``paper_id`` is attached by the caller and ``verification_status`` derived from
    confidence + evidence, while ``new_concepts`` is rebuilt from the typed pairs
    strict mode forced us into.
    """
    from lens.extract.extractor import extraction_response_to_tuple

    response = ExtractionResponse.model_validate(
        {
            "tradeoffs": [
                {
                    "improves": "throughput",
                    "worsens": "memory",
                    "technique": "paged attention",
                    "context": "long context",
                    "confidence": 0.9,
                    "evidence_quote": "throughput improves by 2.2x over baseline",
                    "new_concepts": [{"name": "paged-attn", "description": "block KV cache"}],
                }
            ],
            "architecture": [],
            "agentic": [],
        }
    )

    tradeoffs, architecture, agentic = extraction_response_to_tuple(response, "2501.00001")

    assert tradeoffs[0]["paper_id"] == "2501.00001"
    assert tradeoffs[0]["verification_status"] == "verified"
    assert tradeoffs[0]["new_concepts"] == {"paged-attn": "block KV cache"}
    assert architecture == [] and agentic == []


@pytest.mark.asyncio
async def test_complete_structured_retries_once_with_validation_error_fed_back():
    """A schema-violating response earns one corrective retry.

    The validation error is handed back to the model, which is far more
    actionable than re-asking for "valid JSON" when the JSON was already valid.
    """
    bad = json.dumps({"tradeoffs": [{"improves": "a"}], "architecture": [], "agentic": []})
    fake = _FakeCompletions(
        strict_error=_bad_request("json_schema not supported"),
        payloads=[bad, json.dumps(_VALID)],
    )
    captured: list[list[dict]] = []

    class _Recording(_FakeCompletions):
        async def create(self, **kwargs):
            captured.append(kwargs["messages"])
            return await super().create(**kwargs)

    rec = _Recording(fake.strict_error, fake.payloads)
    client = _client_with(rec)

    result = await client.complete_structured(
        [{"role": "user", "content": "extract"}], ExtractionResponse
    )

    assert result.tradeoffs[0].improves == "latency"
    # last attempt must carry the validation failure back to the model
    last = captured[-1]
    assert any("confidence" in str(m.get("content", "")).lower() for m in last)


@pytest.mark.asyncio
async def test_extract_paper_uses_schema_constrained_call():
    """extract_paper routes through complete_structured, not raw text completion."""
    from lens.extract.extractor import extract_paper

    class _FakeClient:
        def __init__(self):
            self.structured_calls = 0

        async def complete_structured(self, messages, schema, **kwargs):
            self.structured_calls += 1
            assert schema is ExtractionResponse
            return ExtractionResponse.model_validate(_VALID)

        async def complete(self, messages, **kwargs):  # pragma: no cover
            raise AssertionError("must not fall back to unstructured completion")

    client = _FakeClient()
    result = await extract_paper("2501.00002", "T", "A", cast(LLMClient, client))

    assert result is not None
    tradeoffs, _, _ = result
    assert client.structured_calls == 1
    assert tradeoffs[0]["paper_id"] == "2501.00002"


def test_extraction_prompt_does_not_embed_its_own_schema():
    """The schema is supplied by complete_structured, from the Pydantic model.

    Leaving a hand-written copy in the prompt would send two schemas on the
    fallback path — and they disagreed about new_concepts (map vs list), which is
    exactly the drift this change removes.
    """
    from lens.extract.prompts import build_extraction_prompt

    prompt = build_extraction_prompt(title="T", abstract="A")

    assert "Response Format" not in prompt
    assert '"new_concepts": {}' not in prompt


def test_generated_schema_prompt_covers_all_three_arrays():
    """The generated section replaces EXTRACTION_RESPONSE_SCHEMA's role."""
    from lens.llm.structured import schema_prompt_section

    section = schema_prompt_section(ExtractionResponse)

    for key in ("tradeoffs", "architecture", "agentic"):
        assert key in section


@pytest.mark.asyncio
async def test_complete_structured_error_carries_the_raw_text():
    """Callers that degrade gracefully need the text that failed to validate.

    Ideation keeps an unusable card as a free-text hypothesis, so raising a bare
    ValidationError would throw away the only thing it still has a use for.
    """
    from lens.llm.structured import StructuredOutputError

    bad = json.dumps({"tradeoffs": [{"improves": "only this"}]})
    fake = _FakeCompletions(
        strict_error=_bad_request("json_schema not supported"), payloads=[bad, bad]
    )
    client = _client_with(fake)

    with pytest.raises(StructuredOutputError) as excinfo:
        await client.complete_structured([{"role": "user", "content": "x"}], ExtractionResponse)

    assert "only this" in excinfo.value.raw_text

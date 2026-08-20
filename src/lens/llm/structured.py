"""Schema-constrained structured output for :class:`~lens.llm.client.LLMClient`.

Providers that implement OpenAI-style ``response_format={"type": "json_schema"}``
enforce the schema during decoding, so the model *cannot* emit a non-conforming
object. That is a stronger guarantee than repairing text after the fact: repair
makes malformed output parse, but a response that omits ``confidence`` or invents
a field parses perfectly well while still being wrong.

Support is per-endpoint rather than per-model, and OpenRouter rejects the request
outright when an endpoint lacks it, so every entry point here degrades to a
prompt-described schema and validates with Pydantic either way.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class StructuredOutputError(Exception):
    """Raised when a response could not be validated against its schema.

    Carries the offending text so callers can degrade gracefully — ideation, for
    instance, keeps an unusable card as a free-text hypothesis rather than
    discarding the model's work entirely.
    """

    def __init__(self, message: str, *, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def _close_object(node: dict[str, Any]) -> None:
    """Make one object node strict-mode legal, in place.

    Strict mode demands ``additionalProperties: false`` and every property listed
    in ``required`` — including properties Pydantic left out because they carry a
    default. Optionality is expressed by a nullable type, never by absence.
    """
    if node.get("type") != "object" or "properties" not in node:
        return
    node["additionalProperties"] = False
    node["required"] = list(node["properties"])


def _walk(node: Any) -> None:
    """Recursively apply :func:`_close_object` to every object in the schema."""
    if isinstance(node, dict):
        # `default` is an unsupported keyword under strict mode. Dropping it is
        # safe: Pydantic still applies the default when validating our side.
        node.pop("default", None)
        _close_object(node)
        for value in node.values():
            _walk(value)
    elif isinstance(node, list):
        for item in node:
            _walk(item)


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build a strict-mode JSON Schema from a Pydantic model.

    Pydantic's own ``model_json_schema()`` is not directly usable: it omits
    defaulted fields from ``required``, which providers reject under
    ``strict: true``. Defaults still apply on our side at validation time, so
    marking them required costs nothing and keeps the request legal.
    """
    schema = model.model_json_schema()
    _walk(schema)
    return schema


# ---------------------------------------------------------------------------
# Request construction, capability detection, and the validation gate
# ---------------------------------------------------------------------------

# Endpoints proven to reject json_schema, keyed by "api_base|model". Populated on
# the first rejection so the wasted round-trip is paid once per model rather than
# on every call. litellm exposes a static capability table, but it describes the
# model rather than OpenRouter's per-endpoint routing, so an observed rejection is
# the only trustworthy signal.
_NO_SCHEMA_SUPPORT: set[str] = set()

_UNSUPPORTED_MARKERS = (
    "response_format",
    "json_schema",
    "structured output",
    "does not support",
)


def response_format(model: type[BaseModel]) -> dict[str, Any]:
    """Build the OpenAI-style ``response_format`` payload for ``model``."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": strict_schema(model),
        },
    }


def is_schema_unsupported(error: Exception) -> bool:
    """Whether ``error`` means "this endpoint can't do json_schema".

    Deliberately narrow: only a 4xx-shaped complaint that names the response
    format counts. A timeout or a 500 must keep propagating, or a transient
    outage would silently and permanently downgrade the client to prompt mode.
    """
    status = getattr(error, "status_code", None)
    if status is not None and not 400 <= int(status) < 500:
        return False
    text = str(error).lower()
    return any(marker in text for marker in _UNSUPPORTED_MARKERS)


def schema_prompt_section(model: type[BaseModel]) -> str:
    """Describe ``model``'s schema in prose, for endpoints without enforcement.

    Generated from the same model as the request schema, so the prompt cannot
    drift away from what validation will accept.
    """
    import json

    return (
        "## Response Format\n"
        "Return ONLY valid JSON matching this schema:\n"
        f"{json.dumps(strict_schema(model), indent=2)}\n\n"
        "Do not include any text outside the JSON object."
    )


def with_schema_prompt(
    messages: list[dict[str, str]], model: type[BaseModel]
) -> list[dict[str, str]]:
    """Append the generated schema section to the final user message."""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            m["content"] = f"{m['content']}\n\n{schema_prompt_section(model)}"
            break
    return out


def validate[ModelT: BaseModel](model: type[ModelT], text: str, *, repair: bool) -> ModelT:
    """Parse and validate ``text`` against ``model``.

    ``repair`` applies json_repair as a text-cleanup step *before* validation on
    the prompt path, where the provider guarantees nothing about syntax. It is
    never the correctness guarantee — Pydantic is. On the enforced path repair is
    off, since a syntax error there is a real signal rather than noise to paper
    over.
    """
    from lens.llm.utils import strip_code_fences

    cleaned = strip_code_fences(text)
    try:
        return model.model_validate_json(cleaned)
    except Exception:
        if not repair:
            raise
        from json_repair import repair_json

        return model.model_validate(repair_json(cleaned, return_objects=True))


def choice_model(name: str, **choices: list[str]) -> type[BaseModel]:
    """Build a model whose fields are enums over runtime-supplied options.

    The analyzer asks the model to pick a parameter, architecture slot or agentic
    category by name, but the valid names live in the corpus vocabulary and are
    only known at request time. Declaring them as an enum lets strict decoding
    rule out a name that isn't in the corpus, instead of discovering it after the
    call when the lookup returns None.

    Every field is required; pass a single-element list to pin a value. Raises
    ValueError for an empty option list, which cannot be expressed as an enum.
    """
    from pydantic import create_model

    fields: dict[str, Any] = {}
    for field, options in choices.items():
        if not options:
            raise ValueError(f"{name}.{field} needs at least one option")
        # The options are only known at runtime, so this Literal cannot be
        # written statically. Pydantic builds the enum from it correctly.
        fields[field] = (
            Literal[tuple(options)],  # type: ignore[valid-type]  # ty: ignore[invalid-type-form]
            ...,
        )
    return create_model(name, **fields)

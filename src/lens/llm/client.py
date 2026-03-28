"""LLM client — litellm (optional) or openai SDK fallback.

Uses litellm if installed (supports any provider via model routing).
Falls back to the openai SDK with a configurable api_base (works with
any OpenAI-compatible endpoint such as a litellm gateway, vLLM, Ollama).
"""

from __future__ import annotations

import logging
from typing import Any

import openai

try:
    import litellm

    litellm.suppress_debug_info = True  # type: ignore[assignment]
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openrouter/anthropic/claude-sonnet-4-6"


class LLMClient:
    """Async LLM client with litellm (optional) or openai SDK fallback."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = api_base or None  # normalize "" to None
        self.api_key = api_key or None
        self._openai_client: openai.AsyncOpenAI | None = None

    def _get_openai_client(self) -> openai.AsyncOpenAI:
        """Get or create cached async openai client."""
        if self._openai_client is None:
            kwargs: dict[str, Any] = {}
            if self.api_base:
                kwargs["base_url"] = self.api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._openai_client = openai.AsyncOpenAI(**kwargs)
        return self._openai_client

    def _require_backend(self) -> None:
        """Raise if neither litellm nor api_base is available."""
        if HAS_LITELLM:
            return
        if self.api_base:
            return
        raise RuntimeError(
            "No LLM backend available. Either:\n"
            "  1. Install litellm: uv add lens[litellm]\n"
            "  2. Set llm.api_base in ~/.lens/config.yaml to an OpenAI-compatible endpoint"
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send a completion request and return the response text."""
        self._require_backend()

        if HAS_LITELLM:
            llm_kwargs: dict[str, Any] = {}
            if self.api_base:
                llm_kwargs["api_base"] = self.api_base
            if self.api_key:
                llm_kwargs["api_key"] = self.api_key
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **llm_kwargs,
                **kwargs,
            )
            content = response.choices[0].message.content
        else:
            client = self._get_openai_client()
            response = await client.chat.completions.create(  # type: ignore[no-matching-overload]
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content

        if content is None:
            raise ValueError(
                f"LLM returned null content (model={self.model}). "
                "This may indicate content filtering or an unsupported response type."
            )
        return content

"""LLM client — litellm (optional) or openai SDK fallback.

Uses litellm if installed (supports any provider via model routing).
Falls back to the openai SDK with a configurable api_base (works with
any OpenAI-compatible endpoint such as a litellm gateway, vLLM, Ollama).

Includes automatic retry with exponential backoff for rate limit (429) errors.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import openai

try:
    import litellm  # ty: ignore[unresolved-import]

    litellm.suppress_debug_info = True  # type: ignore[assignment]
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openrouter/anthropic/claude-sonnet-4-6"

# Retry config for rate limits
MAX_RETRIES = 5
INITIAL_BACKOFF = 10  # seconds
MAX_BACKOFF = 120  # seconds


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

    @staticmethod
    def _is_rate_limit(error: Exception) -> bool:
        """Check if an error is a rate limit (429) that should be retried."""
        error_str = str(error)
        error_type = type(error).__name__
        return "429" in error_str or "RateLimitError" in error_type

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send a completion request and return the response text.

        Automatically retries with exponential backoff on rate limit (429) errors.
        """
        self._require_backend()

        last_error: Exception | None = None
        backoff = INITIAL_BACKOFF

        for attempt in range(MAX_RETRIES + 1):
            try:
                return await self._call_llm(messages, **kwargs)
            except Exception as e:
                if self._is_rate_limit(e) and attempt < MAX_RETRIES:
                    last_error = e
                    logger.info(
                        "Rate limited (attempt %d/%d), waiting %ds...",
                        attempt + 1,
                        MAX_RETRIES,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                    continue
                raise

        # Should not reach here, but just in case
        assert last_error is not None  # noqa: S101
        raise last_error

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Execute a single LLM call (no retry logic)."""
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
            from openai.types.chat import ChatCompletionMessageParam

            client = self._get_openai_client()
            typed_messages = cast(list[ChatCompletionMessageParam], messages)
            response = await client.chat.completions.create(
                model=self.model,
                messages=typed_messages,
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

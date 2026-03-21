"""LLM client wrapper around litellm.

Provides a simple async interface for LLM completions.
Supports any provider via litellm's model routing.
"""

from __future__ import annotations

from typing import Any

import litellm

DEFAULT_MODEL = "openrouter/anthropic/claude-sonnet-4-6"


class LLMClient:
    """Async LLM client using litellm for provider abstraction."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send a completion request and return the response text."""
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(
                f"LLM returned null content (model={self.model}). "
                "This may indicate content filtering or an unsupported response type."
            )
        return content

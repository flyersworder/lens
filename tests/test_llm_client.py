"""Tests for LLM client wrapper."""

from unittest.mock import AsyncMock, patch

import pytest


def test_llm_client_init():
    from lens.llm.client import LLMClient

    client = LLMClient(model="test/model")
    assert client.model == "test/model"


def test_llm_client_default_model():
    from lens.llm.client import LLMClient

    client = LLMClient()
    assert client.model == "openrouter/anthropic/claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_llm_client_complete():
    from lens.llm.client import LLMClient

    client = LLMClient(model="test/model")
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = '{"result": "test"}'

    with patch("lens.llm.client.litellm.acompletion", return_value=mock_response):
        result = await client.complete(messages=[{"role": "user", "content": "test"}])
        assert result == '{"result": "test"}'


@pytest.mark.asyncio
async def test_llm_client_complete_with_system():
    from lens.llm.client import LLMClient

    client = LLMClient(model="test/model")
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "response"

    with patch("lens.llm.client.litellm.acompletion", return_value=mock_response) as mock_call:
        await client.complete(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ]
        )
        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args
        assert call_kwargs.kwargs["model"] == "test/model"

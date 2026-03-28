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


def test_llm_client_init_with_api_base():
    from lens.llm.client import LLMClient

    client = LLMClient(model="gpt-4", api_base="http://localhost:4000/v1", api_key="sk-test")
    assert client.api_base == "http://localhost:4000/v1"
    assert client.api_key == "sk-test"


def test_llm_client_normalizes_empty_strings():
    from lens.llm.client import LLMClient

    client = LLMClient(api_base="", api_key="")
    assert client.api_base is None
    assert client.api_key is None


@pytest.mark.asyncio
async def test_llm_client_complete_with_litellm():
    from lens.llm.client import HAS_LITELLM, LLMClient

    if not HAS_LITELLM:
        pytest.skip("litellm not installed")

    client = LLMClient(model="test/model")
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = '{"result": "test"}'

    with patch("litellm.acompletion", return_value=mock_response):
        result = await client.complete(messages=[{"role": "user", "content": "test"}])
        assert result == '{"result": "test"}'


@pytest.mark.asyncio
async def test_llm_client_complete_openai_fallback():
    """Test the openai SDK fallback path (when litellm is unavailable)."""
    from lens.llm.client import LLMClient

    client = LLMClient(model="gpt-4", api_base="http://localhost:4000/v1", api_key="sk-test")

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "hello from gateway"

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Force the openai path by temporarily pretending litellm is not available
    with patch("lens.llm.client.HAS_LITELLM", False):
        client._openai_client = mock_client
        result = await client.complete(messages=[{"role": "user", "content": "test"}])
        assert result == "hello from gateway"
        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_llm_client_null_content_raises():
    from lens.llm.client import HAS_LITELLM, LLMClient

    if not HAS_LITELLM:
        pytest.skip("litellm not installed")

    client = LLMClient(model="test/model")
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = None

    with (
        patch("litellm.acompletion", return_value=mock_response),
        pytest.raises(ValueError, match="null content"),
    ):
        await client.complete(messages=[{"role": "user", "content": "test"}])


def test_require_backend_raises_without_litellm_or_api_base():
    from lens.llm.client import LLMClient

    client = LLMClient(model="test/model")
    with (
        patch("lens.llm.client.HAS_LITELLM", False),
        pytest.raises(RuntimeError, match="No LLM backend available"),
    ):
        client._require_backend()

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel


class _Schema(BaseModel):
    message: str
    score: float = 0.0


@pytest.fixture(autouse=True)
def reset_singleton():
    import src.agents.llm as llm_mod
    llm_mod._singleton = None
    yield
    llm_mod._singleton = None


def test_singleton():
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ.setdefault("GEMINI_API_KEY", "fake-key")

    mock_client = MagicMock()
    with patch("google.genai.Client", return_value=mock_client):
        from src.agents.llm import get_llm_client
        a = get_llm_client()
        b = get_llm_client()
        assert a is b


@pytest.mark.asyncio
async def test_fallback_on_bad_response():
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ.setdefault("GEMINI_API_KEY", "fake-key")

    mock_client = MagicMock()
    with patch("google.genai.Client", return_value=mock_client):
        from src.agents.llm import get_llm_client, LLMClient, GeminiProvider

        provider = MagicMock(spec=GeminiProvider)
        provider.call = AsyncMock(side_effect=Exception("bad response"))

        client = LLMClient(provider=provider)
        result = await client.call(
            system="system",
            user="user",
            response_schema=_Schema,
        )
        assert isinstance(result, _Schema)
        assert provider.call.call_count == 2


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_gemini_returns_valid_schema():
    os.environ["LLM_PROVIDER"] = "gemini"
    import src.agents.llm as llm_mod
    llm_mod._singleton = None
    from src.agents.llm import get_llm_client
    client = get_llm_client()
    result = await client.call(
        system="You are a helpful assistant.",
        user='Return a JSON with message="hello" and score=1.0.',
        response_schema=_Schema,
    )
    assert isinstance(result, _Schema)
    assert result.message


@pytest.mark.real_llm
@pytest.mark.asyncio
async def test_claude_returns_valid_schema():
    os.environ["LLM_PROVIDER"] = "claude"
    import src.agents.llm as llm_mod
    llm_mod._singleton = None
    from src.agents.llm import get_llm_client
    client = get_llm_client()
    result = await client.call(
        system="You are a helpful assistant.",
        user='Return a JSON with message="hello" and score=0.5.',
        response_schema=_Schema,
    )
    assert isinstance(result, _Schema)
    assert result.message

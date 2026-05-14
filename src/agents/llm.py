from __future__ import annotations

import logging
import os
import time
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("pce.llm")

_singleton: LLMClient | None = None


def _safe_default(schema: type[BaseModel]) -> BaseModel:
    fields: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if field.default is not None and field.default is not field.default_factory:
            fields[name] = field.default
        elif field.default_factory is not None:
            fields[name] = field.default_factory()
        else:
            annotation = field.annotation
            origin = getattr(annotation, "__origin__", None)
            if annotation is str or annotation == "str":
                fields[name] = "unknown"
            elif annotation is float or annotation == "float":
                fields[name] = 0.0
            elif annotation is int or annotation == "int":
                fields[name] = 0
            elif annotation is bool or annotation == "bool":
                fields[name] = False
            elif origin is list:
                fields[name] = []
            else:
                fields[name] = None
    try:
        return schema(**fields)
    except Exception:
        return schema.model_construct(**fields)


class GeminiProvider:
    def __init__(self, model: str = "gemini-2.5-flash"):
        import google.genai as genai
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._genai = genai
        self.default_model = model

    async def call(
        self,
        system: str,
        user: str,
        response_schema: type[BaseModel],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> BaseModel:
        import asyncio
        from google.genai import types as gtypes

        config = gtypes.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        client = self._client
        chosen_model = model or self.default_model

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=chosen_model,
                contents=user,
                config=config,
            ),
        )
        return response_schema.model_validate_json(response.text)


class ClaudeProvider:
    def __init__(self, model: str = "claude-sonnet-4-5"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.default_model = model

    async def call(
        self,
        system: str,
        user: str,
        response_schema: type[BaseModel],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> BaseModel:
        import asyncio

        schema = response_schema.model_json_schema()
        tool_def = {
            "name": "structured_output",
            "description": "Return structured output matching the schema",
            "input_schema": schema,
        }

        def _call():
            return self._client.messages.create(
                model=model or self.default_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                tools=[tool_def],
                tool_choice={"type": "tool", "name": "structured_output"},
                messages=[{"role": "user", "content": user}],
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _call)
        for block in response.content:
            if block.type == "tool_use":
                return response_schema.model_validate(block.input)
        raise ValueError("No tool_use block in Claude response")


class LLMClient:
    def __init__(self, provider: GeminiProvider | ClaudeProvider):
        self._provider = provider

    async def call(
        self,
        system: str,
        user: str,
        response_schema: type[BaseModel],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> BaseModel:
        provider_name = type(self._provider).__name__
        for attempt in range(1, 3):
            t0 = time.time()
            try:
                result = await self._provider.call(
                    system=system,
                    user=user,
                    response_schema=response_schema,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency = int((time.time() - t0) * 1000)
                logger.info(
                    "LLM call success | provider=%s model=%s temp=%s latency=%dms attempt=%d",
                    provider_name, model or "default", temperature, latency, attempt,
                )
                return result
            except Exception as exc:
                latency = int((time.time() - t0) * 1000)
                logger.warning(
                    "LLM call failed | provider=%s attempt=%d latency=%dms error=%s",
                    provider_name, attempt, latency, exc,
                )
                if attempt == 2:
                    logger.error("LLM failed after 2 attempts — returning safe_default")
                    return _safe_default(response_schema)


def get_llm_client() -> LLMClient:
    global _singleton
    if _singleton is None:
        provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()
        if provider_name == "claude":
            _singleton = LLMClient(ClaudeProvider())
        else:
            _singleton = LLMClient(GeminiProvider())
    return _singleton

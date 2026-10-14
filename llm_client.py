import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

# Transient API failures worth retrying
_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


class StructuredOutputError(Exception):
    """Model kept returning output that does not satisfy the schema."""

    def __init__(self, attempts: int, errors: list[str]) -> None:
        self.attempts = attempts
        self.errors = errors
        super().__init__(
            f"invalid structured output after {attempts} attempt(s): " + "; ".join(errors)
        )


@dataclass
class UsageStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: types.GenerateContentResponseUsageMetadata | None) -> None:
        if usage is None:
            return
        self.calls += 1
        self.prompt_tokens += usage.prompt_token_count or 0
        self.completion_tokens += usage.candidates_token_count or 0
        self.total_tokens += usage.total_token_count or 0

    def summary(self) -> str:
        return (
            f"calls={self.calls} "
            f"prompt={self.prompt_tokens} "
            f"completion={self.completion_tokens} "
            f"total={self.total_tokens}"
        )


class LLMClient:
    """Thin wrapper around the Gemini client for learning-friendly calls."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.1-flash-lite",
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        max_repair_attempts: int = 1,
    ) -> None:
        load_dotenv()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not found in .env")

        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.max_repair_attempts = max_repair_attempts
        self.usage = UsageStats()
        self._client = genai.Client(api_key=key)

    async def generate(self, prompt: str) -> str:
        """Return the full reply as plain text."""
        response = await self._generate_content(prompt)
        return response.text or ""

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield text pieces as the model generates them."""
        chunks = await self._with_retries(
            lambda: self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=prompt,
            )
        )
        last_usage = None
        async for chunk in chunks:
            if chunk.usage_metadata is not None:
                last_usage = chunk.usage_metadata
            if chunk.text:
                yield chunk.text
        self.usage.add(last_usage)

    async def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        context: dict[str, Any] | None = None,
    ) -> T:
        """Return a validated Pydantic model.

        Two layers:
        - response_schema makes Gemini emit JSON of the right *shape*;
        - schema.model_validate_json(..., context=...) checks what the API
          schema cannot express (non-empty strings, values that must match
          our data). Context is passed to the schema's validators.

        On validation failure the errors are sent back to the model once
        ("repair" retry) instead of re-asking the identical prompt.
        """
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        )
        attempt_prompt = prompt
        attempts = 1 + self.max_repair_attempts
        problems: list[str] = []

        for attempt in range(1, attempts + 1):
            response = await self._generate_content(attempt_prompt, config=config)
            try:
                return self._parse_structured(response, schema, context)
            except ValidationError as exc:
                problems = _format_validation_errors(exc)
            except ValueError as exc:
                problems = [str(exc)]

            logger.warning(
                "Structured output rejected (attempt %d/%d): %s",
                attempt, attempts, "; ".join(problems),
            )
            attempt_prompt = _repair_prompt(prompt, response.text or "", problems)

        raise StructuredOutputError(attempts, problems)

    async def _generate_content(
        self,
        prompt: str,
        config: types.GenerateContentConfig | None = None,
    ) -> types.GenerateContentResponse:
        logger.debug("Gemini request: %d chars", len(prompt))

        async def call() -> types.GenerateContentResponse:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            self.usage.add(response.usage_metadata)
            return response

        return await self._with_retries(call)

    async def _with_retries[R](self, call: Callable[[], Awaitable[R]]) -> R:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return await call()
            except errors.APIError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_CODES or attempt + 1 >= self.max_retries:
                    raise
                await asyncio.sleep(self.retry_base_delay * (2**attempt))

        raise RuntimeError("retry loop exited unexpectedly") from last_error

    @staticmethod
    def _parse_structured(
        response: types.GenerateContentResponse,
        schema: type[T],
        context: dict[str, Any] | None,
    ) -> T:
        # Truncated JSON may still parse into something plausible; reject it explicitly.
        candidate = response.candidates[0] if response.candidates else None
        if candidate is not None and candidate.finish_reason == types.FinishReason.MAX_TOKENS:
            raise ValueError("response truncated (MAX_TOKENS)")
        if not response.text:
            reason = candidate.finish_reason if candidate else "no candidates"
            raise ValueError(f"empty response (finish_reason={reason})")
        # Always validate the raw text ourselves: response.parsed skips our context validators.
        return schema.model_validate_json(response.text, context=context)


def _format_validation_errors(exc: ValidationError) -> list[str]:
    return [
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors(include_url=False)
    ]


def _repair_prompt(original: str, bad_output: str, problems: list[str]) -> str:
    issues = "\n".join(f"- {p}" for p in problems)
    return f"""{original}

## Your previous answer was rejected by validation
Previous answer:
{bad_output or "(empty)"}

Problems:
{issues}

Return a corrected answer that fixes every problem and still follows all rules above."""

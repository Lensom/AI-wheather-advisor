import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from pricing import estimate_cost

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-lite"  # override with GEMINI_MODEL in .env
MAX_RETRY_DELAY_S = 60.0  # longer server-requested waits fail fast instead

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
    model: str = DEFAULT_MODEL
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0  # billed as output
    total_tokens: int = 0
    api_seconds: float = 0.0  # time spent waiting for Gemini (successful calls)
    repairs: int = 0  # structured() repair retries sent
    rate_limited: int = 0  # 429 responses that were retried
    wait_seconds: float = 0.0  # time slept before retries

    def add(
        self,
        usage: types.GenerateContentResponseUsageMetadata | None,
        seconds: float = 0.0,
    ) -> None:
        self.api_seconds += seconds
        if usage is None:
            return
        self.calls += 1
        self.prompt_tokens += usage.prompt_token_count or 0
        self.completion_tokens += usage.candidates_token_count or 0
        self.thinking_tokens += usage.thoughts_token_count or 0
        self.total_tokens += usage.total_token_count or 0

    @property
    def cost_usd(self) -> float | None:
        return estimate_cost(
            self.model, self.prompt_tokens, self.completion_tokens + self.thinking_tokens
        )

    def summary(self) -> str:
        cost = self.cost_usd
        return (
            f"calls={self.calls} "
            f"prompt={self.prompt_tokens} "
            f"completion={self.completion_tokens} "
            + (f"thinking={self.thinking_tokens} " if self.thinking_tokens else "")
            + f"total={self.total_tokens} "
            f"api_time={self.api_seconds:.1f}s "
            + (f"retry_wait={self.wait_seconds:.0f}s " if self.wait_seconds else "")
            + f"cost≈{'n/a' if cost is None else f'${cost:.5f}'}"
        )


class LLMClient:
    """Thin wrapper around the Gemini client for learning-friendly calls."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        max_repair_attempts: int = 1,
    ) -> None:
        load_dotenv()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not found in .env")

        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.max_repair_attempts = max_repair_attempts
        self.usage = UsageStats(model=self.model)
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
            if attempt < attempts:
                self.usage.repairs += 1

        raise StructuredOutputError(attempts, problems)

    async def generate_response(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> types.GenerateContentResponse:
        """Raw response for callers that drive their own control flow (tool loops).

        Same retries and usage tracking as the other methods.
        """
        return await self._generate_content(contents, config=config)

    async def _generate_content(
        self,
        prompt: str | list[types.Content],
        config: types.GenerateContentConfig | None = None,
    ) -> types.GenerateContentResponse:
        if isinstance(prompt, str):
            logger.debug("Gemini request: %d chars", len(prompt))
        else:
            logger.debug("Gemini request: %d contents", len(prompt))

        async def call() -> types.GenerateContentResponse:
            started = time.perf_counter()
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            self.usage.add(response.usage_metadata, time.perf_counter() - started)
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
                # 429 says how long to wait; 1-2 s backoff would just burn the retries.
                delay = _server_retry_delay(exc) or self.retry_base_delay * (2**attempt)
                if delay > MAX_RETRY_DELAY_S:
                    raise  # e.g. a daily quota: waiting won't help, fail fast
                if exc.code == 429:
                    self.usage.rate_limited += 1
                self.usage.wait_seconds += delay
                logger.warning(
                    "Gemini %s %s, retrying in %.0f s (attempt %d/%d)",
                    exc.code, exc.status, delay, attempt + 2, self.max_retries,
                )
                await asyncio.sleep(delay)

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


def _server_retry_delay(exc: errors.APIError) -> float | None:
    """Seconds from google.rpc.RetryInfo ("retryDelay": "21s"), if the server sent one."""
    details = exc.details.get("error", {}).get("details", []) if isinstance(exc.details, dict) else []
    for item in details:
        if isinstance(item, dict) and item.get("@type", "").endswith("RetryInfo"):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1)) + 1.0  # small margin past the quota window
    return None


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

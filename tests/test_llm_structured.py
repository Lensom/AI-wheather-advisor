"""LLMClient.structured() retry/repair logic with a fake Gemini. Offline."""

import asyncio
import json

import pytest
from google.genai import errors, types

from advice_models import WeatherAdvice
import llm_client
from llm_client import LLMClient, StructuredOutputError

VALID = {
    "summary": "Cool and dry.",
    "clothing_recommendations": ["Light jacket"],
    "outdoor_activity": "recommended",
    "reasoning": ["temperature_c is 12.5"],
    "rain_risk_explanation": "rain_risk.overall is low",
    "best_time": "2026-10-15T16:00:00",
}
INVALID = {**VALID, "outdoor_activity": "maybe"}


def fake_response(text: str, finish=types.FinishReason.STOP) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                finish_reason=finish,
            )
        ]
    )


def make_client(responses: list[types.GenerateContentResponse]) -> tuple[LLMClient, list[str]]:
    """LLMClient whose Gemini call returns `responses` in order; records prompts."""
    llm = LLMClient(api_key="test-key")
    prompts: list[str] = []
    queue = iter(responses)

    async def fake_generate(prompt, config=None):
        prompts.append(prompt)
        return next(queue)

    llm._generate_content = fake_generate
    return llm, prompts


def run(llm: LLMClient):
    return asyncio.run(llm.structured("PROMPT", WeatherAdvice))


def test_valid_first_try():
    llm, prompts = make_client([fake_response(json.dumps(VALID))])
    assert run(llm).summary == "Cool and dry."
    assert len(prompts) == 1


def test_repair_retry_sends_errors_back():
    llm, prompts = make_client([
        fake_response(json.dumps(INVALID)),
        fake_response(json.dumps(VALID)),
    ])
    assert run(llm).outdoor_activity == "recommended"
    assert len(prompts) == 2
    assert prompts[1].startswith("PROMPT")
    assert "outdoor_activity" in prompts[1] and "rejected by validation" in prompts[1]


def test_gives_up_with_typed_error():
    llm, _ = make_client([fake_response(json.dumps(INVALID))] * 2)
    with pytest.raises(StructuredOutputError) as exc:
        run(llm)
    assert exc.value.attempts == 2
    assert any("outdoor_activity" in e for e in exc.value.errors)


def test_malformed_json_is_repaired():
    llm, prompts = make_client([
        fake_response('{"summary": "Cool'),
        fake_response(json.dumps(VALID)),
    ])
    assert run(llm).summary == "Cool and dry."
    assert "Invalid JSON" in prompts[1]


def test_truncated_response_rejected():
    llm, _ = make_client([
        fake_response(json.dumps(VALID), finish=types.FinishReason.MAX_TOKENS),
    ] * 2)
    with pytest.raises(StructuredOutputError, match="MAX_TOKENS"):
        run(llm)


# --- retries on API errors -----------------------------------------------------------


def api_error(code: int, retry_delay: str | None = None) -> errors.APIError:
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}] if retry_delay else []
    return errors.ClientError(code, {"error": {"code": code, "status": "X", "message": "m", "details": details}})


def run_with_failures(monkeypatch, failures: list[Exception]):
    """Call _with_retries with a call that fails with `failures` first; record sleeps."""
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(llm_client.asyncio, "sleep", fake_sleep)
    llm = LLMClient(api_key="test-key")
    queue = list(failures)

    async def call():
        if queue:
            raise queue.pop(0)
        return "ok"

    return asyncio.run(llm._with_retries(call)), sleeps, llm


def test_429_waits_as_long_as_server_asks(monkeypatch):
    result, sleeps, llm = run_with_failures(monkeypatch, [api_error(429, "21s")])
    assert result == "ok"
    assert sleeps == [22.0]  # 21 s + 1 s margin, not the 1 s backoff
    assert llm.usage.rate_limited == 1 and llm.usage.wait_seconds == 22.0


def test_backoff_without_server_hint(monkeypatch):
    _, sleeps, _ = run_with_failures(monkeypatch, [api_error(503), api_error(503)])
    assert sleeps == [1.0, 2.0]


def test_too_long_server_delay_fails_fast(monkeypatch):
    with pytest.raises(errors.ClientError):
        run_with_failures(monkeypatch, [api_error(429, "3600s")])


def test_non_retryable_error_raises_immediately(monkeypatch):
    with pytest.raises(errors.ClientError):
        run_with_failures(monkeypatch, [api_error(400)])

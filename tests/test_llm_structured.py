"""LLMClient.structured() retry/repair logic with a fake Gemini. Offline."""

import asyncio
import json

import pytest
from google.genai import types

from advice_models import WeatherAdvice
from llm_client import LLMClient, StructuredOutputError

VALID = {
    "summary": "Cool and dry.",
    "clothing_recommendations": ["Light jacket"],
    "outdoor_activity": "recommended",
    "reasoning": ["temperature_c is 12.5"],
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

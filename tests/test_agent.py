"""Agent control flow with a scripted fake Gemini. Offline."""

import asyncio
from pathlib import Path

import pytest
from google.genai import types

from agent import AgentLoopError, WeatherAgent
from tools import GET_WEATHER, WeatherTools
from weather_models import WeatherSnapshot

FIXTURE = Path(__file__).parent / "fixtures" / "snapshot_berlin.json"


def text_response(text: str) -> types.GenerateContentResponse:
    return _response([types.Part(text=text)])


def calls_response(*cities: str) -> types.GenerateContentResponse:
    return _response([
        types.Part(function_call=types.FunctionCall(id=f"id-{c}", name=GET_WEATHER, args={"city": c}))
        for c in cities
    ])


def _response(parts: list[types.Part]) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )


class FakeLLM:
    """Returns scripted responses; records what the agent sent each time."""

    def __init__(self, *responses: types.GenerateContentResponse) -> None:
        self._responses = iter(responses)
        self.requests: list[tuple[list[types.Content], types.GenerateContentConfig]] = []

    async def generate_response(self, contents, config):
        self.requests.append((list(contents), config))
        return next(self._responses)


def fake_loader(args):
    return WeatherSnapshot.model_validate_json(FIXTURE.read_text(encoding="utf-8")), []


def make_agent(*responses, max_steps: int = 4) -> tuple[WeatherAgent, FakeLLM]:
    llm = FakeLLM(*responses)
    return WeatherAgent(llm, WeatherTools(fake_loader), max_steps=max_steps), llm


def ask(agent: WeatherAgent, question: str):
    return asyncio.run(agent.ask(question))


def mode(config: types.GenerateContentConfig) -> types.FunctionCallingConfigMode:
    return config.tool_config.function_calling_config.mode


# --- basic flows ---------------------------------------------------------------------


def test_answer_without_tools():
    agent, llm = make_agent(text_response("I only help with weather."))
    result = ask(agent, "What is 2+2?")
    assert result.steps == 1 and result.tool_calls == []
    assert result.session_has_weather_data is False
    assert [c.role for c in agent.history] == ["user", "model"]


def test_one_call_then_answer():
    agent, llm = make_agent(calls_response("Berlin"), text_response("Walk at 16:00, 16.3 °C."))
    result = ask(agent, "When should I walk in Berlin?")

    assert result.answer == "Walk at 16:00, 16.3 °C."
    assert result.steps == 2 and result.session_has_weather_data
    assert [c.role for c in agent.history] == ["user", "model", "user", "model"]
    response = agent.history[2].parts[0].function_response
    assert response.id == "id-Berlin" and response.response["status"] == "ok"
    # Second request contained the function response.
    sent, _ = llm.requests[1]
    assert sent[-1].parts[0].function_response is not None
    assert result.unsupported_numbers == []


def test_sdk_auto_calling_is_disabled():
    agent, llm = make_agent(text_response("ok"))
    ask(agent, "hi")
    _, config = llm.requests[0]
    assert config.automatic_function_calling.disable is True
    assert mode(config) == types.FunctionCallingConfigMode.AUTO


def test_parallel_calls_answered_in_one_turn_in_order():
    agent, _ = make_agent(calls_response("Kyiv", "Odesa"), text_response("Odesa is better."))
    result = ask(agent, "Kyiv or Odesa?")
    responses = [p.function_response for p in agent.history[2].parts]
    assert [r.id for r in responses] == ["id-Kyiv", "id-Odesa"]
    assert [c.args["city"] for c in result.tool_calls] == ["Kyiv", "Odesa"]


# --- control-flow limits ----------------------------------------------------------------


def test_last_step_disables_tools():
    agent, llm = make_agent(calls_response("Berlin"), text_response("Done."), max_steps=2)
    result = ask(agent, "Berlin?")
    assert result.steps == 2
    assert mode(llm.requests[0][1]) == types.FunctionCallingConfigMode.AUTO
    assert mode(llm.requests[1][1]) == types.FunctionCallingConfigMode.NONE


def test_model_ignoring_limit_raises_and_rolls_back():
    agent, _ = make_agent(
        text_response("First answer."),
        calls_response("Berlin"),
        calls_response("Berlin"),  # still calling on the last step
        max_steps=2,
    )
    ask(agent, "first question")
    assert len(agent.history) == 2

    with pytest.raises(AgentLoopError, match="no final answer"):
        ask(agent, "Berlin?")
    assert len(agent.history) == 2  # half-finished turn removed
    assert agent.session_tool_calls == []


def test_empty_answer_raises():
    agent, _ = make_agent(_response([types.Part(text="")]))
    with pytest.raises(AgentLoopError, match="empty answer"):
        ask(agent, "Berlin?")
    assert agent.history == []


def test_empty_question_rejected():
    agent, llm = make_agent()
    with pytest.raises(ValueError):
        ask(agent, "   ")
    assert llm.requests == []


# --- multi-turn and grounding -----------------------------------------------------------


def test_follow_up_reuses_history_without_new_call():
    agent, llm = make_agent(
        calls_response("Berlin"), text_response("Dry."),
        text_response("Wear a light jacket."),
    )
    ask(agent, "Rain in Berlin?")
    result = ask(agent, "What should I wear?")
    assert result.tool_calls == []
    assert result.session_has_weather_data  # earlier turn's data counts
    sent, _ = llm.requests[-1]
    assert len(sent) == 5  # previous 4 turns + new question


def test_old_tool_results_are_compacted_on_next_question():
    agent, llm = make_agent(
        calls_response("Berlin"), text_response("Dry."),
        text_response("Light jacket."),
    )
    ask(agent, "Rain in Berlin?")
    fresh = agent.history[2].parts[0].function_response.response
    assert "hourly_forecast" in fresh["facts"]

    ask(agent, "What should I wear?")
    old = agent.history[2].parts[0].function_response
    assert old.id == "id-Berlin"  # pairing with the call is preserved
    assert old.response["compacted"] is True
    assert "hourly_forecast" not in old.response["facts"]
    assert "hourly" not in old.response["facts"]["assessment"]
    assert old.response["facts"]["assessment"]["outdoor_verdict"]["level"] == "recommended"
    assert old.response["facts"]["current"]["temperature_c"] == 12.5
    # Second request really sent the smaller payload.
    sent, _ = llm.requests[-1]
    assert sent[2].parts[0].function_response.response["compacted"] is True


def test_threshold_quoted_from_instructions_is_not_flagged():
    agent, _ = make_agent(text_response("The forecast covers only the next 24 hours."))
    assert ask(agent, "Weather next week?").unsupported_numbers == []


def test_invented_number_is_flagged():
    agent, _ = make_agent(calls_response("Berlin"), text_response("UV index is 7.5 today."))
    assert ask(agent, "Berlin?").unsupported_numbers == [7.5]


def test_numbers_from_question_are_not_flagged():
    agent, _ = make_agent(text_response("I have no data for 2030."))
    assert ask(agent, "Weather in 2030?").unsupported_numbers == []

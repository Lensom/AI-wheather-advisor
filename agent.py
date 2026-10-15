"""Tool-calling agent loop (Stage 5).

Control flow, written out by hand instead of the SDK's automatic function
calling so every step is visible and bounded:

    user question
      -> Gemini: text answer?  -> done
               function calls? -> append model turn (keeps thought signatures),
                                  execute all calls (in parallel),
                                  append one user turn with all FunctionResponses,
                                  loop
    last step: tools disabled (mode NONE) -> the model must answer in text.

The conversation history is kept between questions, so follow-ups such as
"and in Odesa?" work. A failed turn is rolled back so the history never
ends with a function call that has no response (the API rejects that).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from google.genai import types

from assessment import RULES_DOC
from grounding import find_unsupported_numbers
from llm_client import LLMClient
from tools import ToolCallRecord, WeatherTools

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 4  # model calls per question, including the final answer

SYSTEM_INSTRUCTION = """You are a weather assistant. You have one tool: get_weather(city, country?, region?).

Rules:
1. For any question about weather, clothing, walks or outdoor plans in a place, call
   get_weather for EVERY place involved before answering. One city per call; for
   comparisons call it for each city in the same turn.
2. Use ONLY data from tool results. Never state temperatures, probabilities, times or
   scores that are not in a tool result. Without a tool result, say you have no data.
3. Values under facts.indicators and facts.assessment were computed in Python and are
   final: explain them, never recompute or contradict them. For the best walk time use
   facts.assessment.walk_candidates; for whether going out is advisable use
   facts.assessment.outdoor_verdict.level.
4. If a tool result has status "error", do not guess. Tell the user what failed and,
   if useful, ask for clarification (for city_not_found: ask for the country or region).
5. Name the place exactly as resolved_location says, never more specifically (e.g. do
   not add a state it does not contain). If other_matches is not empty, mention that
   the name is ambiguous so the user can correct you.
6. If the user names no city, ask which city. Never call the tool with a guessed city.
7. If the question is not about weather, say briefly that you only help with weather.
8. The forecast covers only the next 24 hours. For later dates say the data does not
   cover them.
9. Answer in the user's language, concisely, with units.

Rules Python used to compute facts.assessment (quote them when explaining):
""" + json.dumps(RULES_DOC, indent=2)


class AgentLoopError(Exception):
    """The model did not produce a final answer within the step limit."""


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCallRecord]  # this question only
    steps: int
    unsupported_numbers: list[float]
    session_has_weather_data: bool  # follow-ups may rely on earlier tool results


class WeatherAgent:
    def __init__(
        self,
        llm: LLMClient,
        tools: WeatherTools,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history: list[types.Content] = []
        self.session_tool_calls: list[ToolCallRecord] = []  # everything the model was told

    def _config(self, *, allow_tools: bool) -> types.GenerateContentConfig:
        mode = types.FunctionCallingConfigMode.AUTO if allow_tools else types.FunctionCallingConfigMode.NONE
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=self.tools.declarations,
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode)),
            # We execute tools ourselves; never let the SDK do it behind our back.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    async def ask(self, question: str) -> AgentResult:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        history_mark, calls_mark = len(self.history), len(self.session_tool_calls)
        try:
            return await self._run_turn(question)
        except BaseException:
            # Roll back the half-finished turn: a dangling function call in history
            # would make every later request fail.
            del self.history[history_mark:]
            del self.session_tool_calls[calls_mark:]
            raise

    async def _run_turn(self, question: str) -> AgentResult:
        _compact_old_tool_results(self.history)
        self.history.append(types.Content(role="user", parts=[types.Part(text=question)]))
        records: list[ToolCallRecord] = []

        for step in range(1, self.max_steps + 1):
            last_step = step == self.max_steps
            response = await self.llm.generate_response(
                self.history, self._config(allow_tools=not last_step)
            )
            calls = response.function_calls or []
            model_turn = response.candidates[0].content if response.candidates else None
            logger.debug(
                "step %d/%d: %s", step, self.max_steps,
                ", ".join(f"{c.name}({c.args})" for c in calls) if calls else "text answer",
            )

            if not calls:
                answer = _text_of(model_turn)
                if not answer:
                    reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
                    raise AgentLoopError(f"empty answer from model (finish_reason={reason})")
                self.history.append(model_turn)
                return self._result(question, answer, records, step)

            # Model turn goes into history unchanged: it carries thought signatures
            # that Gemini 3 requires back together with the function responses.
            self.history.append(model_turn)
            executed = await asyncio.gather(*(self.tools.execute(call) for call in calls))
            self.history.append(types.Content(role="user", parts=[part for part, _ in executed]))
            for _, record in executed:
                records.append(record)
                self.session_tool_calls.append(record)

        raise AgentLoopError(f"no final answer after {self.max_steps} steps")

    def _result(
        self, question: str, answer: str, records: list[ToolCallRecord], steps: int
    ) -> AgentResult:
        # Ground against everything the model was told: instructions (24 h window, rule
        # thresholds it may quote), the question, and this session's tool results.
        seen = [SYSTEM_INSTRUCTION, question] + [
            json.dumps(r.result, ensure_ascii=False) for r in self.session_tool_calls
        ]
        return AgentResult(
            answer=answer,
            tool_calls=records,
            steps=steps,
            unsupported_numbers=find_unsupported_numbers(answer, "\n".join(seen)),
            session_has_weather_data=any(r.status == "ok" for r in self.session_tool_calls),
        )


COMPACTED_NOTE = (
    "Hourly data was removed from this earlier result to save tokens. If hourly "
    "details are needed, call get_weather again with the same arguments (served from cache)."
)


def _compact_old_tool_results(history: list[types.Content]) -> None:
    """Shrink tool results from earlier questions; the history is resent on every call.

    Keeps current conditions, indicators and decisions (enough for most follow-ups);
    drops the hourly arrays, which are ~80% of a result. The function_call /
    function_response pairing and ids stay intact.
    """
    for content in history:
        for i, part in enumerate(content.parts or []):
            fr = part.function_response
            if fr is None or not fr.response or fr.response.get("status") != "ok":
                continue
            if fr.response.get("compacted"):
                continue
            facts = fr.response["facts"]
            compact = {
                **{k: v for k, v in fr.response.items() if k != "facts"},
                "compacted": True,
                "note": COMPACTED_NOTE,
                "facts": {
                    **{k: v for k, v in facts.items() if k not in ("hourly_forecast", "assessment")},
                    "assessment": {k: v for k, v in facts["assessment"].items() if k != "hourly"},
                },
            }
            content.parts[i] = types.Part(
                function_response=types.FunctionResponse(id=fr.id, name=fr.name, response=compact)
            )


def _text_of(content: types.Content | None) -> str:
    """Visible answer text (skips thought parts)."""
    if content is None or not content.parts:
        return ""
    return "".join(p.text for p in content.parts if p.text and not p.thought).strip()

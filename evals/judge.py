"""Optional LLM-as-judge for what keyword checks cannot assess.

A second Gemini call grades the advice against the same facts. Caveats:
- the judge is the same model family, so it shares blind spots (self-preference);
- scores drift between judge versions; compare runs only with the same judge;
- it doubles cost, so it is off by default (--judge).
"""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import BaseModel, Field

from advice_models import WeatherAdvice
from llm_client import LLMClient

Score = Annotated[int, Field(ge=1, le=5)]
PASS_SCORE = 4


class JudgeVerdict(BaseModel):
    clothing_score: Score = Field(
        description="1-5: are clothing suggestions appropriate for the temperatures, rain and wind?"
    )
    faithfulness_score: Score = Field(
        description="1-5: is every claim supported by the facts, with no invented data or contradictions?"
    )
    issues: list[str] = Field(default_factory=list, description="Concrete problems found; empty if none")


def judge_prompt(facts_json: str, advice: WeatherAdvice) -> str:
    return f"""You are grading a weather assistant's answer. Be strict and concrete.

## Facts the assistant was given (ground truth)
```json
{facts_json}
```

## Assistant's answer
```json
{json.dumps(advice.model_dump(), ensure_ascii=False, indent=2)}
```

## Grade
- clothing_score (1-5): 5 = clothing fits the temperatures, precipitation and wind in the
  facts; 1 = clearly wrong (e.g. shorts at -5 °C, no rain gear for heavy rain).
- faithfulness_score (1-5): 5 = every statement is supported by the facts and agrees with
  the Python decisions under "assessment"; 1 = invented values or contradicts the facts.
- issues: list each concrete problem. Do not penalise style or length."""


async def judge_advice(llm: LLMClient, facts_json: str, advice: WeatherAdvice) -> JudgeVerdict:
    return await llm.structured(judge_prompt(facts_json, advice), JudgeVerdict)

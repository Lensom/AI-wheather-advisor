"""Programmatic checks of model output against scenario expectations.

hard checks decide pass/fail of a case; soft checks are reported as rates
(heuristics with known false positives, e.g. the numbers check).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from advice_models import WeatherAdvice
from agent import AgentResult
from assessment import WeatherAssessment
from evals.scenarios import AdviceCase, AgentTurn
from grounding import find_unsupported_numbers

# Clothing items written as "No need for an umbrella" must not count as recommending one.
NEGATIONS = (
    "no need", "not need", "don't need", "do not need", "unnecessary", "not necessary",
    "not required", "won't need", "avoid", "leave the", "skip the", "instead of",
)

# Data we never send: mentioning it outside missing_information means it was invented.
ABSENT_FIELDS = {
    "uv index": r"\buv\b",
    "air quality": r"\bair quality\b|\baqi\b",
    "pollen": r"\bpollen\b",
    "sunrise/sunset": r"\bsunrise\b|\bsunset\b",
    "visibility": r"\bvisibility\b",
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    hard: bool = True


def _mentions(text: str, keyword: str) -> bool:
    return keyword.casefold() in text.casefold()


def _recommends(item: str, keyword: str) -> bool:
    low = item.casefold()
    return keyword.casefold() in low and not any(n in low for n in NEGATIONS)


def _groups_check(name: str, texts: list[str], groups: tuple[tuple[str, ...], ...]) -> Check:
    joined = " ".join(texts)
    missing = [g for g in groups if not any(_mentions(joined, k) for k in g)]
    detail = "; ".join("none of " + "/".join(g[:4]) + ("..." if len(g) > 4 else "") for g in missing)
    return Check(name, not missing, detail)


def check_advice(
    case: AdviceCase,
    advice: WeatherAdvice,
    assessment: WeatherAssessment,
    facts_text: str,
    repairs: int,
) -> list[Check]:
    checks = [Check("valid_output", True)]

    expected = case.expected_verdict
    checks.append(Check(
        "verdict_matches_python",
        advice.outdoor_activity == expected == assessment.outdoor_verdict.level,
        f"model={advice.outdoor_activity} python={assessment.outdoor_verdict.level} expected={expected}",
    ))

    candidate_times = {c.time.isoformat() for c in assessment.walk_candidates}
    if case.expect_best_time:
        ok = advice.best_time in candidate_times
        detail = f"best_time={advice.best_time} not in candidates" if not ok else ""
    else:
        ok = advice.best_time is None
        detail = f"expected null, got {advice.best_time}" if not ok else ""
    checks.append(Check("best_time", ok, detail))

    if case.clothing_required:
        checks.append(_groups_check(
            "clothing_required", advice.clothing_recommendations, case.clothing_required
        ))
    if case.clothing_forbidden:
        bad = sorted({
            k for item in advice.clothing_recommendations
            for k in case.clothing_forbidden if _recommends(item, k)
        })
        checks.append(Check("clothing_forbidden", not bad, ", ".join(bad)))
    if case.concerns_required:
        checks.append(_groups_check(
            "concerns_mention",
            advice.weather_concerns + advice.reasoning + [advice.rain_risk_explanation],
            case.concerns_required,
        ))
    if case.missing_required:
        checks.append(_groups_check(
            "missing_information", advice.missing_information, case.missing_required
        ))

    # Constraint: no invented fields outside missing_information.
    claims = " ".join(
        [advice.summary, advice.rain_risk_explanation]
        + advice.clothing_recommendations + advice.weather_concerns + advice.reasoning
    ).casefold()
    invented = [name for name, pattern in ABSENT_FIELDS.items() if re.search(pattern, claims)]
    checks.append(Check("no_absent_fields", not invented, ", ".join(invented)))

    unsupported = find_unsupported_numbers(advice.model_dump_json(), facts_text)
    checks.append(Check("numbers_grounded", not unsupported, str(unsupported), hard=False))
    checks.append(Check("first_try_valid", repairs == 0, f"repairs={repairs}", hard=False))
    return checks


def check_agent_turn(turn: AgentTurn, result: AgentResult) -> list[Check]:
    checks = [Check("answered", bool(result.answer.strip()))]

    called = {
        str(r.args.get("city", "")).casefold() for r in result.tool_calls if r.name == "get_weather"
    }
    if turn.expected_cities is not None:
        checks.append(Check(
            "tool_calls",
            called == set(turn.expected_cities),
            f"called={sorted(called)} expected={sorted(turn.expected_cities)}",
        ))
    if turn.max_steps is not None:
        checks.append(Check(
            "steps", result.steps <= turn.max_steps, f"steps={result.steps} max={turn.max_steps}"
        ))
    if turn.answer_any:
        ok = any(_mentions(result.answer, k) for k in turn.answer_any)
        checks.append(Check(
            "answer_content", ok, "" if ok else "none of " + "/".join(turn.answer_any[:5])
        ))
    checks.append(Check(
        "numbers_grounded", not result.unsupported_numbers, str(result.unsupported_numbers), hard=False
    ))
    return checks

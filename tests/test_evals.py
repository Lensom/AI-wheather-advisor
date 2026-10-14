"""The evaluation harness itself: scenarios, checks, compaction. Offline.

If a scenario's *Python* expectations drift (rules changed), these fail before
any money is spent on Gemini calls.
"""

import pytest

from advice_models import WeatherAdvice, validation_context
from agent import AgentResult
from assessment import assess
from evals.checks import check_advice, check_agent_turn
from evals.scenarios import (
    ADVICE_CASES,
    AGENT_CASES,
    START,
    AgentTurn,
    cities,
    daily_curve,
    scenario_loader,
)
from tools import GetWeatherArgs, ToolCallRecord
from weather_client import CityNotFoundError

CASES = {c.id: c for c in ADVICE_CASES}


# --- scenarios ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ADVICE_CASES, ids=lambda c: c.id)
def test_scenario_python_decisions_match_expectations(case):
    a = assess(case.build())
    assert a.outdoor_verdict.level == case.expected_verdict
    assert bool(a.walk_candidates) == case.expect_best_time


def test_daily_curve_peaks_at_15():
    temps = daily_curve(10, 20)
    assert len(temps) == 24
    assert temps[15 - START.hour] == 20 and min(temps) == 10


def test_cold_scenario_prefers_warmest_hours():
    a = assess(CASES["cold_dry"].build())
    assert a.walk_candidates[0].temperature_c == max(h.temperature_c for h in CASES["cold_dry"].build().hourly)


def test_scenario_loader():
    snapshot, others = scenario_loader(GetWeatherArgs(city="rainford"))
    assert snapshot.location.name == "Rainford" and others == []
    with pytest.raises(CityNotFoundError):
        scenario_loader(GetWeatherArgs(city="Atlantis"))


def test_agent_case_ids_unique():
    ids = [c.id for c in AGENT_CASES] + [c.id for c in ADVICE_CASES]
    assert len(ids) == len(set(ids))


# --- advice checks --------------------------------------------------------------------------


def advice(case_id="all_day_rain", **overrides) -> WeatherAdvice:
    """Validated like in the real flow: with the scenario's Python decisions as context."""
    data = {
        "summary": "Rain all day.",
        "clothing_recommendations": ["Waterproof jacket", "Umbrella"],
        "outdoor_activity": "not_recommended",
        "best_time": None,
        "rain_risk_explanation": "High rain risk all day.",
        "weather_concerns": ["Heavy rain"],
        "reasoning": ["rain_risk.overall is high"],
        "missing_information": [],
    } | overrides
    context = validation_context(assess(CASES[case_id].build()))
    return WeatherAdvice.model_validate(data, context=context)


def run_checks(adv: WeatherAdvice, case_id="all_day_rain", facts="", repairs=0):
    case = CASES[case_id]
    return {c.name: c for c in check_advice(case, adv, assess(case.build()), facts, repairs)}


def test_good_advice_passes_all_hard_checks():
    checks = run_checks(advice())
    assert all(c.passed for c in checks.values() if c.hard)


def test_missing_rain_gear_fails():
    checks = run_checks(advice(clothing_recommendations=["Warm sweater"]))
    assert not checks["clothing_required"].passed


def test_forbidden_item_detected_but_negation_ignored():
    case = "mild_dry"
    a = assess(CASES[case].build())
    best = a.walk_candidates[0].time.isoformat()
    base = {"outdoor_activity": "recommended", "best_time": best}
    ok = run_checks(advice(case, **base, clothing_recommendations=["Light jacket", "No need for an umbrella"]), case)
    bad = run_checks(advice(case, **base, clothing_recommendations=["Light jacket", "Take an umbrella"]), case)
    assert ok["clothing_forbidden"].passed
    assert not bad["clothing_forbidden"].passed and "umbrella" in bad["clothing_forbidden"].detail


def test_absent_field_allowed_only_in_missing_information():
    allowed = run_checks(advice(missing_information=["UV index is not provided"]))
    invented = run_checks(advice(weather_concerns=["Heavy rain", "High UV index"]))
    assert allowed["no_absent_fields"].passed
    assert not invented["no_absent_fields"].passed


def test_soft_checks_do_not_fail_case():
    checks = run_checks(advice(summary="Rain all day, 99 mm."), facts="{}", repairs=1)
    assert not checks["numbers_grounded"].passed and not checks["numbers_grounded"].hard
    assert not checks["first_try_valid"].passed and not checks["first_try_valid"].hard


# --- agent checks ---------------------------------------------------------------------------


def agent_result(answer: str, called: list[str], steps: int = 2) -> AgentResult:
    records = [
        ToolCallRecord("get_weather", {"city": c}, "ok", False, 1, {}) for c in called
    ]
    return AgentResult(answer, records, steps, [], session_has_weather_data=bool(called))


def test_expected_calls_exact_set_case_insensitive():
    turn = AgentTurn("q", cities("Sunmere", "Frostvik"))
    ok = {c.name: c for c in check_agent_turn(turn, agent_result("x", ["sunmere", "FROSTVIK"]))}
    extra = {c.name: c for c in check_agent_turn(turn, agent_result("x", ["Sunmere"]))}
    assert ok["tool_calls"].passed and not extra["tool_calls"].passed


def test_no_calls_expected():
    turn = AgentTurn("q", cities())
    checks = {c.name: c for c in check_agent_turn(turn, agent_result("Which city?", ["Kyiv"]))}
    assert not checks["tool_calls"].passed


def test_steps_and_answer_content():
    turn = AgentTurn("q", None, answer_any=("Sunmere",), max_steps=2)
    checks = {c.name: c for c in check_agent_turn(turn, agent_result("Frostvik", [], steps=3))}
    assert not checks["steps"].passed and not checks["answer_content"].passed

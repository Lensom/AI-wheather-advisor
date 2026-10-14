"""Schema, grounding and decision validation of WeatherAdvice. Offline, no API calls."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from advice_models import OUTDOOR_VERDICT, WALK_CANDIDATES, WeatherAdvice, validation_context
from assessment import assess
from weather_models import WeatherSnapshot

FIXTURE = Path(__file__).parent / "fixtures" / "snapshot_berlin.json"


@pytest.fixture
def context() -> dict:
    """Berlin fixture: verdict 'recommended', candidates 16:00, 17:00, 15:00."""
    snapshot = WeatherSnapshot.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    return validation_context(assess(snapshot))


@pytest.fixture
def payload() -> dict:
    return {
        "summary": "Cool and dry.",
        "clothing_recommendations": ["Light jacket"],
        "outdoor_activity": "recommended",
        "best_time": "2026-10-15T16:00:00",
        "rain_risk_explanation": "rain_risk.overall is low (all probabilities below 30%).",
        "weather_concerns": [],
        "reasoning": ["walk_score 97 at 16:00 >= 75 with low rain risk"],
        "missing_information": [],
    }


def validate(data: dict, context: dict | None = None) -> WeatherAdvice:
    return WeatherAdvice.model_validate_json(json.dumps(data), context=context)


# --- shape -----------------------------------------------------------------


def test_valid_payload(payload, context):
    advice = validate(payload, context)
    assert advice.outdoor_activity == "recommended"
    assert advice.best_time == "2026-10-15T16:00:00"


def test_optional_lists_default_to_empty(payload):
    del payload["weather_concerns"], payload["missing_information"]
    advice = validate(payload)
    assert advice.weather_concerns == [] and advice.missing_information == []


@pytest.mark.parametrize(
    "field",
    ["summary", "clothing_recommendations", "outdoor_activity", "reasoning", "rain_risk_explanation"],
)
def test_required_field_missing(payload, field):
    del payload[field]
    with pytest.raises(ValidationError, match=field):
        validate(payload)


def test_unknown_outdoor_activity(payload):
    payload["outdoor_activity"] = "maybe"
    with pytest.raises(ValidationError, match="outdoor_activity"):
        validate(payload)


@pytest.mark.parametrize("field", ["clothing_recommendations", "reasoning"])
def test_list_must_not_be_empty(payload, field):
    payload[field] = []
    with pytest.raises(ValidationError, match=field):
        validate(payload)


def test_blank_strings_rejected(payload):
    payload["summary"] = "   "
    payload["weather_concerns"] = [""]
    with pytest.raises(ValidationError) as exc:
        validate(payload)
    locs = {err["loc"][0] for err in exc.value.errors()}
    assert locs == {"summary", "weather_concerns"}


def test_strings_are_stripped(payload):
    payload["summary"] = "  Cool and dry.  "
    assert validate(payload).summary == "Cool and dry."


def test_best_time_not_a_timestamp(payload):
    payload["best_time"] = "late afternoon"
    with pytest.raises(ValidationError, match="ISO timestamp"):
        validate(payload)


# --- Python decisions (validation context) ------------------------------------


def test_best_time_must_be_walk_candidate(payload, context):
    # 08:00 is a real forecast hour, but not one of Python's top candidates.
    payload["best_time"] = "2026-10-15T08:00:00"
    with pytest.raises(ValidationError, match="not one of the walk candidates"):
        validate(payload, context)


def test_best_time_normalized(payload, context):
    payload["best_time"] = "2026-10-15 15:00"
    assert validate(payload, context).best_time == "2026-10-15T15:00:00"


def test_any_candidate_accepted_not_only_first(payload, context):
    payload["best_time"] = "2026-10-15T17:00:00"
    assert validate(payload, context).best_time == "2026-10-15T17:00:00"


def test_outdoor_activity_must_match_python_verdict(payload, context):
    payload["outdoor_activity"] = "possible_with_caution"
    with pytest.raises(ValidationError, match="must equal assessment.outdoor_verdict.level"):
        validate(payload, context)


def test_null_best_time_rejected_when_candidates_exist(payload, context):
    payload["best_time"] = None
    payload["missing_information"] = ["something"]
    with pytest.raises(ValidationError, match="walk candidates exist"):
        validate(payload, context)


def test_no_candidates_requires_null_best_time(payload):
    context = {WALK_CANDIDATES: set(), OUTDOOR_VERDICT: "not_recommended"}
    payload["outdoor_activity"] = "not_recommended"
    with pytest.raises(ValidationError, match="best_time must be null"):
        validate(payload, context)

    payload["best_time"] = None
    assert validate(payload, context).best_time is None  # no missing_information needed


# --- without context: generic fallback rules --------------------------------------


def test_best_time_not_checked_without_context(payload):
    payload["best_time"] = "2026-10-20T12:00:00"
    assert validate(payload).best_time == "2026-10-20T12:00:00"


def test_best_time_null_with_explanation(payload):
    del payload["best_time"]
    payload["missing_information"] = ["precipitation_probability_pct is null for all hours"]
    assert validate(payload).best_time is None


def test_best_time_null_without_explanation(payload):
    payload["best_time"] = None
    with pytest.raises(ValidationError, match="missing_information must explain"):
        validate(payload)

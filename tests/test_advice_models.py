"""Schema + grounding validation of WeatherAdvice. Offline, no API calls."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from advice_models import WeatherAdvice, validation_context
from weather_models import WeatherSnapshot

FIXTURE = Path(__file__).parent / "fixtures" / "snapshot_berlin.json"


@pytest.fixture
def snapshot() -> WeatherSnapshot:
    return WeatherSnapshot.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def payload() -> dict:
    return {
        "summary": "Cool and dry.",
        "clothing_recommendations": ["Light jacket"],
        "outdoor_activity": "recommended",
        "best_time": "2026-10-15T16:00:00",
        "weather_concerns": [],
        "reasoning": ["temperature_c peaks at 16.3 at 16:00"],
        "missing_information": [],
    }


def validate(data: dict, snapshot: WeatherSnapshot | None = None) -> WeatherAdvice:
    context = validation_context(snapshot) if snapshot else None
    return WeatherAdvice.model_validate_json(json.dumps(data), context=context)


def test_valid_payload(payload, snapshot):
    advice = validate(payload, snapshot)
    assert advice.outdoor_activity == "recommended"
    assert advice.best_time == "2026-10-15T16:00:00"


def test_optional_lists_default_to_empty(payload):
    del payload["weather_concerns"], payload["missing_information"]
    advice = validate(payload)
    assert advice.weather_concerns == [] and advice.missing_information == []


def test_best_time_null_with_explanation(payload):
    del payload["best_time"]
    payload["missing_information"] = ["precipitation_probability_pct is null for all hours"]
    assert validate(payload).best_time is None


def test_best_time_null_without_explanation(payload):
    payload["best_time"] = None
    with pytest.raises(ValidationError, match="missing_information must explain"):
        validate(payload)


@pytest.mark.parametrize("field", ["summary", "clothing_recommendations", "outdoor_activity", "reasoning"])
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


def test_best_time_outside_forecast(payload, snapshot):
    payload["best_time"] = "2026-10-20T12:00:00"
    with pytest.raises(ValidationError, match="not an hourly_forecast time"):
        validate(payload, snapshot)


def test_best_time_normalized(payload, snapshot):
    payload["best_time"] = "2026-10-15 16:00"
    assert validate(payload, snapshot).best_time == "2026-10-15T16:00:00"


def test_best_time_not_checked_without_context(payload):
    payload["best_time"] = "2026-10-20T12:00:00"
    assert validate(payload).best_time == "2026-10-20T12:00:00"

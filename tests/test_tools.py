"""Tool argument validation, execution, errors, cache and budget. Offline."""

import asyncio
from pathlib import Path

import pytest
from google.genai import types

from tools import (
    GET_WEATHER,
    LocationMismatchError,
    GetWeatherArgs,
    WeatherTools,
    pick_location,
)
from weather_client import CityNotFoundError, WeatherAPIError
from weather_models import Location, WeatherSnapshot

FIXTURE = Path(__file__).parent / "fixtures" / "snapshot_berlin.json"


def berlin() -> WeatherSnapshot:
    return WeatherSnapshot.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def loc(name="Paris", country="France", code="FR") -> Location:
    return Location(
        name=name, country=country, country_code=code, latitude=0, longitude=0, timezone="UTC"
    )


class FakeLoader:
    """Stands in for Open-Meteo; records calls, can raise."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []
        self.error = error

    def __call__(self, args):
        self.calls.append((args.city, args.country, args.region))
        if self.error:
            raise self.error
        return berlin(), [loc("Berlin", "United States", "US")]


def call(args: dict | None, name: str = GET_WEATHER, id_: str = "call-1") -> types.FunctionCall:
    return types.FunctionCall(id=id_, name=name, args=args)


def execute(tools: WeatherTools, fc: types.FunctionCall):
    return asyncio.run(tools.execute(fc))


# --- declaration / args --------------------------------------------------------


def test_schema_sent_to_gemini():
    schema = GetWeatherArgs.model_json_schema()
    assert schema["required"] == ["city"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"city", "country", "region"}


def test_args_are_stripped():
    assert GetWeatherArgs.model_validate({"city": "  Lviv "}).city == "Lviv"


# --- success ---------------------------------------------------------------------


def test_ok_result_and_function_response_part():
    loader = FakeLoader()
    part, record = execute(WeatherTools(loader), call({"city": "Berlin", "country": "DE"}))

    assert loader.calls == [("Berlin", "DE", None)]
    response = part.function_response
    assert response.id == "call-1" and response.name == GET_WEATHER  # id must echo the call
    result = response.response
    assert result["status"] == "ok"
    assert result["resolved_location"] == "Berlin, State of Berlin, Germany"
    assert result["other_matches"] == ["Berlin, United States"]
    assert result["facts"]["assessment"]["outdoor_verdict"]["level"] == "recommended"
    assert "rules" not in result["facts"]["notes"]  # sent once in the system instruction
    hourly = result["facts"]["assessment"]["hourly"]
    assert hourly[0] == {"time": "2026-10-14T21:00:00", "rain_risk": "low", "walk_score": 89}
    assert hourly[1]["walk_score"] is None  # 22:00 is outside walk hours
    assert record.status == "ok" and record.cached is False


# --- errors become results, not exceptions ------------------------------------------


@pytest.mark.parametrize(
    "args",
    [None, {}, {"city": ""}, {"city": "   "}, {"city": "Lviv", "date": "tomorrow"}, {"city": 42}],
)
def test_invalid_arguments(args):
    loader = FakeLoader()
    part, record = execute(WeatherTools(loader), call(args))
    assert record.status == "invalid_arguments"
    assert part.function_response.response["status"] == "error"
    assert loader.calls == []  # never reaches the API


def test_unknown_tool():
    loader = FakeLoader()
    _, record = execute(WeatherTools(loader), call({"city": "Lviv"}, name="get_stock_price"))
    assert record.status == "unknown_tool"
    assert "get_weather" in record.result["message"]
    assert loader.calls == []


def test_city_not_found_is_not_cached():
    loader = FakeLoader(CityNotFoundError("No locations found for 'Atlantis'"))
    tools = WeatherTools(loader)
    _, first = execute(tools, call({"city": "Atlantis"}))
    _, second = execute(tools, call({"city": "Atlantis"}))
    assert first.status == second.status == "city_not_found"
    assert len(loader.calls) == 2


def test_weather_api_failure_tells_model_not_to_guess():
    tools = WeatherTools(FakeLoader(WeatherAPIError("HTTP 503")))
    _, record = execute(tools, call({"city": "Lviv"}))
    assert record.status == "weather_unavailable"
    assert "Do not guess" in record.result["message"]


# --- cache and budget ------------------------------------------------------------------


def test_cache_is_case_insensitive_and_free():
    loader = FakeLoader()
    tools = WeatherTools(loader, max_api_calls=1)
    execute(tools, call({"city": "Berlin"}))
    _, again = execute(tools, call({"city": "  BERLIN "}))
    assert again.cached is True and again.status == "ok"
    assert len(loader.calls) == 1 and tools.api_calls == 1


def test_country_and_region_are_part_of_cache_key():
    loader = FakeLoader()
    tools = WeatherTools(loader)
    execute(tools, call({"city": "Paris"}))
    execute(tools, call({"city": "Paris", "country": "US"}))
    execute(tools, call({"city": "Paris", "country": "US", "region": "Texas"}))
    execute(tools, call({"city": "Paris", "country": "us", "region": "TEXAS"}))  # cached
    assert len(loader.calls) == 3


def test_budget_exhausted():
    loader = FakeLoader()
    tools = WeatherTools(loader, max_api_calls=1)
    execute(tools, call({"city": "Berlin"}))
    _, record = execute(tools, call({"city": "Lviv"}))
    assert record.status == "tool_budget_exhausted"
    assert len(loader.calls) == 1


# --- disambiguation ---------------------------------------------------------------------


def test_pick_location_first_hit_without_filters():
    hits = [loc(), loc(country="United States", code="US")]
    chosen, others = pick_location(hits)
    assert chosen is hits[0] and others == [hits[1]]


@pytest.mark.parametrize("country", ["US", "us", "United States", "united states"])
def test_pick_location_by_country_name_or_code(country):
    hits = [loc(), loc(country="United States", code="US")]
    chosen, others = pick_location(hits, country)
    assert chosen.country_code == "US" and others == [hits[0]]


def us_paris(region: str) -> Location:
    return loc(country="United States", code="US").model_copy(update={"admin1": region})


def test_pick_location_by_region():
    hits = [loc(), us_paris("Texas"), us_paris("Tennessee")]
    chosen, _ = pick_location(hits, "US", "tennessee")
    assert chosen.admin1 == "Tennessee"


def test_pick_location_region_substring():
    hit = loc("Lviv", "Ukraine", "UA").model_copy(update={"admin1": "Lviv Oblast"})
    assert pick_location([hit], region="Lviv")[0] is hit


def test_pick_location_mismatch_lists_what_was_found():
    with pytest.raises(LocationMismatchError, match=r"not found in 'Germany' \(found: Paris, France\)"):
        pick_location([loc()], "Germany")
    with pytest.raises(LocationMismatchError, match="'Ohio, US'"):
        pick_location([us_paris("Texas")], "US", "Ohio")
    assert issubclass(LocationMismatchError, CityNotFoundError)  # -> city_not_found result

"""Tools Gemini may call (Stage 5: function calling).

The model never runs code. It returns a FunctionCall (name + JSON args);
we validate the args, run the function, and send back a FunctionResponse
with the same id. Every failure becomes a structured error *result* the
model can react to ("city not found -> ask for the country"), never an
exception that kills the conversation.

The tool returns the same verified facts as the classic flow:
snapshot + Python indicators + Python decisions (assessment).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

import httpx
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from assessment import assess
from indicators import compute_indicators
from prompts import weather_facts
from weather_client import (
    DEFAULT_TIMEOUT,
    CityNotFoundError,
    WeatherAPIError,
    fetch_weather,
    format_location_label,
    geocode_city,
)
from weather_models import Location, WeatherSnapshot

logger = logging.getLogger(__name__)

GET_WEATHER = "get_weather"
DEFAULT_MAX_API_CALLS = 4  # real executions per session; cache hits are free

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class GetWeatherArgs(BaseModel):
    """Arguments of get_weather. Also the source of the JSON schema sent to Gemini."""

    model_config = ConfigDict(extra="forbid")

    city: Annotated[Name, StringConstraints(max_length=100)] = Field(
        description="One city name only, e.g. 'Lviv'. Call once per city."
    )
    country: Annotated[Name, StringConstraints(max_length=60)] | None = Field(
        default=None,
        description=(
            "Country name or ISO 3166-1 alpha-2 code to disambiguate, e.g. 'France' or 'FR'. "
            "Only if the user named or clearly implied it."
        ),
    )
    region: Annotated[Name, StringConstraints(max_length=60)] | None = Field(
        default=None,
        description=(
            "State / province / oblast to disambiguate, e.g. 'Texas' for Paris, Texas. "
            "Only if the user named or clearly implied it."
        ),
    )


GET_WEATHER_DECLARATION = types.FunctionDeclaration(
    name=GET_WEATHER,
    description=(
        "Get verified current weather and the next 24 hours of hourly forecast for a city, "
        "plus indicators and decisions (rain risk, walk candidates, outdoor verdict) "
        "computed in Python. The only source of weather data."
    ),
    parameters_json_schema=GetWeatherArgs.model_json_schema(),
)


class LocationMismatchError(CityNotFoundError):
    """City exists, but not in the requested country / region."""


# validated args -> (snapshot of the chosen place, other places with that name)
WeatherLoader = Callable[[GetWeatherArgs], tuple[WeatherSnapshot, list[Location]]]


def pick_location(
    locations: list[Location],
    country: str | None = None,
    region: str | None = None,
) -> tuple[Location, list[Location]]:
    """Choose a geocoding hit; non-interactive, so the model is told about alternatives."""
    candidates = locations
    if country:
        wanted = country.casefold()
        candidates = [
            loc for loc in candidates
            if wanted in (loc.country.casefold(), (loc.country_code or "").casefold())
        ]
    if region:
        # Substring: "Lviv" should match admin1 "Lviv Oblast".
        wanted = region.casefold()
        candidates = [loc for loc in candidates if wanted in (loc.admin1 or "").casefold()]
    if not candidates:
        found = "; ".join(format_location_label(loc) for loc in locations)
        where = ", ".join(filter(None, [region, country]))
        raise LocationMismatchError(
            f"'{locations[0].name}' not found in '{where}' (found: {found})"
        )
    chosen = candidates[0]
    return chosen, [loc for loc in locations if loc is not chosen]


def load_from_open_meteo(args: GetWeatherArgs) -> tuple[WeatherSnapshot, list[Location]]:
    narrowing = bool(args.country or args.region)
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        # More hits when filtering, so the right one is likely included.
        locations = geocode_city(args.city, count=10 if narrowing else 5, client=client)
        chosen, others = pick_location(locations, args.country, args.region)
        return fetch_weather(chosen, client=client), others


@dataclass
class ToolCallRecord:
    """One executed (or rejected) call, for the CLI trace and logs."""

    name: str
    args: dict[str, Any]
    status: str  # "ok" or an error code
    cached: bool
    duration_ms: int
    result: dict[str, Any]


class WeatherTools:
    """Validates and executes tool calls; caches results for the session."""

    def __init__(
        self,
        loader: WeatherLoader = load_from_open_meteo,
        max_api_calls: int = DEFAULT_MAX_API_CALLS,
    ) -> None:
        self._load = loader
        self.max_api_calls = max_api_calls
        self.api_calls = 0
        self._cache: dict[tuple[str, ...], dict[str, Any]] = {}

    @property
    def declarations(self) -> list[types.Tool]:
        return [types.Tool(function_declarations=[GET_WEATHER_DECLARATION])]

    async def execute(self, call: types.FunctionCall) -> tuple[types.Part, ToolCallRecord]:
        """Run one FunctionCall and wrap the result as a FunctionResponse part."""
        started = time.perf_counter()
        result, cached = await self._run(call)
        record = ToolCallRecord(
            name=call.name or "",
            args=dict(call.args or {}),
            status=result["status"] if result["status"] == "ok" else result["error"],
            cached=cached,
            duration_ms=int((time.perf_counter() - started) * 1000),
            result=result,
        )
        logger.info(
            "tool %s(%s) -> %s%s in %d ms",
            record.name, record.args, record.status, " (cached)" if cached else "", record.duration_ms,
        )
        part = types.Part(
            function_response=types.FunctionResponse(id=call.id, name=call.name, response=result)
        )
        return part, record

    async def _run(self, call: types.FunctionCall) -> tuple[dict[str, Any], bool]:
        if call.name != GET_WEATHER:
            return _error("unknown_tool", f"No tool named {call.name!r}. Available: {GET_WEATHER}."), False

        try:
            args = GetWeatherArgs.model_validate(call.args or {})
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}"
                for e in exc.errors(include_url=False)
            )
            return _error("invalid_arguments", problems), False

        key = tuple((v or "").casefold() for v in (args.city, args.country, args.region))
        if key in self._cache:
            return self._cache[key], True

        if self.api_calls >= self.max_api_calls:
            return _error(
                "tool_budget_exhausted",
                f"Limit of {self.max_api_calls} weather lookups reached. "
                "Answer with the data you already have and say what is missing.",
            ), False

        self.api_calls += 1
        try:
            # Sync HTTP code in a thread, so parallel calls really run in parallel.
            snapshot, others = await asyncio.to_thread(self._load, args)
        except CityNotFoundError as exc:
            return _error("city_not_found", str(exc)), False
        except WeatherAPIError as exc:
            logger.warning("Weather API failed inside tool: %s", exc)
            return _error(
                "weather_unavailable",
                "The weather service failed. Do not guess the weather; tell the user to retry later.",
            ), False

        assessment = assess(snapshot)
        facts = weather_facts(
            # Rules text lives in the agent's system instruction (sent once, not per city).
            snapshot, compute_indicators(snapshot), assessment, include_rules=False
        )
        # Free-form questions target specific hours ("this evening?"), so the agent
        # gets per-hour decisions too; the classic flow leaves them out.
        facts["assessment"]["hourly"] = [
            {"time": h.time.isoformat(), "rain_risk": h.rain_risk, "walk_score": h.walk_score}
            for h in assessment.hourly
        ]
        result = {
            "status": "ok",
            "resolved_location": format_location_label(snapshot.location),
            "other_matches": [format_location_label(loc) for loc in others],
            "facts": facts,
        }
        self._cache[key] = result
        return result, False


def _error(code: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error": code, "message": message}

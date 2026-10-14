"""Open-Meteo integration: geocode city → fetch forecast → normalize.

Design notes for AI Engineering:
- External APIs are unreliable; fail loudly with typed errors.
- Never pass raw JSON to Gemini — only validated WeatherSnapshot.
- Deterministic parsing lives here; interpretation comes later.
"""

from __future__ import annotations

import logging
from datetime import datetime

import httpx
from pydantic import ValidationError

from weather_models import CurrentConditions, HourlyPoint, Location, WeatherSnapshot

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Keep the request surface explicit — easier to audit and test.
CURRENT_FIELDS = (
    "temperature_2m,"
    "apparent_temperature,"
    "relative_humidity_2m,"
    "precipitation,"
    "wind_speed_10m"
)
HOURLY_FIELDS = "temperature_2m,precipitation,precipitation_probability"

DEFAULT_TIMEOUT = 10.0


class WeatherClientError(Exception):
    """Base error for weather retrieval failures."""


class CityNotFoundError(WeatherClientError):
    """Geocoding returned no matches."""


class WeatherAPIError(WeatherClientError):
    """HTTP / transport / unexpected API payload failure."""


def geocode_city(
    city: str,
    *,
    count: int = 5,
    client: httpx.Client | None = None,
) -> list[Location]:
    """Resolve a city name to one or more Location candidates.

    Returns a list because many names are ambiguous (e.g. "London").
    Callers decide whether to take the first hit or ask the user.
    """
    query = city.strip()
    if not query:
        raise ValueError("City name must not be empty")

    params = {
        "name": query,
        "count": count,
        "language": "en",
        "format": "json",
    }

    data = _get_json(GEOCODING_URL, params, client=client)
    results = data.get("results") or []
    if not results:
        raise CityNotFoundError(f"No locations found for '{query}'")

    locations: list[Location] = []
    for item in results:
        try:
            locations.append(
                Location(
                    name=item["name"],
                    country=item.get("country") or item.get("country_code") or "Unknown",
                    latitude=item["latitude"],
                    longitude=item["longitude"],
                    timezone=item["timezone"],
                    admin1=item.get("admin1"),
                    country_code=item.get("country_code"),
                )
            )
        except (KeyError, TypeError, ValidationError) as exc:
            logger.warning("Skipping malformed geocoding result: %s", exc)
            continue

    if not locations:
        raise WeatherAPIError(f"Geocoding returned unusable data for '{query}'")

    return locations


def fetch_weather(
    location: Location,
    *,
    forecast_hours: int = 24,
    client: httpx.Client | None = None,
) -> WeatherSnapshot:
    """Fetch current + hourly forecast and return a validated snapshot."""
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "current": CURRENT_FIELDS,
        "hourly": HOURLY_FIELDS,
        "forecast_hours": forecast_hours,
        "timezone": location.timezone,
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }

    data = _get_json(FORECAST_URL, params, client=client)
    return _normalize_forecast(location, data)


def format_location_label(location: Location) -> str:
    """Human-readable label for disambiguation menus."""
    parts = [location.name, location.country]
    if location.admin1:
        parts.insert(1, location.admin1)
    return ", ".join(parts)


def get_weather_for_city(
    city: str,
    *,
    forecast_hours: int = 24,
    location_index: int = 0,
) -> WeatherSnapshot:
    """Non-interactive path: geocode → pick by index → fetch weather.

    For ambiguous cities, resolve the index in the CLI (see select_location),
    then call fetch_weather with the chosen Location.
    """
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        locations = geocode_city(city, client=client)
        if location_index < 0 or location_index >= len(locations):
            raise ValueError(
                f"location_index {location_index} out of range "
                f"(0–{len(locations) - 1})"
            )
        return fetch_weather(
            locations[location_index],
            forecast_hours=forecast_hours,
            client=client,
        )


def _get_json(
    url: str,
    params: dict,
    *,
    client: httpx.Client | None = None,
) -> dict:
    owns_client = client is None
    http = client or httpx.Client(timeout=DEFAULT_TIMEOUT)

    try:
        response = http.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise WeatherAPIError(f"Request timed out: {url}") from exc
    except httpx.HTTPStatusError as exc:
        raise WeatherAPIError(
            f"HTTP {exc.response.status_code} from weather API"
        ) from exc
    except httpx.RequestError as exc:
        # Do not embed full URL with query in user-facing text if it grows secrets later.
        raise WeatherAPIError("Network error while contacting weather API") from exc
    except ValueError as exc:
        raise WeatherAPIError("Weather API returned invalid JSON") from exc
    finally:
        if owns_client:
            http.close()

    if not isinstance(payload, dict):
        raise WeatherAPIError("Unexpected weather API payload type")
    return payload


def _normalize_forecast(location: Location, data: dict) -> WeatherSnapshot:
    """Map Open-Meteo arrays into typed models. Fail if required fields missing."""
    try:
        current_raw = data["current"]
        current = CurrentConditions(
            observed_at=_parse_time(current_raw["time"]),
            temperature_c=float(current_raw["temperature_2m"]),
            apparent_temperature_c=float(current_raw["apparent_temperature"]),
            relative_humidity_pct=int(current_raw["relative_humidity_2m"]),
            precipitation_mm=float(current_raw["precipitation"]),
            wind_speed_kmh=float(current_raw["wind_speed_10m"]),
        )

        hourly_raw = data["hourly"]
        times = hourly_raw["time"]
        temps = hourly_raw["temperature_2m"]
        precip = hourly_raw["precipitation"]
        probs = hourly_raw.get("precipitation_probability")

        if not (len(times) == len(temps) == len(precip)):
            raise WeatherAPIError("Hourly forecast arrays have mismatched lengths")

        hourly: list[HourlyPoint] = []
        for i, time_str in enumerate(times):
            prob = None
            if probs is not None and i < len(probs) and probs[i] is not None:
                prob = int(probs[i])
            hourly.append(
                HourlyPoint(
                    time=_parse_time(time_str),
                    temperature_c=float(temps[i]),
                    precipitation_mm=float(precip[i] or 0.0),
                    precipitation_probability_pct=prob,
                )
            )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise WeatherAPIError("Failed to normalize weather payload") from exc

    return WeatherSnapshot(location=location, current=current, hourly=hourly)


def _parse_time(value: str) -> datetime:
    # Open-Meteo returns "2026-10-12T19:45" (no timezone offset in the string).
    return datetime.fromisoformat(value)


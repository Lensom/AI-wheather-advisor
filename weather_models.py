"""Normalized weather domain models.

Raw Open-Meteo JSON is messy (parallel hourly arrays, API field names).
These models are the single source of truth for the rest of the app.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class Location(BaseModel):
    """Resolved place from the geocoding API."""

    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    admin1: str | None = None  # region / oblast when available
    country_code: str | None = None  # ISO 3166-1 alpha-2, e.g. "UA"


class CurrentConditions(BaseModel):
    """Current weather, metric units only."""

    observed_at: datetime
    temperature_c: float
    apparent_temperature_c: float
    relative_humidity_pct: int = Field(ge=0, le=100)
    precipitation_mm: float = Field(ge=0)
    wind_speed_kmh: float = Field(ge=0)


class HourlyPoint(BaseModel):
    """One hour of forecast data."""

    time: datetime
    temperature_c: float
    precipitation_mm: float = Field(ge=0)
    precipitation_probability_pct: int | None = Field(default=None, ge=0, le=100)


class WeatherSnapshot(BaseModel):
    """Validated weather package ready for indicators + Gemini."""

    location: Location
    current: CurrentConditions
    hourly: list[HourlyPoint]
    units: dict[str, str] = Field(
        default_factory=lambda: {
            "temperature": "°C",
            "precipitation": "mm",
            "wind_speed": "km/h",
            "humidity": "%",
            "precipitation_probability": "%",
        }
    )

"""Deterministic weather indicators computed in Python.

These numbers are the source of truth for later Gemini interpretation.
They are heuristics with documented thresholds — not a scientific comfort index.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from weather_models import HourlyPoint, WeatherSnapshot


class WeatherIndicators(BaseModel):
    """Pre-computed facts derived from WeatherSnapshot."""

    temp_delta_to_afternoon_c: float | None = Field(
        description="Current temp minus warmest hour in 12:00–17:00 local window"
    )
    max_precip_probability_pct: int | None
    max_precip_probability_at: datetime | None
    warmest_hour_at: datetime | None
    warmest_hour_temp_c: float | None
    driest_hour_at: datetime | None
    driest_hour_precip_mm: float | None
    outdoor_comfort_score: int = Field(
        ge=0,
        le=100,
        description="Heuristic 0–100; see comfort_score() docstring",
    )
    comfort_notes: list[str] = Field(default_factory=list)


def compute_indicators(snapshot: WeatherSnapshot) -> WeatherIndicators:
    """Derive indicators from validated weather data only."""
    hourly = snapshot.hourly
    current_temp = snapshot.current.temperature_c

    afternoon = _afternoon_hours(hourly)
    warmest_afternoon = max(afternoon, key=lambda h: h.temperature_c) if afternoon else None
    temp_delta = None
    if warmest_afternoon is not None:
        temp_delta = round(current_temp - warmest_afternoon.temperature_c, 1)

    max_prob_point = _max_precip_probability(hourly)
    warmest = max(hourly, key=lambda h: h.temperature_c) if hourly else None
    driest = min(hourly, key=lambda h: h.precipitation_mm) if hourly else None

    score, notes = comfort_score(snapshot)

    return WeatherIndicators(
        temp_delta_to_afternoon_c=temp_delta,
        max_precip_probability_pct=(
            max_prob_point.precipitation_probability_pct if max_prob_point else None
        ),
        max_precip_probability_at=max_prob_point.time if max_prob_point else None,
        warmest_hour_at=warmest.time if warmest else None,
        warmest_hour_temp_c=warmest.temperature_c if warmest else None,
        driest_hour_at=driest.time if driest else None,
        driest_hour_precip_mm=driest.precipitation_mm if driest else None,
        outdoor_comfort_score=score,
        comfort_notes=notes,
    )


def comfort_score(snapshot: WeatherSnapshot) -> tuple[int, list[str]]:
    """Simple outdoor comfort heuristic (0–100).

    Formula (documented, not scientifically validated):
      start at 100
      − up to 30 for |apparent_temp − 21°C| (1.5 pts per °C)
      − up to 25 for precipitation probability in the next 6 hours
      − up to 20 for wind above 15 km/h (1 pt per km/h over 15, capped)
      − up to 15 for humidity outside 30–70%
      − 10 if current precipitation > 0.2 mm
      clamp to [0, 100]

    Gemini should interpret this score, not recompute it.
    """
    notes: list[str] = []
    score = 100.0
    cur = snapshot.current

    ideal_c = 21.0
    temp_penalty = min(30.0, abs(cur.apparent_temperature_c - ideal_c) * 1.5)
    if temp_penalty:
        notes.append(
            f"Apparent temp {cur.apparent_temperature_c} C vs ideal ~{ideal_c} C "
            f"(-{temp_penalty:.0f})"
        )
    score -= temp_penalty

    near = snapshot.hourly[:6]
    probs = [
        h.precipitation_probability_pct
        for h in near
        if h.precipitation_probability_pct is not None
    ]
    if probs:
        peak = max(probs)
        precip_penalty = min(25.0, peak * 0.25)
        if precip_penalty:
            notes.append(
                f"Peak precip probability next 6h: {peak}% (-{precip_penalty:.0f})"
            )
        score -= precip_penalty
    else:
        notes.append("Precipitation probability missing for near-term hours")

    wind = cur.wind_speed_kmh
    if wind > 15:
        wind_penalty = min(20.0, wind - 15)
        notes.append(f"Wind {wind} km/h above 15 km/h (-{wind_penalty:.0f})")
        score -= wind_penalty

    humidity = cur.relative_humidity_pct
    if humidity < 30:
        hum_penalty = min(15.0, (30 - humidity) * 0.5)
        notes.append(f"Low humidity {humidity}% (-{hum_penalty:.0f})")
        score -= hum_penalty
    elif humidity > 70:
        hum_penalty = min(15.0, (humidity - 70) * 0.5)
        notes.append(f"High humidity {humidity}% (-{hum_penalty:.0f})")
        score -= hum_penalty

    if cur.precipitation_mm > 0.2:
        notes.append(f"Current precipitation {cur.precipitation_mm} mm (-10)")
        score -= 10.0

    return int(round(max(0.0, min(100.0, score)))), notes


def _afternoon_hours(hourly: list[HourlyPoint]) -> list[HourlyPoint]:
    return [h for h in hourly if 12 <= h.time.hour <= 17]


def _max_precip_probability(hourly: list[HourlyPoint]) -> HourlyPoint | None:
    candidates = [
        h for h in hourly if h.precipitation_probability_pct is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda h: h.precipitation_probability_pct or 0)

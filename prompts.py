"""Grounded prompt for structured weather advice.

Weather JSON is the only allowed source of numbers.
Gemini interprets; it must not invent values.
Response shape is enforced by WeatherAdvice + Gemini structured output.
"""

from __future__ import annotations

import json

from indicators import WeatherIndicators
from weather_models import WeatherSnapshot


def weather_facts_json(snapshot: WeatherSnapshot, indicators: WeatherIndicators) -> str:
    """Serialize verified data for the model."""
    payload = {
        "location": snapshot.location.model_dump(mode="json"),
        "current": snapshot.current.model_dump(mode="json"),
        "hourly_forecast": [h.model_dump(mode="json") for h in snapshot.hourly],
        "indicators": indicators.model_dump(mode="json"),
        "units": snapshot.units,
        "notes": {
            "outdoor_comfort_score": (
                "Heuristic 0-100 computed in Python, not a scientific index. "
                "Interpret it; do not recompute or contradict the supplied value."
            )
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_advice_prompt(snapshot: WeatherSnapshot, indicators: WeatherIndicators) -> str:
    """Grounded instructions; schema is applied separately via response_schema."""
    facts_json = weather_facts_json(snapshot, indicators)
    return f"""You are a weather advisor for a learning AI Engineering project.

## Verified weather data (source of truth)
The JSON below was fetched from Open-Meteo and validated in Python.
Indicators were computed deterministically in Python.

```json
{facts_json}
```

## Rules (must follow)
1. Use ONLY values present in the JSON. Do not invent temperatures, probabilities, times, or scores.
2. Do not invent UV index, air quality, pollen, sunrise/sunset, or other absent fields.
3. If something needed for a recommendation is missing or null, put it in missing_information — do not guess.
4. Treat outdoor_comfort_score as a provided heuristic; interpret it, do not recalculate it.
5. Every item in reasoning must cite supporting field name(s) from the JSON
   (e.g. temperature_c, apparent_temperature_c, max_precip_probability_pct).
6. best_time must be copied exactly from one hourly_forecast "time" value
   (e.g. "2026-10-15T16:00:00"), or null if you cannot justify one.
   If best_time is null, missing_information must say which data prevented choosing one.
7. outdoor_activity must be exactly one of: recommended, possible_with_caution, not_recommended.

Return the structured WeatherAdvice object only (schema is enforced by the API).
"""

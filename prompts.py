"""Grounded prompt for structured weather advice.

Weather JSON is the only allowed source of numbers.
Decisions (rain risk, walk candidates, outdoor verdict) are made in Python
(assessment.py); Gemini explains them and must not override them.
Response shape is enforced by WeatherAdvice + Gemini structured output.
"""

from __future__ import annotations

import json

from assessment import RULES_DOC, WeatherAssessment
from indicators import WeatherIndicators
from weather_models import WeatherSnapshot


def weather_facts_json(
    snapshot: WeatherSnapshot,
    indicators: WeatherIndicators,
    assessment: WeatherAssessment,
) -> str:
    """Serialize verified data and Python decisions for the model."""
    return json.dumps(weather_facts(snapshot, indicators, assessment), ensure_ascii=False, indent=2)


def weather_facts(
    snapshot: WeatherSnapshot,
    indicators: WeatherIndicators,
    assessment: WeatherAssessment,
    *,
    include_rules: bool = True,
) -> dict:
    """Facts payload as a dict (prompt text above, tool results in tools.py).

    include_rules=False drops the rules text: the agent sends it once in the
    system instruction instead of repeating it in every tool result.
    """
    facts = {
        "location": snapshot.location.model_dump(mode="json"),
        "current": snapshot.current.model_dump(mode="json"),
        "hourly_forecast": [h.model_dump(mode="json") for h in snapshot.hourly],
        "indicators": indicators.model_dump(mode="json"),
        # Per-hour assessment stays out: derivable, and ~600 extra tokens per call.
        "assessment": assessment.model_dump(mode="json", exclude={"hourly"}),
        "units": snapshot.units,
        "notes": {
            "outdoor_comfort_score": (
                "Heuristic 0-100 computed in Python, not a scientific index. "
                "Describes current conditions. Interpret it; do not recompute or contradict it."
            ),
            "rules": RULES_DOC,
        },
    }
    if not include_rules:
        del facts["notes"]["rules"]
    return facts


def build_advice_prompt(
    snapshot: WeatherSnapshot,
    indicators: WeatherIndicators,
    assessment: WeatherAssessment,
) -> str:
    """Grounded instructions; schema is applied separately via response_schema."""
    facts_json = weather_facts_json(snapshot, indicators, assessment)
    return f"""You are a weather advisor for a learning AI Engineering project.

## Verified weather data (source of truth)
The JSON below was fetched from Open-Meteo and validated in Python.
"indicators" and "assessment" were computed deterministically in Python
using the rules in notes.rules.

```json
{facts_json}
```

## Your role
Python has already made the decisions. You explain them in plain language
and add judgement where no rule exists (clothing, concerns, wording).

## Rules (must follow)
1. Use ONLY values present in the JSON. Do not invent temperatures, probabilities, times, or scores.
2. Do not invent UV index, air quality, pollen, sunrise/sunset, or other absent fields.
3. If something needed for a recommendation is missing or null, put it in missing_information — do not guess.
4. Everything under "indicators" and "assessment" is final. Explain it; never recompute,
   override, or contradict it (e.g. do not call rain risk low when assessment says moderate).
5. outdoor_activity must equal assessment.outdoor_verdict.level. At least one reasoning item
   must explain that verdict using assessment.outdoor_verdict.reasons and notes.rules.outdoor_verdict.
6. best_time must be copied exactly from one assessment.walk_candidates[].time value
   (normally the first; choose another only if you state a data-based reason in reasoning).
   If walk_candidates is empty, best_time must be null.
7. rain_risk_explanation: explain how assessment.rain_risk (overall, next_3h,
   first_high_risk_at) affects a walk, quoting the thresholds from notes.rules.rain_risk.
8. Every item in reasoning must cite supporting field name(s) from the JSON
   (e.g. temperature_c, walk_score, rain_risk.overall).

Return the structured WeatherAdvice object only (schema is enforced by the API).
"""

"""Structured Gemini advice schema.

Two kinds of rules live here:
- shape rules (types, required fields, enum) - also sent to Gemini as response_schema;
- grounding rules (best_time must be a real forecast hour) - need our weather
  data, so they run only in Pydantic via validation context. See
  validation_context() and LLMClient.structured().
Invalid payloads raise ValidationError; LLMClient sends the errors back once.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from weather_models import WeatherSnapshot

OutdoorActivity = Literal["recommended", "possible_with_caution", "not_recommended"]

# "" and "   " are valid JSON strings but useless advice.
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

FORECAST_TIMES = "forecast_times"


class WeatherAdvice(BaseModel):
    """Grounded AI recommendations — interpret weather facts, do not invent them."""

    summary: NonEmptyStr = Field(description="Plain-language weather summary using only supplied data")
    clothing_recommendations: list[NonEmptyStr] = Field(
        min_length=1,
        description="Clothing suggestions justified by supplied temperatures/precip/wind",
    )
    outdoor_activity: OutdoorActivity = Field(
        description="Overall outdoor convenience assessment"
    )
    best_time: str | None = Field(
        default=None,
        description=(
            "Best walk time: exactly one hourly_forecast 'time' value, or null if insufficient data"
        ),
    )
    weather_concerns: list[NonEmptyStr] = Field(
        default_factory=list,
        description="Inconveniences or risks supported by the data",
    )
    reasoning: list[NonEmptyStr] = Field(
        min_length=1,
        description="Explanations that cite concrete fields from the weather JSON",
    )
    missing_information: list[NonEmptyStr] = Field(
        default_factory=list,
        description="Fields that would help but are absent; empty list if nothing critical is missing",
    )

    @field_validator("best_time")
    @classmethod
    def best_time_in_forecast(cls, value: str | None, info: ValidationInfo) -> str | None:
        """Parse to a timestamp; with context, require it to be a forecast hour."""
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            raise ValueError("must be an ISO timestamp like 2026-10-15T16:00:00") from None

        allowed: set[datetime] | None = (info.context or {}).get(FORECAST_TIMES)
        if allowed is not None and parsed not in allowed:
            if not allowed:
                raise ValueError("no hourly forecast available; best_time must be null")
            raise ValueError(
                f"{value!r} is not an hourly_forecast time "
                f"(available: {min(allowed).isoformat()} .. {max(allowed).isoformat()})"
            )
        return parsed.isoformat()

    @model_validator(mode="after")
    def no_best_time_requires_explanation(self) -> WeatherAdvice:
        """Cross-field rule: if no walk time is given, say which data was missing."""
        if self.best_time is None and not self.missing_information:
            raise ValueError(
                "best_time is null, so missing_information must explain what data is lacking"
            )
        return self


def validation_context(snapshot: WeatherSnapshot) -> dict[str, Any]:
    """Facts the validators check the AI answer against."""
    return {FORECAST_TIMES: {h.time for h in snapshot.hourly}}

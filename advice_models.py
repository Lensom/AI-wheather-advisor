"""Structured Gemini advice schema.

Three kinds of rules live here:
- shape rules (types, required fields, enum) - also sent to Gemini as response_schema;
- grounding rules (best_time must be a real forecast hour) - need our weather data;
- decision rules (Stage 4): outdoor_activity and best_time are decided in Python
  (assessment.py); Gemini must repeat and explain them, never override them.
The last two run only in Pydantic via validation context, see validation_context()
and LLMClient.structured(). Invalid payloads raise ValidationError; LLMClient
sends the errors back once.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from assessment import OutdoorActivity, WeatherAssessment

# "" and "   " are valid JSON strings but useless advice.
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

WALK_CANDIDATES = "walk_candidates"
OUTDOOR_VERDICT = "outdoor_verdict"


class WeatherAdvice(BaseModel):
    """Grounded AI recommendations — interpret weather facts, do not invent them."""

    summary: NonEmptyStr = Field(description="Plain-language weather summary using only supplied data")
    clothing_recommendations: list[NonEmptyStr] = Field(
        min_length=1,
        description="Clothing suggestions justified by supplied temperatures/precip/wind",
    )
    outdoor_activity: OutdoorActivity = Field(
        description="Copy of assessment.outdoor_verdict.level (decided in Python)"
    )
    best_time: str | None = Field(
        default=None,
        description=(
            "Exactly one assessment.walk_candidates[].time value; null only if there are no candidates"
        ),
    )
    rain_risk_explanation: NonEmptyStr = Field(
        description=(
            "How assessment.rain_risk affects a walk, citing its levels and the rain_risk rule thresholds"
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
    def best_time_is_candidate(cls, value: str | None, info: ValidationInfo) -> str | None:
        """Parse to a timestamp; with context, require it to be a Python walk candidate."""
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            raise ValueError("must be an ISO timestamp like 2026-10-15T16:00:00") from None

        allowed: set[datetime] | None = (info.context or {}).get(WALK_CANDIDATES)
        if allowed is not None and parsed not in allowed:
            if not allowed:
                raise ValueError("there are no walk candidates; best_time must be null")
            options = ", ".join(t.isoformat() for t in sorted(allowed))
            raise ValueError(f"{value!r} is not one of the walk candidates ({options})")
        return parsed.isoformat()

    @field_validator("outdoor_activity")
    @classmethod
    def activity_matches_verdict(cls, value: str, info: ValidationInfo) -> str:
        """The verdict is computed in Python; the model may explain it, not change it."""
        expected = (info.context or {}).get(OUTDOOR_VERDICT)
        if expected is not None and value != expected:
            raise ValueError(
                f"must equal assessment.outdoor_verdict.level {expected!r}, got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def null_best_time_is_justified(self, info: ValidationInfo) -> WeatherAdvice:
        """Cross-field rule for best_time = null.

        With context: null is allowed only when Python found no walk candidates.
        Without context: null must be explained in missing_information.
        """
        if self.best_time is not None:
            return self
        candidates = (info.context or {}).get(WALK_CANDIDATES)
        if candidates:
            raise ValueError("best_time is null but walk candidates exist; pick one of them")
        if candidates is None and not self.missing_information:
            raise ValueError(
                "best_time is null, so missing_information must explain what data is lacking"
            )
        return self


def validation_context(assessment: WeatherAssessment) -> dict[str, Any]:
    """Python decisions the validators check the AI answer against."""
    return {
        WALK_CANDIDATES: {c.time for c in assessment.walk_candidates},
        OUTDOOR_VERDICT: assessment.outdoor_verdict.level,
    }

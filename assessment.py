"""Deterministic decisions: rain risk, walk windows, outdoor verdict.

Stage 4 split: Python *decides* with explicit rules, Gemini *explains*.
indicators.py measures (numbers); this module applies thresholds to them
and produces the decisions that Gemini must not override.

All thresholds are documented heuristics, not meteorological standards.
RULES_DOC is sent to Gemini so it can cite the rules it is explaining.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from weather_models import HourlyPoint, WeatherSnapshot

RiskLevel = Literal["low", "moderate", "high", "unknown"]
OutdoorActivity = Literal["recommended", "possible_with_caution", "not_recommended"]

# Rain risk per hour (either condition is enough)
HIGH_PROB_PCT = 60
HIGH_PRECIP_MM = 1.0
MODERATE_PROB_PCT = 30
MODERATE_PRECIP_MM = 0.2

# Walk score per hour
WALK_HOURS = range(7, 22)  # 07:00-21:00 local time
IDEAL_WALK_TEMP_C = 18.0  # lower than the comfort-score ideal: walking warms you up
TEMP_PENALTY_PER_C = 2.0
MAX_TEMP_PENALTY = 40.0
RAIN_PENALTY: dict[RiskLevel, float] = {"low": 0, "moderate": 25, "unknown": 15, "high": 60}

# Candidates and verdict
MIN_CANDIDATE_SCORE = 50
MAX_CANDIDATES = 3
RECOMMENDED_MIN_SCORE = 75
NEXT_HOURS = 3

_SEVERITY: dict[RiskLevel, int] = {"low": 0, "moderate": 1, "high": 2}

RULES_DOC = {
    "rain_risk": (
        f"Per hour: high if precipitation_probability_pct >= {HIGH_PROB_PCT} "
        f"or precipitation_mm >= {HIGH_PRECIP_MM}; moderate if >= {MODERATE_PROB_PCT}% "
        f"or >= {MODERATE_PRECIP_MM} mm; low otherwise; unknown if probability is missing "
        "and precipitation_mm is below the moderate threshold. "
        "overall = worst known level in the forecast window; "
        f"next_3h = worst known level in the first {NEXT_HOURS} hours."
    ),
    "walk_score": (
        f"Only hours {WALK_HOURS.start:02d}:00-{WALK_HOURS.stop - 1:02d}:00 are scored. "
        f"100 - min({MAX_TEMP_PENALTY:.0f}, |temperature_c - {IDEAL_WALK_TEMP_C:.0f}| x "
        f"{TEMP_PENALTY_PER_C:.0f}) - rain penalty (low 0, unknown 15, moderate 25, high 60), "
        "clamped to 0-100. Heuristic."
    ),
    "walk_candidates": (
        f"Scored hours with walk_score >= {MIN_CANDIDATE_SCORE} and rain risk not high, "
        f"best first (ties: temperature closer to {IDEAL_WALK_TEMP_C:.0f} °C, then earlier hour), "
        f"at most {MAX_CANDIDATES}."
    ),
    "outdoor_verdict": (
        "not_recommended if there are no walk candidates; recommended if the best "
        f"candidate has walk_score >= {RECOMMENDED_MIN_SCORE} and low rain risk; "
        "possible_with_caution otherwise. Describes the best walk window in the next "
        "24 h, not current conditions (that is outdoor_comfort_score)."
    ),
}


class HourAssessment(BaseModel):
    time: datetime
    rain_risk: RiskLevel
    walk_score: int | None  # None outside WALK_HOURS


class RainRisk(BaseModel):
    overall: RiskLevel
    next_3h: RiskLevel
    first_high_risk_at: datetime | None
    hours_by_level: dict[RiskLevel, int]


class WalkCandidate(BaseModel):
    time: datetime
    walk_score: int
    rain_risk: RiskLevel
    temperature_c: float


class OutdoorVerdict(BaseModel):
    level: OutdoorActivity
    reasons: list[str]


class WeatherAssessment(BaseModel):
    rain_risk: RainRisk
    walk_candidates: list[WalkCandidate]
    outdoor_verdict: OutdoorVerdict
    hourly: list[HourAssessment]  # for the CLI table; not sent to Gemini


def assess(snapshot: WeatherSnapshot) -> WeatherAssessment:
    """Apply the rules in RULES_DOC to a validated snapshot."""
    points = snapshot.hourly
    risks = [rain_risk_for_hour(p) for p in points]
    hourly = [
        HourAssessment(time=p.time, rain_risk=r, walk_score=walk_score_for_hour(p, r))
        for p, r in zip(points, risks)
    ]
    candidates = walk_candidates(points, hourly)
    return WeatherAssessment(
        rain_risk=_summarize_rain(hourly),
        walk_candidates=candidates,
        outdoor_verdict=outdoor_verdict(candidates, any_walk_hours=any(
            h.walk_score is not None for h in hourly
        )),
        hourly=hourly,
    )


def rain_risk_for_hour(point: HourlyPoint) -> RiskLevel:
    prob = point.precipitation_probability_pct
    mm = point.precipitation_mm
    if (prob is not None and prob >= HIGH_PROB_PCT) or mm >= HIGH_PRECIP_MM:
        return "high"
    if (prob is not None and prob >= MODERATE_PROB_PCT) or mm >= MODERATE_PRECIP_MM:
        return "moderate"
    if prob is None:
        return "unknown"
    return "low"


def walk_score_for_hour(point: HourlyPoint, risk: RiskLevel) -> int | None:
    if point.time.hour not in WALK_HOURS:
        return None
    temp_penalty = min(
        MAX_TEMP_PENALTY, abs(point.temperature_c - IDEAL_WALK_TEMP_C) * TEMP_PENALTY_PER_C
    )
    score = 100.0 - temp_penalty - RAIN_PENALTY[risk]
    return int(round(max(0.0, min(100.0, score))))


def walk_candidates(
    points: list[HourlyPoint], hourly: list[HourAssessment]
) -> list[WalkCandidate]:
    eligible = [
        WalkCandidate(
            time=h.time,
            walk_score=h.walk_score,
            rain_risk=h.rain_risk,
            temperature_c=p.temperature_c,
        )
        for p, h in zip(points, hourly)
        if h.walk_score is not None
        and h.walk_score >= MIN_CANDIDATE_SCORE
        and h.rain_risk != "high"
    ]
    # Tie-break on temperature: in deep cold the temperature penalty is capped, so all
    # hours score the same and "earliest" would pick the coldest morning hour.
    eligible.sort(key=lambda c: (-c.walk_score, abs(c.temperature_c - IDEAL_WALK_TEMP_C), c.time))
    return eligible[:MAX_CANDIDATES]


def outdoor_verdict(candidates: list[WalkCandidate], *, any_walk_hours: bool) -> OutdoorVerdict:
    if not candidates:
        reason = (
            f"No hour {WALK_HOURS.start:02d}:00-{WALK_HOURS.stop - 1:02d}:00 in the forecast window"
            if not any_walk_hours
            else f"No daytime hour has walk_score >= {MIN_CANDIDATE_SCORE} without high rain risk"
        )
        return OutdoorVerdict(level="not_recommended", reasons=[reason])

    best = candidates[0]
    when = best.time.strftime("%Y-%m-%d %H:%M")
    if best.walk_score >= RECOMMENDED_MIN_SCORE and best.rain_risk == "low":
        return OutdoorVerdict(
            level="recommended",
            reasons=[
                f"Best hour {when}: walk_score {best.walk_score} >= {RECOMMENDED_MIN_SCORE} "
                "and low rain risk"
            ],
        )

    reasons = []
    if best.walk_score < RECOMMENDED_MIN_SCORE:
        reasons.append(
            f"Best hour {when}: walk_score {best.walk_score} < {RECOMMENDED_MIN_SCORE}"
        )
    if best.rain_risk != "low":
        reasons.append(f"Best hour {when}: rain risk is {best.rain_risk}")
    return OutdoorVerdict(level="possible_with_caution", reasons=reasons)


def _worst(levels: list[RiskLevel]) -> RiskLevel:
    known = [lvl for lvl in levels if lvl != "unknown"]
    if not known:
        return "unknown"
    return max(known, key=_SEVERITY.__getitem__)


def _summarize_rain(hourly: list[HourAssessment]) -> RainRisk:
    levels = [h.rain_risk for h in hourly]
    first_high = next((h.time for h in hourly if h.rain_risk == "high"), None)
    return RainRisk(
        overall=_worst(levels),
        next_3h=_worst(levels[:NEXT_HOURS]),
        first_high_risk_at=first_high,
        hours_by_level={lvl: levels.count(lvl) for lvl in ("low", "moderate", "high", "unknown")},
    )

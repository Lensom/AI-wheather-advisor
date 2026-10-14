"""Deterministic rules in assessment.py. Pure Python, no API calls."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from assessment import (
    HourAssessment,
    WalkCandidate,
    assess,
    outdoor_verdict,
    rain_risk_for_hour,
    walk_score_for_hour,
)
from weather_models import CurrentConditions, HourlyPoint, Location, WeatherSnapshot

FIXTURE = Path(__file__).parent / "fixtures" / "snapshot_berlin.json"
NOON = datetime(2026, 10, 15, 12, 0)


def hour(time=NOON, temp=18.0, mm=0.0, prob: int | None = 0) -> HourlyPoint:
    return HourlyPoint(
        time=time, temperature_c=temp, precipitation_mm=mm, precipitation_probability_pct=prob
    )


def snapshot_of(points: list[HourlyPoint]) -> WeatherSnapshot:
    return WeatherSnapshot(
        location=Location(name="X", country="Y", latitude=0, longitude=0, timezone="UTC"),
        current=CurrentConditions(
            observed_at=points[0].time,
            temperature_c=points[0].temperature_c,
            apparent_temperature_c=points[0].temperature_c,
            relative_humidity_pct=50,
            precipitation_mm=0,
            wind_speed_kmh=5,
        ),
        hourly=points,
    )


def day(start_hour: int, n: int, **kwargs) -> list[HourlyPoint]:
    start = NOON.replace(hour=start_hour)
    return [hour(time=start + timedelta(hours=i), **kwargs) for i in range(n)]


# --- rain risk: boundaries of each threshold ---------------------------------


@pytest.mark.parametrize(
    ("prob", "mm", "expected"),
    [
        (0, 0.0, "low"),
        (29, 0.19, "low"),
        (30, 0.0, "moderate"),  # probability threshold
        (0, 0.2, "moderate"),  # amount threshold alone is enough
        (59, 0.99, "moderate"),
        (60, 0.0, "high"),
        (0, 1.0, "high"),
        (None, 0.0, "unknown"),  # cannot rule rain out
        (None, 0.5, "moderate"),  # forecast amount still counts
        (None, 1.5, "high"),
    ],
)
def test_rain_risk_for_hour(prob, mm, expected):
    assert rain_risk_for_hour(hour(prob=prob, mm=mm)) == expected


# --- walk score -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("temp", "risk", "expected"),
    [
        (18.0, "low", 100),
        (8.0, "low", 80),  # 10 °C off x 2
        (-30.0, "low", 60),  # temperature penalty capped at 40
        (28.0, "moderate", 55),  # 20 + 25
        (18.0, "unknown", 85),
        (18.0, "high", 40),
        (-30.0, "high", 0),  # clamped at 0
    ],
)
def test_walk_score(temp, risk, expected):
    assert walk_score_for_hour(hour(temp=temp), risk) == expected


@pytest.mark.parametrize("h", [6, 22, 0, 3])
def test_night_hours_not_scored(h):
    assert walk_score_for_hour(hour(time=NOON.replace(hour=h)), "low") is None


@pytest.mark.parametrize("h", [7, 21])
def test_walk_hours_are_inclusive(h):
    assert walk_score_for_hour(hour(time=NOON.replace(hour=h)), "low") == 100


# --- candidates -------------------------------------------------------------------


def test_candidates_best_first_max_three_ties_earlier():
    temps = [10.0, 18.0, 17.0, 19.0, 18.0]  # scores 84, 100, 98, 98, 100
    points = [hour(time=NOON + timedelta(hours=i), temp=t) for i, t in enumerate(temps)]
    times = [c.time.hour for c in assess(snapshot_of(points)).walk_candidates]
    assert times == [13, 16, 14]


def test_capped_cold_ties_prefer_warmest_hours():
    # Found by evals: every hour scores 60 (penalty cap), "earliest" picked the coldest.
    temps = [-9.0, -8.0, -6.0, -4.0, -3.0, -5.0]
    points = [hour(time=NOON.replace(hour=7) + timedelta(hours=i), temp=t) for i, t in enumerate(temps)]
    candidates = assess(snapshot_of(points)).walk_candidates
    assert {c.walk_score for c in candidates} == {60}
    assert [c.temperature_c for c in candidates] == [-3.0, -4.0, -5.0]


def test_candidates_exclude_high_risk_low_score_and_night():
    points = [
        hour(time=NOON.replace(hour=3)),  # night
        hour(time=NOON.replace(hour=10), prob=80),  # high rain risk
        hour(time=NOON.replace(hour=11), temp=0),  # 100 - 36 = 64 -> kept
        # Temperature alone can't go below 60 (penalty cap 40); rain is needed:
        hour(time=NOON.replace(hour=12), temp=-5, prob=40),  # 100 - 40 - 25 = 35 -> dropped
    ]
    candidates = assess(snapshot_of(points)).walk_candidates
    assert [c.time.hour for c in candidates] == [11]


# --- verdict ----------------------------------------------------------------------


def candidate(score: int, risk="low") -> WalkCandidate:
    return WalkCandidate(time=NOON, walk_score=score, rain_risk=risk, temperature_c=18)


def test_verdict_recommended():
    v = outdoor_verdict([candidate(75)], any_walk_hours=True)
    assert v.level == "recommended"


@pytest.mark.parametrize(
    ("c", "reason"),
    [(candidate(74), "< 75"), (candidate(80, "moderate"), "moderate"), (candidate(90, "unknown"), "unknown")],
)
def test_verdict_caution(c, reason):
    v = outdoor_verdict([c], any_walk_hours=True)
    assert v.level == "possible_with_caution"
    assert any(reason in r for r in v.reasons)


def test_verdict_not_recommended_when_all_daytime_rainy():
    a = assess(snapshot_of(day(8, 12, prob=90)))
    assert a.walk_candidates == []
    assert a.outdoor_verdict.level == "not_recommended"
    assert "high rain risk" in a.outdoor_verdict.reasons[0]


def test_verdict_not_recommended_without_daytime_hours():
    v = outdoor_verdict([], any_walk_hours=False)
    assert v.level == "not_recommended"
    assert "in the forecast window" in v.reasons[0]


# --- rain summary -------------------------------------------------------------------


def test_rain_summary_worst_and_first_high():
    probs = [0, 10, 0, 40, 70, 0]
    points = [hour(time=NOON + timedelta(hours=i), prob=p) for i, p in enumerate(probs)]
    rain = assess(snapshot_of(points)).rain_risk
    assert rain.next_3h == "low"
    assert rain.overall == "high"
    assert rain.first_high_risk_at == NOON + timedelta(hours=4)
    assert rain.hours_by_level == {"low": 4, "moderate": 1, "high": 1, "unknown": 0}


def test_rain_summary_unknown_only_when_nothing_known():
    assert assess(snapshot_of(day(8, 3, prob=None))).rain_risk.overall == "unknown"
    mixed = day(8, 2, prob=None) + [hour(time=NOON.replace(hour=10), prob=40)]
    assert assess(snapshot_of(mixed)).rain_risk.overall == "moderate"


# --- real fixture -------------------------------------------------------------------


def test_berlin_fixture():
    snapshot = WeatherSnapshot.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    a = assess(snapshot)
    assert a.rain_risk.overall == "low"
    assert a.outdoor_verdict.level == "recommended"
    # 15:00 (15.8 °C) and 17:00 (16.2 °C) tie at 96; 17:00 is closer to 18 °C.
    assert [c.time.hour for c in a.walk_candidates] == [16, 17, 15]
    assert len(a.hourly) == len(snapshot.hourly)
    assert all(isinstance(h, HourAssessment) for h in a.hourly)

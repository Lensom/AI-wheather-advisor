"""Evaluation scenarios: synthetic, deterministic weather + expectations.

Synthetic on purpose: live weather changes every hour, so results would not be
comparable between runs. Each scenario targets one behaviour (rain gear, cold,
heat, missing data, a Python verdict the model might want to override...).

Keyword expectations are a cheap rubric, not ground truth: a group like
("umbrella", "waterproof", ...) passes if the model mentions any of them.
The optional LLM judge (evals/judge.py) covers what keywords cannot.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from assessment import OutdoorActivity
from weather_client import CityNotFoundError
from weather_models import CurrentConditions, HourlyPoint, Location, WeatherSnapshot

START = datetime(2026, 10, 15, 6, 0)  # forecast window: 06:00 -> 05:00 next day

RAIN_GEAR = ("umbrella", "raincoat", "rain jacket", "waterproof", "water-resistant", "rain boots")
WARM_OUTER = ("coat", "parka", "down", "insulated", "winter jacket", "puffer")
WARM_ACCESSORIES = ("gloves", "mittens", "hat", "beanie", "scarf")
LIGHT_CLOTHES = ("light", "breathable", "linen", "shorts", "t-shirt", "cotton")
LAYERS = ("jacket", "sweater", "layer", "cardigan", "hoodie", "long sleeve", "long-sleeve", "fleece")
WIND_GEAR = ("windproof", "wind-resistant", "windbreaker", "shell", "jacket")
RAIN_WORDS = ("rain", "precip", "shower", "wet")


def daily_curve(low: float, high: float) -> list[float]:
    """24 hourly temperatures from START: minimum around 03:00, maximum at 15:00."""
    temps = []
    for i in range(24):
        hour = (START.hour + i) % 24
        x = math.cos((hour - 15) / 24 * 2 * math.pi)  # 1 at 15:00, -1 at 03:00
        temps.append(round(low + (high - low) * (x + 1) / 2, 1))
    return temps


def make_snapshot(
    city: str,
    temps: list[float],
    *,
    rain: dict[int, tuple[int | None, float]] | None = None,
    default_prob: int | None = 0,
    wind_kmh: float = 8.0,
    humidity: int = 60,
) -> WeatherSnapshot:
    """rain maps hour-of-day -> (probability %, mm); other hours: (default_prob, 0.0)."""
    rain = rain or {}
    hourly = []
    for i, temp in enumerate(temps):
        time = START + timedelta(hours=i)
        prob, mm = rain.get(time.hour, (default_prob, 0.0))
        hourly.append(
            HourlyPoint(
                time=time, temperature_c=temp, precipitation_mm=mm, precipitation_probability_pct=prob
            )
        )
    return WeatherSnapshot(
        location=Location(
            name=city, country="Testland", country_code="TL",
            latitude=50.0, longitude=10.0, timezone="Europe/Berlin",
        ),
        current=CurrentConditions(
            observed_at=START,
            temperature_c=temps[0],
            apparent_temperature_c=round(temps[0] - 1.5, 1),
            relative_humidity_pct=humidity,
            precipitation_mm=hourly[0].precipitation_mm,
            wind_speed_kmh=wind_kmh,
        ),
        hourly=hourly,
    )


def mild_dry(city: str = "Sunmere") -> WeatherSnapshot:
    return make_snapshot(city, daily_curve(12, 19))


def cold_dry(city: str = "Frostvik") -> WeatherSnapshot:
    return make_snapshot(city, daily_curve(-9, -3), wind_kmh=10, humidity=75)


def hot_dry(city: str = "Sunbake") -> WeatherSnapshot:
    return make_snapshot(city, daily_curve(24, 34), humidity=40)


def rainy_afternoon(city: str = "Showerton") -> WeatherSnapshot:
    return make_snapshot(city, daily_curve(10, 16), rain={h: (85, 1.6) for h in range(12, 19)})


def all_day_rain(city: str = "Rainford") -> WeatherSnapshot:
    return make_snapshot(city, daily_curve(8, 12), rain={h: (90, 2.5) for h in range(24)}, humidity=95)


def drizzle(city: str = "Mistvale") -> WeatherSnapshot:
    return make_snapshot(city, daily_curve(11, 15), rain={h: (40, 0.1) for h in range(24)}, humidity=88)


def missing_probability(city: str = "Blindham") -> WeatherSnapshot:
    return make_snapshot(city, daily_curve(13, 18), default_prob=None)


def windy(city: str = "Breezeport") -> WeatherSnapshot:
    # Hourly data has no wind, so the walk score ignores it: Python says "recommended",
    # the model must keep that verdict but should still flag the wind.
    return make_snapshot(city, daily_curve(9, 13), wind_kmh=48)


@dataclass(frozen=True)
class AdviceCase:
    """Classic flow: one snapshot -> structured WeatherAdvice."""

    id: str
    description: str
    build: Callable[[], WeatherSnapshot]
    expected_verdict: OutdoorActivity
    expect_best_time: bool  # False: best_time must be null
    clothing_required: tuple[tuple[str, ...], ...] = ()  # every group: at least one keyword
    clothing_forbidden: tuple[str, ...] = ()
    concerns_required: tuple[tuple[str, ...], ...] = ()
    missing_required: tuple[tuple[str, ...], ...] = ()


ADVICE_CASES: list[AdviceCase] = [
    AdviceCase(
        "mild_dry", "12-19 °C, no rain", mild_dry, "recommended", True,
        clothing_required=(LAYERS,),
        clothing_forbidden=("umbrella", "raincoat", "parka", "down jacket"),
    ),
    AdviceCase(
        "cold_dry", "-9..-3 °C, dry: temperature penalty capped -> caution", cold_dry,
        "possible_with_caution", True,
        clothing_required=(WARM_OUTER, WARM_ACCESSORIES),
        clothing_forbidden=("shorts", "sandals"),
    ),
    AdviceCase(
        "hot_dry", "24-34 °C, dry", hot_dry, "recommended", True,
        clothing_required=(LIGHT_CLOTHES,),
        clothing_forbidden=("coat", "gloves", "scarf", "thermal", "sweater"),
        concerns_required=(("heat", "hot", "warm", "34"),),
    ),
    AdviceCase(
        "rainy_afternoon", "heavy rain 12-18, dry morning/evening", rainy_afternoon,
        "recommended", True,
        clothing_required=(RAIN_GEAR,),
        concerns_required=(RAIN_WORDS,),
    ),
    AdviceCase(
        "all_day_rain", "90% / 2.5 mm every hour: no walk candidates", all_day_rain,
        "not_recommended", False,
        clothing_required=(RAIN_GEAR,),
        concerns_required=(RAIN_WORDS,),
    ),
    AdviceCase(
        "drizzle", "40% / 0.1 mm all day: moderate risk", drizzle, "possible_with_caution", True,
        clothing_required=(RAIN_GEAR,),
        concerns_required=(RAIN_WORDS,),
    ),
    AdviceCase(
        "missing_probability", "precipitation probability absent: unknown risk", missing_probability,
        "possible_with_caution", True,
        missing_required=(("probab",),),
    ),
    AdviceCase(
        "windy", "48 km/h wind; Python verdict ignores wind (no hourly wind data)", windy,
        "recommended", True,
        clothing_required=(WIND_GEAR,),
        concerns_required=(("wind",),),
    ),
]


# --- agent (tool calling) -----------------------------------------------------------

CITY_SCENARIOS: dict[str, Callable[[str], WeatherSnapshot]] = {
    "sunmere": mild_dry,
    "frostvik": cold_dry,
    "rainford": all_day_rain,
    "breezeport": windy,
}


def scenario_loader(args) -> tuple[WeatherSnapshot, list[Location]]:
    """Replaces Open-Meteo in agent evals: fictional cities -> fixed scenarios."""
    build = CITY_SCENARIOS.get(args.city.casefold())
    if build is None:
        raise CityNotFoundError(f"No locations found for '{args.city}'")
    return build(args.city.title()), []


@dataclass(frozen=True)
class AgentTurn:
    question: str
    expected_cities: frozenset[str] | None = None  # None: don't check; empty: no calls allowed
    answer_any: tuple[str, ...] = ()  # answer must mention at least one (case-insensitive)
    max_steps: int | None = None  # e.g. 2 = all calls in one parallel turn, then answer


@dataclass(frozen=True)
class AgentCase:
    id: str
    description: str
    turns: list[AgentTurn] = field(default_factory=list)


def cities(*names: str) -> frozenset[str]:
    return frozenset(n.casefold() for n in names)


AGENT_CASES: list[AgentCase] = [
    AgentCase("umbrella", "single city, rain all day", [
        AgentTurn("Should I take an umbrella in Rainford today?", cities("Rainford"),
                  answer_any=("yes", "umbrella", "waterproof")),
    ]),
    AgentCase("compare_two", "two cities -> parallel calls in one step", [
        AgentTurn("Sunmere or Frostvik: where is a walk better today?", cities("Sunmere", "Frostvik"),
                  answer_any=("Sunmere",), max_steps=2),
    ]),
    AgentCase("no_city", "no city -> ask, no tool call", [
        AgentTurn("What should I wear today?", cities(), answer_any=("city", "where", "location")),
    ]),
    AgentCase("off_topic", "not about weather -> decline, no tool call", [
        AgentTurn("Who won the 2022 World Cup?", cities(), answer_any=("weather",)),
    ]),
    AgentCase("unknown_city", "tool returns city_not_found -> no invented weather", [
        AgentTurn("Is it going to rain in Atlantis?", cities("Atlantis"),
                  answer_any=("find", "found", "country", "spell")),
    ]),
    AgentCase("beyond_forecast", "asks about next week -> must say data covers 24 h", [
        AgentTurn("What will the weather be in Sunmere next week?", None, answer_any=("24",)),
    ]),
    AgentCase("follow_ups", "multi-turn: history, compaction, cache", [
        AgentTurn("Can I go for a walk in Breezeport today?", cities("Breezeport")),
        AgentTurn("What should I wear there?", None, answer_any=WIND_GEAR + LAYERS),
        AgentTurn("How strong is the wind right now?", None, answer_any=("48",)),
    ]),
]

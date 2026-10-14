"""CLI: city -> weather -> indicators -> grounded Gemini advice."""

from __future__ import annotations

import asyncio
import logging
import sys

import httpx

from advice_models import WeatherAdvice, validation_context
from grounding import find_unsupported_numbers
from indicators import WeatherIndicators, compute_indicators
from llm_client import LLMClient, StructuredOutputError
from prompts import build_advice_prompt, weather_facts_json
from weather_client import (
    CityNotFoundError,
    WeatherAPIError,
    fetch_weather,
    format_location_label,
    geocode_city,
)
from weather_models import Location, WeatherSnapshot

logger = logging.getLogger(__name__)


def _configure_stdout() -> None:
    # Avoid Windows cp1251 crashes on degree signs / dashes in pipes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def select_location(locations: list[Location]) -> Location:
    """Ask the user to pick when geocoding returns multiple matches."""
    if len(locations) == 1:
        return locations[0]

    print("\nSeveral cities match. Choose one:")
    for i, loc in enumerate(locations, start=1):
        print(f"  {i}. {format_location_label(loc)}")

    while True:
        raw = input(f"Enter number (1-{len(locations)}): ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= choice <= len(locations):
            return locations[choice - 1]
        print(f"Out of range. Use 1-{len(locations)}.")


def prompt_city() -> str:
    while True:
        city = input("Enter city name: ").strip()
        if city:
            return city
        print("City name must not be empty.")


def load_weather_for_city(city: str) -> WeatherSnapshot:
    """Geocode (with disambiguation) and fetch a validated snapshot."""
    with httpx.Client(timeout=10.0) as client:
        locations = geocode_city(city, client=client)
        chosen = select_location(locations)
        print(f"\nSelected: {format_location_label(chosen)}")
        return fetch_weather(chosen, client=client)


def display_snapshot(snapshot: WeatherSnapshot, indicators: WeatherIndicators) -> None:
    loc = snapshot.location
    cur = snapshot.current
    units = snapshot.units

    print("\n" + "=" * 60)
    print(f"  {format_location_label(loc)}")
    print(f"  timezone: {loc.timezone}")
    print(f"  coords: {loc.latitude:.4f}, {loc.longitude:.4f}")
    print("=" * 60)

    print("\nCurrent conditions")
    print(f"  observed at:            {cur.observed_at.isoformat()}")
    print(f"  temperature:            {cur.temperature_c} {units['temperature']}")
    print(
        f"  apparent temperature:   {cur.apparent_temperature_c} {units['temperature']}"
    )
    print(f"  relative humidity:      {cur.relative_humidity_pct} {units['humidity']}")
    print(f"  wind speed:             {cur.wind_speed_kmh} {units['wind_speed']}")
    print(f"  precipitation:          {cur.precipitation_mm} {units['precipitation']}")

    print("\nDeterministic indicators (Python, not AI)")
    print(f"  outdoor comfort score:  {indicators.outdoor_comfort_score}/100  [heuristic]")
    if indicators.temp_delta_to_afternoon_c is not None:
        print(
            f"  temp vs afternoon peak: {indicators.temp_delta_to_afternoon_c:+.1f} "
            f"{units['temperature']}"
        )
    else:
        print("  temp vs afternoon peak: n/a (no afternoon hours in window)")
    if indicators.max_precip_probability_pct is not None:
        when = (
            indicators.max_precip_probability_at.strftime("%Y-%m-%d %H:%M")
            if indicators.max_precip_probability_at
            else "?"
        )
        print(
            f"  max precip probability: {indicators.max_precip_probability_pct}"
            f"{units['precipitation_probability']} at {when}"
        )
    else:
        print("  max precip probability: n/a")
    if indicators.warmest_hour_at is not None:
        print(
            f"  warmest hour:           {indicators.warmest_hour_temp_c}"
            f"{units['temperature']} at "
            f"{indicators.warmest_hour_at.strftime('%Y-%m-%d %H:%M')}"
        )
    if indicators.driest_hour_at is not None:
        print(
            f"  driest hour:            {indicators.driest_hour_precip_mm}"
            f"{units['precipitation']} at "
            f"{indicators.driest_hour_at.strftime('%Y-%m-%d %H:%M')}"
        )
    if indicators.comfort_notes:
        print("  score notes:")
        for note in indicators.comfort_notes:
            print(f"    - {note}")

    print(f"\nHourly forecast (next {len(snapshot.hourly)} h)")
    print(
        f"  {'Time':<16} {'Temp':>7} {'Precip':>8} {'Prob':>6}"
    )
    print(
        f"  {'-' * 16} {'-' * 7} {'-' * 8} {'-' * 6}"
    )
    for point in snapshot.hourly:
        prob = (
            f"{point.precipitation_probability_pct}%"
            if point.precipitation_probability_pct is not None
            else "n/a"
        )
        print(
            f"  {point.time.strftime('%Y-%m-%d %H:%M'):<16} "
            f"{point.temperature_c:>6.1f}° "
            f"{point.precipitation_mm:>7.1f}mm "
            f"{prob:>6}"
        )


async def generate_advice(
    snapshot: WeatherSnapshot,
    indicators: WeatherIndicators,
) -> tuple[WeatherAdvice, str, list[float]]:
    """Return validated structured advice, usage summary, unsupported numbers."""
    llm = LLMClient()
    prompt = build_advice_prompt(snapshot, indicators)
    advice = await llm.structured(
        prompt,
        schema=WeatherAdvice,
        context=validation_context(snapshot),
    )
    # Soft check: logged and shown, never a reason to reject (too many false positives).
    unsupported = find_unsupported_numbers(
        advice.model_dump_json(), weather_facts_json(snapshot, indicators)
    )
    return advice, llm.usage.summary(), unsupported


def _print_list(title: str, items: list[str]) -> None:
    print(f"\n{title}")
    if not items:
        print("  (none)")
        return
    for item in items:
        print(f"  - {item}")


def display_advice(advice: WeatherAdvice, usage: str, unsupported: list[float]) -> None:
    print("\n" + "=" * 60)
    print("  AI advice (grounded + structured, validated)")
    print("=" * 60)
    print(f"\nSummary\n  {advice.summary}")
    _print_list("Clothing", advice.clothing_recommendations)
    print(f"\nOutdoor activity\n  {advice.outdoor_activity}")
    print(
        f"\nBest walk time\n  {advice.best_time if advice.best_time else 'insufficient data'}"
    )
    _print_list("Weather concerns", advice.weather_concerns)
    _print_list("Reasoning", advice.reasoning)
    _print_list("Missing information", advice.missing_information)
    if unsupported:
        print(f"\n[grounding check] numbers not found in weather data: {unsupported}")
    print(f"\n[token usage] {usage}")


def main() -> int:
    _configure_stdout()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Gemini SDK is noisy at INFO during normal runs.
    logging.getLogger("google_genai").setLevel(logging.WARNING)

    print("AI Weather Advisor\n")
    city = prompt_city()

    try:
        snapshot = load_weather_for_city(city)
        indicators = compute_indicators(snapshot)
    except CityNotFoundError as exc:
        print(f"\nError: {exc}")
        return 1
    except WeatherAPIError as exc:
        logger.exception("Weather API failure")
        print(f"\nError: could not fetch weather ({exc})")
        return 1
    except ValueError as exc:
        print(f"\nError: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130

    display_snapshot(snapshot, indicators)

    try:
        advice, usage, unsupported = asyncio.run(generate_advice(snapshot, indicators))
    except StructuredOutputError as exc:
        print(f"\nGemini returned invalid advice ({exc.attempts} attempts):")
        for problem in exc.errors:
            print(f"  - {problem}")
        return 1
    except ValueError as exc:
        print(f"\nGemini error: {exc}")
        return 1
    except Exception as exc:
        logger.exception("Gemini advice failed")
        print(f"\nGemini error: {type(exc).__name__}: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130

    display_advice(advice, usage, unsupported)
    return 0


if __name__ == "__main__":
    sys.exit(main())

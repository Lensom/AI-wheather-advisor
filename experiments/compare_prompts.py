"""Stage 2 experiment: same weather dataset, several prompts, compare grounding.

Run from the project root:
    python -m experiments.compare_prompts             # full fixture
    python -m experiments.compare_prompts --degraded  # precip probability removed

--degraded tests the "report missing information" rule: a well-grounded
prompt should say rain probability is unavailable instead of guessing it.

Free-text output on purpose: this is what Stage 3 (structured output) replaces.
Costs one Gemini call per variant.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from pathlib import Path

from grounding import find_unsupported_numbers
from indicators import compute_indicators
from llm_client import LLMClient
from prompts import weather_facts_json
from weather_models import WeatherSnapshot

FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "snapshot_berlin.json"


def naive(facts: str) -> str:
    return f"Here is the weather data:\n{facts}\n\nWhat should I wear and when should I go for a walk?"


def grounded(facts: str) -> str:
    return f"""You are a weather advisor. The JSON below is the ONLY source of truth.

{facts}

Rules:
- Use only values present in the JSON. Never invent numbers, times or probabilities.
- Do not mention UV, air quality, pollen, sunrise/sunset unless present.
- If information needed for advice is missing or null, say so explicitly.

Give: a short summary, clothing advice, whether outdoor activity is advisable,
and the best hour for a walk (an hourly_forecast timestamp)."""


def grounded_cited(facts: str) -> str:
    return grounded(facts) + """

For every claim, append the JSON field(s) it is based on in square brackets,
e.g. "Mild now (12.5 °C) [current.temperature_c]".
End with a section "Missing information:" listing absent data (or "none")."""


VARIANTS: dict[str, Callable[[str], str]] = {
    "naive": naive,
    "grounded": grounded,
    "grounded_cited": grounded_cited,
}


def load_snapshot(degraded: bool) -> WeatherSnapshot:
    snapshot = WeatherSnapshot.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    if degraded:
        for point in snapshot.hourly:
            point.precipitation_probability_pct = None
    return snapshot


async def run(degraded: bool) -> None:
    snapshot = load_snapshot(degraded)
    facts = weather_facts_json(snapshot, compute_indicators(snapshot))
    llm = LLMClient()

    rows: list[tuple[str, int, list[float], int]] = []
    for name, build in VARIANTS.items():
        before = llm.usage.total_tokens
        text = await llm.generate(build(facts))
        tokens = llm.usage.total_tokens - before
        unsupported = find_unsupported_numbers(text, facts)
        rows.append((name, len(text), unsupported, tokens))

        print("\n" + "=" * 70)
        print(f"  {name}")
        print("=" * 70)
        print(text.strip())

    print("\n" + "=" * 70)
    print(f"  Comparison ({'degraded' if degraded else 'full'} dataset)")
    print("=" * 70)
    print(f"  {'variant':<16} {'chars':>6} {'tokens':>7}  unsupported numbers")
    for name, chars, unsupported, tokens in rows:
        print(f"  {name:<16} {chars:>6} {tokens:>7}  {unsupported or '-'}")
    print("\n  Unsupported = number in the answer not found in the facts JSON (heuristic).")
    print("  Also read the outputs: did each variant admit missing data?")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degraded", action="store_true", help="drop precipitation probability")
    asyncio.run(run(parser.parse_args().degraded))


if __name__ == "__main__":
    main()

"""Gemini prices for cost *estimates* (USD per 1M tokens).

Prices change; verify at https://ai.google.dev/gemini-api/docs/pricing.
Checked 2026-10-14: most sources list gemini-3.1-flash-lite at 0.25 / 1.50,
one aggregator lists 0.125 / 0.75. Override without code changes:
    GEMINI_PRICE_INPUT_PER_M=0.25
    GEMINI_PRICE_OUTPUT_PER_M=1.50
Thinking tokens are billed as output tokens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float


PRICES: dict[str, Price] = {
    "gemini-3.1-flash-lite": Price(0.25, 1.50),
}


def price_for(model: str) -> Price | None:
    """Env override first, then the table; None if unknown (cost not shown)."""
    env_in = os.getenv("GEMINI_PRICE_INPUT_PER_M")
    env_out = os.getenv("GEMINI_PRICE_OUTPUT_PER_M")
    if env_in and env_out:
        return Price(float(env_in), float(env_out))
    return PRICES.get(model)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = price_for(model)
    if price is None:
        return None
    return (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1_000_000

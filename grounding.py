"""Cheap hallucination check: numbers in AI text must exist in the supplied facts.

Heuristic, not proof. A flagged number is "worth a human look", e.g.
"24 hours" or "2 layers" are flagged although harmless, while a correct
number used in the wrong context ("rain at 16:00" when 16 is a temperature)
passes. Good enough to compare prompts and to surface soft warnings.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def _numbers(text: str) -> set[float]:
    return {float(m.replace(",", ".")) for m in _NUMBER.findall(text)}


def find_unsupported_numbers(ai_text: str, facts_text: str) -> list[float]:
    """Numbers in ai_text that do not appear in facts_text (abs tolerance 0.05).

    Timestamps in facts ("2026-10-15T16:00:00") contribute 2026, 10, 15, 16, 0,
    so citing an hour or a date is not flagged.
    """
    known = _numbers(facts_text)
    # Accept a leading minus being dropped/added ("-3" vs "minus 3") as supported.
    known |= {abs(n) for n in known}
    unsupported = [
        n for n in _numbers(ai_text)
        if not any(abs(abs(n) - k) <= 0.05 for k in known)
    ]
    return sorted(unsupported)

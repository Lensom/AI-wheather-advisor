# AI Weather Advisor

Learning project for AI Engineering: real weather → deterministic indicators → grounded Gemini advice.

## Status

**Stage 1 done:** geocoding, forecast, Pydantic normalization, errors, disambiguation, indicators, CLI.

**Stage 2 done:** grounded prompt (no invented values), token usage tracking, prompt comparison experiment (`experiments/compare_prompts.py`).

**Stage 3 done:** structured output + Pydantic — shape rules via Gemini `response_schema`, grounding rules via Pydantic validation context (`best_time` must be a real forecast hour), one repair retry with validation errors fed back, `StructuredOutputError`, soft numeric grounding check, offline tests.

**Later ideas:** tool calling, evaluation tests, caching, web UI.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`.env`:

```
GEMINI_API_KEY=your_key_here
```

## Run

```bash
python main.py
```

1. Enter a city  
2. Disambiguate if needed  
3. See weather + Python indicators  
4. See validated structured AI advice  

## Architecture

| Module | Responsibility |
|--------|----------------|
| `weather_models.py` | Weather domain models |
| `weather_client.py` | Open-Meteo geocoding + forecast |
| `indicators.py` | Deterministic metrics + comfort heuristic |
| `advice_models.py` | `WeatherAdvice` structured output schema |
| `prompts.py` | Facts JSON + grounding rules |
| `llm_client.py` | Gemini client, retries, usage, structured parse + repair retry |
| `grounding.py` | Heuristic check: numbers in AI text must exist in the facts |
| `experiments/compare_prompts.py` | Stage 2: compare prompt variants on one fixed dataset |
| `tests/` | Offline tests (schema validation, repair loop with fake Gemini) |
| `cli.py` | CLI orchestration |
| `main.py` | Entry point |

### Data flow

```
city
  → geocode + select_location
  → fetch_weather → WeatherSnapshot
  → compute_indicators
  → display weather
  → build_advice_prompt (facts + grounding rules)
  → Gemini structured(WeatherAdvice)      response_schema = shape
  → Pydantic validate + context            types, enum, non-empty, best_time ∈ forecast
      ↳ invalid → 1 repair retry with errors → else StructuredOutputError
  → grounding check (soft warning)
  → display advice
```

Weather data is the source of truth. `outdoor_comfort_score` is a documented heuristic, not a scientific index.

## Grounding rules (prompt)

- Use only values from the supplied JSON  
- Do not invent UV / AQI / pollen / sunrise unless present  
- Report gaps in `missing_information`  
- Cite concrete fields in `reasoning`  
- `best_time` must come from `hourly_forecast` (or null)  

## Validation layers

| Layer | Enforced by | On failure |
|-------|-------------|-----------|
| JSON shape, types, enum | Gemini `response_schema` + Pydantic | repair retry, then error |
| Non-empty strings/lists | Pydantic (`NonEmptyStr`, `min_length`) | repair retry, then error |
| `best_time` is a forecast hour | Pydantic validator + `validation_context(snapshot)` | repair retry, then error |
| Truncated output | `finish_reason == MAX_TOKENS` check | repair retry, then error |
| Numbers exist in facts | `grounding.find_unsupported_numbers` | warning only (heuristic) |

## Tests and experiments

```bash
python -m pytest                              # offline, no API calls
python -m experiments.compare_prompts         # 3 Gemini calls
python -m experiments.compare_prompts --degraded
```

`tests/fixtures/snapshot_berlin.json` is a real captured snapshot, so experiments are repeatable.

## APIs

- Open-Meteo Geocoding + Forecast (metric units)  
- Gemini via `google-genai` (`GEMINI_API_KEY`)

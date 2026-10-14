# AI Weather Advisor

Learning project for AI Engineering: real weather → deterministic indicators and decisions (Python) → grounded Gemini explanation.

## Status

**Stage 1 done:** geocoding, forecast, Pydantic normalization, errors, disambiguation, indicators, CLI.

**Stage 2 done:** grounded prompt (no invented values), token usage tracking, prompt comparison experiment (`experiments/compare_prompts.py`).

**Stage 3 done:** structured output + Pydantic — shape rules via Gemini `response_schema`, grounding rules via Pydantic validation context (`best_time` must be a real forecast hour), one repair retry with validation errors fed back, `StructuredOutputError`, soft numeric grounding check, offline tests.

**Stage 4 done:** Python calculations + AI reasoning — rain risk, hourly walk score, walk candidates and the outdoor verdict are computed by documented rules in `assessment.py`; Gemini only explains them (`rain_risk_explanation`, `reasoning`) and validation rejects any answer that overrides them.

**Stage 5 done:** tool calling — `python main.py --ask` lets Gemini call `get_weather(city, country?, region?)` itself. Hand-written agent loop (no SDK auto-calling): validated arguments, errors returned as tool results, parallel calls, session cache, call budget, step limit, multi-turn history with rollback.

**Stage 6 done:** evaluation — 8 weather scenarios + 7 tool-calling scenarios, programmatic checks, optional LLM judge, cost/latency metrics, baseline regression check (`python -m evals.run`). Also: cost estimate in usage, rate-limit-aware retries, compaction of old tool results, pinned dependencies, `.env.example`.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

```bash
copy .env.example .env               # Windows (cp on macOS/Linux), then set GEMINI_API_KEY
```

Optional `.env` settings: `GEMINI_MODEL`, `GEMINI_PRICE_INPUT_PER_M`, `GEMINI_PRICE_OUTPUT_PER_M` (see `pricing.py`).

## Run

```bash
python main.py                                  # classic: city -> structured advice
python main.py --ask                            # interactive questions, Gemini calls tools
python main.py --ask "Kyiv or Lviv for a walk?"  # one question
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
| `assessment.py` | Deterministic decisions: rain risk, walk score, walk candidates, outdoor verdict |
| `advice_models.py` | `WeatherAdvice` structured output schema |
| `prompts.py` | Facts JSON + grounding rules |
| `llm_client.py` | Gemini client, retries (honours 429 `retryDelay`), usage/cost/latency, structured parse + repair retry |
| `pricing.py` | Price table for cost estimates (env-overridable) |
| `grounding.py` | Heuristic check: numbers in AI text must exist in the facts |
| `experiments/compare_prompts.py` | Stage 2: compare prompt variants on one fixed dataset |
| `tests/` | Offline tests: rules, schema, repair loop, tools, agent loop, eval harness (fake Gemini / fake weather) |
| `tools.py` | `get_weather` declaration, argument validation, execution, cache, budget |
| `agent.py` | Tool-calling loop, system instruction, history, step limit |
| `cli.py` | CLI orchestration (classic + `--ask`) |
| `evals/` | Stage 6: scenarios, checks, LLM judge, runner, `baseline.json` |
| `main.py` | Entry point |

### Data flow

```
city
  → geocode + select_location
  → fetch_weather → WeatherSnapshot
  → compute_indicators                    numbers (Python)
  → assess                                decisions (Python rules)
  → display weather + decisions
  → build_advice_prompt (facts + decisions + rules text + grounding rules)
  → Gemini structured(WeatherAdvice)      response_schema = shape
  → Pydantic validate + context            types, enum, non-empty,
                                           outdoor_activity == verdict, best_time ∈ candidates
      ↳ invalid → 1 repair retry with errors → else StructuredOutputError
  → grounding check (soft warning)
  → display advice
```

Weather data is the source of truth. `outdoor_comfort_score` is a documented heuristic, not a scientific index.

## Tool calling (Stage 5)

```
question
  → Gemini (tools: get_weather, mode AUTO)
      text?            → answer (+ grounding check against all tool results)
      function calls?  → append model turn as is (keeps Gemini 3 thought signatures)
                       → validate args (Pydantic) → execute in parallel (cache, budget)
                       → append ONE user turn with all FunctionResponses (same ids)
                       → loop
  last step: mode NONE → the model must answer in text
```

| Situation | What happens |
|---|---|
| Invalid / extra / missing args | `invalid_arguments` result, API not called |
| Unknown tool name | `unknown_tool` result |
| City not found / not in the given country or region | `city_not_found` result → model asks for clarification |
| Weather API failure | `weather_unavailable` result, model told not to guess |
| Same city again in the session | served from cache, no API call, no budget used |
| More than 4 real lookups per session | `tool_budget_exhausted` result |
| Model still calling tools at the step limit | `AgentLoopError`, the turn is rolled back from history |
| No city in the question / off-topic | model asks / declines without a tool call |

Errors are returned to the model as *results* (`{"status": "error", "error": code, "message": ...}`) rather than raised, so the model can react: ask for the country, say the service is down.

Tool results contain the same facts as the classic flow, plus per-hour `rain_risk` and `walk_score` (free-form questions ask about specific hours). The rules text is sent once in the system instruction, not in every tool result.

**Cost:** history is resent on every request. To keep follow-ups cheap, tool results from *earlier* questions are compacted when a new question starts: hourly arrays are dropped, current conditions / indicators / decisions stay, and a note tells the model it can call the tool again (served from cache) for hourly details. Measured in evals: a follow-up costs ~1.8k input tokens instead of ~4.7k.

## Who decides what (Stage 4)

| Python decides (`assessment.py`) | Gemini adds |
|---|---|
| rain risk per hour, `overall`, `next_3h`, `first_high_risk_at` | how that risk affects a walk (`rain_risk_explanation`) |
| walk score per hour, top-3 walk candidates | which candidate to name (normally the first) |
| `outdoor_verdict` (level + reasons) | plain-language explanation of the verdict |
| — | clothing, concerns, summary wording, missing information |

### Rules (heuristics, not meteorological standards)

**Rain risk per hour** (either condition is enough):
- high: probability ≥ 60% or precipitation ≥ 1.0 mm
- moderate: probability ≥ 30% or precipitation ≥ 0.2 mm
- low: otherwise
- unknown: probability missing and precipitation below 0.2 mm

`overall` is the worst known level over 24 h, `next_3h` the worst known level over the first 3 h.

**Walk score per hour** (only 07:00–21:00 local):
`100 − min(40, |temperature − 18 °C| × 2) − rain penalty` (low 0, unknown 15, moderate 25, high 60), clamped to 0–100.
18 °C is lower than the comfort-score ideal (21 °C) because walking warms you up. Hourly data has no wind or apparent temperature, so the score uses air temperature and rain only.
With low rain risk an hour never scores below 60, so cold alone gives "caution", never "not recommended".

**Walk candidates:** scored hours with walk score ≥ 50 and rain risk not high; best first; ties go to the temperature closer to 18 °C, then the earlier hour; max 3.
(The temperature tie-break was added after evals showed that in deep cold every hour scores 60 and "earliest" picked the coldest morning hour.)

**Outdoor verdict:** `not_recommended` if there are no candidates; `recommended` if the best candidate has score ≥ 75 and low rain risk; otherwise `possible_with_caution`.
The verdict describes the best walk window in the next 24 h. `outdoor_comfort_score` describes conditions right now. A rainy day with a dry evening is therefore still `recommended`, with the rain shown as a concern.

The same rules text (`RULES_DOC`) is sent to Gemini so it can quote the thresholds.

## Grounding rules (prompt)

- Use only values from the supplied JSON  
- Do not invent UV / AQI / pollen / sunrise unless present  
- Report gaps in `missing_information`  
- Cite concrete fields in `reasoning`  
- `outdoor_activity` must equal the Python verdict  
- `best_time` must be one of the Python walk candidates (null only if there are none)  
- never recompute or contradict `indicators` / `assessment`  

## Validation layers

| Layer | Enforced by | On failure |
|-------|-------------|-----------|
| JSON shape, types, enum | Gemini `response_schema` + Pydantic | repair retry, then error |
| Non-empty strings/lists | Pydantic (`NonEmptyStr`, `min_length`) | repair retry, then error |
| `outdoor_activity` equals Python verdict | Pydantic validator + `validation_context(assessment)` | repair retry, then error |
| `best_time` is a Python walk candidate | Pydantic validator + `validation_context(assessment)` | repair retry, then error |
| `best_time` null only without candidates | Pydantic `model_validator` | repair retry, then error |
| Truncated output | `finish_reason == MAX_TOKENS` check | repair retry, then error |
| Numbers exist in facts | `grounding.find_unsupported_numbers` | warning only (heuristic) |

## Tests and experiments

```bash
python -m pytest                              # 122 offline tests, no API calls
python -m experiments.compare_prompts         # 3 Gemini calls
python -m experiments.compare_prompts --degraded
```

`tests/fixtures/snapshot_berlin.json` is a real captured snapshot, so experiments are repeatable.

## Evaluation (Stage 6)

```bash
python -m evals.run                     # all cases, compare with evals/baseline.json
python -m evals.run --repeat 3 --judge  # flakiness + LLM-as-judge
python -m evals.run --case drizzle      # single case while iterating
python -m evals.run --save-baseline     # accept current behaviour as the new baseline
```

The runner exercises the real app code (`cli.generate_advice`, `WeatherAgent`). Only Open-Meteo is replaced, by **synthetic scenarios** (`evals/scenarios.py`): live weather changes every hour, so results would not be comparable between runs.

| Suite | Scenarios | Checked |
|---|---|---|
| advice (8) | mild, cold, hot, rainy afternoon, rain all day, drizzle, missing probability, windy | verdict = Python, `best_time` ∈ candidates, clothing keywords (required / forbidden, negation-aware), concerns, missing information, no invented UV / pollen / AQI / sunrise |
| agent (7) | umbrella, compare two cities, no city, off-topic, unknown city, next week, 3-turn follow-ups | which tools were called with which cities, parallel calls in one step, answer content |

- **Hard checks** decide pass/fail. **Soft checks** are reported as rates: `numbers_grounded` (heuristic with false positives), `first_try_valid` (no repair retry needed), `judge`.
- **LLM judge** (`--judge`): a second Gemini call scores clothing and faithfulness 1–5. It shares the model's blind spots and doubles cost, so it's optional and soft.
- **Metrics per case:** latency (rate-limit waits excluded), LLM calls, input/output tokens, estimated cost. Agent multi-turn cases also record tokens and latency per question.
- **Regression gate** (exit code 1): pass rate drops > 10 points, a case that always passed now always fails, cost per case +25%, or p50 latency +50%.
- Each run is saved to `evals/results/` with the model's outputs, for debugging failures.

### Baseline (gemini-3.1-flash-lite, 1 run per case)

| | |
|---|---|
| Pass rate (hard checks) | 15/15 |
| Latency p50 / p95 | 3.2 s / 9.0 s per case |
| Tokens per case | ~3.4k in / ~280 out |
| Cost | ~$0.0013 per request → **~$1.3 per 1000 requests** (judge +~$0.001 per case) |
| Soft: numbers grounded | 13/15 (see findings) |

### What the evals found (and what was fixed)

1. **Rule bug (fixed).** In deep cold every daytime hour scored 60 (penalty cap), so the tie-break "earliest hour" recommended the *coldest* morning. The tie-break now prefers the temperature closer to 18 °C. Caught by the scenario sanity check, before any API call.
2. **Rate limits (fixed).** Free tier allows 15 requests/min. A 3× run crashed 12 of 45 cases with HTTP 429: the retry slept 1–2 s, while the server asked for ~20–40 s. `LLMClient` now waits the server's `retryDelay` (capped at 60 s; longer → fail fast) and reports waits separately from latency.
3. **Token growth in follow-ups (fixed).** Older tool results are compacted: follow-up questions dropped from ~4.7k to ~1.8k input tokens.
4. **Model arithmetic (open).** In `missing_probability` the model wrote "8.8 °C below ideal" (21.0 − 12.2): not invented, but computed by the model, against the "Python calculates" rule. The judge scored it 5/5, the numbers check flagged it, which shows why both kinds of checks are useful. Next step: a prompt rule against derived numbers, verified with `--repeat`.
5. **False positive (fixed).** "The forecast covers 24 hours" was flagged because the grounding source missed the system instruction. The agent now grounds against everything the model was told.

## Extensions

- **Hourly wind and apparent temperature** from Open-Meteo, so the walk score stops ignoring wind (the `windy` scenario shows the gap).
- **Response caching** for identical snapshots (e.g. same city within 10 minutes), plus Gemini context caching for the static system prompt.
- **CI**: run `pytest` on every change and `evals.run` on prompt or model changes, with the baseline gate.
- **Model comparison**: `evals.run --model <other>` with the same scenarios and a cost/quality table.
- **Web UI** (FastAPI + a small frontend) once the CLI behaviour is stable. The core modules don't depend on the CLI.

## APIs

- Open-Meteo Geocoding + Forecast (metric units)  
- Gemini via `google-genai` (`GEMINI_API_KEY`)

"""Evaluation runner: quality, cost and latency, with baseline regression check.

    python -m evals.run                         # all suites, compare with baseline
    python -m evals.run --suite advice --repeat 3
    python -m evals.run --judge                 # + LLM-as-judge (doubles advice cost)
    python -m evals.run --save-baseline         # accept this run as the new baseline
    python -m evals.run --case drizzle --case umbrella

Runs the real app code (cli.generate_advice, WeatherAgent) against synthetic
scenarios; only Open-Meteo is replaced. Cases run sequentially so latency is
not distorted by concurrent requests. Exit code 1 on regression (CI-friendly).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median

from agent import AgentLoopError, WeatherAgent
from assessment import assess
from cli import generate_advice
from evals.checks import Check, check_advice, check_agent_turn
from evals.judge import PASS_SCORE, judge_advice
from evals.scenarios import ADVICE_CASES, AGENT_CASES, AdviceCase, AgentCase, scenario_loader
from indicators import compute_indicators
from llm_client import LLMClient, StructuredOutputError
from prompts import weather_facts_json
from tools import WeatherTools

EVALS_DIR = Path(__file__).parent
BASELINE = EVALS_DIR / "baseline.json"
RESULTS_DIR = EVALS_DIR / "results"

# Regression thresholds vs baseline
MAX_PASS_RATE_DROP = 0.10  # absolute, e.g. 95% -> 84% fails
MAX_COST_INCREASE = 0.25  # relative, per case on average
MAX_P50_LATENCY_INCREASE = 0.50  # relative; latency is noisy, so generous


@dataclass
class CaseResult:
    suite: str
    case_id: str
    repeat: int
    passed: bool
    checks: list[dict]
    latency_s: float  # wall time minus rate-limit waits
    llm_calls: int
    input_tokens: int
    output_tokens: int  # completion + thinking
    cost_usd: float | None
    turn_input_tokens: list[int] = field(default_factory=list)  # agent: per question
    turn_latency_s: list[float] = field(default_factory=list)  # agent: per question
    output: dict | list | None = None  # what the model said, for debugging failures
    rate_limited: int = 0  # 429s retried (waits excluded from latency_s)
    judge: dict | None = None
    judge_cost_usd: float | None = None
    error: str | None = None


def _result(suite, case_id, repeat, checks: list[Check], llm: LLMClient, latency, **extra) -> CaseResult:
    u = llm.usage
    return CaseResult(
        suite=suite,
        case_id=case_id,
        repeat=repeat,
        passed=all(c.passed for c in checks if c.hard),
        checks=[asdict(c) for c in checks],
        latency_s=round(max(0.0, latency - u.wait_seconds), 2),
        rate_limited=u.rate_limited,
        llm_calls=u.calls,
        input_tokens=u.prompt_tokens,
        output_tokens=u.completion_tokens + u.thinking_tokens,
        cost_usd=u.cost_usd,
        **extra,
    )


async def run_advice_case(case: AdviceCase, repeat: int, model: str | None, judge: bool) -> CaseResult:
    snapshot = case.build()
    indicators, assessment = compute_indicators(snapshot), assess(snapshot)
    facts = weather_facts_json(snapshot, indicators, assessment)
    llm = LLMClient(model=model)

    started = time.perf_counter()
    try:
        advice, _, _ = await generate_advice(snapshot, indicators, assessment, llm=llm)
    except StructuredOutputError as exc:
        latency = time.perf_counter() - started
        return _result("advice", case.id, repeat, [Check("valid_output", False, str(exc))], llm, latency,
                       error=str(exc))
    latency = time.perf_counter() - started
    checks = check_advice(case, advice, assessment, facts, llm.usage.repairs)

    extra: dict = {}
    if judge:
        judge_llm = LLMClient(model=model)
        verdict = await judge_advice(judge_llm, facts, advice)
        ok = min(verdict.clothing_score, verdict.faithfulness_score) >= PASS_SCORE
        checks.append(Check(
            "judge", ok,
            f"clothing={verdict.clothing_score} faithful={verdict.faithfulness_score} "
            + "; ".join(verdict.issues),
            hard=False,
        ))
        extra = {"judge": verdict.model_dump(), "judge_cost_usd": judge_llm.usage.cost_usd}
    return _result("advice", case.id, repeat, checks, llm, latency, output=advice.model_dump(), **extra)


async def run_agent_case(case: AgentCase, repeat: int, model: str | None) -> CaseResult:
    llm = LLMClient(model=model)
    agent = WeatherAgent(llm, WeatherTools(loader=scenario_loader))
    checks: list[Check] = []
    turn_tokens: list[int] = []
    turn_latency: list[float] = []
    transcript: list[dict] = []

    started = time.perf_counter()
    for i, turn in enumerate(case.turns, start=1):
        before = llm.usage.prompt_tokens
        turn_started = time.perf_counter()
        try:
            result = await agent.ask(turn.question)
        except AgentLoopError as exc:
            checks.append(Check(f"t{i}.answered", False, str(exc)))
            break
        turn_tokens.append(llm.usage.prompt_tokens - before)
        turn_latency.append(round(time.perf_counter() - turn_started, 2))
        transcript.append({
            "question": turn.question,
            "tool_calls": [{"args": r.args, "status": r.status, "cached": r.cached} for r in result.tool_calls],
            "answer": result.answer,
        })
        prefix = f"t{i}." if len(case.turns) > 1 else ""
        for check in check_agent_turn(turn, result):
            check.name = prefix + check.name
            checks.append(check)
    latency = time.perf_counter() - started
    return _result("agent", case.id, repeat, checks, llm, latency, turn_input_tokens=turn_tokens,
                   turn_latency_s=turn_latency, output=transcript)


# --- reporting ------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def summarize(results: list[CaseResult]) -> dict:
    by_case: dict[str, list[CaseResult]] = {}
    for r in results:
        by_case.setdefault(f"{r.suite}/{r.case_id}", []).append(r)

    check_stats: dict[str, list[int]] = {}
    for r in results:
        for c in r.checks:
            name = c["name"].split(".", 1)[-1]  # t2.tool_calls -> tool_calls
            stat = check_stats.setdefault(name, [0, 0])
            stat[0] += c["passed"]
            stat[1] += 1

    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    judge_costs = [r.judge_cost_usd for r in results if r.judge_cost_usd is not None]
    latencies = [r.latency_s for r in results]
    return {
        "runs": len(results),
        "pass_rate": sum(r.passed for r in results) / len(results),
        "case_pass_rates": {k: sum(r.passed for r in v) / len(v) for k, v in by_case.items()},
        "check_pass_rates": {k: p / n for k, (p, n) in sorted(check_stats.items())},
        "latency_p50_s": round(median(latencies), 2),
        "latency_p95_s": round(_p95(latencies), 2),
        "avg_input_tokens": round(sum(r.input_tokens for r in results) / len(results)),
        "avg_output_tokens": round(sum(r.output_tokens for r in results) / len(results)),
        "llm_calls": sum(r.llm_calls for r in results),
        "rate_limited": sum(r.rate_limited for r in results),
        "total_cost_usd": round(sum(costs), 6) if costs else None,
        "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "judge_cost_usd": round(sum(judge_costs), 6) if judge_costs else None,
    }


def print_report(results: list[CaseResult], summary: dict) -> None:
    print(f"\n{'case':<30} {'pass':>5} {'lat s':>6} {'calls':>5} {'in tok':>7} {'out':>5} {'cost $':>9}  failed checks")
    print("-" * 110)
    by_case: dict[str, list[CaseResult]] = {}
    for r in results:
        by_case.setdefault(f"{r.suite}/{r.case_id}", []).append(r)
    for key, runs in by_case.items():
        n = len(runs)
        failed = sorted({
            f"{c['name']}({c['detail'][:60]})" if c["detail"] else c["name"]
            for r in runs for c in r.checks if not c["passed"]
        })
        cost = [r.cost_usd for r in runs if r.cost_usd is not None]
        print(
            f"{key:<30} {sum(r.passed for r in runs)}/{n:<3} "
            f"{sum(r.latency_s for r in runs) / n:>6.1f} "
            f"{sum(r.llm_calls for r in runs) / n:>5.1f} "
            f"{sum(r.input_tokens for r in runs) // n:>7} "
            f"{sum(r.output_tokens for r in runs) // n:>5} "
            f"{(sum(cost) / len(cost)) if cost else float('nan'):>9.5f}  "
            + ("; ".join(failed) or "-")
        )
        multi = [r for r in runs if len(r.turn_input_tokens) > 1]
        if multi:
            print(f"{'':<30} per question: input tokens {multi[0].turn_input_tokens}, "
                  f"latency s {multi[0].turn_latency_s}")

    s = summary
    print("\nSummary")
    print(f"  runs: {s['runs']}   pass rate (all hard checks): {_pct(s['pass_rate'])}")
    print("  check pass rates: " + ", ".join(f"{k} {_pct(v)}" for k, v in s["check_pass_rates"].items()))
    print(f"  latency: p50 {s['latency_p50_s']} s, p95 {s['latency_p95_s']} s "
          f"(per case, all LLM calls, rate-limit waits excluded; {s['rate_limited']} x 429 retried)")
    print(f"  tokens per case: {s['avg_input_tokens']} in / {s['avg_output_tokens']} out;  LLM calls: {s['llm_calls']}")
    if s["total_cost_usd"] is not None:
        print(f"  cost: ${s['total_cost_usd']:.4f} total, ${s['avg_cost_usd']:.5f} per case"
              + (f"; judge ${s['judge_cost_usd']:.4f}" if s["judge_cost_usd"] else ""))
        print(f"  -> about ${s['avg_cost_usd'] * 1000:.2f} per 1000 requests (estimate, see pricing.py)")


def compare(summary: dict, baseline: dict) -> list[str]:
    """Return regression messages (empty = OK) and print the diff."""
    base = baseline["summary"]
    problems: list[str] = []
    print(f"\nBaseline ({baseline['created']}, model {baseline['model']})")

    drop = base["pass_rate"] - summary["pass_rate"]
    print(f"  pass rate: {_pct(base['pass_rate'])} -> {_pct(summary['pass_rate'])}")
    if drop > MAX_PASS_RATE_DROP:
        problems.append(f"pass rate dropped by {_pct(drop)}")

    for case, rate in summary["case_pass_rates"].items():
        old = base["case_pass_rates"].get(case)
        if old is not None and rate < old:
            print(f"  worse: {case} {_pct(old)} -> {_pct(rate)}")
            if old == 1.0 and rate == 0.0:
                problems.append(f"{case} always passed, now always fails")
        elif old is not None and rate > old:
            print(f"  better: {case} {_pct(old)} -> {_pct(rate)}")

    if base.get("avg_cost_usd") and summary.get("avg_cost_usd"):
        change = summary["avg_cost_usd"] / base["avg_cost_usd"] - 1
        print(f"  cost per case: {change:+.0%}")
        if change > MAX_COST_INCREASE:
            problems.append(f"cost per case up {change:.0%}")

    change = summary["latency_p50_s"] / base["latency_p50_s"] - 1
    print(f"  latency p50: {base['latency_p50_s']} s -> {summary['latency_p50_s']} s ({change:+.0%})")
    if change > MAX_P50_LATENCY_INCREASE:
        problems.append(f"p50 latency up {change:.0%}")
    return problems


# --- main ---------------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    advice_cases = [c for c in ADVICE_CASES if not args.case or c.id in args.case]
    agent_cases = [c for c in AGENT_CASES if not args.case or c.id in args.case]
    results: list[CaseResult] = []
    model = args.model

    async def guarded(suite: str, case_id: str, repeat: int, coro) -> CaseResult:
        print(f"  {suite}/{case_id} #{repeat} ...", flush=True)
        try:
            return await coro
        except Exception as exc:  # API outage etc.: record, keep evaluating the rest
            logging.getLogger(__name__).exception("case crashed")
            return CaseResult(suite, case_id, repeat, False,
                              [asdict(Check("crashed", False, f"{type(exc).__name__}: {exc}"))],
                              0.0, 0, 0, 0, None, error=f"{type(exc).__name__}: {exc}")

    for repeat in range(1, args.repeat + 1):
        if args.suite in ("advice", "all"):
            for case in advice_cases:
                results.append(await guarded(
                    "advice", case.id, repeat, run_advice_case(case, repeat, model, args.judge)
                ))
        if args.suite in ("agent", "all"):
            for case in agent_cases:
                results.append(await guarded("agent", case.id, repeat, run_agent_case(case, repeat, model)))

    if not results:
        print("No cases selected.")
        return 2

    summary = summarize(results)
    print_report(results, summary)

    used_model = LLMClient(model=model).model
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": used_model,
        "args": {k: v for k, v in vars(args).items() if k != "save_baseline"},
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out.relative_to(EVALS_DIR.parent)}")

    if args.save_baseline:
        BASELINE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Baseline updated: {BASELINE.relative_to(EVALS_DIR.parent)}")
        return 0

    if BASELINE.exists() and not args.case:
        problems = compare(summary, json.loads(BASELINE.read_text(encoding="utf-8")))
        if problems:
            print("\nREGRESSION: " + "; ".join(problems))
            return 1
        print("\nNo regression vs baseline.")
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate AI Weather Advisor")
    parser.add_argument("--suite", choices=["advice", "agent", "all"], default="all")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case (measures flakiness)")
    parser.add_argument("--case", action="append", help="only this case id (repeatable)")
    parser.add_argument("--judge", action="store_true", help="add LLM-as-judge to advice cases")
    parser.add_argument("--model", help="Gemini model (default: GEMINI_MODEL or llm_client default)")
    parser.add_argument("--save-baseline", action="store_true")
    sys.exit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()

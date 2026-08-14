#!/usr/bin/env python3
"""Emit a `test-time-bench`-shaped run summary for an opencode benchmark run.

## Why this exists

`shsym/test-time-bench` (TTB) already solved the reporting problem this
integration kept hitting by hand: what did the run *actually* boot, what was it
*allowed* to spend, and when a case produced nothing, was that the agent failing
or a budget expiring? Its `inferletbench.run_summary.v2` artifact answers all
three in one file, and its shape is adopted here rather than reinvented.

Three of its ideas are the whole point:

  * **Provenance is cryptographic, not narrative.** TTB stamps
    `engine_config_sha256`, `pie_version`, and image digests into every summary.
    This integration learned the same lesson the expensive way — four
    measurements were once taken against a server booted with different flags,
    while `/health` answered `ok` throughout — and answered it with boot helpers
    that prove liveness. A hash in the artifact is the durable form of that.
  * **Budgets are declared AND enforced, or not declared.** TTB's own audit
    `H1-agentic-budget-not-plumbed` finds `max_steps`/`wall_time_seconds`
    validated but never plumbed, with the run summary reporting the hardcoded
    constant as though it were the declared contract. So every budget here
    carries an `enforced` flag, and anything unenforced is reported as
    `null` — an unenforceable budget is not quietly printed as though it held.
  * **Exhaustion is a first-class outcome.** `wall_time_exhausted_cases` and
    friends separate "the agent worked and failed" from "the clock ran out".
    Our headline miss to date is an instance that produced no patch after 21
    minutes, and nothing in the artifact said which of those it was.

## Where the numbers come from

opencode keeps real per-session accounting in its own SQLite store
(`~/.local/share/opencode/opencode.db`): `tokens_input`, `tokens_output`, and
the message chain per session, keyed by the workspace `directory`. That is a
better instrument than the wall-clock this harness reported before — on the
graded runs it shows the vLLM arm ending a case after 2 messages and 97 output
tokens, which reads as an agent giving up, where seconds alone read as "fast".

Token counts are the **agent's** accounting, not the server's. They agree with
the server only if every request was served; a rejected or errored call still
appears as a message. Treated as agent-side telemetry, which is what it is.

Usage:
    python3 ttb_summary.py --cases cases.jsonl --label pie \\
        --suite swe-bench-lite-first-20 [--report report.json] \\
        [--config bench-config.json] --out run-summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import uuid
from pathlib import Path

OPENCODE_DB = Path.home() / ".local/share/opencode/opencode.db"
SCHEMA_VERSION = "inferletbench.run_summary.v2"


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def git_rev(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        rev = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        # A dirty tree is recorded, not hidden: a summary that names a clean
        # revision it was not built from is worse than one that admits drift.
        return f"pie_rev:{rev}{'+dirty' if dirty else ''}" if rev else "pie_rev:unknown"
    except Exception:
        return "pie_rev:unknown"


def opencode_telemetry(directory: str, db: Path = OPENCODE_DB,
                       started_ms: int | None = None,
                       seconds: float | None = None) -> dict:
    """Per-case agent accounting, keyed by the workspace directory.

    Sessions are directory-scoped (verified against the store: each `opencode
    run` creates its own `ses_…` carrying its own `directory`), so a case's
    telemetry is the sum over sessions whose directory is this workspace. A
    case that never started a session yields zeros and `sessions: 0`, which is
    distinguishable from a case that ran and emitted nothing.
    """
    if not db.exists():
        return {"sessions": 0, "tokens_in": 0, "tokens_out": 0, "model_calls": 0,
                "telemetry_source": "unavailable"}
    # Resolve BOTH sides: opencode stores the realpath, and a caller that
    # recorded `/tmp/...` on macOS would otherwise join against nothing and
    # report a case that ran as zero-session.
    try:
        directory = str(Path(directory).resolve())
    except OSError:
        pass
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select id, tokens_input, tokens_output, tokens_cache_read, time_created "
            "from session where directory = ? order by time_created", (directory,)).fetchall()
        note = None
        if started_ms:
            lo = started_ms - 5_000
            hi = started_ms + int((seconds or 3600) * 1000) + 120_000
            rows = [r for r in rows if lo <= (r[4] or 0) <= hi]
        elif len(rows) > 1:
            # No window recorded (a pre-`started_ms` cases file). Summing every
            # session at this path would fold in earlier runs, so take the most
            # recent and say that is what happened.
            rows = rows[-1:]
            note = "multiple sessions at this path; used the most recent (no started_ms recorded)"
        calls = 0
        for sid, *_ in rows:
            # One assistant message is one model call: opencode writes the
            # assistant turn per completion, tool results ride as parts of it.
            calls += con.execute(
                "select count(*) from message where session_id = ? "
                "and json_extract(data, '$.role') = 'assistant'", (sid,)).fetchone()[0]
        # Per-CALL timing, which is the only way to separate serving from the
        # agent's own work. A case's wall clock is model time plus tool time —
        # running pytest, editing files, waiting on a build — and those are not
        # the server's. Dividing a case's tokens by its wall clock makes a task
        # that happens to trigger a slow test suite look like a slow engine.
        # opencode records `time.created`/`time.completed` and per-call token
        # counts on every assistant message, so the split is measurable rather
        # than estimated.
        calls_ms, tok_in_c, tok_out_c, tok_cache_c = [], 0, 0, 0
        for sid, *_ in rows:
            for (md,) in con.execute(
                    "select data from message where session_id = ? "
                    "and json_extract(data, '$.role') = 'assistant'", (sid,)):
                m = json.loads(md)
                tm, tk = m.get("time") or {}, m.get("tokens") or {}
                if tm.get("created") and tm.get("completed"):
                    calls_ms.append(tm["completed"] - tm["created"])
                tok_in_c += tk.get("input") or 0
                tok_out_c += tk.get("output") or 0
                # `cache.read` is what the SERVER said it served from a parked
                # prefix (usage.prompt_tokens_details.cached_tokens on the
                # wire). Summed per call, not per session, so the reuse
                # fraction below is over the same population as the latencies.
                tok_cache_c += (tk.get("cache") or {}).get("read") or 0
        return {
            "sessions": len(rows),
            "call_ms": calls_ms,
            "call_tokens_in": tok_in_c,
            "call_tokens_out": tok_out_c,
            "call_tokens_cache_read": tok_cache_c,
            "tokens_in": sum(r[1] or 0 for r in rows),
            "tokens_out": sum(r[2] or 0 for r in rows),
            "tokens_cache_read": sum(r[3] or 0 for r in rows),
            "model_calls": calls,
            "telemetry_source": "opencode.db",
            "telemetry_note": note,
        }
    finally:
        con.close()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    # Linear interpolation, matching `statistics.quantiles`' inclusive method
    # so a p50 on an even count is the midpoint rather than a sample.
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def load_provenance(path: Path | None) -> None:
    """Fold a run's recorded provenance into the environment.

    Provenance is known while the server is UP (config path, KV pool, artifact)
    but the summary must be re-emitted AFTER grading, which happens later and in
    another shell. Passing it through env alone means the graded artifact — the
    one anybody actually quotes — silently loses every provenance field it has.
    Observed: a resolved run whose summary carried `model_artifact: unknown` and
    `kv_pool_pages: null` because grading ran in a different process.

    Env still wins, so an explicit override beats a stale file.
    """
    if not path or not path.exists():
        return
    for k, v in json.loads(path.read_text()).items():
        if v is not None and not os.environ.get(k):
            os.environ[k] = str(v)


def build(cases: list[dict], label: str, suite: str, report: dict | None,
          config: dict | None, repo: Path) -> dict:
    enriched = []
    for c in cases:
        t = (opencode_telemetry(c["workspace"], started_ms=c.get("started_ms"),
                                seconds=c.get("seconds"))
             if c.get("workspace") else {})
        enriched.append({**c, **t})

    n = len(enriched)
    lat = [c.get("seconds", 0.0) for c in enriched]
    t_in = [c.get("tokens_in", 0) for c in enriched]
    t_out = [c.get("tokens_out", 0) for c in enriched]
    calls = [c.get("model_calls", 0) for c in enriched]
    all_calls_s = [ms / 1000.0 for c in enriched for ms in (c.get("call_ms") or [])]
    call_in = sum(c.get("call_tokens_in", 0) for c in enriched)
    call_out = sum(c.get("call_tokens_out", 0) for c in enriched)
    call_cache = sum(c.get("call_tokens_cache_read", 0) for c in enriched)

    resolved = None
    if report:
        ids = set(report.get("resolved_ids") or [])
        resolved = sum(1 for c in enriched if c["case_id"] in ids)

    budgets = (config or {}).get("budgets", {})
    # Declared only where enforced. TTB's H1 audit is precisely this failure:
    # a schema that accepts a budget nothing consumes, then a summary that
    # prints it as the governing contract.
    def budget(name):
        b = budgets.get(name)
        if not isinstance(b, dict):
            return None
        return b.get("value") if b.get("enforced") else None

    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid5(uuid.NAMESPACE_URL,
                                 f"{suite}/{label}/{[c['case_id'] for c in enriched]}")),
        "agent": {
            "kind": "opencode",
            "version": os.environ.get("OPENCODE_VERSION", "unknown"),
            "note": "external agent process; TTB's own agent_loop.rs is NOT used",
        },
        "runtime": {
            "pie_version": git_rev(repo),
            "provider": "local",
            "backend": os.environ.get("TTB_BACKEND", "Apple Metal (unified memory)"),
            "engine_endpoint": os.environ.get("TTB_ENGINE_ENDPOINT", "unknown"),
            "engine_config_ref": os.environ.get("TTB_ENGINE_CONFIG", "unknown"),
            # A digest recorded at run time wins: the engine config is an
            # mktemp file and is gone by the time a graded summary is emitted.
            "engine_config_sha256": os.environ.get("TTB_ENGINE_CONFIG_SHA256")
            or (sha256_file(Path(os.environ["TTB_ENGINE_CONFIG"]))
                if os.environ.get("TTB_ENGINE_CONFIG")
                and Path(os.environ["TTB_ENGINE_CONFIG"]).exists() else None),
            "model_artifact": os.environ.get("TTB_MODEL_ARTIFACT", "unknown"),
            "kv_pool_pages": int(os.environ["TTB_KV_POOL_PAGES"])
            if os.environ.get("TTB_KV_POOL_PAGES", "").isdigit() else None,
        },
        "benchmark": {"suite": suite, "suite_version": (config or {}).get(
            "identity", {}).get("version", "local"), "split": "local"},
        "budget": {
            "max_agent_steps": budget("max_agent_steps"),
            "max_model_calls": budget("max_model_calls"),
            "max_output_tokens": budget("max_output_tokens"),
            "max_wall_time_s": budget("max_wall_time_s"),
        },
        "metrics": {
            "cases": n,
            "success_rate": (resolved / n) if (resolved is not None and n) else None,
            "resolved": resolved,
            "nonempty_patch_rate": (sum(1 for c in enriched if c.get("patch_bytes", 0) > 0) / n)
            if n else 0.0,
            "tokens_in": sum(t_in),
            "tokens_out": sum(t_out),
            "mean_tokens_in": statistics.fmean(t_in) if t_in else 0,
            "mean_tokens_out": statistics.fmean(t_out) if t_out else 0,
            "model_calls": sum(calls),
            "mean_model_calls": statistics.fmean(calls) if calls else 0,
            # ── serving vs the agent's own work ──────────────────────────
            "model_call_latency_p50_s": percentile(all_calls_s, 0.50),
            "model_call_latency_p95_s": percentile(all_calls_s, 0.95),
            "model_call_latency_mean_s": statistics.fmean(all_calls_s) if all_calls_s else 0,
            "model_time_s": round(sum(all_calls_s), 1),
            # Everything the case spent NOT waiting on the model: tool
            # execution, test suites, the agent's own bookkeeping.
            "tool_and_overhead_s": round(max(0.0, sum(lat) - sum(all_calls_s)), 1),
            "model_time_fraction": (round(sum(all_calls_s) / sum(lat), 3)
                                    if sum(lat) else None),
            # ── how much of the prompt never reached the GPU ─────────────
            # A FRACTION, never a hit flag: a prefix cache that resumes on a
            # shallow cut and re-prefills most of the history still reports a
            # hit, which is how a regression hides behind a green check.
            #
            # Denominator is input + cache_read, because opencode's `input`
            # counts only what the server said it actually processed. Reading
            # `in_call_prompt_tok_s` without this number is misleading in BOTH
            # directions: an arm with reuse looks slow (its input shrank while
            # its per-call latency includes a real prefill of the delta), and
            # the arms are not comparable unless you know how much each one
            # skipped.
            "prompt_tokens_cached": call_cache,
            "prompt_reuse_fraction": (round(call_cache / (call_in + call_cache), 3)
                                      if (call_in + call_cache) else None),
            # In-CALL rates: tokens divided by time actually spent in the
            # server, not by the case's wall clock.
            "in_call_prompt_tok_s": (round(call_in / sum(all_calls_s), 1)
                                     if sum(all_calls_s) else None),
            "in_call_output_tok_s": (round(call_out / sum(all_calls_s), 1)
                                     if sum(all_calls_s) else None),
            "case_latency_mean_s": statistics.fmean(lat) if lat else 0,
            "case_latency_p50_s": percentile(lat, 0.50),
            "case_latency_p95_s": percentile(lat, 0.95),
            # The distinction the old artifact could not make: an empty patch
            # after the clock expired is not the same result as an empty patch
            # from an agent that stopped on its own.
            "wall_time_exhausted_cases": sum(1 for c in enriched if c.get("state") == "timeout"),
            "agent_error_cases": sum(1 for c in enriched if c.get("state") == "agent-error"),
            "zero_session_cases": sum(1 for c in enriched if c.get("sessions", 0) == 0),
            "timeout_rate": (sum(1 for c in enriched if c.get("state") == "timeout") / n)
            if n else 0.0,
        },
        "cases": [
            {k: c.get(k) for k in ("case_id", "state", "seconds", "patch_bytes",
                                   "tokens_in", "tokens_out", "model_calls", "sessions")}
            for c in enriched
        ],
    }
    if resolved is None:
        summary["metrics"]["success_rate_note"] = (
            "no grader report supplied; success_rate is null rather than assumed")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="cases.jsonl from run_swebench.py")
    ap.add_argument("--label", required=True)
    ap.add_argument("--suite", default="swe-bench-known-solvable")
    ap.add_argument("--report", help="official grader report.json, for success_rate")
    ap.add_argument("--config", help="benchmark config json, for declared budgets")
    ap.add_argument("--out", default="run-summary.json")
    ap.add_argument("--provenance", help="provenance.json written by ttb/run.sh "
                                         "while the server was up")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    load_provenance(Path(args.provenance) if args.provenance else
                    Path(args.cases).parent / "provenance.json")
    cases = [json.loads(l) for l in Path(args.cases).read_text().splitlines() if l.strip()]
    report = json.loads(Path(args.report).read_text()) if args.report else None
    config = json.loads(Path(args.config).read_text()) if args.config else None
    summary = build(cases, args.label, args.suite, report, config, Path(args.repo))
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")

    m = summary["metrics"]
    print(f"wrote {args.out}")
    print(f"  cases {m['cases']}  resolved {m['resolved']}  "
          f"nonempty {m['nonempty_patch_rate']:.0%}")
    print(f"  tokens in/out {m['tokens_in']:,}/{m['tokens_out']:,}  "
          f"model calls {m['model_calls']} (mean {m['mean_model_calls']:.1f})")
    print(f"  case latency mean/p50/p95 {m['case_latency_mean_s']:.0f}/"
          f"{m['case_latency_p50_s']:.0f}/{m['case_latency_p95_s']:.0f}s")
    print(f"  model call latency p50/p95 {m['model_call_latency_p50_s']:.1f}/"
          f"{m['model_call_latency_p95_s']:.1f}s  "
          f"| in-call {m['in_call_prompt_tok_s']} prompt tok/s, "
          f"{m['in_call_output_tok_s']} out tok/s")
    if m.get("prompt_reuse_fraction") is not None:
        print(f"  prefix reuse {m['prompt_reuse_fraction']:.1%} "
              f"({m['prompt_tokens_cached']:,} of "
              f"{m['prompt_tokens_cached'] + m['tokens_in']:,} prompt tokens "
              f"served from parked KV)")
    print(f"  time split: {m['model_time_s']}s in the model, "
          f"{m['tool_and_overhead_s']}s in tools/overhead "
          f"({m['model_time_fraction']:.0%} model)" if m['model_time_fraction'] is not None else "")
    print(f"  wall-time-exhausted {m['wall_time_exhausted_cases']}  "
          f"agent-error {m['agent_error_cases']}  zero-session {m['zero_session_cases']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

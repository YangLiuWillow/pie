"""Fixed-trajectory replay: mask- vs rebuild-condensation on a REAL agent run.

The Stage-2 live A/B (job 19059956) was CONFOUNDED — running the agent twice
produced two DIFFERENT trajectories (layer-B engine nondeterminism), so the
wall/quality delta measured trajectory variance, not the condenser. This harness
removes the confound in two phases:

  PHASE 1 — CAPTURE (once, GPU). Run openhands-agent on one real SWE-bench
    instance with a HIGH context_token_limit so it does NOT condense: a clean,
    linear trajectory. `dump_trajectory=True` returns every turn's assistant JSON
    + observation. Saved to a JSON file (reused on reruns).

  PHASE 2 — REPLAY (GPU, cheap, deterministic). Tokenize that fixed transcript
    once into token_ids + per-turn offsets, then feed it to the mask-condense-
    bench `traj_replay` mode, which imposes a LOW context_limit and, at every
    turn that overflows, probes the model's prediction of the REAL continuation
    three ways: FULL (whole history), MASK (drop middle, original positions,
    Pie — free), REBUILD (drop middle, re-positioned compactly — stock/APC).

Both condensers see byte-identical tokens ⇒ any perplexity gap is purely the
condensation mechanism. Headline = mask/rebuild perplexity ratio (quality) +
rebuild re-prefill ms (the compounding cost mask makes 0).

Env: INSTANCE_ID, PIE_PORT, CAPTURE_CONTEXT_LIMIT, MAX_STEPS, REPLAY_CONTEXT_LIMIT,
     KEEP_RECENT_TURNS, PROBE_TOKENS, TRAJ_JSON (capture cache path), HF_HOME.
"""
from __future__ import annotations

import asyncio
import glob
import json
import math
import os
import statistics as st
import sys

INSTANCE_ID = os.environ.get("INSTANCE_ID", "django__django-14373")
PIE_PORT = int(os.environ.get("PIE_PORT", "18098"))
CAPTURE_CONTEXT_LIMIT = int(os.environ.get("CAPTURE_CONTEXT_LIMIT", "60000"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "50"))
REPLAY_CONTEXT_LIMIT = int(os.environ.get("REPLAY_CONTEXT_LIMIT", "16000"))
KEEP_RECENT_TURNS = int(os.environ.get("KEEP_RECENT_TURNS", "12"))
PROBE_TOKENS = int(os.environ.get("PROBE_TOKENS", "48"))
TRAJ_JSON = os.environ.get(
    "TRAJ_JSON", f"logs/traj_capture_{INSTANCE_ID.replace('/', '_')}.json"
)
AGENT_INFERLET = os.environ.get("AGENT_INFERLET", "openhands-agent@0.1.0")
BENCH_INFERLET = os.environ.get("BENCH_INFERLET", "mask-condense-bench@0.1.0")
BENCH_WASM = os.environ.get("WASM", "")
BENCH_MANIFEST = os.environ.get("MANIFEST", "")


# ── PHASE 1: capture ──────────────────────────────────────────────────────────
def capture_trajectory() -> dict:
    """Run the agent once and return {system, task, turns:[{assistant,observation}]}."""
    if os.path.exists(TRAJ_JSON):
        print(f"[capture] reusing cached trajectory {TRAJ_JSON}")
        with open(TRAJ_JSON) as f:
            return json.load(f)

    from datasets import load_dataset
    from benchmarks.swe_bench import (
        Problem, checked_out_repo, _run_agent_inferlet, _format_user_prompt,
    )
    from tool_server import start_tool_server

    row = next(
        (r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
         if r["instance_id"] == INSTANCE_ID),
        None,
    )
    if row is None:
        raise SystemExit(f"instance {INSTANCE_ID!r} not found")
    problem = Problem.from_row(row)

    print(f"[capture] running {INSTANCE_ID} (no-condense, limit={CAPTURE_CONTEXT_LIMIT})")
    with checked_out_repo(problem) as ws:
        server, port = start_tool_server(str(ws))
        try:
            result = asyncio.run(_run_agent_inferlet(
                f"ws://127.0.0.1:{PIE_PORT}", AGENT_INFERLET,
                task=_format_user_prompt(problem, use_cwd=True),
                tool_server_url=f"http://127.0.0.1:{port}",
                max_steps=MAX_STEPS,
                context_token_limit=CAPTURE_CONTEXT_LIMIT,
                dump_trajectory=True,
                idle_timeout_s=600.0,
                instance_timeout_s=3000.0,
            ))
        finally:
            server.shutdown()

    traj = result.get("trajectory")
    if not traj or not traj.get("turns"):
        raise SystemExit(f"capture produced no trajectory (finished={result.get('finished')})")
    traj["_meta"] = {
        "instance_id": INSTANCE_ID,
        "finished": result.get("finished"),
        "steps": result.get("steps"),
        "num_turns": len(traj["turns"]),
    }
    os.makedirs("logs", exist_ok=True)
    with open(TRAJ_JSON, "w") as f:
        json.dump(traj, f)
    print(f"[capture] {len(traj['turns'])} turns → {TRAJ_JSON}")
    return traj


# ── PHASE 2a: tokenize the fixed transcript ──────────────────────────────────
def _tokenizer():
    from tokenizers import Tokenizer
    tj = sorted(glob.glob(
        os.environ["HF_HOME"]
        + "/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/*/tokenizer.json"
    ))
    if not tj:
        raise SystemExit("Qwen3-Coder-30B tokenizer.json not found under HF_HOME")
    return Tokenizer.from_file(tj[0])


def build_tokens(traj: dict) -> tuple[list[int], list[int]]:
    """Render the transcript in Qwen chat format and return (token_ids, turn_starts).

    A 'turn' = one assistant message + its observation; turn_starts[i] = token
    offset where turn i's assistant segment begins (the drop-middle boundary).
    Segments are concatenated so offsets are exactly additive.
    """
    tok = _tokenizer()

    def seg(role: str, content: str) -> list[int]:
        text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return tok.encode(text, add_special_tokens=False).ids

    ids: list[int] = []
    ids += seg("system", traj["system"])
    ids += seg("user", traj["task"])
    turn_starts: list[int] = []
    for turn in traj["turns"]:
        turn_starts.append(len(ids))  # assistant segment of this turn starts here
        ids += seg("assistant", turn.get("assistant", ""))
        obs = turn.get("observation", "")
        if obs:
            ids += seg("user", f"Observation:\n{obs}")
    return ids, turn_starts


# ── PHASE 2b: replay through the inferlet ────────────────────────────────────
async def run_replay(token_ids: list[int], turn_starts: list[int]) -> dict:
    from pie_client import Event, PieClient

    cfg = {
        "token_ids": token_ids,
        "turn_starts": turn_starts,
        "keep_recent_turns": KEEP_RECENT_TURNS,
        "num_queries": PROBE_TOKENS,
        "context_limit": REPLAY_CONTEXT_LIMIT,
    }
    async with PieClient(f"ws://127.0.0.1:{PIE_PORT}") as c:
        await c.authenticate("local-dev")
        if BENCH_WASM and BENCH_MANIFEST:
            await c.install_program(BENCH_WASM, BENCH_MANIFEST, force_overwrite=True)
        proc = await c.launch_process(BENCH_INFERLET, input=cfg)
        while True:
            ev, val = await proc.recv()
            if ev == Event.Return:
                s = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
                return json.loads(s)
            if ev == Event.Error:
                raise SystemExit(f"replay inferlet error: {val!r}")
            if ev == Event.Stdout:
                line = (val.decode() if isinstance(val, (bytes, bytearray)) else str(val)).rstrip()
                if line:
                    print(f"  [inferlet] {line}")


def ppl(mean_logprob: float) -> float:
    return math.exp(-mean_logprob) if mean_logprob == mean_logprob else float("nan")


def report(res: dict) -> None:
    full = res["full_lp"]; mask = res["mask_lp"]; rebuild = res["rebuild_lp"]
    reprefill = res["rebuild_reprefill_ms"]; pturns = res["probe_turns"]
    hlen = res["history_len"]

    print(f"\n=== TRAJ-REPLAY: {res['num_probes']} probes over {res['num_turns']} turns "
          f"(nq={res['nq']}, context_limit={res['context_limit']}, "
          f"keep_recent_turns={res['keep_recent_turns']}) ===")
    print(f"{'turn':>5} {'histlen':>8} {'full_ppl':>9} {'mask_ppl':>9} "
          f"{'rebld_ppl':>9} {'m/r':>6} {'reprefill_ms':>12}")
    ratios = []
    for i in range(len(pturns)):
        fp, mp, rp = ppl(full[i]), ppl(mask[i]), ppl(rebuild[i])
        r = mp / rp if rp == rp and rp > 0 else float("nan")
        if r == r:
            ratios.append(r)
        print(f"{pturns[i]:>5} {hlen[i]:>8} {fp:>9.4f} {mp:>9.4f} {rp:>9.4f} "
              f"{r:>6.3f} {reprefill[i]:>12.1f}")

    def ms(xs):
        xs = [x for x in xs if x == x]
        if not xs:
            return float("nan"), 0.0
        m = st.mean(xs)
        sem = (st.pstdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
        return m, sem

    rm, rsem = ms(ratios)
    total_reprefill = sum(x for x in reprefill if x == x)
    better = sum(1 for r in ratios if r < 1.0)
    print(f"\n  mean mask/rebuild perplexity ratio = {rm:.4f} +/- {rsem:.4f} "
          f"(mask better in {better}/{len(ratios)} probes)")
    print(f"  rebuild re-prefill total = {total_reprefill:.0f} ms over {len(pturns)} "
          f"condensations; MASK = 0 ms (structural)")
    print("\nREADING: ratio +/- SEM straddling 1.0 => mask == rebuild QUALITY on the "
          "real trajectory (and mask is FREE). The re-prefill ms is the compounding "
          "cost APC must pay to drop the middle; mask avoids it entirely.")


def main() -> None:
    traj = capture_trajectory()
    token_ids, turn_starts = build_tokens(traj)
    print(f"[tokenize] {len(token_ids)} tokens, {len(turn_starts)} turns; "
          f"last turn starts at {turn_starts[-1]} "
          f"(replay context_limit={REPLAY_CONTEXT_LIMIT})")
    if turn_starts[-1] <= REPLAY_CONTEXT_LIMIT:
        print("WARNING: trajectory never exceeds REPLAY_CONTEXT_LIMIT — no condensation "
              "will fire. Lower REPLAY_CONTEXT_LIMIT or capture a longer run.",
              file=sys.stderr)
    res = asyncio.run(run_replay(token_ids, turn_starts))
    with open(TRAJ_JSON.replace(".json", "_replay.json"), "w") as f:
        json.dump(res, f, indent=2)
    report(res)
    print("=== done ===")


if __name__ == "__main__":
    main()

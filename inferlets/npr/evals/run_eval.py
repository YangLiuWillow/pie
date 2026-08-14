"""Concurrent avg@k evaluation driver for the NPR inferlet.

Speaks the same JSON-over-WS turn protocol as `../client.py`, but one
**connection per run**: the gateway's session loop replaces the in-flight turn
whenever a new client frame arrives (`gateway/src/ingress/ws.rs:96` sets
`cur = Some(new_rx)`), so a second `launch_process` on a live connection would
silently drop the first run's event stream. Per-run connections also give a
clean turn termination — the worker treats the launched process's `return`
event as terminal (`worker/src/link/gateway.rs::turn_terminal`).

The wasm is uploaded once up front; later connections launch by name, because
uploaded programs are installed at `add_program` time and the registry is
server-side.

Arms (see `--arms`):
  refill      join_mode=refill    — the faithful KV-equivalent join (default)
  adopt       join_mode=adopt     — phase-3 faithful join via adopt_kv KV graft
                                    (no recomputation; falls back to refill per
                                    sibling — check `adopt_fallbacks` in results)
  textual     join_mode=textual   — phase-1 baseline, steps concatenated causally
  sequential  max_plans=0         — never forks; the model emits the same
                                    <guideline>/<step> schema but decodes it in
                                    one causal stream (NPR's degrade-to-
                                    sequential path). The speed reference.

Results stream to a JSONL file, one line per run, so the sweep is resumable
(`--resume` skips keys already present) and survives a crash.

NOTE on timing: `elapsed_ms` is per-run wall clock, so it is only a speed
measurement at `--concurrency 1`. Accuracy sweeps should use high concurrency
(the engine batches them); speed passes must be serial.

Usage:
    python run_eval.py --arms refill,textual,sequential --k 8 --concurrency 24 \
        --out results/aime25.jsonl
    python run_eval.py --arms refill,sequential --k 1 --concurrency 1 \
        --limit 8 --tag speed --out results/speed.jsonl
"""

import argparse
import asyncio
import json
import pathlib
import sys
import time

import blake3
import websockets

HERE = pathlib.Path(__file__).resolve().parent
INFERLET = HERE.parent

ARMS = {
    "refill": {"join_mode": "refill"},
    "adopt": {"join_mode": "adopt"},
    "textual": {"join_mode": "textual"},
    "sequential": {"join_mode": "refill", "max_plans": 0},
}


def load_problems(path: pathlib.Path, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows[:limit] if limit else rows


async def upload_program(url: str, identity: str) -> None:
    wasm = (INFERLET / "target/wasm32-wasip2/release/npr.wasm").read_bytes()
    manifest = (INFERLET / "Pie.toml").read_text()
    async with websockets.connect(
        url, additional_headers={"x-pie-identity": identity}, max_size=None
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "add_program",
                    "corr_id": 1,
                    "program_hash": blake3.blake3(wasm).hexdigest(),
                    "manifest": manifest,
                    "force_overwrite": True,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "chunk_data": list(wasm),
                }
            )
        )
        async for raw in ws:
            m = json.loads(raw)
            if m.get("type") == "response":
                if not m.get("ok"):
                    raise RuntimeError(f"add_program failed: {m.get('result')}")
                print(f"[upload] {len(wasm)} bytes ok", flush=True)
                return
            if m.get("type") == "error":
                raise RuntimeError(f"add_program error: {m.get('message')}")


async def one_run(url: str, identity: str, payload: dict, timeout: float) -> dict:
    """Launch one process on a fresh connection; return its parsed result."""
    t0 = time.monotonic()
    stderr: list[str] = []
    async with websockets.connect(
        url, additional_headers={"x-pie-identity": identity}, max_size=None
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "launch_process",
                    "corr_id": 2,
                    "inferlet": "npr@0.1.0",
                    "input": json.dumps(payload),
                    "capture_outputs": True,
                }
            )
        )
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no return after {timeout}s")
            raw = await asyncio.wait_for(ws.recv(), timeout=left)
            m = json.loads(raw)
            t = m.get("type")
            if t == "response" and not m.get("ok"):
                raise RuntimeError(f"launch failed: {m.get('result')}")
            elif t == "process_event":
                ev, val = m.get("event"), m.get("value", "")
                if ev == "return":
                    out = json.loads(val)
                    out["client_wall_ms"] = int((time.monotonic() - t0) * 1000)
                    return out
                if ev == "error":
                    raise RuntimeError(f"process error: {val[:2000]}")
                if ev == "stderr":
                    stderr.append(val)
            elif t == "error":
                raise RuntimeError(f"ws error: {m.get('message')} {' '.join(stderr[-3:])}")


async def worker(name: int, queue: asyncio.Queue, args, out_lock, out_file, state):
    while True:
        # Circuit breaker: a wedged/dead server makes every remaining run
        # fail fast, silently voiding the sweep while looking busy (the
        # first 720-run sweep burned 499 runs this way; a parallel session
        # nearly published dead-server results as model scores). Stop
        # pulling work once the consecutive-failure streak trips.
        if state.get("tripped"):
            return
        try:
            task = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        key, arm, prob, sample = task
        payload = dict(ARMS[arm])
        payload["question"] = prob["problem"]
        payload["max_new_tokens"] = args.max_new_tokens
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        if args.top_p is not None:
            payload["top_p"] = args.top_p
        if args.prompt_cache:
            payload["prompt_cache"] = True

        record = None
        for attempt in range(args.retries + 1):
            try:
                res = await one_run(args.url, args.identity, payload, args.timeout)
                record = {
                    "key": key,
                    "arm": arm,
                    "problem_id": prob["id"],
                    "sample": sample,
                    "gold": prob["answer"],
                    "attempt": attempt,
                    "error": None,
                    **res,
                }
                break
            except Exception as e:  # noqa: BLE001 — record and continue the sweep
                err = f"{type(e).__name__}: {e}"
                if attempt == args.retries:
                    record = {
                        "key": key,
                        "arm": arm,
                        "problem_id": prob["id"],
                        "sample": sample,
                        "gold": prob["answer"],
                        "attempt": attempt,
                        "error": err,
                        "answer": None,
                    }
                else:
                    print(f"[w{name}] {key} retry {attempt + 1}: {err}", flush=True)
                    await asyncio.sleep(2.0)

        async with out_lock:
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()
            state["done"] += 1
            if record.get("error") is not None:
                state["err_streak"] = state.get("err_streak", 0) + 1
                if (
                    args.abort_after > 0
                    and state["err_streak"] >= args.abort_after
                    and not state.get("tripped")
                ):
                    state["tripped"] = True
                    print(
                        f"[ABORT] {state['err_streak']} consecutive failed runs — "
                        "the server is likely wedged or dead (check its log for "
                        "'Insufficient Memory' / compute failures / warn-level "
                        "rejections). Stopping the sweep; restart the server and "
                        "re-run with --resume.",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                state["err_streak"] = 0
            ok = record.get("answer") is not None and str(record["answer"]).strip() == str(
                record["gold"]
            ).strip()
            print(
                f"[{state['done']}/{state['total']}] {key} "
                f"ans={record.get('answer')} gold={record['gold']} "
                f"{'OK ' if ok else '.  '}"
                f"tok={record.get('tokens_generated')} "
                f"blocks={record.get('parallel_blocks')} "
                f"wall={record.get('client_wall_ms')}ms"
                + (f" ERR {record['error'][:120]}" if record.get("error") else ""),
                flush=True,
            )
        queue.task_done()


async def main_async(args) -> int:
    problems = load_problems(pathlib.Path(args.data), args.limit)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            print(f"unknown arm {a!r}; known: {list(ARMS)}", file=sys.stderr)
            return 2

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("error") is None or args.keep_errors:
                    seen.add(rec["key"])
        print(f"[resume] {len(seen)} runs already recorded", flush=True)

    queue: asyncio.Queue = asyncio.Queue()
    for arm in arms:
        for prob in problems:
            for k in range(args.k):
                key = f"{arm}/{prob['id']}/{k}"
                if key not in seen:
                    queue.put_nowait((key, arm, prob, k))
    total = queue.qsize()
    print(
        f"[plan] {len(arms)} arms x {len(problems)} problems x k={args.k} "
        f"= {total} runs to go, concurrency {args.concurrency}",
        flush=True,
    )
    if total == 0:
        return 0

    if not args.no_upload:
        await upload_program(args.url, args.identity)

    state = {"done": 0, "total": total}
    lock = asyncio.Lock()
    t0 = time.monotonic()
    with out_path.open("a") as f:
        await asyncio.gather(
            *[
                worker(i, queue, args, lock, f, state)
                for i in range(min(args.concurrency, total))
            ]
        )
    if state.get("tripped"):
        print(
            f"[done-ABORTED] {state['done']}/{total} runs in "
            f"{time.monotonic() - t0:.0f}s -> {out_path} (circuit breaker tripped)",
            flush=True,
        )
        return 3

    # End-of-sweep canary: liveness proved only at boot is worthless — a
    # server that is up at t=0 and dead at t=60 produces a summary line
    # identical to a healthy one. Prove it can still serve after the last
    # real run before trusting the sweep.
    try:
        await one_run(
            args.url,
            args.identity,
            {"max_plans": 0, "max_new_tokens": 8, "question": "canary: reply with any token"},
            min(args.timeout, 120.0),
        )
        print("[canary] server alive and generating at sweep end", flush=True)
    except Exception as e:  # noqa: BLE001
        print(
            f"[canary-FAILED] server did not complete a trivial run after the "
            f"sweep: {type(e).__name__}: {e} — treat late results with "
            "suspicion and check the server log.",
            file=sys.stderr,
            flush=True,
        )
        print(f"[done] {total} runs in {time.monotonic() - t0:.0f}s -> {out_path}", flush=True)
        return 4

    print(f"[done] {total} runs in {time.monotonic() - t0:.0f}s -> {out_path}", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8092/v1/ws")
    ap.add_argument("--identity", default="default/npr")
    ap.add_argument("--data", default=str(HERE / "aime25.jsonl"))
    ap.add_argument("--arms", default="refill,textual,sequential")
    ap.add_argument("--k", type=int, default=8, help="samples per problem")
    ap.add_argument("--limit", type=int, default=None, help="first N problems only")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=1800.0, help="per-run seconds")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=30000)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument(
        "--abort-after",
        type=int,
        default=10,
        help="stop the sweep after this many CONSECUTIVE failed runs (a wedged "
        "server fails everything fast while looking busy; 0 disables)",
    )
    ap.add_argument(
        "--prompt-cache",
        action="store_true",
        help="reuse the prompt prefill across repeats of a problem via a "
        "content-addressed context snapshot (one snapshot per problem is "
        "retained server-side for the life of the server; check the "
        "prompt_cache field in results for hit/miss)",
    )
    ap.add_argument("--out", default=str(HERE / "results/aime25.jsonl"))
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--keep-errors", action="store_true", help="do not retry failed keys")
    ap.add_argument("--no-upload", action="store_true", help="program already installed")
    ap.add_argument("--tag", default=None, help="unused marker for run bookkeeping")
    args = ap.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()

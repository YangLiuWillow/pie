#!/usr/bin/env python3
"""Minutes-scale repro for the c8 overcommit livelock (2026-07-30).

Drives N_SESSIONS concurrent live-context conversations against a running
`pie serve` (overcommit toml), each with a UNIQUE ~TARGET_TOKENS-token history
-- uniqueness matters: the KV trie is content-addressed, so identical fillers
would dedupe and never overcommit the pool. Turns run concurrently in rounds,
mimicking the benchmark's c8 shape at ~1/10 the wall time.

Wedge criterion: any call exceeding CALL_TIMEOUT_S (healthy contended calls
run seconds; the benchmark's failures were 900 s starvation). On wedge, prints
which sessions/rounds stalled and exits 2 so the wrapper can snapshot server
state. Exit 0 = all rounds completed.

Usage:
  pie serve --config pie_cuda_native_config_30b_moe_h100_overcommit.toml \
      --port 18097 --no-auth &            # PIE_CUDA_KV_PAGE_SIZE=32
  /root/venvs/harness/bin/python 50_overcommit_repro.py \
      [--sessions 8] [--rounds 12] [--target-tokens 28000] \
      [--idle-suspend] [--uri ws://127.0.0.1:18097]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time

from pie_client import Event, PieClient

INFERLET = "openhands-coder-session@0.1.0"

WORDS = (
    "diagnostic scheduler restore admission eviction market bid rent page "
    "context snapshot suffix prefix replay working committed pinned stashed "
    "suspended queue drain clearing price dividend wallet horizon overcommit"
).split()


def filler(seed: str, n_chars: int) -> str:
    rng = random.Random(seed)
    out = []
    total = 0
    while total < n_chars:
        w = rng.choice(WORDS)
        out.append(w)
        total += len(w) + 1
    return " ".join(out)


def as_text(v) -> str:
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


class Session:
    def __init__(self, idx: int, target_tokens: int, idle_suspend: bool, max_tokens: int = 32):
        self.max_tokens = max_tokens
        self.idx = idx
        self.sid = f"repro-{idx}"
        self.idle_suspend = idle_suspend
        # ~4 chars/token for this word list; unique per session via seed.
        self.messages = [
            {"role": "system", "content": filler(f"sys-{idx}", target_tokens * 4)}
        ]
        self.proc = None
        self.cm = None
        self.latencies: list[float] = []
        self.modes: dict[str, int] = {}
        self.wedged_at: int | None = None

    async def connect(self, uri: str, boot_timeout: float) -> None:
        self.cm = PieClient(uri)
        client = await self.cm.__aenter__()
        await client.authenticate(f"repro{self.idx}")
        self.proc = await client.launch_process(INFERLET, input={"daemon": True})
        while True:
            event, value = await asyncio.wait_for(self.proc.recv(), timeout=boot_timeout)
            if event == Event.Message and json.loads(as_text(value)).get("ready"):
                return
            if event in (Event.Error, Event.Return):
                raise RuntimeError(f"s{self.idx} daemon failed: {value!r}")

    async def turn(self, rnd: int, timeout: float) -> float:
        self.messages.append(
            {"role": "user", "content": f"round {rnd}: " + filler(f"u-{self.idx}-{rnd}", 2000)}
        )
        payload = {
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "use_grammar": False,
            "session_id": self.sid,
            "kv_verify": True,
            "live_context": True,
            "live_idle_suspend": self.idle_suspend,
        }
        t0 = time.monotonic()
        await self.proc.signal(json.dumps(payload))
        while True:
            event, value = await asyncio.wait_for(self.proc.recv(), timeout=timeout)
            if event == Event.Message:
                frame = json.loads(as_text(value))
                if not frame.get("ok"):
                    raise RuntimeError(f"s{self.idx} inferlet error: {frame.get('error')!r}")
                out = frame["result"]
                dt = time.monotonic() - t0
                self.latencies.append(dt)
                sess = out.get("session") or {}
                m = sess.get("mode", "?")
                self.modes[m] = self.modes.get(m, 0) + 1
                self.messages.append(
                    {"role": "assistant", "content": out.get("text") or "(empty)"}
                )
                return dt
            if event in (Event.Error, Event.Return):
                raise RuntimeError(f"s{self.idx} proc event {event}: {value!r}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:18097")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--target-tokens", type=int, default=28000)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--call-timeout", type=float, default=120.0)
    ap.add_argument("--idle-suspend", action="store_true")
    ap.add_argument("--boot-timeout", type=float, default=180.0)
    args = ap.parse_args()

    sessions = [
        Session(i, args.target_tokens, args.idle_suspend, args.max_tokens) for i in range(args.sessions)
    ]
    print(f"connecting {args.sessions} daemons …", flush=True)
    await asyncio.gather(*(s.connect(args.uri, args.boot_timeout) for s in sessions))
    print("connected. running rounds …", flush=True)

    wedged = False
    for rnd in range(args.rounds):
        t0 = time.monotonic()

        async def one(s: Session):
            try:
                return await s.turn(rnd, args.call_timeout)
            except asyncio.TimeoutError:
                s.wedged_at = rnd
                return None

        results = await asyncio.gather(*(one(s) for s in sessions))
        line = " ".join(
            f"s{s.idx}:{'WEDGE' if r is None else f'{r:5.1f}s'}"
            for s, r in zip(sessions, results)
        )
        print(f"round {rnd:2d}  [{time.monotonic()-t0:6.1f}s]  {line}", flush=True)
        if any(r is None for r in results):
            wedged = True
            break

    print("\n=== summary")
    for s in sessions:
        med = sorted(s.latencies)[len(s.latencies) // 2] if s.latencies else 0
        print(
            f"  s{s.idx}: calls={len(s.latencies)} med={med:.1f}s "
            f"max={max(s.latencies, default=0):.1f}s modes={s.modes} "
            f"wedged_at={s.wedged_at}"
        )
    if wedged:
        stuck = [s.idx for s in sessions if s.wedged_at is not None]
        print(f"\nWEDGED: sessions {stuck} exceeded {args.call_timeout:.0f}s — "
              "snapshot the server log NOW (scheduler state at stall).")
        return 2
    print("\nALL ROUNDS COMPLETED — no wedge at this configuration.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

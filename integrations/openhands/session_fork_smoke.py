"""Local smoke of the openhands-coder-session *fork* protocol (delegation KV reuse).

Drives the inferlet directly through PieClient (no OpenHands agent). Proves that
a child session can fork a parent session's KV snapshot and prefill only its own
suffix, with from-scratch fidelity, and without disturbing the parent snapshot:

  1. parent fresh call     -> mode=fresh
  2. child forks parent    -> mode=forked, prefill == child_len - parent_len
  3. divergent child       -> fork prefix test fails -> mode=fresh (full prefill)
  4. parent still extends   -> mode=extended (fork did not consume/mutate parent)
  5. deletes               -> mode=deleted

All generating calls run with kv_verify on, so a fork that did not reproduce the
canonical from-scratch render byte-for-byte would error instead of pass.
"""
import asyncio
import json
import sys

from pie_client import Event, PieClient

URI = "ws://127.0.0.1:18099"
INFERLET = "openhands-coder-session@0.1.0"
WASM = "/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-coder-session/target/wasm32-wasip2/release/openhands_coder_session.wasm"
MANIFEST = "/nfs/roberts/project/pi_ql324/ly337/pie/inferlets/openhands-coder-session/Pie.toml"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
}]


async def call(client, payload):
    proc = await client.launch_process(INFERLET, input=payload)
    while True:
        event, value = await asyncio.wait_for(proc.recv(), timeout=120)
        if event == Event.Return:
            if isinstance(value, (bytes, bytearray)):
                value = value.decode()
            if isinstance(value, str):
                value = json.loads(value)
            return value
        if event == Event.Error:
            raise RuntimeError(f"inferlet error: {value!r}")


async def main():
    async with PieClient(URI) as client:
        await client.authenticate("local-dev")
        await client.install_program(WASM, MANIFEST, force_overwrite=True)

        base = {
            "tools": TOOLS,
            "max_tokens": 8,
            "temperature": 0.0,
            "kv_verify": True,
        }

        # 1. Parent builds a real conversation prefix.
        parent_msgs = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Investigate the failing test in foo.py"},
        ]
        rp = await call(client, {
            **base, "session_id": "fork_parent",
            "messages": parent_msgs, "session_prev_len": 0,
        })
        sp = rp["session"]
        print("parent:", sp)
        assert sp["mode"] == "fresh", sp
        assert sp["prefill_tokens"] == sp["len"]

        # 2. Child forks the parent. Its render is the parent's render plus one
        #    extra user turn, so the parent snapshot is a genuine token-prefix.
        child_msgs = parent_msgs + [
            {"role": "user", "content": "Focus only on the regression in bar()."},
        ]
        rc = await call(client, {
            **base, "session_id": "fork_child",
            "messages": child_msgs,
            "session_prev_len": 0,  # child has no snapshot of its own
            "session_fork_from": "fork_parent",
            "session_fork_prev_len": sp["len"],
            "session_fork_prev_hash": sp["hash"],
        })
        sc = rc["session"]
        print("child (fork):", sc)
        assert sc["mode"] == "forked", sc
        assert sc["prefill_tokens"] == sc["len"] - sp["len"], sc
        assert 0 < sc["prefill_tokens"] < sc["len"], sc

        # 3. A child whose render does NOT share the parent prefix must not
        #    fork — the prefix test fails and it falls through to a cold build.
        divergent_msgs = [
            {"role": "system", "content": "You are a specialized web researcher."},
            {"role": "user", "content": "Summarize the HTTP spec."},
        ]
        rd = await call(client, {
            **base, "session_id": "fork_divergent",
            "messages": divergent_msgs,
            "session_prev_len": 0,
            "session_fork_from": "fork_parent",
            "session_fork_prev_len": sp["len"],
            "session_fork_prev_hash": sp["hash"],
        })
        sd = rd["session"]
        print("child (divergent):", sd)
        assert sd["mode"] == "fresh", sd
        assert sd["prefill_tokens"] == sd["len"], sd

        # 4. The parent snapshot must be intact after being forked: a genuine
        #    extension of the parent still opens the parent's own snapshot.
        parent_ext_msgs = parent_msgs + [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c0", "function": {"name": "terminal", "arguments": '{"command": "pytest"}'}},
            ]},
            {"role": "tool", "tool_call_id": "c0", "content": "1 failed"},
        ]
        rpe = await call(client, {
            **base, "session_id": "fork_parent",
            "messages": parent_ext_msgs,
            "session_prev_len": sp["len"], "session_prev_hash": sp["hash"],
        })
        spe = rpe["session"]
        print("parent extend (post-fork):", spe)
        assert spe["mode"] == "extended", spe
        assert spe["prefill_tokens"] == spe["len"] - sp["len"], spe

        # 5. Cleanup.
        for sid in ("fork_child", "fork_divergent", "fork_parent"):
            r = await call(client, {"session_id": sid, "session_action": "delete"})
            assert r["session"]["mode"] == "deleted", (sid, r)

    print("\nALL FORK SMOKE CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

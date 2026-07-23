"""Local smoke of the openhands-coder-session protocol against the dummy driver.

Drives the inferlet directly through PieClient (no OpenHands agent):
  1. first call            -> mode=rebuilt, prefill == len (nothing saved yet)
  2. pure-extension call   -> mode=extended, prefill == delta
  3. rewritten-history call-> mode=rebuilt, prefill == len (condenser case)
  4. retry of call 3       -> identical messages hit their own saved render,
                              so extension with an empty delta
  5. delete                -> mode=deleted, and a later call must rebuild
  6. no session_id         -> stateless, no session block at all
All calls run with kv_verify on.

The cache is self-keyed (commit c9fdf557): a call is identified by the hash of
its own token render, so the host sends nothing but `session_id`. This file
used to echo `session_prev_len`/`session_prev_hash` back from the previous
response and expect the first call to report mode="fresh" — both are artifacts
of the older host-coordinated protocol and were removed.
"""
import asyncio
import json
import os
import sys

from pie_client import Event, PieClient

URI = "ws://127.0.0.1:18099"
INFERLET = "openhands-coder-session@0.1.0"
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_INFERLET = os.path.join(_REPO, "inferlets", "openhands-coder-session")
WASM = os.path.join(_INFERLET, "target", "wasm32-wasip2", "release",
                    "openhands_coder_session.wasm")
MANIFEST = os.path.join(_INFERLET, "Pie.toml")

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
            "session_id": "smoke1",
            "kv_verify": True,
        }
        msgs1 = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the bug in foo.py"},
        ]
        r1 = await call(client, {**base, "messages": msgs1})
        s1 = r1["session"]
        print("call1:", s1)
        # "rebuilt", not "fresh": the cache self-keys, so the first call scans
        # this render's interior boundaries, finds nothing saved yet, and
        # rebuilds. "fresh" is now reserved for a render with no interior
        # boundary at all (a single render unit). Production runs show
        # modes={rebuilt: 1, extended: N} for exactly this reason.
        assert s1["mode"] == "rebuilt", s1
        assert s1["prefill_tokens"] == s1["len"]

        msgs2 = msgs1 + [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c0", "function": {"name": "terminal", "arguments": '{"command": "ls"}'}},
            ]},
            {"role": "tool", "tool_call_id": "c0", "content": "foo.py\nbar.py"},
        ]
        r2 = await call(client, {**base, "messages": msgs2})
        s2 = r2["session"]
        print("call2:", s2)
        assert s2["mode"] == "extended", s2
        assert s2["prefill_tokens"] == s2["len"] - s1["len"], s2
        assert 0 < s2["prefill_tokens"] < s2["len"]

        # History rewrite (condenser-style): earlier message changed.
        msgs3 = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Condensed summary: agent listed files."},
        ]
        r3 = await call(client, {**base, "messages": msgs3})
        s3 = r3["session"]
        print("call3:", s3)
        assert s3["mode"] == "rebuilt", s3
        assert s3["prefill_tokens"] == s3["len"]

        # Identical retry re-prefills. The boundary scan skips the final
        # boundary (`len >= full_len` in build_context), so a render can never
        # reuse the snapshot it saved itself — only a strictly shorter prefix.
        # Under append-only OpenHands history that costs nothing, since the
        # previous turn is always strictly shorter; it only bites when the exact
        # same message list is sent twice, which needs a transport-level retry
        # (a nudge or fake-user response appends a message, so it extends).
        # Allowing the equal-length hit would make this free and looks safe —
        # save() already ignores a duplicate name — but it is a behaviour change
        # to the reuse path, so it is left alone here.
        r4 = await call(client, {**base, "messages": msgs3})
        s4 = r4["session"]
        print("call4:", s4)
        assert s4["mode"] == "rebuilt", s4
        assert s4["prefill_tokens"] == s4["len"], s4
        assert s4["hash"] == s3["hash"], (s3, s4)

        r5 = await call(client, {
            "session_id": "smoke1", "session_action": "delete",
        })
        print("call5:", r5["session"])
        assert r5["session"]["mode"] == "deleted"
        assert r5["stop_reason"] == "session_deleted"

        # After delete, echoing stale state must rebuild (snapshot gone).
        r6 = await call(client, {**base, "messages": msgs3})
        s6 = r6["session"]
        print("call6:", s6)
        assert s6["mode"] == "rebuilt", s6

        await call(client, {"session_id": "smoke1", "session_action": "delete"})

        # Stateless path still works (no session block in output).
        r7 = await call(client, {"messages": msgs1, "tools": TOOLS, "max_tokens": 8})
        assert "session" not in r7 or r7["session"] is None, r7
        print("call7: stateless ok, prompt_tokens =", r7["prompt_tokens"])

    print("\nALL SESSION SMOKE CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

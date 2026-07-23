"""Slice 3 — measure the prefill saving from session-fork on delegation.

Exercises the real Slice-2 glue (`PieLLM.model_copy`) end-to-end against the
live coder-session inferlet on GPU. A parent PieLLM runs a short multi-turn
trajectory (building a KV snapshot); a child is derived exactly as the OpenHands
SDK derives a delegated sub-agent's LLM (``parent.agent.llm.model_copy(...)``,
see openhands/tools/task/manager.py:306) and continues from the parent's full
context. The fork-stamped child forks the parent's prompt KV and prefills only
its own suffix. A no-fork control (a freshly constructed PieLLM, never copied)
runs the identical child prompt cold — the litellm baseline's behavior.

Reports tokens prefilled forked-vs-control = the prefill saving that the litellm
baseline architecturally cannot achieve.

Scope note: this measures the *continuation* delegation family (critic / resume /
full-context sub-agents), where the child's prompt contains the parent's full
prefix — which the current full-prefix fork supports. Stock TaskToolSet spawns a
*fresh* sub-agent that shares only the system+tools preamble; capturing that
needs longest-common-prefix fork (a follow-up that requires runtime token
readback). See the session handoff.
"""
import sys

from pie_openhands import PieLLM

URI = "ws://127.0.0.1:18099"
INFERLET = "openhands-coder-session@0.1.0"

SYSTEM = (
    "You are a coding agent. Investigate and fix issues with minimal, "
    "well-tested changes. Use the tools available to inspect and edit files."
)


def _llm(**overrides) -> PieLLM:
    defaults = dict(
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        pie_uri=URI,
        pie_username="local-dev",
        pie_inferlet=INFERLET,
        pie_session=True,
        native_tool_calling=True,
        num_retries=1,
        retry_min_wait=0,
        retry_max_wait=0,
    )
    defaults.update(overrides)
    return PieLLM(**defaults)


def _call(llm: PieLLM, messages: list[dict]) -> dict:
    """One transport round-trip; return this call's session telemetry."""
    llm._transport_call(messages=messages, tools=[], max_tokens=4, temperature=0.0)
    return llm._pie_session_stats[-1]


def main() -> int:
    # ── Parent trajectory: build a real multi-turn KV snapshot ───────────
    m1 = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Fix the failing test in foo.py."},
    ]
    m2 = m1 + [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c0", "function": {"name": "terminal", "arguments": '{"command": "pytest -x"}'}},
        ]},
        {"role": "tool", "tool_call_id": "c0", "content": "E   AssertionError in test_bar"},
    ]
    m3 = m2 + [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "function": {"name": "terminal", "arguments": '{"command": "sed -n 1,40p foo.py"}'}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "def bar(x):\n    return x + 1"},
    ]

    parent = _llm()
    s1 = _call(parent, m1)
    s2 = _call(parent, m2)
    s3 = _call(parent, m3)
    print("parent call1:", s1)
    print("parent call2:", s2)
    print("parent call3:", s3)
    assert s1["mode"] == "fresh" and s2["mode"] == "extended" and s3["mode"] == "extended"
    parent_snapshot_len = s3["prompt_len"]

    # The child continues from the parent's full context plus one instruction.
    child_msgs = m3 + [
        {"role": "user", "content": "Now also add a regression guard for the negative-input case in bar()."},
    ]

    # ── Fork child: derived exactly as the SDK derives a delegate's LLM ───
    # (openhands/tools/task/manager.py:306,309)
    child = parent.model_copy(update={"stream": False})
    child.reset_metrics()
    assert child._pie_fork_from == parent._pie_session_id, "Slice-2 glue did not stamp fork source"
    cf = _call(child, child_msgs)
    print("child (forked):", cf)

    # ── Control: fresh PieLLM, never copied → no fork → cold prefill ──────
    control = _llm()
    cc = _call(control, child_msgs)
    print("control (no fork):", cc)

    # ── Report ───────────────────────────────────────────────────────────
    assert cf["mode"] == "forked", f"expected forked, got {cf['mode']}"
    assert cc["mode"] == "fresh", f"expected fresh control, got {cc['mode']}"
    assert cf["prompt_len"] == cc["prompt_len"], "renders must match for a fair comparison"

    child_len = cf["prompt_len"]
    forked_prefill = cf["prefill_tokens"]
    control_prefill = cc["prefill_tokens"]
    saved = control_prefill - forked_prefill
    pct = 100.0 * saved / control_prefill if control_prefill else 0.0

    print("\n=== fork delegation prefill saving ===")
    print(f"parent snapshot len         : {parent_snapshot_len}")
    print(f"child render len            : {child_len}")
    print(f"control prefill (no fork)   : {control_prefill}")
    print(f"forked prefill (reused KV)  : {forked_prefill}")
    print(f"prefill tokens saved        : {saved}  ({pct:.1f}%)")

    assert forked_prefill < control_prefill, "fork did not reduce prefill"
    assert forked_prefill == child_len - parent_snapshot_len, "forked prefill should be suffix-only"

    for llm in (parent, child, control):
        llm.close_pie_session()

    print("\nFORK DELEGATION MEASUREMENT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

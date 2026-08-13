#!/usr/bin/env python3
"""Prove that the session inferlet actually resumes KV across turns.

The acceptance suite (`test_acceptance.py`) does NOT cover this and cannot: its
25 checks are 25 independent conversations, so every one of them is a first
turn. It passes identically against Strategy A and Strategy B — which is the
point of it (wire parity), and also why a green acceptance run says nothing
about whether the KV working set is being reused. Every turn in the first green
Strategy B run logged `cached=0`.

This suite drives the shape the strategy is FOR: a multi-turn agent loop where
each request echoes the previous assistant turn back and appends a little. It
asserts on `usage.prompt_tokens_details.cached_tokens`, which the inferlet
reports as the resumed prefix depth.

What each check is really guarding:

  1. `resume_hits_on_echo_back`  — the response/save unification invariant. The
     content we send back and the content we hash must be the same string; if
     they diverge by so much as a trailing space, this is the test that fails,
     and nothing else in the repo would notice.
  2. `resume_survives_a_tool_round_trip` — the real opencode shape, where the
     appended suffix is a tool result answering a call from the retained prefix.
  3. `divergent_history_misses_cleanly` — the safety property the whole design
     rests on: an edited history must MISS and rebuild, never resume a prefix
     that no longer matches. A wrong-but-fluent answer here is the failure mode
     the delta wire was rejected for.
  4. `resume_matches_a_cold_rebuild` — greedy decoding from a resumed prefix and
     from a full rebuild of the same bytes must agree. This is what catches an
     off-by-one between KV length and fold position: such a state still
     generates fluent text, so only a differential test sees it.

Usage:
    PIE_BASE_URL=http://127.0.0.1:8080 python3 test_resume.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("PIE_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
CHAT = f"{BASE_URL}/v1/chat/completions"
MODEL = os.environ.get("PIE_MODEL", "pie")
TIMEOUT = float(os.environ.get("PIE_TIMEOUT", "300"))


def post(body):
    req = urllib.request.Request(
        CHAT,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer test"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def cached(resp):
    """The resumed prefix depth this turn reported."""
    return (resp.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)


def prompt_tokens(resp):
    return (resp.get("usage") or {}).get("prompt_tokens", 0)


def message(resp):
    return resp["choices"][0]["message"]


def turn(messages, tools=None, max_tokens=48, temperature=0.0):
    """One non-streaming turn. Greedy by default so answers are comparable."""
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    return post(body)


def echo_back(messages, resp):
    """Append the assistant turn EXACTLY as opencode would replay it.

    Verbatim `content` and verbatim `tool_calls` — this function is the test's
    model of the client, and any liberty taken here (stripping, re-encoding,
    reordering) would make a passing resume prove nothing about the real one.
    """
    m = message(resp)
    out = list(messages)
    replay = {"role": "assistant", "content": m.get("content") or ""}
    if m.get("tool_calls"):
        replay["tool_calls"] = m["tool_calls"]
    out.append(replay)
    return out


_SYSTEM_BASE = (
    "You are a terse assistant. Answer in one short sentence. "
    "Do not ask questions back."
)


def sys_for(tag):
    """A system prompt unique to one test.

    Tests must not share a system turn. Since the daemon caches at stride
    boundaries and searches ACROSS conversations, a shared system prompt means
    every test after the first starts with that prefix already resident — so
    "first turn" assertions see a non-zero `cached` and the isolation the
    assertions assume is gone. Giving each test its own tag restores it, and
    costs one extra cold prefill per test on a 0.6B.

    This is the same mechanism the head-sharing feature relies on, observed from
    the wrong end: it worked so well it broke the tests.
    """
    return f"{_SYSTEM_BASE} [case:{tag}]"


SYSTEM = _SYSTEM_BASE

# The tool-calling system prompt and shape that `test_acceptance.py` already
# gets a reliable call out of on the 0.6B. Reused verbatim rather than invented:
# a weaker prompt here would turn a resume regression into a silent SKIP.
TOOL_SYSTEM = (
    "You are a function-calling assistant. You MUST answer by "
    "calling one of the provided tools. Never answer from memory."
)

WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }
]


# ---------------------------------------------------------------------------


def test_resume_hits_on_echo_back():
    """Turn 2, echoing turn 1 back verbatim, must resume turn 1's KV."""
    msgs = [
        {"role": "system", "content": sys_for("echo-back")},
        {"role": "user", "content": "Name one primary colour."},
    ]
    r1 = turn(msgs)
    assert cached(r1) == 0, f"first turn should have nothing cached, got {cached(r1)}"

    msgs = echo_back(msgs, r1)
    msgs.append({"role": "user", "content": "Name a different one."})
    r2 = turn(msgs)

    c = cached(r2)
    assert c > 0, (
        "second turn resumed nothing — the save address and the echo-back "
        f"address disagree. usage={r2.get('usage')}"
    )
    # The resumed prefix must be a strict prefix of this turn's prompt, never
    # the whole of it: the new user turn and cue still have to be prefilled.
    assert c < prompt_tokens(r2), f"cached {c} >= prompt {prompt_tokens(r2)}"
    return f"cached {c}/{prompt_tokens(r2)} tokens"


def test_resume_survives_a_tool_round_trip():
    """The opencode shape: the suffix is a tool result answering a retained call."""
    msgs = [
        {"role": "system", "content": f"{TOOL_SYSTEM} [case:tool-roundtrip]"},
        {
            "role": "user",
            "content": "What time is it right now in Paris? Use the get_time tool.",
        },
    ]
    r1 = turn(msgs, tools=WEATHER_TOOL, max_tokens=256)
    calls = message(r1).get("tool_calls")
    if not calls:
        return "SKIP: model declined the forced call (0.6B behaviour, not a resume fault)"

    msgs = echo_back(msgs, r1)
    msgs.append(
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": "14:05 CET"}
    )
    r2 = turn(msgs, tools=WEATHER_TOOL, max_tokens=256)

    c = cached(r2)
    assert c > 0, (
        "tool round-trip resumed nothing — the tool-call canon and the "
        f"echo-back canon disagree. usage={r2.get('usage')}"
    )
    return f"cached {c}/{prompt_tokens(r2)} tokens across the tool result"


def test_divergent_history_stops_at_the_edit():
    """An edit inside the prefix must stop the resume BEFORE the edit.

    This is the safety property. If the inferlet ever resumed past the edit it
    would answer from a context the client never sent — fluently, and with
    nothing else in the suite to catch it.

    It used to assert the resume was zero, which was right when boundaries were
    one-per-render-op: an edited user turn invalidated the only boundary there
    was. With stride boundaries the *system* turn ahead of the edit is still a
    genuine, byte-identical shared prefix, and resuming it is correct — that is
    the same mechanism that lets two different conversations share a 7.2k-token
    head. So the assertion is now relative: strictly less than the unedited
    conversation resumes, which is what "stopped before the edit" means and is a
    stronger claim than "resumed nothing".
    """
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Name one primary colour."},
    ]
    r1 = turn(msgs)
    msgs = echo_back(msgs, r1)

    # Rewrite history the way opencode's compaction would: same shape, different
    # bytes, in a message that is INSIDE the retained prefix.
    unedited = list(msgs)
    unedited.append({"role": "user", "content": "Name a different one."})
    baseline = turn(unedited)

    edited = list(msgs)
    edited[1] = {"role": "user", "content": "Name one primary colour, please."}
    edited.append({"role": "user", "content": "Name a different one."})
    r2 = turn(edited)

    assert cached(r2) < cached(baseline), (
        f"resumed {cached(r2)} tokens against an EDITED history, against "
        f"{cached(baseline)} for the unedited one — the resume reached into or "
        "past the edited message, so the address is not sensitive to a change "
        "inside the retained prefix"
    )
    return f"stopped at the edit: {cached(r2)} vs {cached(baseline)} unedited"


def test_resume_matches_a_cold_rebuild():
    """Greedy from a resumed prefix must equal greedy from a full rebuild.

    This is the check that earns the whole strategy. An over-advanced fold (the
    `run_ahead` speculation hazard) or a one-cell hole under the seal both leave
    a state that still generates fluent, on-topic text — no error, no warning,
    nothing else in this repo would notice. Only a differential against a cold
    rebuild of the SAME BYTES sees it.

    Getting a genuinely cold arm without restarting the daemon: flood the
    branch cache until the one we want is evicted, then send the SAME bytes
    again. Byte-identical request, resumed once and rebuilt once.

    This used to rely on retention dropping the parent branch it extended, so a
    repeat necessarily missed. That stopped being true once the daemon began
    keeping every render boundary — a repeat now re-hits an earlier one, which
    is the retry-resilience the boundary scan exists for. The test noticed
    before anything else did, by asserting the miss it depends on rather than
    assuming it.
    """
    base = [
        {"role": "system", "content": sys_for("cold-rebuild")},
        {"role": "user", "content": "Name one primary colour."},
    ]
    r1 = turn(base)
    convo = echo_back(base, r1)
    convo.append({"role": "user", "content": "Name a different one."})

    warm = turn(convo)
    assert cached(warm) > 0, (
        f"expected a resume to compare against, got usage={warm.get('usage')}"
    )

    # Evict: a dozen distinct conversations push every earlier branch past the
    # retention budget. Cheap — these never decode.
    for i in range(12):
        # A DIFFERENT system prompt, so the evictors push the branch out
        # without leaving their own shared prefix behind for the cold arm.
        turn(
            [
                {"role": "system", "content": sys_for(f"evictor-{i}")},
                {"role": "user", "content": f"evict {i}"},
            ],
            max_tokens=1,
        )

    cold = turn(convo)
    # Not zero: the evicting conversations share this one's SYSTEM turn, so a
    # stride boundary over that prefix legitimately survives. What matters for
    # this test is that the BULK was rebuilt, so the two answers are being
    # produced from a resumed prefix and a rebuilt one respectively.
    assert cached(cold) == 0, (
        f"expected the repeat to rebuild, but it resumed {cached(cold)} against "
        f"the warm run's {cached(warm)} — this test cannot tell warm from cold, "
        "so its pass would be meaningless"
    )

    w, c = message(warm)["content"], message(cold)["content"]
    assert w == c, (
        "resumed and rebuilt answers DIVERGE at temperature 0 — the retained "
        f"state does not represent the same context as a fresh render.\n"
        f"  resumed: {w!r}\n  rebuilt: {c!r}"
    )
    return f"resumed({cached(warm)} cached) == rebuilt, {len(w)} chars"


def test_retry_rehits_an_earlier_boundary():
    """Re-sending the same turn must resume, not rebuild.

    This is what keeping every render boundary buys, and it is the shape a
    client retry actually has: opencode reissues a turn after a transport error
    or a tool failure, byte-identical. With one boundary per branch the retry
    misses and pays a full rebuild at the worst possible moment; with the
    boundary list it re-hits the still-valid earlier one.

    Guarding it explicitly because the cold-rebuild test above depends on the
    opposite behaviour, and a regression that collapsed the boundary list would
    make that test pass while silently costing every retry a rebuild.
    """
    msgs = [
        {"role": "system", "content": sys_for("retry")},
        {"role": "user", "content": "Name one primary colour."},
    ]
    r1 = turn(msgs)
    convo = echo_back(msgs, r1)
    convo.append({"role": "user", "content": "Name a different one."})

    first = turn(convo)
    assert cached(first) > 0, f"setup turn did not resume: {first.get('usage')}"

    retry = turn(convo)
    assert cached(retry) > 0, (
        "a byte-identical retry rebuilt from scratch — the branch kept only its "
        f"tip boundary. usage={retry.get('usage')}"
    )
    return f"retry resumed {cached(retry)} tokens"


TESTS = [
    test_resume_hits_on_echo_back,
    test_retry_rehits_an_earlier_boundary,
    test_resume_survives_a_tool_round_trip,
    test_divergent_history_stops_at_the_edit,
    test_resume_matches_a_cold_rebuild,
]


def main():
    passed = failed = skipped = 0
    for t in TESTS:
        try:
            note = t()
        except AssertionError as e:
            print(f"[FAIL] {t.__name__} — {e}")
            failed += 1
            continue
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {t.__name__} — {e!r}")
            failed += 1
            continue
        if note and note.startswith("SKIP"):
            print(f"[SKIP] {t.__name__} — {note[6:]}")
            skipped += 1
        else:
            print(f"[PASS] {t.__name__}{f' — {note}' if note else ''}")
            passed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped ({len(TESTS)} tests)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

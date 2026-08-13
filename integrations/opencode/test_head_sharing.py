#!/usr/bin/env python3
"""B1: does a NEW conversation reuse another conversation's system+tools head?

## Why this is worth its own test

`test_resume.py` proves a conversation resumes *itself*. That is the easy half:
history is append-only, so each turn's render contains the last one. Sharing
across two *different* conversations is the hard half, and it is where the cold
starts are — OpenHands measured **71% of concurrency-1 prefill as unavoidable
cold starts**, which no amount of within-conversation caching can touch.

opencode's head is ~7.2k tokens of system prompt plus tool schemas, and it is
re-prefilled from scratch for every new session. At pie's measured 297 tok/s on
the Coder-30B that is ~24 s of pure repetition per conversation.

## What had to be true, and what nearly wasn't

OpenHands built this and it **never fired** — their head hash differed per
instance because `FileEditorTool` embedded the working directory in its
description (`d93f6cffe`). So the first thing this checks is that opencode's
head is genuinely shareable.

It is, but not entirely. Measured on the two captured sessions:

- system prompt: 9,648 chars, of which the first **8,695 are invariant**;
- then a per-session block — `Working directory:`, `Is directory a git repo:`,
  `Platform:`, `Today's date:`;
- then 21,188 chars of tool schemas that are **byte-identical** across sessions
  but sit *downstream* of that block, so a prefix cache cannot reach them.

So the guaranteed floor is the 8,695-char invariant prefix (~25% of the head,
shareable across any project and any day), and the ceiling is ~100% for two
sessions in the same directory on the same date — which is the common case for
one developer in one repo.

Both are large, and neither is reachable with one boundary per render op: the
head is a single `EquipAfterSystem` op, so its only boundary is the whole head,
which never matches. Stride boundaries are what make it findable.

Usage:
    PIE_BASE_URL=http://127.0.0.1:8080 python3 test_head_sharing.py
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

BASE = os.environ.get("PIE_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
FIXTURES = Path(__file__).resolve().parents[2] / "tests/inferlets/fixtures/opencode/wire"
TIMEOUT = float(os.environ.get("PIE_TIMEOUT", "600"))


def body(name):
    d = json.loads((FIXTURES / name).read_text())
    b = d["body"]
    b = json.loads(b) if isinstance(b, str) else b
    b = dict(b)
    b.update(max_tokens=8, stream=False, temperature=0.0)
    b.pop("stream_options", None)
    return b


def post(b):
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer share"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def cached(x):
    return (x.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)


def prompt(x):
    return (x.get("usage") or {}).get("prompt_tokens", 0)


def test_same_project_and_day_shares_almost_everything():
    """Two real opencode sessions from the same checkout share their whole head."""
    a = post(body("req-002.json"))   # session Q8GID7
    b = post(body("req-004.json"))   # session Zyo2k8 — a DIFFERENT conversation
    share = cached(b)
    assert share > 0, (
        f"no cross-conversation sharing at all: usage={b.get('usage')}. "
        "Either boundaries are per-op again, or only the tip is being retained."
    )
    frac = share / prompt(b)
    assert frac > 0.5, f"expected most of the head to share, got {frac:.1%}"
    return f"{share}/{prompt(b)} tokens ({frac:.1%}) across two conversations"


def test_different_project_and_day_still_shares_the_invariant_prefix():
    """The floor: the environment block diverges, the prose before it does not.

    This is the case that matters for a fleet — different repos, different days —
    and it is the one a naive "is the head identical?" check would call a miss.
    """
    b = body("req-004.json")
    s = b["messages"][0]["content"]
    s = s.replace(
        "/private/tmp/claude-501/-Users-yangliu-Desktop-Lin-startup/"
        "547ad305-3772-4f62-b37d-d7cff2202cda/scratchpad/oc-project",
        "/Users/someone/an-entirely-different-repo",
    ).replace("Tue Aug 11 2026", "Fri Dec 25 2026")
    b["messages"][0]["content"] = s
    r = post(b)
    share = cached(r)
    assert share > 0, (
        "a different project on a different day shared nothing — the invariant "
        f"prose ahead of the environment block should still hit. usage={r.get('usage')}"
    )
    return f"{share}/{prompt(r)} tokens ({share/prompt(r):.1%}) — the guaranteed floor"


TESTS = [
    test_same_project_and_day_shares_almost_everything,
    test_different_project_and_day_still_shares_the_invariant_prefix,
]


def main():
    passed = failed = 0
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
        print(f"[PASS] {t.__name__} — {note}")
        passed += 1
    print(f"\n{passed} passed, {failed} failed ({len(TESTS)} tests)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

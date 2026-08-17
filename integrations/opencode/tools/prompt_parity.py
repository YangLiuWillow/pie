#!/usr/bin/env python3
"""Do the four engines actually answer the SAME prompt?

Every cross-engine number in `results-qwen36-four-way.md` rests on one
assumption: that opencode's request renders to the same tokens whichever server
receives it. That assumption is not safe, and on this model it was false twice
before this script existed:

  1. **Thinking.** Qwen3.6's chat template prefills `<think>` unless
     `enable_thinking=false`. mlx-lm and vLLM followed the template default;
     pie's renderer hard-codes the no-think cue. The same question reached the
     model as two different prompts, and the whole answer went into a
     `reasoning` field on two arms and into `content` on the third.

  2. **Tool dialect.** Qwen3.6 mandates the XML form
     `<tool_call><function=NAME><parameter=P>`. pie's registry keyed the dialect
     off Coder lineage, which Qwen3.6 is explicitly NOT, and so prompted Hermes
     JSON -- one dialect rendered, another parsed.

Both were invisible in latency and throughput and would have surfaced only as
accuracy, attributed to the engine.

## What it checks

The servers share a tokenizer, so `usage.prompt_tokens` on an identical request
is a cheap fingerprint of the rendered prompt. Two requests are sent to each
live arm:

    plain   -- messages only
    tools   -- the same messages plus a tool schema

`plain` alone would pass while every tool schema was being dropped, which is a
failure this repo has already had (`every_qwen_release_reaches_the_tool_capable_instruct`).
The delta between them is reported per arm: an arm whose tool delta is 0 is not
rendering the schema at all.

Agreement here is evidence, not proof -- two different renderings can collide on
a token count. It is the cheap check that runs before a 20-hour run, not a
substitute for `parity/check_render.py`, which diffs the token IDs themselves.

Usage:
    prompt_parity.py                       # probe the default four
    prompt_parity.py --arm pie=http://127.0.0.1:8080/v1:qwen3.6-35b-a3b:pie-local
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

# name -> (base URL, model id, bearer token)
DEFAULT_ARMS = [
    ("pie", "http://127.0.0.1:8080/v1", "qwen3.6-35b-a3b", "pie-local"),
    ("mlx", "http://127.0.0.1:8001/v1", "mlx-community/Qwen3.6-35B-A3B-4bit", "mlx-local"),
    ("vllm", "http://127.0.0.1:8000/v1", "qwen3.6-35b-a3b", "vllm-local"),
    ("llamacpp", "http://127.0.0.1:8002/v1", "qwen3.6-35b-a3b", "lcpp-local"),
]

MESSAGES = [
    {"role": "system", "content": "You are a terse coding assistant."},
    {"role": "user", "content": "Read the file README.md and summarize it in one line."},
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the working tree.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to read"}},
            "required": ["path"],
        },
    },
}]

NOTHINK = {"enable_thinking": False}


def ask(base, model, token, tools, timeout):
    """One non-streaming call; returns (prompt_tokens, note)."""
    body = {
        "model": model,
        "messages": MESSAGES,
        # One token, not zero: some servers reject max_tokens=0, and the
        # completion is irrelevant -- only the PROMPT accounting is read.
        "max_tokens": 1,
        "stream": False,
        "temperature": 0,
        "chat_template_kwargs": NOTHINK,
    }
    if tools:
        body["tools"] = TOOLS
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:120].decode(errors='replace')}"
    except Exception as e:  # connection refused = arm not running
        return None, f"{type(e).__name__}: {e}"
    pt = (d.get("usage") or {}).get("prompt_tokens")
    return pt, "" if pt else f"no usage.prompt_tokens in response: {str(d)[:120]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="name=baseurl:model:token (repeatable); replaces the defaults")
    ap.add_argument("--timeout", type=float, default=600)
    # The four arms can never be live together: each needs ~20 GiB on a 48 GiB
    # box, which is why every other tool here restarts servers between arms. So
    # results accumulate across boots instead of being collected in one pass.
    ap.add_argument("--append", metavar="FILE",
                    help="append this run's live arms to FILE as JSONL")
    ap.add_argument("--summarize", metavar="FILE",
                    help="compare the arms accumulated in FILE and exit")
    a = ap.parse_args()

    if a.summarize:
        seen = {}
        for line in open(a.summarize):
            line = line.strip()
            if line:
                r = json.loads(line)
                seen[r["arm"]] = r          # last boot of an arm wins
        rows = [(r["arm"], r["plain"], r["tools"], "") for r in seen.values()]
        return report(rows)

    arms = DEFAULT_ARMS
    if a.arm:
        arms = []
        for spec in a.arm:
            name, _, rest = spec.partition("=")
            base, _, rest = rest.partition(":")
            # the URL carries its own colon, so rebuild it
            if rest.startswith("//"):
                scheme = base
                base, _, rest = rest.partition(":")
                base = f"{scheme}:{base}"
            model, _, token = rest.rpartition(":")
            arms.append((name, base, model, token))

    rows = []
    for name, base, model, token in arms:
        plain, e1 = ask(base, model, token, False, a.timeout)
        tools, e2 = ask(base, model, token, True, a.timeout)
        rows.append((name, plain, tools, e1 or e2))

    if a.append:
        with open(a.append, "a") as fh:
            for name, plain, tools, _ in rows:
                if plain and tools:
                    fh.write(json.dumps(
                        {"arm": name, "plain": plain, "tools": tools}) + "\n")

    return report(rows)


def report(rows):
    print(f"\n{'arm':<10} {'plain':>7} {'w/ tools':>9} {'tool delta':>11}  note")
    for name, plain, tools, err in rows:
        d = (tools - plain) if (plain and tools) else None
        print(f"{name:<10} {str(plain or '-'):>7} {str(tools or '-'):>9} "
              f"{str(d if d is not None else '-'):>11}  {err}")

    live = [r for r in rows if r[1] and r[2]]
    if len(live) < 2:
        print("\nfewer than two arms answered; nothing to compare", file=sys.stderr)
        return 2

    # An arm that renders no tool schema is broken in a way equal token counts
    # would otherwise hide.
    dead_tools = [r[0] for r in live if r[2] == r[1]]
    if dead_tools:
        print(f"\nFAIL: {', '.join(dead_tools)} rendered the tool schema as ZERO tokens "
              "-- the model never sees the tools")
        return 1

    # The PLAIN render is the strict check, and it is the one that caught the
    # thinking divergence: same messages, same template, so any difference is a
    # difference in how the arm framed the conversation.
    plains = {r[1] for r in live}
    toolset = {r[2] for r in live}
    if len(plains) != 1:
        print("\nFAIL: the arms do NOT render the same conversation.")
        print("  Every cross-engine number below this point compares different questions.")
        print(f"  plain prompt_tokens: {sorted(plains)}")
        return 1

    print(f"\nPASS (conversation): all {len(live)} arms render the plain prompt to "
          f"{next(iter(plains))} tokens")

    # The TOOLS render is NOT expected to match across engines, and asserting
    # that it does would be wrong. mlx-lm, vLLM and llama.cpp all render the
    # checkpoint's own `chat_template.jinja`; pie authors its tool preamble in
    # Rust (`QwenInstruct::build_tool_system_prompt`). Same dialect, different
    # wording, so different token counts BY DESIGN.
    #
    # What must hold is weaker and still worth checking: every arm renders the
    # schema as something, and the spread stays small enough to be a wording
    # difference rather than a dropped schema or the wrong dialect entirely.
    if len(toolset) == 1:
        print(f"PASS (tools):        all arms render the tool schema to "
              f"{next(iter(toolset))} tokens")
    else:
        lo, hi = min(toolset), max(toolset)
        print(f"NOTE (tools):        {lo}-{hi} tokens, spread {hi - lo} "
              f"({(hi - lo) / lo * 100:.0f}%)")
        print("  Expected: pie authors its own tool preamble in Rust, the other three")
        print("  render the checkpoint's chat_template.jinja. Same dialect, different")
        print("  wording. A LARGE spread instead means a dropped schema or wrong dialect.")
        for name, plain, tools, _ in live:
            print(f"    {name:<10} {tools - plain:>5} tokens of tool preamble")
    return 0


if __name__ == "__main__":
    sys.exit(main())

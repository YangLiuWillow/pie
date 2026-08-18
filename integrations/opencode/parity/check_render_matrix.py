#!/usr/bin/env python3
"""Bit-identity: pie's rendered token ids vs the checkpoint's own chat template.

`check_render.py` diffs ONE model against five captured wire requests. This
diffs a MATRIX -- every checkpoint whose template actually differs, crossed
with both thinking modes, crossed with the message shapes an agent produces --
and reports a cell count per arm.

## Why a matrix, and why these axes

Five divergences between pie's hand-written renderer and Qwen3.6's
`chat_template.jinja` were found by reading the two side by side, one at a
time, over a day. Each was invisible to everything that existed:

  * the throughput benchmarks feed a fixed prompt and measure decode rate,
    so they never render a tools turn or replay an assistant turn;
  * `check_render.py` defaults to `Qwen/Qwen3-0.6B` -- a plain Qwen3, whose
    template is the one pie gets RIGHT -- with `enable_thinking=False`
    hardcoded on both sides;
  * the in-crate `*_matches_reference` tests compare against strings a human
    transcribed from the template, so they assert that the author agreed with
    themselves.

So the axes are chosen to be the ones that were unwatched:

  arms      the three templates that genuinely differ. Qwen3 writes JSON
            schemas and JSON calls; Qwen3-Coder writes XML schemas and XML
            calls; Qwen3.6 writes JSON schemas and XML calls, which is
            neither, and is why serving it as `Coder` was wrong.
  thinking  both modes. `generation_suffix: "<think>\\n"` -- Qwen3.6 opens the
            assistant turn INSIDE a reasoning block -- was missed entirely
            because no harness ever rendered the thinking-on cue.
  shapes    tool declarations, assistant replay, and multi-round agent loops.
            `empty_reasoning_header` and `system_before_tools` only show up
            once a conversation has a replayed assistant turn or a system
            message beside tools; a one-user-message fixture renders neither.

## What a failing cell means

A mismatch is pie asking the model a different question than the checkpoint's
own template asks. It does not show up in tok/s, and it does not show up as an
error -- it shows up as accuracy, attributed to the engine.

## The measured baseline, 2026-08-17

    arm            cells   what fails
    qwen3          24/26   `empty_system` only -- pie drops an empty system
                           message where the template renders an empty system
                           turn. Both thinking modes.
    qwen3_coder    12/26
    qwen3_5         5/26   every shape that declares tools or replays an
                           assistant turn

The qwen3_5 failures are the four `ChatMLConfig` fields upstream `dev-sslee`
has and this branch does not, and the harness quantifies each one:

    system_user_tools  no-think   351 vs 351   pure reordering: the template
                                               writes the tools block BEFORE
                                               the system content
                                               (`system_before_tools`)
    system_user_tools  thinking   347 vs 349   -2 tokens: the template opens
                                               the turn inside a reasoning
                                               block (`generation_suffix`)
    agent_loop_2       no-think   454 vs 462   -8 tokens: 2 replayed assistant
                                               turns x 4 tokens of
                                               `<think>\n\n</think>\n\n`
                                               (`empty_reasoning_header`)

Those were first found by reading the Rust against the Jinja by hand. This
reproduces all three by a different method and puts a number on each, which is
the point -- the next one will be found by running this instead.

## One deliberate divergence the harness accounts for

`tool_schema_envelopes` NAME-SORTS the tool list, so a client that reorders its
tools between turns still hashes to the same KV snapshot address -- the same
strings feed the prompt and the address. opencode already name-sorts on the
wire, so in production the two agree. The reference side sorts too; comparing
against an unsorted list would test a condition that never occurs and bury 37
real cells under one design decision.

Usage:
    check_render_matrix.py --bin target/debug/render-tokens
    check_render_matrix.py --bin ... --arm qwen3_5 --shape agent_loop_2 -v
"""
import argparse
import copy
import glob
import json
import os
import pathlib
import subprocess
import sys
import tempfile

HUB = os.path.expanduser("~/.cache/huggingface/hub")

# (arm, HF cache dir, the deployment name pie's registry keys the dialect off)
ARMS = [
    ("qwen3", "models--mlx-community--Qwen3-8B-4bit",
     "mlx-community--Qwen3-8B-4bit"),
    ("qwen3_coder", "models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit",
     "mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit"),
    ("qwen3_5", "models--mlx-community--Qwen3.6-35B-A3B-4bit",
     "mlx-community--Qwen3.6-35B-A3B-4bit"),
]

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the working tree.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "Path"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"},
                                      "timeout": {"type": "integer"}},
                       "required": ["command"]}}},
]

SYS = "You are a terse coding agent."


def call(cid, name, args):
    """An assistant tool call in OPENAI WIRE form: arguments is a JSON string."""
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def result(cid, text):
    return {"role": "tool", "tool_call_id": cid, "content": text}


R1 = call("c1", "read_file", {"path": "README.md"})
R2 = call("c2", "bash", {"command": "ls -la", "timeout": 30})

# 13 shapes. Names kept close to the ones upstream's own run reported so the
# two are comparable.
SHAPES = {
    "user_only": ([{"role": "user", "content": "What does this repo do?"}], None),
    "system_user": ([{"role": "system", "content": SYS},
                     {"role": "user", "content": "What does this repo do?"}], None),
    "empty_system": ([{"role": "system", "content": ""},
                      {"role": "user", "content": "hi"}], None),
    "multiturn_chat": ([{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "Hello."},
                        {"role": "user", "content": "and now?"}], None),
    "tools_no_system": ([{"role": "user", "content": "Read README.md"}], TOOLS),
    "system_user_tools": ([{"role": "system", "content": SYS},
                           {"role": "user", "content": "Read README.md"}], TOOLS),
    "tool_call_one": ([{"role": "system", "content": SYS},
                       {"role": "user", "content": "Read README.md"},
                       {"role": "assistant", "content": "", "tool_calls": [R1]},
                       result("c1", "# Title\nsome text")], TOOLS),
    "tool_call_two_results": ([{"role": "system", "content": SYS},
                               {"role": "user", "content": "Read and list"},
                               {"role": "assistant", "content": "",
                                "tool_calls": [R1, R2]},
                               result("c1", "# Title"),
                               result("c2", "a\nb")], TOOLS),
    "content_and_call": ([{"role": "system", "content": SYS},
                          {"role": "user", "content": "Read README.md"},
                          {"role": "assistant", "content": "I'll read it.",
                           "tool_calls": [R1]},
                          result("c1", "# Title")], TOOLS),
    "assistant_no_tools": ([{"role": "system", "content": SYS},
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "Hello."},
                            {"role": "user", "content": "bye"}], None),
    "tool_result_then_user": ([{"role": "system", "content": SYS},
                               {"role": "user", "content": "Read README.md"},
                               {"role": "assistant", "content": "", "tool_calls": [R1]},
                               result("c1", "# Title"),
                               {"role": "user", "content": "now summarise it"}], TOOLS),
    "agent_loop_2": ([{"role": "system", "content": SYS},
                      {"role": "user", "content": "Investigate"},
                      {"role": "assistant", "content": "Reading.", "tool_calls": [R1]},
                      result("c1", "# Title"),
                      {"role": "assistant", "content": "Listing.", "tool_calls": [R2]},
                      result("c2", "a\nb")], TOOLS),
    "agent_loop_3": ([{"role": "system", "content": SYS},
                      {"role": "user", "content": "Investigate"},
                      {"role": "assistant", "content": "Reading.", "tool_calls": [R1]},
                      result("c1", "# Title"),
                      {"role": "assistant", "content": "", "tool_calls": [R2]},
                      result("c2", "a\nb"),
                      {"role": "assistant", "content": "One more.", "tool_calls": [R1]},
                      result("c1", "# Title again")], TOOLS),
}


def hf_tools(tools):
    """Name-sorted, because pie sorts and the sort is load-bearing.

    `tool_schema_envelopes` orders tools by function name so a client that
    reorders its list between turns still hashes to the same KV snapshot
    address -- the same strings feed the prompt AND the address. opencode
    already name-sorts on the wire, so in production the two agree; handing
    the reference an unsorted list would test a condition that never occurs
    and bury the real divergences under one known design decision.
    """
    if not tools:
        return None
    return sorted(tools, key=lambda t: t["function"]["name"])


def hf_messages(messages):
    """The wire form a Jinja template expects.

    OpenAI carries `arguments` as a JSON STRING; every Qwen template treats it
    as a mapping (`tool_call.arguments|items`, `|tojson`). A real server parses
    it before templating, so the harness does too -- feeding the string through
    would compare pie against a template failure, not against the template.
    """
    out = copy.deepcopy(messages)
    for m in out:
        for c in m.get("tool_calls") or []:
            a = c["function"].get("arguments")
            if isinstance(a, str):
                c["function"]["arguments"] = json.loads(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--shape", action="append", default=[])
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the first divergence for each failing cell")
    a = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("needs `transformers` (tokenizer only, no torch)", file=sys.stderr)
        return 2

    arms = [x for x in ARMS if not a.arm or x[0] in a.arm]
    shapes = {k: v for k, v in SHAPES.items() if not a.shape or k in a.shape}
    rows, total_ok, total = [], 0, 0

    for arm, cache_dir, deploy_name in arms:
        snaps = glob.glob(f"{HUB}/{cache_dir}/snapshots/*/")
        if not snaps:
            rows.append((arm, None, f"not in the HF cache ({cache_dir})"))
            continue
        snap = pathlib.Path(snaps[0])
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        tokenizer_json = snap / "tokenizer.json"

        ok = 0
        cells = len(shapes) * 2
        for name, (messages, tools) in shapes.items():
            for thinking in (False, True):
                total += 1
                hf_text = tok.apply_chat_template(
                    hf_messages(messages), tools=hf_tools(tools),
                    add_generation_prompt=True, enable_thinking=thinking,
                    tokenize=False)
                hf_ids = tok(hf_text, add_special_tokens=False)["input_ids"]

                body = {"model": deploy_name, "messages": messages}
                if tools:
                    body["tools"] = tools
                with tempfile.NamedTemporaryFile("w", suffix=".json",
                                                 delete=False) as fh:
                    json.dump(body, fh)
                    req = fh.name
                env = dict(os.environ,
                           PARITY_MODEL_NAME=deploy_name,
                           PARITY_THINKING="1" if thinking else "0")
                proc = subprocess.run([a.bin, str(tokenizer_json), req],
                                      capture_output=True, text=True, env=env)
                os.unlink(req)
                if proc.returncode != 0:
                    if a.verbose:
                        print(f"  [{arm}/{name}/think={thinking}] BIN ERROR\n"
                              f"{proc.stderr[:400]}")
                    continue
                pie_ids = json.loads(proc.stdout)
                if pie_ids == hf_ids:
                    ok += 1
                    total_ok += 1
                elif a.verbose:
                    n = min(len(pie_ids), len(hf_ids))
                    i = next((k for k in range(n) if pie_ids[k] != hf_ids[k]), n)
                    # Decode the ids rather than reading `proc.stderr`: stderr
                    # carries render-tokens' own diagnostic line, which lands
                    # ahead of the prompt and makes every char-level view look
                    # like a divergence at offset 0.
                    pie_text = tok.decode(pie_ids, skip_special_tokens=False)
                    p = 0
                    lim = min(len(pie_text), len(hf_text))
                    while p < lim and pie_text[p] == hf_text[p]:
                        p += 1
                    print(f"  [{arm}/{name}/think={thinking}] "
                          f"pie {len(pie_ids)} vs hf {len(hf_ids)} tokens, "
                          f"first diff @{i}")
                    print(f"      shared tail: {pie_text[max(0,p-70):p]!r}")
                    print(f"      pie next   : {pie_text[p:p+90]!r}")
                    print(f"      hf  next   : {hf_text[p:p+90]!r}")
        rows.append((arm, (ok, cells), None))

    print(f"\n{'arm':<14} {'cells':>9}   status")
    for arm, score, err in rows:
        if err:
            print(f"{arm:<14} {'-':>9}   {err}")
        else:
            ok, cells = score
            mark = "OK" if ok == cells else f"{cells - ok} MISMATCH"
            print(f"{arm:<14} {f'{ok}/{cells}':>9}   {mark}")
    print(f"\n{total_ok}/{total} cells token-exact "
          f"({len(shapes)} shapes x 2 thinking modes x {len(arms)} arms)")
    if total_ok != total:
        print("\nA mismatch is pie asking a different question than the "
              "checkpoint's own\ntemplate asks. Re-run with -v for the first "
              "divergence in each cell.")
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())

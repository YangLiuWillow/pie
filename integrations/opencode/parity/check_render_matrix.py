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

## The measured baseline

    arm                     2026-08-17    now
    qwen3                     24/26      30/30
    qwen3_5                    5/26      30/30
    qwen3_coder_upstream          -      30/30
    qwen3_5_upstream              -      30/30
    qwen3_coder               12/26      10/30   <- deliberate, see below
                                        120/120 graded

## "Which template" is a real question, and the answer differs per model

The `_upstream` arms render against QWEN'S OWN repos rather than the
mlx-community conversions pie actually serves. For Qwen3.6 that is the same
file -- byte-identical, 7764 bytes -- and both tokenizers encode identically,
so `qwen3_5_upstream` passing 30/30 says the renderer matches the model
author's template, not merely a converter's copy of it.

Qwen3-Coder is NOT the same file. mlx-community ships 6722 bytes where Qwen
currently publishes 6211, and the difference is not cosmetic: Qwen's revision
opens the tools turn with a `# Tools\n\n` heading that the mlx checkpoint's
does not. Every shape declaring tools diverges by exactly those three tokens;
every shape without tools passes. Hence 10/30.

pie tracks QWEN's file. A checkpoint redistributed with a patched or older
copy does not get to become the reference, so `qwen3_coder_upstream` is the
graded arm and the mlx conversion's is carried as an expected divergence --
visible, sized, and not failing the run.

Matching Qwen took two changes, and only the first was obvious: the `# Tools`
heading, and then `render_extra_keys`. Qwen has no special handling for
`enum` or `required` at all -- every unhandled schema key goes through one
generic passthrough, JSON for containers and plain text otherwise, with the key
used RAW as the tag. The mlx conversion added a `render_item_list` macro that
writes `[`tz`]` where Qwen writes `["tz"]`, and a `normed_json_key` that
rewrites tag names. Adding the heading alone left both Coder arms at 10/30;
only replacing the special-cased lists with the generic rule closed it.

The cost is real and stated rather than hidden: serving
`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, pie now renders three tokens
of heading and JSON-quoted lists that THAT checkpoint's template does not, so
mlx-lm and vLLM serving the same files render something different. Twenty of
its thirty cells. That is the trade the choice buys, in the direction chosen.

Every arm renders byte-for-byte what its own `chat_template.jinja` renders, in
both thinking modes, across all thirteen shapes. What closed the gap, in the
order it was closed and with the cells each was worth:

    system_before_tools        +2   tools block leads the system content on 3.5/3.6
    empty_reasoning_header     +5   post-query assistant turns carry <think></think>
    generation_suffix         +12   3.5/3.6 opens the turn INSIDE a reasoning block
    empty system turn          +4   a system message of "" is a turn, not an absence
    tool_response framing     +12   Coder puts the newline after the block, not before
    is_last                    +2   Qwen3 writes the block for a post-query turn
                                    only when it is LAST or carries reasoning

The last two were found by this harness rather than by reading, and neither
could have come from upstream: `dev-sslee` has no Qwen3-Coder row at all, so
its arm never rendered a Coder tool response, and its own measurement reports
qwen3_5 at 26/26 through a registry the live server does not reach.

The flag reaches the server now: `chat.assistant` and `chat.assistant-call`
both carry `after-query` and `is-last`, so the serving path renders what this
measures. It did not until task #28 landed, and a green matrix said nothing
about the server until then -- `render-tokens` links the renderer directly.

## What 90/90 does NOT say

The cells are the ones written here, and a shape nobody wrote is a shape
nobody checks -- `reasoning_mid_loop` passed the moment it was added, but the
branch it covers was unmeasured until then, and `plain_assistant_after_query`
FAILED when added and cost a `is_last` fact to fix. Known-unmeasured today:

  * multimodal turns. Qwen3.6 carries a vision tower and its template has
    image/video content parts; every shape here is text.
  * `preserve_thinking`, the template's other route to a reasoning header.
  * non-ASCII content. `python_json` mirrors `json.dumps`, which escapes
    non-ASCII by default; that path is noted as unhandled until parity says
    otherwise, and no shape here has a non-ASCII character to say it with.
  * a user turn whose content is itself wrapped in `<tool_response>`, which is
    the `multi_step_tool` guard the last-query walk carries.
  * conversations long enough to cross whatever the serving layer truncates at.

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
    # EXPECTED to diverge. pie tracks Qwen's published Coder template, and the
    # mlx conversion ships a different one -- see `qwen3_coder_upstream`. The
    # arm stays so the size and shape of that divergence is visible rather than
    # forgotten, but it does not fail the run.
    ("qwen3_coder", "models--mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit",
     "mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit", "expected"),
    ("qwen3_5", "models--mlx-community--Qwen3.6-35B-A3B-4bit",
     "mlx-community--Qwen3.6-35B-A3B-4bit"),
    # Qwen's OWN repos, because the served checkpoint's template is not always
    # the model author's. mlx-community ships a Coder template that differs
    # from Qwen's currently-published one (6722 bytes against 6211: their tool
    # schema renderer uses `param_fields`/`normed_json_key`/`</return>` where
    # Qwen's uses `json_dict`/`json_key`). Qwen3.6's is byte-identical between
    # the two, and both tokenizers encode identically, so only Coder is really
    # a second reference -- but the arm is here for both, because "which
    # template" is a question this harness should answer rather than assume.
    ("qwen3_coder_upstream", "models--Qwen--Qwen3-Coder-30B-A3B-Instruct",
     "Qwen--Qwen3-Coder-30B-A3B-Instruct"),
    ("qwen3_5_upstream", "models--Qwen--Qwen3.6-35B-A3B",
     "Qwen--Qwen3.6-35B-A3B"),
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
    # A post-query assistant turn with NO tool calls. Rare in an agent loop --
    # a text-only reply usually ends the turn and a user message follows it --
    # and it is exactly the case a `reasoning-header` flag carried only on the
    # tool-call path would miss. Upstream's `assistant-call` has that gap; this
    # cell is here so ours cannot acquire it silently.
    "plain_assistant_after_query": ([{"role": "system", "content": SYS},
                                     {"role": "user", "content": "Investigate"},
                                     {"role": "assistant", "content": "", "tool_calls": [R1]},
                                     result("c1", "# Title"),
                                     {"role": "assistant", "content": "Found it."}], TOOLS),
    # A replayed turn that actually CARRIES reasoning, mid-loop so it is not
    # last. Both templates branch on `reasoning_content` being non-empty and
    # every other shape here leaves it empty, so this is the branch that was
    # measured only in its empty form.
    "reasoning_mid_loop": ([{"role": "system", "content": SYS},
                            {"role": "user", "content": "Investigate"},
                            {"role": "assistant",
                             "content": "<think>\nCheck the README first.\n</think>\n\nReading.",
                             "tool_calls": [R1]},
                            result("c1", "# Title"),
                            {"role": "assistant", "content": "", "tool_calls": [R2]},
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

    for arm, cache_dir, deploy_name, *rest in arms:
        expected = bool(rest and rest[0] == "expected")
        snaps = glob.glob(f"{HUB}/{cache_dir}/snapshots/*/")
        if not snaps:
            rows.append((arm, None, f"not in the HF cache ({cache_dir})", expected))
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

                # The mode travels in the REQUEST, the same way a real client
                # sends it and the same way `plan_render` reads it. It was an
                # env var while `cue`/`cue-no-think` were two WIT functions.
                body = {"model": deploy_name, "messages": messages,
                        "chat_template_kwargs": {"enable_thinking": thinking}}
                if tools:
                    body["tools"] = tools
                with tempfile.NamedTemporaryFile("w", suffix=".json",
                                                 delete=False) as fh:
                    json.dump(body, fh)
                    req = fh.name
                env = dict(os.environ, PARITY_MODEL_NAME=deploy_name)
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
        rows.append((arm, (ok, cells), None, expected))

    print(f"\n{'arm':<22} {'cells':>9}   status")
    graded = 0
    graded_ok = 0
    for arm, score, err, expected in rows:
        if err:
            print(f"{arm:<22} {'-':>9}   {err}")
            continue
        ok, cells = score
        if expected:
            mark = "OK" if ok == cells else f"{cells - ok} EXPECTED DIVERGENCE"
        else:
            mark = "OK" if ok == cells else f"{cells - ok} MISMATCH"
            graded += cells
            graded_ok += ok
        print(f"{arm:<22} {f'{ok}/{cells}':>9}   {mark}")
    print(f"\n{graded_ok}/{graded} graded cells token-exact "
          f"({len(shapes)} shapes x 2 thinking modes), "
          f"{total_ok}/{total} including expected divergences")
    total_ok, total = graded_ok, graded
    if total_ok != total:
        print("\nA mismatch is pie asking a different question than the "
              "checkpoint's own\ntemplate asks. Re-run with -v for the first "
              "divergence in each cell.")
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())

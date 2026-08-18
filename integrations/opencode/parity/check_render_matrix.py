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
    qwen3_coder               12/26      30/30
    qwen3_5                    5/26      30/30
    qwen3_coder_upstream          -      30/30
    qwen3_5_upstream              -      30/30
                                        150/150

## "Which template" is a real question, and it has two right answers

The `_upstream` arms render against QWEN'S OWN repos rather than the
redistributed conversions pie serves. For Qwen3.6 that is the same file --
byte-identical at 7764 bytes -- and both tokenizers encode identically, so
`qwen3_5_upstream` passing says the renderer matches the model author, not
merely a converter's copy.

Qwen3-Coder is two different files, both live today:

    MlxGguf   mlx-community/…-4bit  6722 bytes  sha 672e747c…
              unsloth/…-GGUF        same variant, embedded in the GGUF
              no `# Tools` heading; `render_item_list` writes [`a`]

    QwenMain  Qwen/Qwen3-Coder-…    6211 bytes  sha 5a38bfa0…
              `# Tools` heading; generic `render_extra_keys` writes ["a"]

pie renders both -- `CoderSchema`, defaulting to `MlxGguf` because that is what
every redistributed checkpoint carries and because a three-way benchmark where
one engine sends a different prompt is not measuring engines. The arms differ
only in which variant they ask for, so 150/150 is the claim that BOTH are
exact, not that one was chosen.

Getting there needed two changes, and only the first was visible in a diff: the
heading, and then the list rendering. Setting the heading alone left both Coder
arms at 10/30 -- it moved the divergence rather than removing it.

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
    # Qwen's OWN repos, because the served checkpoint's template is not always
    # the model author's. mlx-community ships a Coder template that differs
    # from Qwen's currently-published one (6722 bytes against 6211: their tool
    # schema renderer uses `param_fields`/`normed_json_key`/`</return>` where
    # Qwen's uses `json_dict`/`json_key`). Qwen3.6's is byte-identical between
    # the two, and both tokenizers encode identically, so only Coder is really
    # a second reference -- but the arm is here for both, because "which
    # template" is a question this harness should answer rather than assume.
    # Same renderer, the other variant: pie supports both, so the harness
    # proves both rather than picking a winner.
    ("qwen3_coder_upstream", "models--Qwen--Qwen3-Coder-30B-A3B-Instruct",
     "Qwen--Qwen3-Coder-30B-A3B-Instruct", "qwen"),
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
    # `agent_loop_3` was here and is gone: its catch profile was IDENTICAL to
    # `agent_loop_2` across every mutation, so a third round detected nothing a
    # second did not. Run `--mutation-matrix` before adding a shape, and again
    # before keeping one.
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
}


def render_batch(binary, tokenizer_json, bodies, env):
    """All of an arm's cells in ONE invocation.

    `render-tokens` parses the tokenizer once per process, and an 11-20 MB
    `tokenizer.json` costs ~0.7 s. Spawning per cell spends that 140 times a
    run; batching spends it five times, once per arm.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(bodies, fh)
        path = fh.name
    try:
        proc = subprocess.run([binary, str(tokenizer_json), path],
                              capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            return None, proc.stderr
        return json.loads(proc.stdout), None
    finally:
        os.unlink(path)


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


# Every renderer fact the harness can deliberately break. Keep in step with
# `PARITY_MUTATE` in render-tokens: a fact with no mutation here is a fact the
# suite cannot prove it checks.
MUTATIONS = [
    "system_before_tools", "empty_reasoning_header", "generation_suffix",
    "tool_response_trailing_newline", "coder_schema", "thinking_off_suffix",
    "drop_empty_system", "ignore_is_last", "header_on_pre_query",
]


def mutation_matrix(a, arms, shapes, mutations, AutoTokenizer):
    """Which shapes catch which deliberate mistake.

    The point is not coverage for its own sake. A shape that catches nothing
    another shape does not catch is REDUNDANT and should go -- `agent_loop_3`
    went that way, its profile identical to `agent_loop_2`. A mutation no shape
    catches is a HOLE: the suite cannot tell that fact is wrong.

    Tokenizers load once here. Sweeping by invoking the whole harness per
    (mutation, shape) reloads them 126 times and takes longer than the sweep
    deserves.
    """
    loaded = []
    for arm, cache_dir, deploy_name, *rest in arms:
        snaps = glob.glob(f"{HUB}/{cache_dir}/snapshots/*/")
        if not snaps:
            continue
        snap = pathlib.Path(snaps[0])
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        # The reference does not depend on the mutation, so render it ONCE for
        # the whole sweep instead of nine times.
        refs = {}
        for name, (messages, tools) in shapes.items():
            for thinking in (False, True):
                txt = tok.apply_chat_template(
                    hf_messages(messages), tools=hf_tools(tools),
                    add_generation_prompt=True, enable_thinking=thinking,
                    tokenize=False)
                refs[(name, thinking)] = tok(txt, add_special_tokens=False)["input_ids"]
        loaded.append((
            arm, deploy_name, tok, snap / "tokenizer.json",
            "qwen" if (rest and rest[0] == "qwen") else "shipped", refs,
        ))

    names = list(shapes)
    print(f"\n{'':<32}" + "".join(f"{i + 1:>3}" for i in range(len(names))))
    for i, n in enumerate(names):
        print(f"  {i + 1:>2}  {n}")
    print()

    caught = {m: set() for m in mutations}
    for m in mutations:
        # ONE invocation per (mutation, arm) covering every shape, not one per
        # (mutation, shape, arm). Same batching argument as the main run: with
        # 9 mutations that is 45 tokenizer parses instead of 630.
        for arm, deploy, tok, tokenizer_json, cs, refs in loaded:
            order = [(n, t) for n in names for t in (False, True)]
            bodies = []
            for name, thinking in order:
                messages, tools = shapes[name]
                b = {"model": deploy, "messages": messages,
                     "chat_template_kwargs": {"enable_thinking": thinking}}
                if tools:
                    b["tools"] = tools
                bodies.append(b)
            env = dict(os.environ, PARITY_MODEL_NAME=deploy,
                       PARITY_CODER_SCHEMA=cs, PARITY_MUTATE=m)
            got, _ = render_batch(a.bin, tokenizer_json, bodies, env)
            if got is None:
                caught[m].update(names)
                continue
            for (name, thinking), ids in zip(order, got):
                if ids != refs[(name, thinking)]:
                    caught[m].add(name)

    for m in mutations:
        row = "".join("  X" if n in caught[m] else "  ." for n in names)
        print(f"{m:<32}{row}")

    print()
    holes = [m for m in mutations if not caught[m]]
    if holes:
        print("HOLES -- no shape detects these, the suite cannot see them wrong:")
        for m in holes:
            print(f"  {m}")
    # Which shapes would a MINIMAL suite keep? Greedy set cover: repeatedly
    # take the shape catching the most still-uncaught mutations.
    #
    # "Is this subsumed by some other shape" is the wrong question and gives a
    # useless answer -- four shapes that all catch the same four mutations each
    # subsume the others, so all four get flagged when you need to keep one.
    per_shape = {n: {m for m in mutations if n in caught[m]} for n in names}
    need = {m for m in mutations if caught[m]}
    keep = []
    while need:
        best = max(names, key=lambda n: len(per_shape[n] & need))
        if not (per_shape[best] & need):
            break
        keep.append(best)
        need -= per_shape[best]
    extra = [n for n in names if n not in keep]
    print(f"A MINIMAL suite for these mutations is {len(keep)} shape(s): "
          f"{', '.join(keep)}")
    if extra:
        print("The rest catch nothing the minimal set misses BY THIS MUTATION")
        print("SET -- which is not the same as useless. Read each shape's own")
        print("comment before deleting: a shape may guard a bug no mutation")
        print("here expresses, and the mutation is then the thing to add.")
        for n in extra:
            print(f"    {n:<30} {sorted(per_shape[n]) or 'catches nothing'}")
    return 0 if not holes else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--shape", action="append", default=[])
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the first divergence for each failing cell")
    ap.add_argument("--mutation-matrix", action="store_true",
                    help="sweep every renderer mutation and report which shapes "
                         "catch it; a shape catching nothing another does not is "
                         "redundant, a mutation nothing catches is a hole")
    a = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("needs `transformers` (tokenizer only, no torch)", file=sys.stderr)
        return 2

    arms = [x for x in ARMS if not a.arm or x[0] in a.arm]
    shapes = {k: v for k, v in SHAPES.items() if not a.shape or k in a.shape}

    if a.mutation_matrix:
        return mutation_matrix(a, arms, shapes, MUTATIONS, AutoTokenizer)

    rows, total_ok, total = [], 0, 0

    for arm, cache_dir, deploy_name, *rest in arms:
        expected = bool(rest and rest[0] == "expected")
        coder_schema = rest[0] if rest and rest[0] == "qwen" else "shipped"
        snaps = glob.glob(f"{HUB}/{cache_dir}/snapshots/*/")
        if not snaps:
            rows.append((arm, None, f"not in the HF cache ({cache_dir})", expected))
            continue
        snap = pathlib.Path(snaps[0])
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        tokenizer_json = snap / "tokenizer.json"

        ok = 0
        cells = len(shapes) * 2
        # One invocation for the whole arm; see `render_batch`.
        order = [(n, t) for n in shapes for t in (False, True)]
        bodies = []
        for name, thinking in order:
            messages, tools = shapes[name]
            b = {"model": deploy_name, "messages": messages,
                 "chat_template_kwargs": {"enable_thinking": thinking}}
            if tools:
                b["tools"] = tools
            bodies.append(b)
        env = dict(os.environ, PARITY_MODEL_NAME=deploy_name,
                   PARITY_CODER_SCHEMA=coder_schema)
        batched, batch_err = render_batch(a.bin, tokenizer_json, bodies, env)
        by_cell = dict(zip(order, batched)) if batched else {}

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
                pie_ids = by_cell.get((name, thinking))
                if pie_ids is None:
                    if a.verbose:
                        print(f"  [{arm}/{name}/think={thinking}] BIN ERROR\n"
                              f"{(batch_err or '')[:400]}")
                    continue
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

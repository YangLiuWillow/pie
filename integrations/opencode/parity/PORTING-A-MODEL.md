# Porting a model to pie: making the renderer provably right

pie does not execute a checkpoint's `chat_template.jinja`. It reimplements it in
Rust, and the reimplementation is what serves every request. So the question
"did I port this correctly" has a precise answer, and CI can hold you to it.

## The five-minute version

1. Add a row to `model/src/instruct.rs::create` for the new `arch_name`.
2. Add an arm to `ARMS` in `check_render_matrix.py`.
3. `--emit-fixtures`, look at the diff, commit it.
4. `cargo test -p render-tokens --test render_parity` — and so does CI.

## What the arm needs

    ("arm_name", "models--<hf-cache-dir>", "<deployment-name>", "<arch-stem>")

**The arch stem is the part people get wrong.** The engine passes
`instruct::create` the *driver's* stem: `architectures[0]`, lowercased, with
the task suffix removed — `arch_stem` in `driver/metal/src/model_facts.cpp` and
its twin in `worker/src/embedded_driver.rs`. CamelCase boundaries carry no
separator through that:

    Qwen3_5MoeForConditionalGeneration  ->  qwen3_5moe     NOT qwen3_5_moe
    Qwen3MoeForCausalLM                 ->  qwen3moe       NOT qwen3_moe

A registry keyed only on the HF `model_type` spelling misses, falls to the `_`
arm, and gets `has_tools: false` — every tool schema silently dropped, no error
anywhere. That has happened here, and upstream's registry still has it.

Get the stem from the checkpoint:

    python -c "import json;a=json.load(open('config.json'))['architectures'][0];\
    s=a.lower();print([s[:-len(x)] for x in ('forconditionalgeneration','forcausallm') \
    if s.endswith(x)] or [s])"

## Why the test builds through `instruct::create`

`render_parity.rs` calls the registry, not a copy of the config. An earlier
version reconstructed `ChatMLConfig` field by field and would have passed
against a row nobody serves. If your new model needs a config the registry
cannot express, that is a signal about the registry — not a reason to special-
case the test. When pie genuinely cannot know a fact -- which of
Qwen3-Coder's two published templates a checkpoint ships is written only in the
`chat_template.jinja` the artifact does not import -- it becomes an entry in
`InstructOverrides`, an operator's answer, and `create_with` applies it. The
test passes the same override the fixture records, so every arm still renders
through the registry.

## What "correct" means, and how to find out when it is not

    cargo test -p render-tokens --test render_parity   112 cells, ~0.5 s
    python check_render_matrix.py --bin … -v           token-level first diff

The Python harness is the one to reach for when a cell fails: it prints the
shared prefix and the first divergent text on both sides.

Facts a Qwen-family template can disagree on, all of them found the hard way:

| field | what differs |
|---|---|
| `tool_dialect` | JSON schemas + JSON calls / XML + XML / JSON + XML |
| `system_before_tools` | does the caller's system message lead the tools turn? |
| `empty_reasoning_header` | `<think></think>` on a post-query turn with no reasoning? |
| `generation_suffix` | does the cue open INSIDE a reasoning block? |
| `thinking_off_suffix` | what the cue emits with thinking off |
| `tool_response_trailing_newline` | newline before or after `<tool_response>` |
| `coder_schema` | which Qwen3-Coder template revision |

## Before adding or keeping a shape

    python check_render_matrix.py --bin … --mutation-matrix    ~27 s

It breaks one renderer fact at a time and reports which shapes catch it. A
shape catching nothing another catches is redundant — `agent_loop_3` was
removed that way. A mutation nothing catches is a hole: the suite cannot tell
that fact is wrong, and the fix is a new shape.

Add a mutation whenever you add a config field. A field with no mutation is a
field the suite cannot prove it checks.

## When the fixtures change

A fixture diff means a checkpoint's template moved. That is news — read it
before committing. Qwen revised the Coder template after mlx-community and
unsloth had snapshotted it, and pie found out days later by hand. The weekly
`render-drift` job exists to make that a scheduled signal instead.

Never hand-edit ids to make the test pass.

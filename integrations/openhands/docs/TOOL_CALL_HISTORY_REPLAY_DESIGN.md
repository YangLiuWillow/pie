# Design question: replaying tool-call history in the native-tool-calling rewrite

**Status:** decided — going with **Option B** (extend `Instruct`/runtime, not
inferlet-side hand-formatting). See "Decision" at the end. Recorded before starting
the native-tool-calling rewrite of `openhands-completion` (see chat log /
`docs/openhands-integration.md` for the broader Phase 1 → native-tool-calling
motivation: `pie_openhands/llm.py` currently forces `native_tool_calling=False`, which
routes every tool call through OpenHands SDK's prompt-mocked, regex-parsed non-native
path — the mechanism behind the escape-bug and part of the stuck-loop failures found
in the 2026-07-05 GPU sessions).

This doc scopes *one* sub-problem of that rewrite: how the inferlet reconstructs
multi-turn conversation history — specifically **past assistant tool calls and past
tool-result turns** — given Pie's inferlet has no persistent KV state across requests.

## Why this is a real problem, not a formality

`inferlets/openhands-completion/src/lib.rs`'s own header comment says:

> Phase 1 does NOT pin KV state across requests — that's `openhands-coder-session`
> in Phase 2. We exit after every completion and the runtime releases the pages.

So every single request to the inferlet must rebuild the **entire** conversation from
scratch — every past user message, every past assistant text/tool-call turn, every
past tool result — not just the newest turn. For a plain-text chat this is trivial
(`ctx.system(...)`, `ctx.user(...)`, `ctx.assistant(...)` per past turn). For a
tool-calling agent loop it isn't, because replaying a past *tool call* isn't just
"replay this string" — it has to come back out byte-identical to what the model's
own chat template would have produced, or the model sees a shifted/foreign-looking
prompt on every follow-up turn.

## What Pie's runtime already provides

`runtime/src/model/instruct/qwen3.rs` implements `Instruct` for Qwen (both `qwen3` and,
via `qwen2.rs:76-82`, the `qwen2` arch that Qwen2.5-Coder uses — same ChatML/tool
format, no thinking support):

```rust
// qwen3.rs:290-339 (impl Instruct for QwenInstruct)
fn system(&self, msg: &str) -> Vec<u32>      // wraps msg in <|im_start|>system...<|im_end|>
fn user(&self, msg: &str) -> Vec<u32>        // same, role=user
fn assistant(&self, msg: &str) -> Vec<u32>   // same, role=assistant — PLAIN STRING ONLY
fn equip(&self, tools: &[String]) -> Vec<u32>        // builds the <tools>...</tools> system block
fn answer(&self, _name: &str, value: &str) -> Vec<u32>  // wraps value in a user-turn <tool_response>
```

Exposed to inferlets via the WIT `pie:instruct/tool-use` interface
(`runtime/wit/instruct/wit/tool-use.wit`) and the Rust SDK
(`sdk/rust/inferlet/src/tools.rs`: `equip_prefix`, `answer_prefix`, `native_grammar`,
`Decoder`/`parse_call`). There's a working single-turn reference loop already in the
repo at `inferlets/marketing-tab1-agent/src/lib.rs`: equip → generate → feed tokens to
`tools::Decoder` → on `Event::Call` execute the tool → `ctx.append(&tools::answer_prefix(...))`.

That reference loop is a **live agent loop within one running inferlet** — it never
needs to *replay* a past tool call, because the KV state is already there from the
turn that produced it. Our case is different: every request is a cold restart that
has to reconstruct N previous turns, some of which were tool calls.

## The actual gap

Reading `qwen3.rs` closely surfaces two concrete mismatches between what's available
and what a faithful history replay needs:

**1. `assistant(&self, msg: &str)` takes a plain string — there is no way to pass
structured `tool_calls`.** The reference Jinja template that `qwen2.rs`'s own
docstring embeds (`qwen2.rs:37-52`) formats a past assistant-with-tool-calls turn as:

```
<|im_start|>assistant
{content, if any}
<tool_call>
{"name": "...", "arguments": {...}}
</tool_call><|im_end|>
```

`Instruct::assistant()` has no parameter for `tool_calls` — it just wraps whatever
string you hand it. So replaying a past tool-calling assistant turn means **the
caller** (our inferlet code) has to hand-format that `<tool_call>...</tool_call>`
block into a string first, then pass the whole thing to `ctx.assistant(...)`. This
duplicates formatting logic that already exists correctly in `qwen2.rs`'s
`build_tool_system_prompt`/template, just not exposed for the "replay a completed
call" case (only for equip's *tool-schema announcement*, not for replaying the
*model's own past output*).

**2. `answer(&self, _name: &str, value: &str)` always opens a fresh user turn — it
doesn't merge consecutive tool results the way the reference template does.**
Two things stand out reading `qwen3.rs:327-339`:

- The `name` parameter is literally unused (`_name`) — Qwen's tool-response format
  doesn't include the tool name in the wrapper, only the value. Fine, matches the
  reference template.
- But the reference Jinja template (`qwen2.rs:53-63`) only opens a new
  `<|im_start|>user` block when the *previous* message wasn't also a tool result —
  consecutive tool results (e.g. two tool calls in one assistant turn, both
  answered) get merged into **one** user turn with multiple `<tool_response>` blocks
  inside it. `Instruct::answer()` has no such lookahead/lookbehind — calling it twice
  in a row (once per tool result) produces **two separate**
  `<|im_start|>user...<|im_end|>` blocks instead of one merged block. This is a
  real, byte-level divergence from the model's expected format, not just a style
  nit — Qwen was fine-tuned on the merged form.

**3. `Context::equip` (the ergonomic wrapper, `sdk/rust/inferlet/src/context.rs:320`)
requires `&[&dyn crate::tools::Tool]`** — compile-time Rust types implementing
`Tool::{name,description,schema}` (via the `#[tool]` macro, see `tool-test`/
`marketing-tab1-agent`). OpenHands supplies tool schemas as **runtime JSON** from
Python (whatever tools the agent is configured with that session) — there's no
Rust type to implement `Tool` for. We'd need the lower-level, dynamic-schema entry
point instead: `tools::equip_prefix(model, tool_schemas: &[String]) -> Result<Vec<u32>>`
(`tools.rs:62-64`) called directly and `ctx.append()`'d, not `ctx.equip()`. Not a
blocker, just a detail that changes which function the rewrite actually calls.

## Options

**Option A — hand-format each historical turn in the inferlet, in Rust.**
Walk the incoming `messages` list turn by turn; for a plain user/system/text-assistant
turn, call `ctx.user/system/assistant(content)` as normal. For an assistant turn with
`tool_calls`, hand-build the `<tool_call>{...}</tool_call>` string(s) (mirroring
`qwen2.rs`'s template) and pass the whole thing through `ctx.assistant(...)`. For a run
of consecutive `tool` role messages, merge them ourselves into one hand-built
`<tool_response>...</tool_response><tool_response>...</tool_response>` blob before
wrapping in a single user turn (i.e., don't call `answer()` per-message when there's
more than one consecutive tool result — build the merged string once and push it via
`ctx.user(...)` instead, bypassing `Instruct::answer()`'s one-block-per-call behavior).
Use the raw `tools::equip_prefix()` (not `Context::equip`) for the dynamic tool schemas.
*Cost:* duplicates a chunk of `qwen2.rs`'s template knowledge into the inferlet in
Rust; if Qwen's template changes or we add a second model family, both places need
updating. *Benefit:* no changes to Pie's runtime/SDK; fully contained in the one
inferlet we're already rewriting; ships fastest.

**Option B — extend `Instruct`/`Context` in Pie's runtime itself**, e.g. add
`assistant_with_tool_calls(&self, content: Option<&str>, calls: &[(String, String)]) -> Vec<u32>`
and fix `answer()` (or add a `answer_batch(&self, results: &[(String, String)])`) to
merge consecutive results into one block, matching the reference template exactly.
*Cost:* touches shared runtime code (`runtime/src/model/instruct/*`), needs updating
for every `Instruct` impl that supports tools (currently only `qwen2`/`qwen3`), needs
its own unit tests (existing pattern: `qwen3.rs`'s `#[cfg(test)]` block already has
`answer_format`/`tool_decoder_parses_call` tests to extend). *Benefit:* the formatting
logic lives in exactly one place (next to the template it must match byte-for-byte),
reusable by any future inferlet that needs to replay tool-calling history, not just
this one — and removes the risk of the Rust-side hand-formatting in Option A silently
drifting from the real template.

**Option C — push formatting to the Python side** (PieLLM pre-renders each historical
turn into the literal string blob the Jinja template would produce, inferlet just
tokenizes/appends). Rejected: this is exactly the "flatten in Python" approach the
rewrite is trying to move *away* from (see prior discussion — losing structure is what
led to the escape-bug/regex-fragility problems in the first place), and it duplicates
model-specific template knowledge in Python that already exists correctly in
`qwen2.rs`, in a language that can't share code with it.

## Recommendation

Option B is the more correct fix (one source of truth for the template, next to
existing tests it can extend), but it's the only option that touches shared runtime
code rather than being contained in the inferlet + PieLLM changes already scoped.
Given this is still a two-model surface (`qwen2`, `qwen3`) and the mismatch is
concrete and well-understood, Option B is not large — but it does mean the
native-tool-calling rewrite has a runtime-side PR in addition to the inferlet +
PieLLM PRs already scoped, not just the two.

**Decision:** Option B. The mismatch is already fully diagnosed and the surface area
(two `Instruct` impls, both in `qwen3.rs`) is small right now — better to put the
formatting logic in one place next to the template it must match, with test coverage,
than to duplicate it into the inferlet and risk drift.

### Concrete follow-up work for Option B

1. `runtime/src/model/instruct.rs`: extend the `Instruct` trait with something like
   `assistant_with_tool_calls(&self, content: Option<&str>, calls: &[(String, String)]) -> Vec<u32>`
   and a merging tool-result method, e.g. `answer_batch(&self, results: &[(String, String)]) -> Vec<u32>`
   (replacing the one-block-per-call behavior of today's `answer()` for runs of
   consecutive tool results). Default-`None`/empty-safe fallback for `Instruct` impls
   that don't support tools (mirrors the existing `tool_call_grammar` default at
   `instruct.rs:103-105`).
2. `runtime/src/model/instruct/qwen3.rs`: implement both for `QwenInstruct`, matching
   `qwen2.rs:37-63`'s reference Jinja template exactly (content + `<tool_call>` blocks
   per call in the assistant turn; merged `<tool_response>` blocks in one user turn
   for consecutive tool results).
3. Extend `qwen3.rs`'s existing `#[cfg(test)]` block (next to `answer_format`,
   `tool_decoder_parses_call`, `equip_format_matches_reference`) with tests for the
   new methods — especially a test asserting consecutive tool results merge into one
   user turn, since that's the concrete bug being fixed.
4. WIT / inferlet SDK: decide whether to expose these as new `pie:instruct/chat` or
   `pie:instruct/tool-use` WIT functions (`runtime/wit/instruct/wit/*.wit`) with
   corresponding wrappers in `sdk/rust/inferlet/src/tools.rs` / `context.rs`, or keep
   them runtime-internal and have the inferlet call through whatever the WIT surface
   ends up being — needs the same WIT → wit-bindgen → SDK wrapper plumbing as the
   existing `equip`/`answer` functions.
5. Only once 1–4 land: `inferlets/openhands-completion/src/lib.rs` can do history
   replay by calling these instead of hand-formatting strings.

## Pointers

- Inferlet to rewrite: `inferlets/openhands-completion/src/lib.rs`
- Reference single-turn agent loop (equip/decode/answer, no history replay needed):
  `inferlets/marketing-tab1-agent/src/lib.rs`
- Reference dynamic-schema tool-call smoke test: `inferlets/tool-test/src/lib.rs`
  (uses `Context::equip` with compile-time `#[tool]`-derived types — not directly
  reusable for our runtime-JSON-schema case, but shows the call/schema plumbing)
- `Instruct` trait + Qwen impl: `runtime/src/model/instruct.rs`,
  `runtime/src/model/instruct/qwen3.rs` (shared by qwen2 via `qwen2.rs:76-82`)
- Reference Jinja template (ground truth for exact byte formatting):
  `runtime/src/model/instruct/qwen2.rs:13-68` (embedded as a doc comment/string)
- WIT interface: `runtime/wit/instruct/wit/tool-use.wit`
- Inferlet SDK surface: `sdk/rust/inferlet/src/tools.rs`, `sdk/rust/inferlet/src/context.rs:267-325`
- Existing `Instruct` unit tests to extend if going with Option B:
  `runtime/src/model/instruct/qwen3.rs` `#[cfg(test)]` block (`answer_format`,
  `tool_decoder_parses_call`, `equip_format_matches_reference`, etc.)

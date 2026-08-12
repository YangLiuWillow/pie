# Renderer-parity harness (P0.4)

Verifies that pie's rendering of an OpenAI chat-completions request — the real
serving path: `pie_openai_serving::plan_render` → `QwenInstruct` (the exact
`ChatMLConfig` that `model/src/instruct.rs` binds for `arch_name = "qwen3"`)
→ `pie-tokenizer` — produces token ids matching HuggingFace
`tokenizer.apply_chat_template(messages, tools=…, add_generation_prompt=True,
enable_thinking=False)` for `Qwen/Qwen3-0.6B`, driven by the real opencode
wire captures in `tests/inferlets/fixtures/opencode/wire/`.

## Layout

- `render-tokens/` — Rust bin (root-workspace member). Loads a HF
  `tokenizer.json` via `pie_tokenizer::Tokenizer::from_file`, parses the
  request (raw body or wire-capture wrapper with a `.body` field), runs
  `plan_render`, maps `RenderOp`s 1:1 onto `Instruct` calls
  (`EquipAfterSystem`→`equip_after_system`, `User`→`user`,
  `Assistant`→`assistant`, `AssistantWithToolCalls`→`assistant_with_tool_calls`,
  `AnswerBatch`→`answer_batch`, `Cue`→`cue`), concatenates. stdout: token ids
  as a JSON array; stderr: the decoded prompt for eyeballing.
- `check_render.py` — driver. Downloads Qwen/Qwen3-0.6B **tokenizer files
  only** (never weights), renders each fixture on both sides, diffs the token
  sequences, and prints a first-divergence report (token index, id/piece both
  sides, ±5 tokens context, decoded windows) plus the **complete char-level
  divergence list**, grouped (`difflib.SequenceMatcher`, autojunk off).

## How to run

```sh
# 1. Build the bin (see the progress log's shared-target-dir note):
CARGO_TARGET_DIR=$HOME/Documents/Liszt_ai/pie/target cargo build -p render-tokens

# 2. Python side needs transformers (tokenizer-only, no torch) + huggingface_hub:
python3 -m venv /path/to/venv && /path/to/venv/bin/pip install transformers huggingface_hub

# 3. Run against all fixtures (add --fixture req-005.json for one,
#    --dump-dir DIR to keep both decoded prompts, --cache-dir DIR to
#    redirect the HF download cache):
/path/to/venv/bin/python integrations/opencode/parity/check_render.py \
    --bin $HOME/Documents/Liszt_ai/pie/target/debug/render-tokens
```

Exit code 0 iff every fixture is token-exact.

## Current parity status (2026-08-11, Qwen/Qwen3-0.6B, transformers 5.15.0)

No fixture is token-exact yet; the divergence list is **exactly three
systematic template differences** — nothing else in ~7.5k-token prompts:

| fixture | shape | divergences |
|---|---|---|
| req-001 / req-003 | title call: system + 2 users, no tools | D1 only (token-exact up to the generation cue) |
| req-002 / req-004 | system + user + 10 tools | D1, D2, D3 |
| req-005 | tool-history replay (assistant+tool_calls, tool) | D1, D2, D3 — the replayed `<tool_call>`/`<tool_response>` turns themselves are **exact** |

- **D1 — empty think block after the cue** (all 5 fixtures, 4 tokens). HF
  with `enable_thinking=False` ends `…<|im_start|>assistant\n<think>\n\n</think>\n\n`
  (ids `151667, 271, 151668, 271`); pie's `cue()` ends at
  `…<|im_start|>assistant\n`. Pie's planned no-think channel is `/no_think`
  user-turn decoration (H17), applied by the inferlet, not this path — and
  opencode never sends `chat_template_kwargs.enable_thinking` anyway. A PA.1
  decision, not a harness bug.
- **D2 — extra `\n` before `# Tools`** (tools fixtures, 1 char). pie renders
  `…</available_skills>\n\n` + `\n# Tools`, HF renders `…\n\n# Tools`.
  Cause: `build_tool_system_prompt` (model/qwen_3/src/chat.rs) starts with
  `"\n# Tools"` *and* `equip_after_system` merges `{content}\n\n{tools_block}`.
  HF: `content + '\n\n'` then `"# Tools…"` (and `<|im_start|>system\n# Tools…`
  when there is no system message — the leading `\n` is wrong in that case
  too).
- **D3 — compact vs spaced JSON separators in the `<tools>` entries**
  (tools fixtures, ×293 = every separator in the 10 schemas). pie:
  `{"type": "function", "function": {"name":"bash","description":…}}`
  (inner envelope from `tool_schema_envelopes`, compact
  `serde_json::to_string`); HF: `{"type": "function", "function": {"name":
  "bash", "description": …}}` (transformers' `tojson` = `json.dumps` with
  default separators `', '`/`': '`, `ensure_ascii=False`, wire key order).
  Verified: HF has 480 `", "`/`": "` occurrences inside the block, pie 187,
  Δ = 293. Note the envelope string also feeds the snapshot address
  (`session.rs`), so a fix must change both together (response/save
  unification invariant).

Confirmed non-divergences: role scaffolding, `<|im_end|>\n` turn suffixes and
the absence of a trailing newline after the last history turn; the
tool-call replay (`<tool_call>\n{"name": "read", "arguments": {…}}\n</tool_call>`
— arguments string passed through verbatim both sides); merged consecutive
`<tool_response>` turns; tool name-sort (opencode's wire order is already
sorted); the `$schema`/`maximum: 2^53−1` schema noise round-trip.

## Harness notes

- transformers 5.x changed `apply_chat_template(tokenize=True)`'s return
  shape; the driver renders text and tokenizes with
  `tok(text, add_special_tokens=False)` (what tokenize=True does internally).
- HF renders opencode's assistant tool-call turn (`content: ""`) identically
  for `""`, `null`, and an absent `content` key — the fixture shape can be
  passed to `apply_chat_template` as-is.
- No prior renderer-parity harness existed to port: on
  `openhands-integration-updated`, `integrations/qwen-code/` has no `parity/`
  dir (the qwen-code C3 check was never committed; only GPU/loader parity
  tests exist there). This harness is written fresh against the plan's spec.

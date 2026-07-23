# Autoresearch Inferlet for Pie

## What you're building

Autoresearch is Karpathy's autonomous ML research agent. It loops: edit
`train.py` -> run a 5-min training experiment -> check `val_bpb` -> keep or
discard -> repeat. Currently it runs through Claude's chat UI. You're going to
move that agent loop into a Pie inferlet so the LLM runs locally on your GPU
with KV-cache continuity across all steps.

## How Pie inferlets work (the mental model)

An inferlet is a WASM binary that Pie loads and runs on the server. Think of it
as a **program that controls the LLM** rather than the LLM controlling itself.
You write it in Rust (compiled to `wasm32-wasip2`), and it gets three
superpowers from the Pie runtime:

1. **`Context`** -- the KV cache. You call `ctx.system(...)`, `ctx.user(...)`,
   `ctx.cue()` to build up the conversation, and `ctx.generate(...)` to make
   the model produce tokens. The KV cache persists across all steps -- no
   re-encoding the full history each turn (this is a major advantage over
   Pattern B / the OpenHands SDK approach).

2. **`constrain_with(JsonSchema)`** -- grammar-constrained decoding. You give
   it a JSON schema and the model is *forced* to produce valid JSON matching
   that schema. This is what makes `openhands-agent` reliable -- the model
   can't narrate instead of acting.

3. **`ctx.idle()`** -- market-based GPU scheduling. During HTTP calls or
   subprocess waits, you drop your GPU bid so other sessions can use the
   memory.

The inferlet communicates with the outside world via HTTP (WASI networking).
Since WASM can't run shell commands directly, you write a lightweight Python
"tool server" that receives HTTP requests and executes them on the host.

## The architecture (3 files)

```
inferlets/autoresearch-agent/
  Cargo.toml          # Rust dependencies (inferlet SDK, serde, serde_json)
  Pie.toml            # Manifest: name, version, parameters
  src/lib.rs          # The agent loop (~200 lines)

integrations/autoresearch/
  tool_server.py      # Python HTTP server that runs bash, git, file edits
  run_autoresearch.sh # Orchestration: pie serve -> install inferlet -> start tool server -> launch
```

## Step-by-step plan

### Step 1: Understand the existing pattern

Read these files in order -- they're the template you'll adapt:

| File | What to learn |
|------|--------------|
| `inferlets/openhands-agent/Pie.toml` | How to declare parameters (task, tool\_server\_url, max\_steps) |
| `inferlets/openhands-agent/Cargo.toml` | The three dependencies you always need: `inferlet`, `serde`, `serde_json` |
| `inferlets/openhands-agent/src/lib.rs` | The full agent loop pattern: Input -> system prompt -> loop { generate -> parse -> call tool server -> feed observation } |
| `integrations/openhands/tool_server.py` | How the Python tool server receives and executes actions |
| `inferlets/agent-react/src/lib.rs` | A simpler agent (calculator/date tools) that shows the same pattern with less code |

The key insight: `openhands-agent` and your autoresearch agent are structurally
almost identical. Both are "generate JSON -> execute action -> feed observation
-> repeat" loops. The differences are:

- **Different actions**: autoresearch needs `bash`, `edit`, `read_file`,
  `finish` vs openhands-agent's `bash`, `edit`, `finish`
- **Different system prompt**: autoresearch's workflow (experiment loop with
  val\_bpb tracking)
- **Different tool server**: autoresearch's needs longer timeouts for 5-min
  training runs

### Step 2: Write the `Pie.toml` manifest

```toml
[package]
name = "autoresearch-agent"
version = "0.1.0"
description = "Autonomous ML research agent (Karpathy's autoresearch) with JSON-schema-constrained generation"

[runtime]
core = "^0.2.0"

[parameters]
program = {type = "string", description = "Contents of program.md -- the research instructions"}
tool_server_url = {type = "string", description = "URL of the tool execution server"}
max_experiments = {type = "int", optional = true, description = "Maximum experiment iterations (default: 100)"}
max_tokens_per_step = {type = "int", optional = true, description = "Max tokens per generation step (default: 4096)"}
```

The `[parameters]` section defines the JSON input your inferlet's `main()`
receives. Compare this to `openhands-agent`'s: we replaced `task` (a single
problem statement) with `program` (the full `program.md` contents that guide
the research direction).

### Step 3: Write `Cargo.toml`

Copy `inferlets/openhands-agent/Cargo.toml` exactly (change the package name).
The three dependencies are always the same:

- `inferlet` -- the SDK (provides `Context`, `Model`, `Sampler`, HTTP client)
- `serde` -- for deserializing the JSON input
- `serde_json` -- for parsing the model's JSON output

### Step 4: Write the JSON schemas

Autoresearch has a different action space than SWE-Bench. Define two schemas:

**`ACTION_SCHEMA`** -- the model must output one of these each step:

```json
{
  "type": "object",
  "properties": {
    "thought":    { "type": "string", "minLength": 1 },
    "action":     { "type": "string", "enum": ["bash", "edit", "read_file", "finish"] },
    "command":    { "type": "string" },
    "path":       { "type": "string" },
    "old_str":    { "type": "string" },
    "new_str":    { "type": "string" },
    "message":    { "type": "string" }
  },
  "required": ["thought", "action", "command", "path", "old_str", "new_str", "message"],
  "additionalProperties": false
}
```

**`FINISH_SCHEMA`** -- forced on the last step (same as openhands-agent's
pattern):

```json
{
  "type": "object",
  "properties": {
    "thought":  { "type": "string", "minLength": 1 },
    "action":   { "type": "string", "const": "finish" },
    "command":  { "type": "string" },
    "path":     { "type": "string" },
    "old_str":  { "type": "string" },
    "new_str":  { "type": "string" },
    "message":  { "type": "string", "minLength": 1 }
  },
  "required": ["thought", "action", "command", "path", "old_str", "new_str", "message"],
  "additionalProperties": false
}
```

The `constrain_with(JsonSchema(...))` call guarantees every generation matches
this shape. Without it, the model might narrate, produce malformed JSON, or
skip fields -- all failure modes we hit repeatedly with `openhands-completion`
before adding grammar constraints.

### Step 5: Write the system prompt

This is where autoresearch diverges from SWE-Bench. Adapt `program.md`'s
instructions into a system prompt. Key elements:

- You can only modify `train.py`
- Each experiment is `uv run train.py > run.log 2>&1` (5 min budget)
- Success metric: `val_bpb` (lower = better), extracted via
  `grep "^val_bpb:" run.log`
- Git workflow: commit before each experiment, advance branch if improved,
  reset if not
- Log results to `results.tsv`
- Include 1-2 worked examples in the same JSON format as your schema

### Step 6: Write `src/lib.rs` -- the agent loop

The structure mirrors `openhands-agent` almost line-for-line:

```rust
#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    // 1. Load model + create context
    let model = Model::load(&runtime::models().first().unwrap())?;
    let mut ctx = Context::new(&model)?;

    // 2. Set up conversation
    ctx.system(SYSTEM_PROMPT);
    ctx.user(&format!("program.md contents:\n{}\n\nBegin.", input.program));
    ctx.cue();

    // 3. Agent loop
    for step in 1..=input.max_experiments {
        // Pick schema (force finish on last step)
        let schema = if step == input.max_experiments {
            FINISH_SCHEMA
        } else {
            ACTION_SCHEMA
        };

        // Generate with grammar constraint
        let raw = ctx
            .generate(Sampler::Multinomial { temperature: 0.7, draws: 0 })
            .max_tokens(input.max_tokens_per_step)
            .constrain_with(inferlet::JsonSchema(schema))?
            .collect_text()
            .await?;

        // Parse JSON, extract fields
        let v: Value = serde_json::from_str(&raw)?;
        let action = v["action"].as_str().unwrap_or("");

        if action == "finish" { break; }

        // Call tool server (GPU goes idle during HTTP)
        let _idle = ctx.idle();
        let observation = call_tool_server(...).await;
        drop(_idle);

        // Feed observation back
        ctx.user(&format!("Observation:\n{observation}"));
        ctx.cue();
    }
}
```

Key things to understand about each line:

- **`ctx.system()` / `ctx.user()` / `ctx.cue()`** -- these build the prompt
  using the model's chat template (e.g., `<|im_start|>system...`). `cue()`
  adds the assistant prefix so the model knows to generate.
- **`ctx.generate(...).constrain_with(...).collect_text()`** -- this is one
  generation step. The KV cache retains everything from prior steps, so only
  the new user message + observation gets encoded (O(n) total, not O(n^2)).
- **`ctx.idle()`** -- returns a guard. While held, your GPU pages can be
  evicted for other sessions. `drop(_idle)` restores priority.
- **`call_tool_server()`** -- HTTP POST to the Python tool server. Copy this
  function verbatim from `openhands-agent`.

### Step 7: Write the tool server

Adapt `integrations/openhands/tool_server.py`. Autoresearch needs these
actions:

| Action | What it does |
|--------|-------------|
| `bash` | Run a shell command (same as openhands) -- covers `uv run train.py`, `grep`, `git commit`, `git reset`, etc. |
| `edit` | str\_replace or create a file (same as openhands) |
| `read_file` | Read a file's contents (simpler than bash + cat, less error-prone) |

You can likely reuse `tool_server.py` as-is -- the `bash` and `edit` actions
already cover everything autoresearch needs. The git operations (`git commit`,
`git reset --hard`, `git branch`) are just bash commands.

**Important:** Increase `BASH_TIMEOUT_S` from 120 to at least 360 -- training
runs take 5 minutes.

### Step 8: Write the orchestration script

Model after `integrations/openhands/run_pie_backend.sh`:

```bash
# 1. Start pie serve with the model config
pie serve --config $CONFIG &

# 2. Install the inferlet WASM
pie install --path inferlets/autoresearch-agent/target/wasm32-wasip2/release/autoresearch_agent.wasm

# 3. Start the tool server (Python, in the autoresearch repo dir)
python tool_server.py --working-dir /path/to/autoresearch &

# 4. Launch the agent
pie run --name autoresearch-agent@0.1.0 --input '{
  "program": "...(contents of program.md)...",
  "tool_server_url": "http://127.0.0.1:PORT",
  "max_experiments": 100
}'
```

### Step 9: Build and test

```bash
# Build the WASM (from inferlets/autoresearch-agent/)
cargo build --target wasm32-wasip2 --release

# Test locally with the dummy driver first (no GPU needed)
pie run --path target/wasm32-wasip2/release/autoresearch_agent.wasm \
  --input '{"program":"test","tool_server_url":"http://127.0.0.1:9999","max_experiments":2}'
```

The dummy driver generates random tokens -- you won't get real tool calls, but
you'll verify the WASM builds, the JSON schema constraint compiles, and the
HTTP plumbing works. Then move to a real model on GPU.

## What to watch out for

1. **WASI HTTP uses chunked transfer encoding** -- your tool server must handle
   it. The existing `tool_server.py` already has `_read_chunked()` for this.

2. **`println!()` during `ctx.idle()` is swallowed** -- debug prints inside
   `call_tool_server()` won't appear because the WASM guest's stdout is only
   delivered between idle periods.

3. **Long-running commands (5-min training runs)** -- the `BASH_TIMEOUT_S` in
   the tool server needs to be at least 360. The model will be idle the whole
   time (good -- `ctx.idle()` yields the GPU), but the HTTP request timeout
   also needs to be long enough.

4. **The `force_overwrite=True` gotcha** -- if you change the WASM and
   re-install it, `pie serve` does NOT hot-reload. You must restart the server.

5. **Cluster considerations** -- autoresearch needs a GPU for both Pie (running
   the LLM) and the training run (`uv run train.py`). If they share one GPU,
   you'll need to budget VRAM carefully. An alternative: run `pie serve` on one
   GPU and the training on another (requires a 2-GPU allocation, e.g.
   `--gpus=h200:2`).

## Where to start right now

```bash
cp -r inferlets/openhands-agent inferlets/autoresearch-agent
```

Then:

1. Edit `Pie.toml` -- change name, description, parameters
2. Edit `Cargo.toml` -- change package name
3. Edit `src/lib.rs` -- change the system prompt and action schemas, keep
   everything else
4. `cargo build --target wasm32-wasip2 --release` to verify it compiles

The tool server and orchestration script can wait until the WASM compiles and
you're ready for a GPU test.

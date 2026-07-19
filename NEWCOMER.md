# Pie: A Newcomer's Guide

This document explains Pie's design principles, core primitives, and how to get
started writing your first inferlet. It complements the main
[README](README.md) and the [online guide](https://pie-project.org/docs/guide/install).

---

## 1. What Problem Does Pie Solve?

Standard LLM serving engines (vLLM, SGLang, TensorRT-LLM) expose a single
interface: prompt in, tokens out. The engine owns the KV cache, the sampler,
and the scheduling — you cannot touch any of it.

This works for simple chat, but agents need more:

- **Multi-turn with KV continuity.** An agent that loops 30 times over bash
  output re-sends the entire history on every call, paying O(n^2) in prompt
  tokens. You want to *append* to the KV cache, not rebuild it.
- **Structured output.** You need the model to emit valid JSON every time, not
  "probably valid JSON" that you parse-and-retry.
- **Custom scheduling.** While an agent waits for a tool response, its KV pages
  should yield priority to other workloads, then reclaim them when the response
  arrives.
- **Branching and forking.** Best-of-N, tree search, and speculative decoding
  all share a prefix — you want to prefill once, then fork N times.

Pie's answer: run small user-supplied WebAssembly programs — **inferlets** —
directly next to the model, giving them access to the KV cache, forward pass,
sampler, grammar constraints, and scheduling market.

## 2. Design Principles

### Programs, Not Prompts

Traditional serving engines accept prompts. Pie accepts *programs*. An inferlet
is a compiled WebAssembly component that can call the forward pass, manipulate
the KV cache, fork contexts, apply grammar constraints, and communicate with
external services — all from inside the runtime. The engine doesn't just
generate tokens for you; it gives you the primitives to build whatever
inference strategy you need.

### KV-Cache as a First-Class Object

In Pie, the KV cache is not hidden inside an opaque engine. It is a
**Context** — a resource your code creates, fills, flushes, forks, saves, and
restores. Contexts are paged: they consist of committed pages (immutable,
shareable across forks) and working pages (mutable, per-context). You manage
them explicitly:

```rust
let mut ctx = Context::new(&model)?;
ctx.system("You are helpful.").user("Hello!").cue();
ctx.flush().await?;           // commits the prompt to pages
let fork_a = ctx.fork()?;    // shares committed pages, copies working
let fork_b = ctx.fork()?;    // same — one prefill, N decode streams
```

### Structured Output by Construction

Pie doesn't "hope" the model emits valid JSON and retry on failure. It
**constrains** every token choice to match a grammar. Two constraint types:

- **`JsonSchema(schema)`** — every generated token is masked against a
  JSON-schema state machine. The model can only produce tokens that lead to
  valid JSON. No retries, no parsing errors.
- **`Ebnf(grammar)`** — same idea, but with arbitrary EBNF grammars.

```rust
let text = ctx
    .generate(Sampler::Argmax)
    .max_tokens(512)
    .constrain_with(inferlet::JsonSchema(r#"{"type":"object","properties":{"name":{"type":"string"}}}"#))?
    .collect_text()
    .await?;
// `text` is guaranteed to be valid JSON matching the schema.
```

### Market-Based Scheduling

When multiple inferlets compete for GPU memory, Pie uses a market mechanism.
Each context has a **bid** (willingness to pay per page per step). Under
contention, lower-bid contexts get their pages evicted first. The SDK computes
a budget-exhausting bid automatically, but you can override it.

The practical API for agents: `ctx.idle()` drops the bid to zero while you wait
for an HTTP response, then restores it when you're done:

```rust
let _idle = ctx.idle();
let result = http_get(url).await?;
drop(_idle);  // bid restored, pages prioritized again
```

### Sandbox Safety via WebAssembly

Inferlets run as `wasm32-wasip2` components. They cannot access the host
filesystem, spawn processes, or touch GPU memory directly. All capabilities
come through the WIT (WebAssembly Interface Types) contract, which the runtime
controls. This means untrusted inferlets can run on shared infrastructure
without compromising the host.

## 3. Core Primitives

### Context

The KV cache state for one sequence. Operations:

| Method | What it does |
|---|---|
| `Context::new(&model)` | Create an empty context |
| `ctx.system(msg)` / `ctx.user(msg)` / `ctx.assistant(msg)` | Buffer chat-template tokens for each role |
| `ctx.cue()` | Insert the assistant-turn prefix ("cue" the model to respond) |
| `ctx.seal()` | Close the current turn without starting a new one |
| `ctx.flush().await` | Drain buffer through a forward pass, commit pages |
| `ctx.generate(sampler)` | Start a generation loop (returns a `Generator`) |
| `ctx.fork()` | Fork into a new context (shared committed pages) |
| `ctx.idle()` | Drop bid to zero (RAII guard restores on drop) |
| `ctx.save(name)` / `Context::open(model, name)` | Persist / restore snapshots |
| `ctx.truncate(n)` | Roll back the last `n` working-page tokens |
| `ctx.buffer()` | View the current unflushed token buffer |

### Model & Tokenizer

Loaded by name from the runtime. The model provides tokenize/detokenize, vocab
metadata, and chat template information. Most inferlets start with:

```rust
let model = Model::load(&runtime::models().first().ok_or("No models")?)?;
```

### Generator

Created by `ctx.generate(sampler)`. Controls the token-by-token loop:

```rust
let mut g = ctx.generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
    .max_tokens(256)
    .stop(&chat::stop_tokens(&model));

while let Some(step) = g.next()? {
    let out = step.execute().await?;
    // out.tokens contains the generated token IDs
}
```

Add `.constrain_with(JsonSchema(s))` or `.constrain_with(Ebnf(g))` to the
builder to enable grammar-constrained decoding.

Convenience collectors run the full loop and return the result:

```rust
let text = ctx.generate(Sampler::Argmax)
    .max_tokens(512)
    .collect_text()
    .await?;

// Or deserialize directly into a struct (pairs well with JsonSchema constraint):
let action: MyAction = ctx.generate(Sampler::Argmax)
    .max_tokens(512)
    .constrain_with(inferlet::JsonSchema(SCHEMA))?
    .collect_json()
    .await?;
```

### Sampler

Controls how the next token is chosen from the logit distribution:

| Variant | Description |
|---|---|
| `Sampler::Argmax` | Greedy (highest probability) |
| `Sampler::TopP { temperature, p }` | Nucleus sampling |
| `Sampler::TopK { temperature, k }` | Top-k sampling |
| `Sampler::TopKTopP { temperature, k, p }` | Combined top-k then nucleus |
| `Sampler::MinP { temperature, p }` | Min-p sampling |
| `Sampler::Multinomial { temperature, draws }` | Full-distribution sampling |

### Grammar & Matcher

The constrained-decoding engine. Two entry points:

- **`JsonSchema(schema_str)`** — compiles a JSON schema string into a token
  mask applied at every generation step.
- **`Ebnf(grammar_str)`** — same, but for arbitrary EBNF grammars.

Both are passed to `Generator::constrain_with()`. The runtime intercepts the
logit vector before sampling and zeros out any token that would lead to an
invalid output.

### Chat & Tool-Use Templates

Pie ships model-aware chat template support:

- **`chat::system/user/assistant/cue/seal`** — tokenize messages according to
  the model's chat template (Llama, Qwen, Mistral, etc.).
- **`chat::Decoder`** — classifies generated tokens into
  `Delta(text)` / `Interrupt(token_id)` / `Done(full_text)`.
- **`tool_use::equip`** — registers tools in the model's tool-call template.
- **`tool_use::format`** — returns the grammar that constrains tool-call
  output.
- **`tool_use::Decoder`** — detects `Start` / `Call(name, args_json)` events
  in generated tokens.

### Client Communication

Inferlets communicate with the remote client via two channels:

- **`println!()`** — writes to stdout, received by the client as `Event.Stdout`.
  This is the most common channel; most inferlets use it for streaming output.
- **WIT session interface** — `session::send` and `session::send-file` deliver
  structured `Event.Message` and `Event.File` events to the client. These are
  lower-level and less commonly used.

On the client side, `Process.recv()` yields `(event, value)` tuples:

```python
while True:
    event, value = await proc.recv()
    if event == Event.Stdout:
        print(value, end="")
    elif event == Event.Return:
        print(f"Result: {value}")
        break
```

### Inter-Inferlet Messaging

Inferlets running on the same engine can communicate via pub/sub topics:

```rust
use inferlet::messaging;

// Producer
messaging::broadcast("my-topic", &output_text);

// Consumer
let sub = messaging::subscribe("my-topic");
let msg = sub.get();  // non-blocking poll
```

This enables multi-stage pipelines where one inferlet's output feeds another's
input (see `demo-messaging` and `agent-swarm` in the `inferlets/` directory).

### WASI HTTP

Inferlets can make outbound HTTP requests via `wstd::http::Client`:

```rust
use inferlet::wstd::http::{Client, Request, body::IntoBody};

let body = serde_json::to_vec(&payload)?;
let req = Request::post("http://tool-server:8080/execute")
    .header("Content-Type", "application/json")
    .body(body.into_body())?;
let resp = Client::new().send(req).await?;
```

**Gotcha:** WASI HTTP always uses chunked transfer encoding for request bodies,
regardless of Content-Length. Your HTTP server must handle this.

### LoRA Adapters

Create and load LoRA adapters at runtime:

```rust
use inferlet::adapter::Adapter;
let adapter = Adapter::create(&model, "my-lora")?;
adapter.load("/path/to/lora/weights")?;
// Apply to a forward pass via the generator's adapter() method
```

Adapters can also be saved (`adapter.save(path)`), reopened
(`Adapter::open(&model, name)`), and forked (`adapter.fork(name)`).

## 4. Project Layout

```
pie/
├── runtime/          # Core runtime + WIT interface definitions
│   └── wit/core/     # The WIT contract (context, inference, model, etc.)
├── server/           # CLI binary: `pie serve`, `pie run`, `pie build`, etc.
├── sdk/
│   ├── rust/         # Rust inferlet SDK (`inferlet` crate)
│   ├── python/       # Python inferlet SDK
│   └── javascript/   # JS/TS inferlet SDK
├── client/
│   ├── python/       # Python client library (PieClient)
│   └── rust/         # Rust client library
├── driver/
│   ├── vllm/         # vLLM subprocess driver
│   ├── cuda/         # Native CUDA driver (from-scratch engine)
│   ├── portable/     # ggml/llama.cpp backend
│   ├── sglang/       # SGLang driver
│   └── dummy/        # Test/dev driver (no GPU)
├── inferlets/        # ~50 example inferlets (see below)
└── website/          # pie-project.org docs site
```

### Key directories explained

**`runtime/wit/`** — The WIT interface definitions are the formal contract
between inferlets and the Pie runtime. Every SDK (Rust, Python, JS) generates
its bindings from these `.wit` files. Reading them is the authoritative
reference for what an inferlet can do.

**`sdk/rust/inferlet/`** — The ergonomic Rust SDK. Wraps the raw WIT bindings
into types like `Context`, `Generator`, `Model`, `Sampler`. This is what you
`use inferlet::*` in your code.

**`driver/`** — Backend implementations. The driver translates Pie's abstract
operations (forward pass, page management, tokenization) into concrete GPU
calls. You pick a driver via the config TOML:

```toml
[[model]]
name = "default"
hf_repo = "Qwen/Qwen2.5-Coder-32B-Instruct"

[model.driver]
type = "vllm"            # or "cuda", "portable", "dummy"
device = ["cuda:0"]

[model.driver.options]
enforce_eager = true
gpu_memory_utilization = 0.85
```

**`client/python/`** — The `PieClient` connects to a running `pie serve`
instance over WebSocket. It can install programs, launch processes, and receive
events (stdout, messages, files, return values).

## 5. Writing Your First Inferlet

### Step 1: Create the project

```bash
pie new my-inferlet
cd my-inferlet
```

This creates a Rust project with `Cargo.toml`, `Pie.toml`, and `src/lib.rs`.

### Step 2: Write the code

Every inferlet has one entry point — an async function annotated with
`#[inferlet::main]`:

```rust
use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_max")]
    max_tokens: usize,
}

fn default_max() -> usize { 256 }

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model = Model::load(&runtime::models().first().ok_or("No model")?)?;

    let mut ctx = Context::new(&model)?;
    ctx.system("You are helpful.").user(&input.prompt).cue();

    let text = ctx
        .generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
        .max_tokens(input.max_tokens)
        .collect_text()
        .await?;

    Ok(text)
}
```

### Step 3: Declare parameters in Pie.toml

```toml
[package]
name = "my-inferlet"
version = "0.1.0"

[runtime]
core = "^0.2.0"

[parameters]
prompt = { type = "string", description = "The user message" }
max_tokens = { type = "int", optional = true, description = "Max tokens (default 256)" }
```

### Step 4: Build

```bash
pie build
```

This compiles to a `.wasm` component targeting `wasm32-wasip2`.

### Step 5: Run

One-shot mode (engine boots, runs, exits):

```bash
pie run my-inferlet@0.1.0 --config my-config.toml -- --prompt "Explain KV caches in 3 sentences"
```

Or with a local `.wasm` path (no registry):

```bash
pie run --path target/wasm32-wasip2/release/my_inferlet.wasm \
        --manifest Pie.toml \
        --config my-config.toml \
        --stdout \
        -- --prompt "Hello"
```

Or against a long-running server:

```bash
# Terminal 1: start the engine
pie serve --config my-config.toml --port 8080

# Terminal 2: install and run via the Python client (see section 7)
```

## 6. Example Inferlets Worth Reading

The `inferlets/` directory has ~50 examples. Here are the ones that best
illustrate Pie's unique capabilities:

| Inferlet | What it demonstrates |
|---|---|
| `helloworld` | Minimal structure: `#[inferlet::main]`, typed I/O, runtime info |
| `text-completion` | Chat-style generation with `Decoder`, stop tokens, `TopP` sampler |
| `constrained-decoding` | EBNF grammar-constrained output |
| `agent-react` | ReAct agent loop with JSON-schema constraint + per-step schema switching |
| `demo-parallel-fork` | Best-of-N via `Context::fork()` — one prefill, N concurrent decode streams |
| `best-of-n` | Fork + pairwise similarity ranking for consensus answers |
| `openhands-agent` | Full SWE-Bench agent with HTTP tool execution, `ctx.idle()`, schema constraints |
| `demo-persistent-kv` | KV cache save/restore across requests |
| `watermarking` | Custom sampling with watermark embedding |
| `jacobi-decoding` | Speculative decoding with `truncate()` for rollback |

## 7. The Client Side (Python)

When running against `pie serve`, you interact through `PieClient`:

```python
from pie_client import PieClient, Event

async with PieClient("ws://127.0.0.1:8080") as client:
    await client.authenticate("local-dev")

    # Install the WASM binary + manifest
    await client.install_program("path/to/inferlet.wasm", "path/to/Pie.toml")

    # Launch a process with parameters
    proc = await client.launch_process("my-inferlet@0.1.0", input={
        "prompt": "What is 2+2?",
        "max_tokens": 64,
    })

    # Receive events
    while True:
        event, value = await proc.recv()
        if event == Event.Stdout:
            chunk = value.decode("utf-8") if isinstance(value, bytes) else value
            print(chunk, end="")
        elif event == Event.Return:
            print(f"Result: {value}")
            break
        elif event == Event.Error:
            print(f"Error: {value}")
            break
```

## 8. Drivers

Pie abstracts the GPU backend behind drivers. Pick one in your config:

| Driver | When to use |
|---|---|
| `vllm` | Production. Runs vLLM as a subprocess, well-tested. |
| `cuda` | Native CUDA engine — maximum performance, tighter Pie integration. |
| `portable` | CPU or mixed environments via ggml/llama.cpp. |
| `sglang` | SGLang backend. |
| `dummy` | Testing and development — no GPU required. |

## 9. Key Gotchas

1. **WASI HTTP uses chunked transfer encoding.** If your inferlet makes HTTP
   POST requests, the receiving server must handle `Transfer-Encoding: chunked`
   bodies (Python's `BaseHTTPRequestHandler` does not by default).

2. **`println!()` during `ctx.idle()` may not reach stdout.** The runtime
   buffers output during idle periods. Print before or after the idle guard,
   not during.

3. **WASM binaries are cached by `pie serve`.** After rebuilding, you must
   either restart `pie serve` or re-install with `install_program()`. The
   `force_overwrite=True` flag updates the stored program but doesn't affect
   already-running server instances.

4. **Inferlets are single-threaded.** WebAssembly is single-threaded, but you
   can use `async` for cooperative concurrency (e.g., `future::join_all` for
   parallel fork decode streams).

5. **Grammar constraints can truncate.** If `max_tokens` is hit mid-grammar,
   the output will be a valid prefix but may not parse as complete JSON. Always
   handle the `serde_json::from_str` error case.

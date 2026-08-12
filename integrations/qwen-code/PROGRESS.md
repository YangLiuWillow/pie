# qwen-code ↔ Pie on the rewritten engine — progress log

Running status of the port and its follow-ups. Design + detailed results:
`docs/qwen-code-dev-port.md`. Branch: `liu/qwen-code-dev` (pushed to fork
`YangLiuWillow/pie`). Never commit to `dev`/`main`.

## Done

**2026-08-11 — port complete and verified**
- `chat-completions` inferlet ported to `pie:inferlet@0.3.0`
  (`tests/inferlets/chat-completions/`): session-loop daemon, PTIR
  generation, in-process KV working-set reuse under the old content
  addresses. 22/22 native unit tests; wasm builds clean.
- OpenAI surface moved client-side: `shim.py` (HTTP/SSE ↔ gateway WS).
- Verified on RTX 3090 (RunPod, CUDA 12.9 toolkit, sm_86):
  **acceptance 33/33** (full old-integration parity: native tool calls,
  unique ids, usage, KV resume) and **stock qwen-code v0.21.6 e2e green**
  (`run_shell_command` executed, clean exit; resume hits cached
  8,841–8,968 tokens on follow-up turns).
- Bring-up automated in `pod_bootstrap.sh` (CUDA ≥ 12.9 toolkit install,
  arch autodetect, build, model import). Dummy-driver profile for
  GPU-free transport work.

**2026-08-11 — C3 renderer parity closed (docs §9)**
- `echo_tokens` debug flag + `parity/check_render.py`:
  **23/23 wire fixtures byte-exact** vs HF `apply_chat_template`.
- Fixed two inherited renderer bugs: the `\n\n\n# Tools` seam (old-engine
  bug) and missing jinja-`tojson` re-serialization of tool schemas +
  replayed tool-call arguments. Both plausibly contributed to the old
  H200 trajectory divergences.
- Open divergence class: `/no_think` soft switch vs HF's empty think
  block — no fixture sets `enable_thinking:false`; checker reports it as
  KNOWN-DIV when it fires.

## In progress

**30B A/B benchmark rerun on the rewritten engine** — pie (shim +
chat-completions daemon) vs vLLM, Qwen3-Coder-30B-A3B-Instruct, qwen-code
task battery. Old-engine result (old plan §5b): pie 1.47× slower overall,
~1.1× trajectory-matched, with two confounds now removed — render
divergence (C3: byte-exact) and per-request daemon overhead (long-lived
instance). Steps:
- [x] Branch pushed to fork
- [x] `bench/` harness carried over + arm boot scripts codified
      (`start_pie_arm.sh`, `start_vllm_arm.sh`, H200 30B config)
- [x] First H200 pod (f5nlisg6y3jxcr) bootstrapped — then handed over to
      the parallel NPR session, which checked out its branch over ours;
      benchmark needs exclusive GPU, so it moved to a fresh pod. Fixes
      that came out of that bring-up: pod_bootstrap artifact check
      (prefetch made import self-skip), 300s `silence_timeout` (30B first
      request outlives the 30s default and the kill takes the WS down),
      shim full-reconnect with backoff.
- [x] Replacement pod 4ld75rjrqf0znl (H200 secure): bootstrapped, 30B
      imported, pie arm ran (5/5 rc=0, 61.5 s total wall)
- [ ] **BLOCKED — RunPod credit exhausted mid-run.** Balance hit $0 while
      four pods across sessions burned $7.92–11.51/hr; RunPod terminated
      every pod, including ours, before the vLLM arm ran. Balance $2.66,
      no pods live, nothing further can run until the account is topped
      up. The `results/` directory (wire captures, per-task logs) died
      with the pod — only the console summary survived (recorded in
      docs §10).
- [ ] vLLM arm: never ran. Root cause found and now guarded in
      `start_vllm_arm.sh`: vllm 0.25.1 ships CUDA-13 wheels and needs
      driver ≥ 580; this pod had 575.57.08. Swapping torch to cu128 fixes
      torch but not vllm's own extensions (`libcudart.so.13`). Next run:
      provision a driver-≥580 pod, or pin a cu12-era vllm and document
      the version delta.
- [ ] Trajectory diff, summary, results into `docs/qwen-code-dev-port.md`

## Later / parked

- `/no_think` parity fixtures (`enable_thinking:false` corpus).
- Upstream candidates: CUDA ≥ 12.9 gate for `gemm.cpp`, python-client
  auth-sentinel fix, the integration itself.
- Engine-index KV storage if sub-page tails ever become indexable;
  Route 2 (server-side port) remains spec-only.
- Local Metal e2e: works in principle; gated on ~4.5 GiB reclaimable
  (close apps) on the 8 GB M2.

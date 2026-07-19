# Session handoff — 2026-07-18 (coder-session Phases 1+2, upstream merge, test fixes)

Written at session end so work can continue in a fresh session. Companion
memory files: `project_coder_session.md`, `project_tool_server_test_failures.md`,
`project_qwen3_coder_tool_format.md` in the Claude memory directory.

**IMPORTANT for the next session:** this session had background watchers that
died with it. Nothing will auto-fire. The "Pending steps" section below must be
driven manually.

---

## 1. What was delivered (responses given to the user, condensed)

### Phase 1 — openhands-coder-session (implemented, verified)

Design doc: `docs/OPENHANDS_CODER_SESSION_DESIGN.md` (its "Implementation
notes" section records all decisions). Goal: stock OpenHands agent unchanged,
Pie replaces only the LLM transport *statefully* — same tokens, N× less
prefill.

- **`inferlets/openhands-coder-session/`** (new crate, UNTRACKED — not yet
  committed): same contract as `openhands-completion` plus session protocol
  (`session_id`, `session_prev_len`, `session_prev_hash` (FNV-1a-64 hex),
  `session_action="delete"`, `kv_verify`, `use_grammar`). Response adds
  `session {id, mode: fresh|extended|rebuilt|stateless|deleted, len, hash,
  prefill_tokens}`. Wasm builds clean.
  - Key decisions: snapshot always equals the **canonical prompt render**
    (saved after prefill, before generation — never the model's sampled
    bytes); snapshot **excludes the generation cue** (message-boundary ends
    keep renders concatenative, avoiding BPE-merge traps); host stores no
    tokens, only echoes (len, hash); ANY mismatch rebuilds silently (retry
    after lost response is a legitimate stale-snapshot case, not an error).
  - Bug found: `tools::native_matcher` traps host-side on an empty tools
    list — the new inferlet gates on `has_tools`; openhands-completion has
    the same latent bug (unfixed there, OpenHands always sends tools).
- **`pie_openhands/llm.py`**: `pie_session` / `pie_kv_verify` fields, echo
  bookkeeping in private attrs, `close_pie_session()` (idempotent, called in
  harness `finally`), `pie_session_summary()` telemetry.
- **Harness**: `--pie-session` / `--kv-verify` in `benchmarks/run_swe_bench.py`;
  `BACKEND=pie-session` (+`KV_VERIFY=1`) in `run_pie_backend.sh`; per-instance
  `pie_session` stats in predictions `_metadata`.
- **Tests**: `tests/test_pie_session.py` (10 tests). Dummy-driver smoke
  (scratchpad script, ephemeral) exercised fresh → extended (39-token delta of
  175) → rebuilt-on-rewrite → zero-delta retry → delete → rebuild-after-delete
  → stateless, plus a PieLLM end-to-end run — all passed with kv-verify on.
- Known deferred issue: the condenser shares the agent's LLM instance, so a
  condensation costs two session rebuilds (correct, slow; Phase 4 fixes).
- Ops gotcha: `pie serve` caches installed programs by name@version —
  **restart the server after rebuilding a wasm**.

### Phase 2 — kv-verify + equivalence runs

- `benchmarks/compare_equivalence.py` + `bench_session_equiv_smoke.sbatch`
  (5 instances × 2 arms at t=0: stateless openhands-completion vs
  pie-session with kv-verify).
- **Job 18709682 (Qwen3-Coder-30B MoE): completed.** Session machinery
  perfect — 10.99× total prefill reduction (49,176 of 540,496 prompt tokens),
  modes all fresh/extended, 0 kv-verify errors, patches byte-equal — but
  **vacuous: ALL patches empty in BOTH arms.**
- **Root cause:** Qwen3-Coder's chat template uses the XML tool format
  (`<tool_call>\n<function=name>\n<parameter=key>...`), while Pie's
  qwen2/qwen3 `Instruct` renders/parses ChatML JSON
  (`<tool_call>\n{"name": ...}`). The model never emits a parseable tool
  call → every turn looks content-only → 10 fake-user nudges → empty patch.
  Affects ANY past `BACKEND=pie` run on Qwen3-Coder. Pattern A (own JSON
  protocol) and litellm baseline (vLLM `qwen3_coder` parser) are unaffected.
  Follow-up work: implement the qwen3-coder XML format in the runtime
  Instruct (equip preamble, replay, answer wrapper, Decoder, grammar).
- **Rerun: job 18731262 on Qwen2.5-Coder-7B** (template matches) — RUNNING at
  handoff time. ⚠️ Its log showed `scikit-learn__scikit-learn-25973` emitting
  content-only turns (fake response 3/10) in arm A — check whether that's
  instance-specific chat behavior or something systematic when analyzing.
  Log: `logs/session_equiv_18731262.out`; predictions
  `predictions/session_equiv_{stateless,session}_*.jsonl`; comparison runs
  automatically at job end (`benchmarks.compare_equivalence`).

### tool_server test failures (user's question: ours or upstream's?)

**Ours, definitively** — `integrations/openhands/` doesn't exist on
`pie-project/pie:main`. Both underlying features ARE necessary (fidelity with
stock OpenHands tools), so they were **fixed, not removed**:

1. `TestBash::test_nonzero_exit` — uncommitted `PersistentBash` returned
   exit 1 when `exit 42` killed the shell. Fixed: on stdout EOF, harvest
   `proc.wait()` (bash's exit status IS the command's status) before restart.
   Second bug found underneath: `_restart()` pre-assigned `cwd` to the
   literal `/proc/<pid>/cwd` string → dead shell made `Popen` raise
   FileNotFoundError; also `/tmp` fallback dropped the agent out of its
   workspace → now falls back to `self._init_cwd` (workspace root).
2. `TestEdit::test_no_match` — FileEditor wrap (commit ea36d778) changed the
   wording; stale assertion updated to accept both ("not found" / "no
   replacement"), same pattern as `test_no_match_returns_error`.

**Full Python suite now: 80 passed, 0 failed** (was 78+2 failed).
Caveat: agent job 18690721 was mid-run when tool_server.py changed; its
imported module keeps old behavior unless its harness restarts.

### Git sync with upstream

- Remote `upstream` = https://github.com/pie-project/pie.git added.
- **fork `main`: fast-forwarded to upstream 550de0f3 and PUSHED** (was 7
  behind; done via `git fetch upstream main:main` — no worktree switch).
- **`openhands-integration` is NOT 7 behind — it's 225 commits behind**
  (forked from an older main). Full merge performed on branch
  **`merge-upstream-20260718`** in worktree **`/nfs/roberts/project/pi_ql324/ly337/pie-merge-wt`**,
  merge commit `86e3617c` (amended once). Conflict resolutions:
  - `runtime/src/model/instruct/qwen3.rs`: kept our Option-B replay token
    constants / no-newline prefixes; adopted upstream's
    generation_header-with-`config.generation_suffix`; kept our `func`
    binding in grammar builder. Also added `generation_suffix: ""` to 3 of
    our test ChatMLConfig literals (new required field).
  - `runtime/src/inference/structured/json_schema.rs`: file contains a NUL
    byte → git "binary", no text merge. Took upstream body + re-applied our
    38-line patch (`json_schema_to_ebnf_named` + `namespace` field).
  - `driver/vllm/src/pie_driver_vllm/engine.py`: upstream's new
    DriverCapabilities fields (max_forward_tokens etc. — bridge protocol
    CHANGED) + kept MoE WorkspaceManager init + kept
    `_normalize_arch_name(self.arch_type)` (merged instruct.rs still matches
    short names like "qwen3_moe").
  - `driver/dev/worker.py`: accepted upstream's deletion of the whole dev
    driver (nothing in integrations/ uses it; smokes use `driver/dummy`).
- **Validation of merged tree: release build OK; `cargo test -p pie
  --release` = 821 passed, 0 failed; openhands-completion wasm builds
  against merged SDK.**
- **NOT yet applied to the live `openhands-integration` branch** — deliberately.

## 2. Why the merge is not applied yet + exact apply procedure

The live checkout's driver is imported **editable** by the pie vllm env
(`/nfs/roberts/scratch/pi_ql324/ly337/pie-vllm-env` imports
`pie_driver_vllm` from `driver/vllm/src/`). Upstream changed the
driver↔runtime capabilities protocol. Running jobs (18690721 agent-32b-200,
18731262 session-equiv) auto-restart `target/release/pie` + harness on
failure; swapping source under them → old binary + new driver = mismatch →
job death. **Apply only after `squeue -j 18690721,18731262` is empty.**

Apply steps (from `/nfs/roberts/project/pi_ql324/ly337/pie`):

1. `git add -A && git commit` the working tree on `openhands-integration`
   (contains: coder-session inferlet + tests + sbatch + docs [untracked],
   tool_server/test fixes, PieLLM session support, harness flags, prior
   session's uncommitted FileEditor/PersistentBash/config work — including
   `driver/dev/src/pie_driver_dev/batching.py`, which the merge will delete;
   committing first preserves it in history).
2. `git merge merge-upstream-20260718` — expect a modify/delete conflict on
   `driver/dev/**` (accept deletion: `git rm -r driver/dev`), possibly
   `driver/vllm/src/pie_driver_vllm/config.py` (uncommitted gpu_util edits vs
   upstream — inspect and combine). Everything else was resolved in 86e3617c.
3. Rebuild: `cargo build --release -p pie` (the live binary MUST match the
   new driver source), rebuild the three openhands inferlet wasms
   (`cargo build --release --target wasm32-wasip2` in each).
4. Re-run validation: Python suite
   (`PYTHONPATH="" .venv/bin/python -m pytest tests/ -q` in
   integrations/openhands — expect 80 pass) and the dummy-driver session
   smoke (boot `target/release/pie serve --config
   integrations/openhands/tests/fixtures/pie_dummy_config.toml --port 18099
   --no-auth`, then from integrations/openhands run
   `PYTHONPATH="" HF_HOME=/nfs/roberts/scratch/pi_ql324/ly337/hf_cache
   .venv/bin/python session_smoke.py` — the smoke script was saved into the
   repo at `integrations/openhands/session_smoke.py` for exactly this).
   Driver env may need `uv sync` in driver/vllm (deps changed: flashinfer
   imported directly now).
5. `git push origin openhands-integration` (user asked for the branch to be
   updated; main is already pushed).
6. Clean up: `git worktree remove ../pie-merge-wt` and delete branch
   `merge-upstream-20260718` after the merge lands.

## 3. Job/experiment state at handoff

- 18690721 agent-32b-200: RUNNING (~5.5h). 18599876 baseline-32b-200:
  finished/left queue (check `sacct`; it was litellm-based, unaffected by pie
  changes). See memory `project_32b_200_experiment.md`.
- 18731262 session-equiv (7B): RUNNING (~14 min in, arm A). On completion,
  read the tail of `logs/session_equiv_18731262.out` — the comparison prints
  per-instance EQUIVALENT/MISMATCH + prefill reduction. Success criteria:
  patches non-empty and byte-equal, 0 kv-verify errors, prefill ~10×+ lower
  in arm B. If patches are again all empty on 7B, the tool-call problem is
  NOT just the Qwen3-Coder template — investigate openhands-completion's
  equip/decode path on GPU directly.
- 18709682 (30B equivalence): done, results analyzed (see §1).

## 4. Phase roadmap status (design doc table)

- Phase 1 (session contexts + delta append): DONE (pending GPU-validated
  non-vacuous equivalence).
- Phase 2 (kv-verify + 5-instance equivalence): code DONE; meaningful run in
  flight (18731262).
- Phase 3 (shared prefix via fork), Phase 4 (fork-based condensation),
  Phase 5 (suspend during tools + concurrency sweep): NOT STARTED.
- New prerequisite discovered for the "headline" benchmarks on Qwen3-Coder:
  qwen3-coder XML tool format support in runtime Instruct.

# Python-free CI for render bit-parity  — BUILT

## The goal

`ci.yml` has **no Python job today** — seven jobs, all cargo/cmake/shell. Adding
the render-parity check must not change that. (`release-pypi.yml` does use
Python, but it publishes the Python SDK and never runs on a PR.)

The check itself needs HuggingFace's `apply_chat_template` as an **independent
oracle** — a Rust Jinja engine would be pie's Rust renderer compared against a
Rust Jinja engine, and a shared assumption would pass silently. So the oracle
stays Python and moves OFF the PR path.

## The shape, which the repo already uses twice

`model/config/tests/differential.rs` compares the Rust config normalizer against
**committed goldens** recorded from the C++ it replaces. `grep Command::new`
there returns nothing: it reads checked-in JSON. `scripts/sync-wit.sh` +
`wit-drift` is the same trade for vendored WIT.

    expensive/foreign oracle  ->  regeneration, by a human, occasionally
    cheap comparison          ->  cargo test, every PR

## Three pieces

### 1. Regeneration (Python, run by hand)  — PARTLY BUILT

`render-tokens --compile <tokenizer.json> <out.pietok>`  **DONE**
  Emits `pie.tokenizer/1` in a flat container: per object `u32` name length,
  name, `u32` data length, data, little-endian. NOT JSON — serialising raw
  bytes as JSON integer arrays turned 6.8 MB into 20.6 MB.

  Measured: 11.4 MB tokenizer.json -> 4.0 MB, 20.0 MB -> 6.8 MB (~35%), and
  loading is ~420x faster than parsing HF JSON (96-138 ms -> under 1 ms).

`check_render_matrix.py --emit-fixtures <dir>`  **DONE**
  For every (arm, shape, thinking): render the reference with
  `apply_chat_template`, tokenize, and write the request body beside the
  expected ids. Shapes live in Python today; the fixture carries the exact
  BODY so the Rust side needs no copy of them.

### 2. Committed fixtures under `tests/fixtures/render/`  — DONE

    tok/<sha16>.pietok      compiled tokenizers, ~10.8 MB total
    cases.json              { tokenizers, arms, cases: [{arm, shape, thinking,
                              body, ids}] }

  Dedup by compiled-blob hash. NOTE: mlx-community and Qwen tokenizers compile
  to DIFFERENT blobs (metadata) but were verified to ENCODE identically over
  the whole corpus, so one blob per encode-equivalence class is sound — record
  that decision in the fixture file rather than leaving it implicit.

### 3. Rust test  — DONE

  Lives beside the deps it needs (`render-tokens` already links
  `pie-openai-serving`, `pie-model-qwen-3`, `pie-tokenizer`), so
  `integrations/opencode/parity/render-tokens/tests/render_parity.rs`.

  Load `.pietok` via `CanonicalTokenizer::from_objects` ->
  `Tokenizer::from_canonical`, run `plan_render` + `QwenInstruct` over each
  case's body, compare ids. No HF cache, no network, no Python.

### 4. `ci.yml` job  — DONE

  `cargo test -p render-tokens`. Path filter on `model/**`,
  `inferlets/openai-serving/**`, `interface/**`.

## What the fixtures must cover

14 shapes x 2 thinking modes x 5 arms = 140 cells, all currently token-exact.
Arms: qwen3, qwen3_coder, qwen3_5, qwen3_coder_upstream, qwen3_5_upstream.

Shape set is mutation-tested (`--mutation-matrix`): 9 mutations, no holes,
`agent_loop_3` already dropped as provably redundant. Re-run it before adding
or removing a shape.

## Staleness

Goldens can go stale if an upstream template changes (Qwen revised the Coder
template after mlx-community and unsloth snapshotted it — found by hand, days
late). A weekly scheduled job re-running the Python harness against live HF is
the intended tripwire; a red build there means the world moved, not the PR.

## As built

    tests/fixtures/render/cases.json                140 cases, 1.0 MB
    tests/fixtures/render/tok/30ecc0eb….pietok      4.0 MB  qwen3, qwen3_coder,
                                                            qwen3_coder_upstream
    tests/fixtures/render/tok/a6b82869….pietok      6.8 MB  qwen3_5
    tests/fixtures/render/tok/a86f6a2e….pietok      6.8 MB  qwen3_5_upstream
                                                   ------
                                                   17.6 MB

Three blobs, not the two estimated: Coder's mlx and Qwen tokenizers compile
IDENTICALLY (one blob serves three arms), while Qwen3.6's two differ in
metadata. They were verified to encode identically over the whole corpus, so
one blob could serve both — kept separate anyway, because correctness is worth
6.8 MB and a fixture that quietly substitutes one tokenizer for another is a
fixture that can lie.

    cargo test -p render-tokens --test render_parity     0.48 s, 140 cells

Verified to FAIL: flipping `system_before_tools` in the test's config broke 80
of 140 cells with the first divergence located per cell. A test that cannot
fail is worth nothing, so this was checked rather than assumed.

`ci.yml` gains `render-parity` and remains Python-free — the only `python` in
the file is inside the error message that tells a human how to regenerate.

## Built, final shape

    ci.yml  render-parity          cargo test, 112 cells, ~0.5 s, no Python
    render-drift.yml               weekly, Python, live HuggingFace

    tests/fixtures/render/         112 cases, 3 blobs, 17.6 MB
    PORTING-A-MODEL.md             the workflow for the next model

The golden test builds its renderer through `instruct::create`, NOT a
reconstructed `ChatMLConfig`. The first version copied the qwen row's ten
fields and would have passed against config nobody serves — which defeats the
purpose, since porting a model IS adding a row. The fixture therefore records
each arm's ARCH STEM (`qwen3_5moe`, not `qwen3_5_moe`), because only the
generator knows it.

`qwen3_coder_upstream` WAS `python_only` -- it needed `CoderSchema::QwenMain`
where the registry hardcoded `MlxGguf`, so `create()` could not produce it, and
the golden test covered 112 of 140 cells. That gap was a signal about the
registry, and the registry was fixed: `InstructOverrides` carries the facts pie
cannot read off anything it holds, and `create_with` applies them. 140/140 in
both now.

Both failure modes verified rather than assumed:

    flip system_before_tools in the TEST's config   80/140 cells (old version)
    flip it in model/src/instruct.rs (the REGISTRY) 32/112 cells, right arm only

The second is the one that matters: it is what a bad port looks like.

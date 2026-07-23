"""End-to-end OpenEvolve run driven entirely by the Pie backend.

Programmatic driver (init_client is a Callable, so it can't live in YAML): loads
the function_minimization example config, routes every LLM call through PieLLM
via the config's init_client hook, disables external embedding/novelty calls,
and runs a few evolution iterations against a live pie server.

Success = the process-parallel loop completes and at least one Pie-generated
child program is evaluated and added (generation > 0), proving the controller
wiring (sample parent → generate_evolve → PieClient → inferlet → diff → apply →
evaluate → add) works end-to-end across worker processes.
"""
import asyncio
import os
import sys
import time

OE_ROOT = "/nfs/roberts/project/pi_ql324/ly337/gemini/openevolve"
sys.path.insert(0, OE_ROOT)

from pie_client import PieClient  # noqa: E402
from openevolve.config import load_config  # noqa: E402
from openevolve.controller import OpenEvolve  # noqa: E402
from openevolve.llm.pie import PieLLM  # noqa: E402

EX_NAME = os.environ.get("OE_EXAMPLE", "function_minimization")
EX = f"{OE_ROOT}/examples/{EX_NAME}"
# Example config filename (circle_packing ships config_phase_1.yaml, not config.yaml).
EX_CONFIG = os.environ.get("OE_CONFIG", "config.yaml")
URI = os.environ.get("PIE_URI", "ws://127.0.0.1:18099")
USER = os.environ.get("PIE_USER", "local-dev")


async def preinstall() -> None:
    wasm = os.environ["PIE_OE_WASM"]
    manifest = os.environ["PIE_OE_MANIFEST"]
    async with PieClient(URI) as c:
        await c.authenticate(USER)
        await c.install_program(wasm, manifest, force_overwrite=True)
    print("inferlet installed", flush=True)


async def main() -> None:
    backend = os.environ.get("PIE_OE_BACKEND", "pie")
    if backend == "pie":
        await preinstall()
        # Workers inherit env at fork; drop the WASM path so PieLLM in workers
        # does NOT re-install (already installed above).
        os.environ.pop("PIE_OE_WASM", None)
        os.environ.pop("PIE_OE_MANIFEST", None)

    cfg = load_config(f"{EX}/{EX_CONFIG}")
    cfg.max_iterations = int(os.environ.get("OE_ITERS", "8"))
    cfg.checkpoint_interval = 10_000
    cfg.random_seed = int(os.environ.get("OE_SEED", "0"))

    # The example config sets max_tokens=16000, which makes each child decode
    # for minutes and blow the request timeout. Diffs are short — cap it and
    # give generation room under the timeout.
    max_tokens = int(os.environ.get("OE_MAX_TOKENS", "2048"))
    timeout = int(os.environ.get("OE_TIMEOUT", "300"))
    cfg.llm.max_tokens = max_tokens
    cfg.llm.timeout = timeout

    # Route LLM traffic to the selected backend.
    #   pie  → PieLLM fork inferlet (B arms)
    #   vllm → OpenAI-compatible vLLM server (A arms); A1 two-phase via
    #          PIE_OE_A1_TWOPHASE=1, APC on the server side.
    vllm_base = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
    vllm_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    vllm_model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    for m in list(cfg.llm.models) + list(cfg.llm.evaluator_models):
        m.max_tokens = max_tokens
        m.timeout = timeout
        if backend == "pie":
            m.init_client = PieLLM
        else:  # vllm / openai
            m.init_client = None
            m.api_base = vllm_base
            m.api_key = vllm_key
            m.name = vllm_model
    print(f"backend={backend} models={[m.name for m in cfg.llm.models]}", flush=True)
    # Avoid external OpenAI calls (embeddings / novelty judge).
    cfg.database.embedding_model = None
    cfg.database.novelty_llm = None

    seed = cfg.random_seed
    out_dir = os.environ.get("OE_OUT", f"{OE_ROOT}/pie_run_out/{backend}_{EX_NAME}_s{seed}")
    print(
        f"=== OE_RUN backend={backend} example={EX_NAME} seed={seed} "
        f"iters={cfg.max_iterations} ===",
        flush=True,
    )
    oe = OpenEvolve(
        f"{EX}/initial_program.py",
        f"{EX}/evaluator.py",
        config=cfg,
        output_dir=out_dir,
    )
    t0 = time.perf_counter()
    best = await oe.run(iterations=cfg.max_iterations)
    elapsed = time.perf_counter() - t0

    programs = oe.database.programs
    n_total = len(programs)
    n_children = sum(1 for p in programs.values() if getattr(p, "generation", 0) > 0)
    print(f"\nDB: {n_total} programs, {n_children} Pie-generated children (gen>0)", flush=True)
    if best is not None:
        print("BEST:", best.id, best.metrics, flush=True)
    # Single parseable summary line per run (the report script keys on this).
    best_combined = (best.metrics.get("combined_score") if best is not None else None)
    print(
        f"OE_MEASURE backend={backend} example={EX_NAME} seed={seed} "
        f"iters={cfg.max_iterations} elapsed_s={elapsed:.2f} "
        f"programs={n_total} children={n_children} best_combined={best_combined}",
        flush=True,
    )

    assert best is not None, "no best program (initial eval failed)"
    assert n_children >= 1, (
        "no Pie-generated child was added — check the log for diff-parse errors "
        "or PieClient failures in the workers"
    )
    print("\nPIE E2E EVOLUTION OK", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"PIE E2E FAIL: {e}", file=sys.stderr)
        sys.exit(1)

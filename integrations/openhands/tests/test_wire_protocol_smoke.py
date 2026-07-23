"""Wire-protocol smoke test.

Boots a real ``pie serve`` (dummy driver, random tokens — needs a HuggingFace
tokenizer download on first run for the Qwen3-0.6B vocab), uploads our
``openhands-completion`` inferlet via ``client.install_program``, and verifies
that:

  1. PieLLM can connect via PieClient.
  2. The inferlet receives our structured messages/tools input, generates
     *some* tokens (random), and returns the contract dict {text, tool_calls,
     stop_reason, prompt_tokens, tokens_generated}.
  3. PieLLM wraps that into a litellm.ModelResponse and (via the parent
     LLM.completion path) returns an LLMResponse.

The test is GATED behind the ``PIE_BIN`` environment variable. If unset,
it is skipped — so the default test suite stays hermetic.

Run with:

  cd pie/integrations/openhands
  PIE_BIN=$(realpath ../../target/release/pie) \\
  PIE_WASM=$(realpath ../../inferlets/openhands-completion/target/wasm32-wasip2/release/openhands_completion.wasm) \\
  PIE_MANIFEST=$(realpath ../../inferlets/openhands-completion/Pie.toml) \\
  PIE_CONFIG=$(realpath tests/fixtures/pie_dummy_config.toml) \\
  .venv/bin/pytest tests/test_wire_protocol_smoke.py -s
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
from pie_client import PieClient

from openhands.sdk import Message, TextContent
from pie_openhands import PieLLM


PIE_BIN = os.environ.get("PIE_BIN")
PIE_WASM = os.environ.get("PIE_WASM")
PIE_MANIFEST = os.environ.get("PIE_MANIFEST")
PIE_CONFIG = os.environ.get("PIE_CONFIG")

# Resolved from the manifest's [package] section — keep in sync with
# inferlets/openhands-completion/Pie.toml.
INFERLET_NAME = "openhands-completion@0.1.0"

pytestmark = pytest.mark.skipif(
    not (PIE_BIN and PIE_WASM and PIE_MANIFEST and PIE_CONFIG),
    reason="PIE_BIN / PIE_WASM / PIE_MANIFEST / PIE_CONFIG not set",
)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"pie serve never bound to {host}:{port} — last error: {last_err}")


async def _install_inferlet(uri: str) -> None:
    async with PieClient(uri) as client:
        await client.authenticate("local-dev")
        await client.install_program(PIE_WASM, PIE_MANIFEST, force_overwrite=True)


# ─── Fixture: a live pie serve with the inferlet installed ──────────────


@pytest.fixture(scope="module")
def pie_server():
    port = _free_port()
    log_path = Path("/tmp/pie_serve_smoke.log")
    log = log_path.open("w")
    proc = subprocess.Popen(
        [
            PIE_BIN, "serve",
            "--config", PIE_CONFIG,
            "--port", str(port),
            "--no-auth",
        ],
        env=os.environ.copy(),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        try:
            _wait_for_port("127.0.0.1", port)
        except Exception:
            proc.terminate()
            proc.wait(timeout=5)
            raise RuntimeError(
                f"pie serve failed to come up — see {log_path} for output"
            )
        uri = f"ws://127.0.0.1:{port}"
        asyncio.run(_install_inferlet(uri))
        yield uri
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


# ─── The smoke test ──────────────────────────────────────────────────────


def test_pie_llm_against_real_pie_serve(pie_server):
    uri = pie_server
    llm = PieLLM(
        model="default",                       # matches dummy config's [[model]].name
        pie_uri=uri,
        pie_username="local-dev",
        pie_inferlet=INFERLET_NAME,
        num_retries=1,
        retry_min_wait=0,
        retry_max_wait=0,
        retry_multiplier=1.0,
    )

    response = llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hello")])],
        max_tokens=16,
        temperature=0.0,
    )

    # Dummy driver returns random tokens — we don't care about content,
    # only that the full wire delivered an LLMResponse with assistant text.
    assert response.message.role == "assistant"
    text = "".join(p.text for p in response.message.content if hasattr(p, "text"))
    assert isinstance(text, str)
    # Some tokens must have been emitted (the inferlet runs until max_tokens
    # under random sampling — stop tokens are rarely hit).
    assert response.raw_response.usage.completion_tokens > 0

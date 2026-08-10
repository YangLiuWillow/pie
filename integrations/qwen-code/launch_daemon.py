#!/usr/bin/env python3
"""Install chat-completions.wasm and launch it as a Pie HTTP daemon.

The daemon lives in the engine's global registry, so it keeps serving after
this script exits. Point qwen-code (or any OpenAI-SDK client) at
http://127.0.0.1:<port>/v1 afterwards.
"""

import argparse
import asyncio
from pathlib import Path

from pie_client import PieClient

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WASM = REPO / "inferlets/chat-completions/target/wasm32-wasip2/release/chat_completions.wasm"
DEFAULT_MANIFEST = REPO / "inferlets/chat-completions/Pie.toml"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:18080", help="pie serve control URI")
    ap.add_argument("--username", default="local-dev")
    ap.add_argument("--wasm", default=str(DEFAULT_WASM))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--inferlet", default="chat-completions@0.1.0")
    ap.add_argument("--port", type=int, default=8123, help="HTTP port for the daemon")
    args = ap.parse_args()

    async with PieClient(args.uri) as client:
        await client.authenticate(args.username)
        await client.install_program(args.wasm, args.manifest, force_overwrite=True)
        await client.launch_daemon(args.inferlet, args.port)
        print(f"daemon {args.inferlet} serving on http://127.0.0.1:{args.port}")


asyncio.run(main())

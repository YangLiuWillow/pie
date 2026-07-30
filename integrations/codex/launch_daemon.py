#!/usr/bin/env python3
"""Install codex-responses.wasm and launch it as a Pie HTTP daemon.

The daemon lives in the engine's global registry, so it keeps serving after
this script exits.
"""

import argparse
import asyncio

from pie_client import PieClient


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:18080", help="pie serve control URI")
    ap.add_argument("--username", default="local-dev")
    ap.add_argument("--wasm", required=True, help="path to codex_responses.wasm")
    ap.add_argument("--manifest", required=True, help="path to the inferlet's Pie.toml")
    ap.add_argument("--inferlet", default="codex-responses@0.1.0")
    ap.add_argument("--port", type=int, default=8123, help="HTTP port for the daemon")
    args = ap.parse_args()

    async with PieClient(args.uri) as client:
        await client.authenticate(args.username)
        await client.install_program(args.wasm, args.manifest, force_overwrite=True)
        await client.launch_daemon(args.inferlet, args.port)
        print(f"daemon {args.inferlet} serving on http://127.0.0.1:{args.port}")


asyncio.run(main())

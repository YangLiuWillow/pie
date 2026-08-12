"""Install + launch the npr inferlet against a running dev-branch `pie serve`.

Speaks the gateway's JSON-over-WS turn protocol directly (the vendored
pie_client still speaks msgpack, which this gateway does not parse):
  - identity via the `x-pie-identity` trust-edge header at the upgrade,
  - Text frames of ClientMessage JSON in, ServerMessage JSON out,
  - `{"type":"turn_done"}` markers between turns (ignored here),
  - the wasm uploaded as a single chunk (intermediate chunks are not acked
    under the one-turn-per-frame model).

Usage:
    python client.py --input '{"selftest": true}'
    python client.py --input '{"max_new_tokens": 30000, "question": "..."}'
    python client.py --url ws://127.0.0.1:8092/v1/ws --timeout 3600 --input '{}'

Requires: pip install websockets blake3   (both come with `pip install -e client/python`)
"""

import argparse
import asyncio
import json
import pathlib

import blake3
import websockets

HERE = pathlib.Path(__file__).resolve().parent


def print_return(msg: str, full: bool) -> None:
    print("=== RETURN ===", flush=True)
    try:
        parsed = json.loads(msg)
        traj = parsed.pop("trajectory", "")
        print(json.dumps(parsed, indent=2))
        print(f"--- trajectory ({len(traj)} chars) ---")
        print(traj if full else traj[:4000])
    except Exception:
        print(msg if full else msg[:4000])


async def run(args: argparse.Namespace) -> None:
    wasm = (HERE / "target/wasm32-wasip2/release/npr.wasm").read_bytes()
    manifest = (HERE / "Pie.toml").read_text()
    extra = json.loads(args.input)

    ws = await websockets.connect(
        args.url, additional_headers={"x-pie-identity": "default/npr"}, max_size=None
    )
    await ws.send(
        json.dumps(
            {
                "type": "add_program",
                "corr_id": 1,
                "program_hash": blake3.blake3(wasm).hexdigest(),
                "manifest": manifest,
                "force_overwrite": True,
                "chunk_index": 0,
                "total_chunks": 1,
                "chunk_data": list(wasm),
            }
        )
    )
    async for raw in ws:
        m = json.loads(raw)
        if m.get("type") == "response":
            print(f"[install] ok={m.get('ok')} result={m.get('result')}", flush=True)
            if not m.get("ok"):
                return
            break
        if m.get("type") == "error":
            print(f"[install error] {m.get('message')}", flush=True)
            return

    await ws.send(
        json.dumps(
            {
                "type": "launch_process",
                "corr_id": 2,
                "inferlet": "npr@0.1.0",
                "input": json.dumps(extra),
                "capture_outputs": True,
            }
        )
    )
    async for raw in ws:
        m = json.loads(raw)
        t = m.get("type")
        if t == "response":
            print(f"[launched] ok={m.get('ok')} pid={m.get('result')}", flush=True)
            if not m.get("ok"):
                return
        elif t == "process_event":
            ev = m.get("event")
            val = m.get("value", "")
            if ev in ("stdout", "stderr", "message"):
                print(f"[{ev}] {val}", end="" if val.endswith("\n") else "\n", flush=True)
            elif ev == "return":
                print_return(val, args.full)
                return
            elif ev == "error":
                print(f"=== PROCESS ERROR ===\n{val}", flush=True)
                return
        elif t == "error":
            print(f"[ws error] {m.get('message')}", flush=True)
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8092/v1/ws")
    ap.add_argument("--input", default="{}", help="inferlet input JSON")
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--full", action="store_true", help="print the full trajectory")
    args = ap.parse_args()
    try:
        asyncio.run(asyncio.wait_for(run(args), timeout=args.timeout))
    except asyncio.TimeoutError:
        print(f"CLIENT TIMEOUT after {args.timeout}s")


if __name__ == "__main__":
    main()

"""Run TerminalBench tasks inside Apptainer containers with a PIE agent.

Converts each task's Dockerfile to an Apptainer definition file, builds a SIF
image, then runs the openhands-agent inferlet against an Apptainer instance
(so agent state persists across exec calls).

Usage:
    python apptainer_runner.py --task-dir /path/to/terminal-bench/original-tasks/task-name
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import yaml

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 10_000
BASH_TIMEOUT_S = 300

SIF_CACHE_DIR = Path(
    os.environ.get(
        "TBENCH_SIF_CACHE",
        "/nfs/roberts/scratch/pi_ql324/ly337/tbench-images",
    )
)


# ─── Task model ───────────────────────────────────────────────────────


class Task:
    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        self.name = task_dir.name
        meta = yaml.safe_load((task_dir / "task.yaml").read_text())
        self.instruction = meta["instruction"]
        self.timeout = meta.get("max_agent_timeout_sec", 900)
        self.test_timeout = meta.get("max_test_timeout_sec", 180)
        self.difficulty = meta.get("difficulty", "unknown")
        self.category = meta.get("category", "unknown")

    @property
    def dockerfile_dir(self) -> Path:
        client_dir = self.task_dir / "client"
        if (client_dir / "Dockerfile").exists():
            return client_dir
        return self.task_dir


# ─── Dockerfile → Apptainer def conversion ───────────────────────────


def _dockerfile_to_def(dockerfile: Path, build_context: Path) -> str:
    """Convert a Dockerfile to an Apptainer definition file string."""
    lines = dockerfile.read_text().splitlines()
    from_image = None
    post_cmds: list[str] = []
    env_vars: list[tuple[str, str]] = []
    files_entries: list[tuple[str, str]] = []
    workdir = "/app"
    arg_defaults: dict[str, str] = {}

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        while line.endswith("\\") and i + 1 < len(lines):
            i += 1
            line = line[:-1] + " " + lines[i].strip()
        stripped = line.strip()
        i += 1

        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split(None, 1)
        instruction = parts[0].upper()
        rest = parts[1] if len(parts) > 1 else ""

        if instruction == "FROM":
            tokens = rest.split()
            if tokens and tokens[0].startswith("--platform"):
                tokens = tokens[1:]
            if tokens:
                img = tokens[0]
                from_image = img.split(" AS ")[0].split(" as ")[0].strip()

        elif instruction == "ARG":
            if "=" in rest:
                k, v = rest.split("=", 1)
                arg_defaults[k.strip()] = v.strip().strip('"').strip("'")
            else:
                arg_defaults[rest.strip()] = ""

        elif instruction == "RUN":
            cmd = rest
            for k, v in arg_defaults.items():
                cmd = cmd.replace(f"${{{k}}}", v).replace(f"${k}", v)
            post_cmds.append(f"cd {workdir} && {cmd}")

        elif instruction == "COPY":
            try:
                copy_parts = shlex.split(rest)
            except ValueError:
                copy_parts = rest.split()
            flags = [p for p in copy_parts if p.startswith("--")]
            args = [p for p in copy_parts if not p.startswith("--")]
            if len(args) >= 2:
                srcs = args[:-1]
                dst_raw = args[-1]
                if not dst_raw.startswith("/"):
                    dst_raw = f"{workdir}/{dst_raw}".replace("/./", "/")
                for src in srcs:
                    host_path = build_context / src
                    if host_path.exists():
                        if host_path.is_dir():
                            files_entries.append((str(host_path.resolve()), dst_raw))
                        else:
                            if dst_raw.endswith("/"):
                                dest = f"{dst_raw}{host_path.name}"
                            else:
                                dest = dst_raw
                            files_entries.append((str(host_path.resolve()), dest))
                    else:
                        post_cmds.append(f"echo 'COPY src not found: {src}'")

        elif instruction == "WORKDIR":
            workdir = rest.strip()
            post_cmds.append(f"mkdir -p {workdir}")

        elif instruction == "ENV":
            m = re.match(r"(\w+)[= ](.+)", rest)
            if m:
                env_vars.append((m.group(1), m.group(2).strip().strip('"')))

        elif instruction in ("CMD", "ENTRYPOINT", "EXPOSE", "VOLUME", "USER",
                             "LABEL", "SHELL", "STOPSIGNAL", "HEALTHCHECK"):
            pass

    if not from_image:
        raise ValueError(f"No FROM instruction found in {dockerfile}")

    def_lines = [
        f"Bootstrap: docker",
        f"From: {from_image}",
        "",
    ]

    if files_entries:
        def_lines.append("%files")
        for host, container in files_entries:
            def_lines.append(f"    {host} {container}")
        def_lines.append("")

    if env_vars:
        def_lines.append("%environment")
        for k, v in env_vars:
            def_lines.append(f"    export {k}=\"{v}\"")
        def_lines.append("")

    if post_cmds:
        def_lines.append("%post")
        for cmd in post_cmds:
            def_lines.append(f"    {cmd}")
        def_lines.append("")

    return "\n".join(def_lines)


def _get_sif_path(task_name: str) -> Path:
    return SIF_CACHE_DIR / f"{task_name}.sif"


def build_task_image(task: Task, force: bool = False) -> Path:
    """Build an Apptainer SIF image from a TerminalBench task's Dockerfile."""
    sif = _get_sif_path(task.name)
    if sif.exists() and not force:
        logger.info("Using cached SIF: %s", sif)
        return sif

    SIF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dockerfile_dir = task.dockerfile_dir
    dockerfile = dockerfile_dir / "Dockerfile"
    logger.info("Building SIF for %s from %s", task.name, dockerfile)

    def_content = _dockerfile_to_def(dockerfile, dockerfile_dir)
    logger.debug("Generated .def:\n%s", def_content)

    sif_tmp = sif.with_suffix(".sif.tmp")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".def", dir=str(SIF_CACHE_DIR), delete=False
    ) as f:
        f.write(def_content)
        def_path = Path(f.name)

    try:
        result = subprocess.run(
            ["apptainer", "build", "--force", str(sif_tmp), str(def_path)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            logger.error("apptainer build failed:\n%s\n%s", result.stdout, result.stderr)
            raise RuntimeError(f"apptainer build failed for {task.name}: {result.stderr[-500:]}")
        sif_tmp.rename(sif)
    finally:
        def_path.unlink(missing_ok=True)
        sif_tmp.unlink(missing_ok=True)

    logger.info("Built SIF: %s (%.1f MB)", sif, sif.stat().st_size / 1e6)
    return sif


# ─── Apptainer instance management ───────────────────────────────────


class ApptainerInstance:
    """Manages an Apptainer instance with persistent writable state."""

    def __init__(self, sif_path: Path, task: Task, working_dir: str = "/app"):
        self.sif_path = sif_path
        self.task = task
        self.working_dir = working_dir
        self.instance_name = f"tbench-{task.name}-{uuid.uuid4().hex[:8]}"
        self._started = False

    def start(self) -> None:
        tests_dir = self.task.task_dir / "tests"
        run_tests = self.task.task_dir / "run-tests.sh"
        binds = []
        if tests_dir.exists():
            binds.extend(["--bind", f"{tests_dir}:/tests"])
        if run_tests.exists():
            binds.extend(["--bind", f"{run_tests}:/run-tests.sh"])

        cmd = [
            "apptainer", "instance", "start",
            "--writable-tmpfs",
            *binds,
            str(self.sif_path),
            self.instance_name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start instance {self.instance_name}: {result.stderr}"
            )
        self._started = True
        logger.info("Started instance %s", self.instance_name)

    def stop(self) -> None:
        if not self._started:
            return
        subprocess.run(
            ["apptainer", "instance", "stop", self.instance_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self._started = False
        logger.info("Stopped instance %s", self.instance_name)

    def exec(self, command: str, timeout: int = BASH_TIMEOUT_S) -> dict[str, Any]:
        if not command.strip():
            return {"observation": "(empty command)", "exit_code": 1}

        cmd = [
            "apptainer", "exec",
            "--pwd", self.working_dir,
            f"instance://{self.instance_name}",
            "bash", "-c", command,
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"observation": f"Command timed out after {timeout}s", "exit_code": 124}

        output = proc.stdout + proc.stderr
        if len(output) > MAX_OUTPUT_CHARS:
            half = MAX_OUTPUT_CHARS // 2
            output = (
                output[:half]
                + f"\n\n... ({len(output) - MAX_OUTPUT_CHARS} chars truncated) ...\n\n"
                + output[-half:]
            )
        return {"observation": output, "exit_code": proc.returncode}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


# ─── Tool server for agent ───────────────────────────────────────────


def start_container_tool_server(
    instance: ApptainerInstance,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> tuple[HTTPServer, int]:
    """HTTP tool server that proxies actions into an Apptainer instance."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self._read_body()
            try:
                req = json.loads(body)
            except json.JSONDecodeError as e:
                self._reply(400, {"error": f"bad json: {e}"})
                return

            action = req.get("action", "")
            try:
                if action == "bash":
                    result = instance.exec(req.get("command", ""))
                elif action == "read_file":
                    path = req.get("path", "")
                    if not path:
                        result = {"observation": "Error: path required", "exit_code": 1}
                    else:
                        result = instance.exec(f"cat -n {shlex.quote(path)}")
                elif action == "edit":
                    result = self._handle_edit(req)
                elif action == "finish":
                    result = {"observation": "Task finished.", "exit_code": 0}
                else:
                    result = {
                        "observation": f"Unknown action: {action!r}",
                        "exit_code": 1,
                    }
            except Exception as e:
                result = {"observation": f"Error: {e}", "exit_code": 1}

            self._reply(200, result)

        def _handle_edit(self, req: dict) -> dict[str, Any]:
            fpath = req.get("path", "")
            old_str = req.get("old_str", "")
            new_str = req.get("new_str", "")
            if not fpath:
                return {"observation": "Error: path required", "exit_code": 1}

            b64_path = base64.b64encode(fpath.encode()).decode()
            b64_old = base64.b64encode(old_str.encode()).decode()
            b64_new = base64.b64encode(new_str.encode()).decode()

            script = (
                "import sys, base64\n"
                f"path = base64.b64decode('{b64_path}').decode()\n"
                f"old = base64.b64decode('{b64_old}').decode()\n"
                f"new = base64.b64decode('{b64_new}').decode()\n"
                "import os\n"
                "if not old:\n"
                "    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)\n"
                "    open(path, 'w').write(new)\n"
                "    print('File created: ' + path)\n"
                "    sys.exit(0)\n"
                "content = open(path).read()\n"
                "count = content.count(old)\n"
                "if count == 0:\n"
                "    print('Error: old_str not found in ' + path)\n"
                "    sys.exit(1)\n"
                "if count > 1:\n"
                "    print(f'Error: old_str found {count} times (must be unique)')\n"
                "    sys.exit(1)\n"
                "open(path, 'w').write(content.replace(old, new, 1))\n"
                "print('File edited: ' + path)\n"
            )
            b64_script = base64.b64encode(script.encode()).decode()
            cmd = f"echo {b64_script} | base64 -d | python3"
            return instance.exec(cmd)

        def _read_body(self) -> bytes:
            te = self.headers.get("Transfer-Encoding", "").lower()
            if "chunked" in te:
                buf = bytearray()
                while True:
                    line = self.rfile.readline().strip()
                    chunk_len = int(line, 16)
                    if chunk_len == 0:
                        self.rfile.readline()
                        break
                    buf.extend(self.rfile.read(chunk_len))
                    self.rfile.readline()
                return bytes(buf)
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length)

        def _reply(self, code: int, body: dict[str, Any]):
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass

    server = HTTPServer((host, port), Handler)
    actual_port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port


# ─── Agent runner ─────────────────────────────────────────────────────


async def run_agent(
    pie_uri: str,
    pie_inferlet: str,
    task: Task,
    tool_server_url: str,
    *,
    max_steps: int = 50,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    """Run the openhands-agent inferlet on a TerminalBench task."""
    from pie_client import Event, PieClient

    input_payload = {
        "task": task.instruction,
        "tool_server_url": tool_server_url,
        "max_steps": max_steps,
        "max_tokens_per_step": 4096,
    }

    async with PieClient(pie_uri) as client:
        await client.authenticate("local-dev")
        proc = await client.launch_process(pie_inferlet, input=input_payload)
        stdout_chunks: list[str] = []
        while True:
            event, value = await asyncio.wait_for(proc.recv(), timeout=timeout_s)
            if event == Event.Stdout:
                chunk = (
                    value.decode("utf-8", "replace")
                    if isinstance(value, (bytes, bytearray))
                    else str(value)
                )
                stdout_chunks.append(chunk)
                logger.debug("inferlet: %s", chunk.rstrip())
            elif event == Event.Return:
                if isinstance(value, dict):
                    return value
                s = (
                    value.decode("utf-8", "replace")
                    if isinstance(value, (bytes, bytearray))
                    else str(value)
                )
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    return {"finished": False, "message": s}
            elif event == Event.Error:
                raise RuntimeError(f"Inferlet error: {value!r}")


# ─── Test runner ──────────────────────────────────────────────────────


def run_tests(
    task: Task,
    instance: ApptainerInstance,
) -> dict[str, Any]:
    """Run the task's test script inside the instance."""
    test_script = task.task_dir / "run-tests.sh"
    if not test_script.exists():
        return {"passed": False, "error": "No run-tests.sh found"}

    result = instance.exec(
        "TEST_DIR=/tests bash /run-tests.sh",
        timeout=int(task.test_timeout),
    )

    passed = result["exit_code"] == 0
    return {
        "passed": passed,
        "exit_code": result["exit_code"],
        "output": result["observation"],
    }


# ─── Full pipeline ────────────────────────────────────────────────────


def evaluate_task(
    task: Task,
    pie_uri: str = "ws://127.0.0.1:8080",
    pie_inferlet: str = "openhands-agent@0.1.0",
    max_steps: int = 50,
) -> dict[str, Any]:
    """Build image, run agent, score results for one task."""
    t0 = time.monotonic()

    sif_path = build_task_image(task)

    with ApptainerInstance(sif_path, task) as instance:
        server, port = start_container_tool_server(instance)
        try:
            agent_result = asyncio.run(
                run_agent(
                    pie_uri,
                    pie_inferlet,
                    task,
                    f"http://127.0.0.1:{port}",
                    max_steps=max_steps,
                    timeout_s=task.timeout,
                )
            )
        finally:
            server.shutdown()

        test_result = run_tests(task, instance)

    wall_clock = time.monotonic() - t0

    return {
        "task_id": task.name,
        "difficulty": task.difficulty,
        "category": task.category,
        "passed": test_result["passed"],
        "agent_finished": agent_result.get("finished", False),
        "agent_steps": agent_result.get("steps", 0),
        "wall_clock_s": wall_clock,
        "test_output": test_result.get("output", "")[:2000],
    }


def evaluate_tasks(
    task_dirs: list[Path],
    pie_uri: str = "ws://127.0.0.1:8080",
    pie_inferlet: str = "openhands-agent@0.1.0",
    max_steps: int = 50,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate multiple tasks sequentially."""
    results = []
    for i, task_dir in enumerate(task_dirs, 1):
        task = Task(task_dir)
        logger.info(
            "[%d/%d] %s (%s, %s)",
            i,
            len(task_dirs),
            task.name,
            task.difficulty,
            task.category,
        )
        try:
            result = evaluate_task(
                task,
                pie_uri=pie_uri,
                pie_inferlet=pie_inferlet,
                max_steps=max_steps,
            )
        except Exception as e:
            logger.exception("Failed: %s", task.name)
            result = {
                "task_id": task.name,
                "passed": False,
                "error": f"{type(e).__name__}: {e}",
            }
        results.append(result)
        logger.info(
            "  -> %s (%.1fs)",
            "PASS" if result.get("passed") else "FAIL",
            result.get("wall_clock_s", 0),
        )

        if output_path:
            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")

    passed = sum(1 for r in results if r.get("passed"))
    logger.info(
        "Results: %d/%d passed (%.0f%%)",
        passed, len(results), 100 * passed / max(len(results), 1),
    )
    return results


# ─── CLI ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Run TerminalBench tasks with PIE + Apptainer",
    )
    p.add_argument("--task-dir", type=Path, action="append", required=True)
    p.add_argument("--pie-uri", default="ws://127.0.0.1:8080")
    p.add_argument("--pie-inferlet", default="openhands-agent@0.1.0")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--output", "-o", type=Path, default=None)
    args = p.parse_args()

    results = evaluate_tasks(
        args.task_dir,
        pie_uri=args.pie_uri,
        pie_inferlet=args.pie_inferlet,
        max_steps=args.max_steps,
        output_path=args.output,
    )

    print(json.dumps(results, indent=2))

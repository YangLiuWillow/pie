"""Tests for the Pattern A tool server."""

from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

import pytest

from tool_server import start_tool_server


@pytest.fixture()
def server_and_dir():
    with tempfile.TemporaryDirectory(prefix="toolsrv-test-") as tmpdir:
        server, port = start_tool_server(tmpdir)
        yield server, port, tmpdir
        server.shutdown()


def _post(port: int, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/execute",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


class TestBash:
    def test_echo(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "bash", "command": "echo hello"})
        assert result["exit_code"] == 0
        assert "hello" in result["observation"]

    def test_cwd(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        result = _post(port, {"action": "bash", "command": "pwd"})
        assert result["exit_code"] == 0
        assert tmpdir in result["observation"]

    def test_nonzero_exit(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "bash", "command": "exit 42"})
        assert result["exit_code"] == 42

    def test_empty_command(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "bash", "command": ""})
        assert result["exit_code"] == 1

    def test_stderr_captured(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "bash", "command": "echo err >&2"})
        assert "err" in result["observation"]


class TestEdit:
    def test_create_file(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        result = _post(port, {
            "action": "edit",
            "path": "hello.txt",
            "old_str": "",
            "new_str": "hello world",
        })
        assert result["exit_code"] == 0
        assert (Path(tmpdir) / "hello.txt").read_text() == "hello world"

    def test_str_replace(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "test.py").write_text("x = 1\ny = 2\n")
        result = _post(port, {
            "action": "edit",
            "path": "test.py",
            "old_str": "x = 1",
            "new_str": "x = 42",
        })
        assert result["exit_code"] == 0
        assert "x = 42" in (Path(tmpdir) / "test.py").read_text()

    def test_ambiguous_match(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "dup.txt").write_text("a\na\n")
        result = _post(port, {
            "action": "edit",
            "path": "dup.txt",
            "old_str": "a",
            "new_str": "b",
        })
        assert result["exit_code"] == 1
        assert "2 times" in result["observation"]

    def test_no_match(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "empty.txt").write_text("")
        result = _post(port, {
            "action": "edit",
            "path": "empty.txt",
            "old_str": "not here",
            "new_str": "anything",
        })
        assert result["exit_code"] == 1
        assert "not found" in result["observation"]

    def test_missing_file(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {
            "action": "edit",
            "path": "nonexistent.txt",
            "old_str": "x",
            "new_str": "y",
        })
        assert result["exit_code"] == 1

    def test_create_nested_dirs(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        result = _post(port, {
            "action": "edit",
            "path": "a/b/c.txt",
            "old_str": "",
            "new_str": "nested",
        })
        assert result["exit_code"] == 0
        assert (Path(tmpdir) / "a" / "b" / "c.txt").read_text() == "nested"


class TestMisc:
    def test_unknown_action(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "unknown"})
        assert result["exit_code"] == 1

    def test_finish_action(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "finish"})
        assert result["exit_code"] == 0

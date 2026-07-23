"""Tests for the autoresearch tool server."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from pathlib import Path

import pytest

# Add parent to path for import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_server import start_tool_server


@pytest.fixture()
def server_and_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        server, port = start_tool_server(tmpdir)
        yield server, port, tmpdir
        server.shutdown()


def _post(port: int, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/execute",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


class TestBashAction:
    def test_echo(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "bash", "command": "echo hello"})
        assert result["exit_code"] == 0
        assert "hello" in result["observation"]

    def test_empty_command(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "bash", "command": ""})
        assert result["exit_code"] == 1

    def test_working_dir(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        result = _post(port, {"action": "bash", "command": "pwd"})
        assert tmpdir in result["observation"]

    def test_nonzero_exit(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "bash", "command": "exit 42"})
        assert result["exit_code"] == 42


class TestEditAction:
    def test_create_file(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        result = _post(port, {
            "action": "edit",
            "path": "train.py",
            "old_str": "",
            "new_str": "print('hello')\n",
        })
        assert result["exit_code"] == 0
        assert "File created" in result["observation"]
        assert (Path(tmpdir) / "train.py").read_text() == "print('hello')\n"

    def test_replace_text(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "train.py").write_text("lr = 1e-3\nbatch_size = 32\n")
        result = _post(port, {
            "action": "edit",
            "path": "train.py",
            "old_str": "lr = 1e-3",
            "new_str": "lr = 3e-4",
        })
        assert result["exit_code"] == 0
        assert "File edited" in result["observation"]
        content = (Path(tmpdir) / "train.py").read_text()
        assert "lr = 3e-4" in content
        assert "lr = 1e-3" not in content

    def test_old_str_not_found(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "train.py").write_text("lr = 1e-3\n")
        result = _post(port, {
            "action": "edit",
            "path": "train.py",
            "old_str": "lr = 1e-4",
            "new_str": "lr = 3e-4",
        })
        assert result["exit_code"] == 1
        assert "not found" in result["observation"]

    def test_ambiguous_match(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "train.py").write_text("x = 1\nx = 1\n")
        result = _post(port, {
            "action": "edit",
            "path": "train.py",
            "old_str": "x = 1",
            "new_str": "x = 2",
        })
        assert result["exit_code"] == 1
        assert "2 times" in result["observation"]

    def test_syntax_error_warning(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "train.py").write_text("x = 1\n")
        result = _post(port, {
            "action": "edit",
            "path": "train.py",
            "old_str": "x = 1",
            "new_str": "x = [",
        })
        assert result["exit_code"] == 0
        assert "SyntaxError" in result["observation"]

    def test_no_path(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {
            "action": "edit",
            "path": "",
            "old_str": "",
            "new_str": "hello",
        })
        assert result["exit_code"] == 1


class TestReadFileAction:
    def test_read_existing(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "train.py").write_text("import torch\nmodel = None\n")
        result = _post(port, {"action": "read_file", "path": "train.py"})
        assert result["exit_code"] == 0
        assert "import torch" in result["observation"]
        # Should have line numbers
        assert "1 |" in result["observation"]

    def test_read_missing(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "read_file", "path": "nonexistent.py"})
        assert result["exit_code"] == 1
        assert "not found" in result["observation"]

    def test_no_path(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "read_file", "path": ""})
        assert result["exit_code"] == 1


class TestFinishAction:
    def test_finish(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "finish"})
        assert result["exit_code"] == 0
        assert "finished" in result["observation"].lower()


class TestUnknownAction:
    def test_unknown(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "teleport"})
        assert result["exit_code"] == 1
        assert "Unknown action" in result["observation"]

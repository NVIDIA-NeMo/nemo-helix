# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO
from unittest.mock import Mock

import httpx
import pytest
from nhx.platform_runner.community_plugins import runtime
from nhx.platform_runner.community_plugins.runtime import LocalTrustedRuntime, PreparedPlugin, create_runtime


@pytest.fixture
def wheel(tmp_path: Path) -> Path:
    # Preparation tests exercise artifact verification/commands, not pip's wheel parser.
    wheel = tmp_path / "sample-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel fixture")
    return wheel


def digest(wheel: Path) -> str:
    return hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_explicit_trust_required(tmp_path: Path, wheel: Path) -> None:
    backend = LocalTrustedRuntime(tmp_path / "store")
    with pytest.raises(PermissionError, match="explicit trust_code"):
        backend.prepare("sample", wheel, digest(wheel))
    assert not backend.root.exists()


@pytest.mark.parametrize("name", ["../escape", "a/b", "UPPER", "", "--option"])
def test_invalid_name(tmp_path: Path, wheel: Path, name: str) -> None:
    backend = LocalTrustedRuntime(tmp_path / "store")
    with pytest.raises(ValueError, match="kebab-case"):
        backend.prepare(name, wheel, digest(wheel), trust_code=True)
    assert not backend.root.exists()


def test_hash_mismatch_never_runs_installer(tmp_path: Path, wheel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = LocalTrustedRuntime(tmp_path / "store")
    install = Mock()
    monkeypatch.setattr(backend, "_install_command", install)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        backend.prepare("sample", wheel, "0" * 64, trust_code=True)
    install.assert_not_called()
    assert list(backend.root.iterdir()) == []


def test_prepare_commands_and_cleanup(tmp_path: Path, wheel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = LocalTrustedRuntime(tmp_path / "store")
    commands: list[list[str]] = []

    def install(command: list[str], log: BinaryIO, *, output: BinaryIO | None = None) -> None:
        commands.append(command)
        if output is not None:
            output.write(b"sample==1.0\n")

    monkeypatch.setattr(backend, "_install_command", install)
    revision = backend.prepare("sample", wheel, digest(wheel), trust_code=True)
    assert revision.sha256 == digest(wheel)
    assert (revision.directory / wheel.name).read_bytes() == wheel.read_bytes()
    assert revision.directory.stat().st_mode & 0o077 == 0
    assert commands[0] == ["uv", "venv", "--python", sys.executable, str(revision.directory / "venv")]
    assert commands[1][:7] == ["uv", "pip", "install", "--python", str(revision.python), "--only-binary", ":all:"]
    assert str(revision.directory / wheel.name) in commands[1]
    assert "--system-site-packages" not in commands[0]
    assert (revision.directory / "resolved-requirements.txt").read_text() == "sample==1.0\n"
    backend.remove(revision)
    assert not revision.directory.exists()


def test_failed_install_is_removed(tmp_path: Path, wheel: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = LocalTrustedRuntime(tmp_path / "store")

    def fail(command: list[str], log: BinaryIO) -> None:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(backend, "_install_command", fail)
    with pytest.raises(subprocess.CalledProcessError):
        backend.prepare("sample", wheel, digest(wheel), trust_code=True)
    assert list(backend.root.iterdir()) == []


@pytest.mark.parametrize("kind", ["isolated", "kubernetes"])
def test_unsupported_backends_never_fall_back(tmp_path: Path, wheel: Path, kind: runtime.RuntimeKind) -> None:
    backend = create_runtime(kind, tmp_path / "store")
    revision = PreparedPlugin("sample", "0" * 64, tmp_path / "revision")
    for operation in [
        lambda: backend.prepare("sample", wheel, digest(wheel), trust_code=True),
        lambda: backend.start(revision),
        lambda: backend.remove(revision),
    ]:
        with pytest.raises(NotImplementedError, match="no trusted fallback"):
            operation()
    assert not (tmp_path / "store").exists()


def test_cannot_remove_outside_store(tmp_path: Path) -> None:
    backend = LocalTrustedRuntime(tmp_path / "store")
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="direct child"):
        backend.remove(PreparedPlugin("sample", "0" * 64, outside))
    assert outside.exists()
    backend.root.mkdir()
    (backend.root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="direct child"):
        backend.remove(PreparedPlugin("sample", "0" * 64, backend.root / "link"))


@pytest.fixture
def host_revision(tmp_path: Path) -> PreparedPlugin:
    """Run the real host/protocol with a fixture service, without network installs.

    A shim uses the test environment's Python and supplies fixture entry-point
    metadata in the CHILD only. This tests supervision, not dependency resolution.
    """
    directory = tmp_path / "store" / "revision"
    (directory / "venv" / "bin").mkdir(parents=True)
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_text("""
import importlib.metadata
import runpy
import sys
from types import SimpleNamespace
from fastapi import APIRouter
from nemo_helix_plugin.service import NemoService, RouterSpec

class Sample(NemoService):
    name = "sample"
    def get_routers(self):
        router = APIRouter()
        @router.post("/echo")
        async def echo(body: dict):
            return body
        return [RouterSpec(router)]

original = importlib.metadata.entry_points
def entries(**kwargs):
    if kwargs.get("group") == "nemo.services":
        return [SimpleNamespace(name="sample", load=lambda: Sample)]
    return original(**kwargs)
importlib.metadata.entry_points = entries
host = sys.argv.pop(2)
sys.argv.pop(1)
sys.argv[0] = host
runpy.run_path(host, run_name="__main__")
""")
    python = directory / "venv" / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -I {shlex.quote(str(bootstrap))} "$@"\n')
    python.chmod(0o700)
    return PreparedPlugin("sample", "0" * 64, directory)


@pytest.mark.skipif(os.name != "posix", reason="Unix sockets/process groups")
def test_real_host_start_invoke_stop_and_remove(host_revision: PreparedPlugin) -> None:
    backend = LocalTrustedRuntime(host_revision.directory.parent, startup_timeout=15)
    with backend:
        running = backend.start(host_revision)
        assert running.socket_directory.stat().st_mode & 0o077 == 0
        assert running.process.poll() is None
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=str(running.socket_path)), base_url="http://plugin"
        ) as client:
            assert client.post("/apis/sample/echo", json={"query": "hello"}).json() == {"query": "hello"}
            assert client.get("/_helix/descriptor").json()["name"] == "sample"
        with pytest.raises(ValueError, match="already running"):
            backend.start(host_revision)
        with pytest.raises(ValueError, match="Stop"):
            backend.remove(host_revision)
    assert running.process.poll() is not None
    assert not running.socket_directory.exists()
    backend.remove(host_revision)
    assert not host_revision.directory.exists()


def test_failed_start_cleans_up(host_revision: PreparedPlugin) -> None:
    python = host_revision.python
    python.write_text("#!/bin/sh\nexit 23\n")
    backend = LocalTrustedRuntime(host_revision.directory.parent, startup_timeout=5)
    with pytest.raises(RuntimeError, match="exited before readiness"):
        backend.start(host_revision)
    assert backend._running == {}


def test_start_timeout_cleans_up(host_revision: PreparedPlugin) -> None:
    backend = LocalTrustedRuntime(host_revision.directory.parent, startup_timeout=0)
    with pytest.raises(TimeoutError, match="did not become ready"):
        backend.start(host_revision)
    assert backend._running == {}

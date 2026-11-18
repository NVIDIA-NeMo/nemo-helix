# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime boundary for externally installed HTTP plugins.

``local-trusted`` provides dependency/crash isolation, NOT security isolation.
Installation and execution have the privileges and environment of the operator.
This module is a building block, not a remotely accessible installation API.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, NoReturn, Protocol

RuntimeKind = Literal["local-trusted", "isolated", "kubernetes"]
_NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PreparedPlugin:
    """An installed revision. Paths are server-owned, never client-supplied."""

    name: str
    sha256: str
    directory: Path

    @property
    def python(self) -> Path:
        return self.directory / "venv" / "bin" / "python"


@dataclass
class RunningPlugin:
    """An owned process handle; never reconstruct ownership from a saved PID."""

    revision: PreparedPlugin
    process: subprocess.Popen[bytes]
    socket_directory: Path

    @property
    def socket_path(self) -> Path:
        return self.socket_directory / "http.sock"


class PluginRuntime(Protocol):
    def prepare(self, name: str, wheel: Path, sha256: str, *, trust_code: bool = False) -> PreparedPlugin: ...

    def start(self, revision: PreparedPlugin) -> RunningPlugin: ...

    def stop(self, running: RunningPlugin) -> None: ...

    def remove(self, revision: PreparedPlugin) -> None: ...


class UnsupportedRuntime:
    """Placeholder: isolated execution must never silently fall back to trusted."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def _fail(self) -> NoReturn:
        raise NotImplementedError(f"Plugin runtime {self.kind!r} is not implemented; no trusted fallback is permitted")

    def prepare(self, name: str, wheel: Path, sha256: str, *, trust_code: bool = False) -> PreparedPlugin:
        self._fail()

    def start(self, revision: PreparedPlugin) -> RunningPlugin:
        self._fail()

    def stop(self, running: RunningPlugin) -> None:
        self._fail()

    def remove(self, revision: PreparedPlugin) -> None:
        self._fail()


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: Path) -> None:
        super().__init__("plugin", timeout=0.2)
        self.path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.path))


class LocalTrustedRuntime:
    """POSIX-only, one virtualenv and HTTP subprocess per revision.

    The caller owns lifecycle serialization and must stop workloads before
    removal. Context exit stops every child started by this runtime instance.
    No plugin is discovered/imported in the controlling interpreter.
    """

    def __init__(
        self,
        root: Path,
        *,
        uv: str = "uv",
        startup_timeout: float = 30,
        stop_timeout: float = 5,
        install_timeout: float = 300,
    ) -> None:
        if os.name != "posix":
            raise NotImplementedError("local-trusted requires POSIX process groups and Unix sockets")
        self.root = root.resolve()
        self.uv = uv
        self.startup_timeout = startup_timeout
        self.stop_timeout = stop_timeout
        self.install_timeout = install_timeout
        self._running: dict[Path, RunningPlugin] = {}

    def __enter__(self) -> LocalTrustedRuntime:
        return self

    def __exit__(self, *args: object) -> None:
        for running in list(self._running.values()):
            self.stop(running)

    def _check_revision(self, revision: PreparedPlugin) -> None:
        # Prevent accidental deletion/launch outside the runtime's private store.
        if revision.directory.is_symlink() or revision.directory.resolve().parent != self.root:
            raise ValueError("Revision must be a direct child of this runtime's artifact store")
        if not _NAME.fullmatch(revision.name) or not _DIGEST.fullmatch(revision.sha256):
            raise ValueError("Invalid revision identity")

    def prepare(self, name: str, wheel: Path, sha256: str, *, trust_code: bool = False) -> PreparedPlugin:
        if not trust_code:
            raise PermissionError(
                "local-trusted executes third-party code with the Helix user's filesystem, network, "
                "and credential access; explicit trust_code=True is required"
            )
        if not _NAME.fullmatch(name):
            raise ValueError("Plugin name must be lowercase kebab-case")
        if not _DIGEST.fullmatch(sha256):
            raise ValueError("A lowercase SHA-256 digest is required")
        if wheel.suffix != ".whl" or not wheel.is_file():
            raise ValueError("Expected a local wheel file; URLs and source distributions are not accepted")

        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.root))
        revision = PreparedPlugin(name, sha256, directory)
        try:
            artifact = directory / wheel.name
            digest = hashlib.sha256()
            with wheel.open("rb") as source, artifact.open("xb") as destination:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    destination.write(chunk)
            if not hmac.compare_digest(digest.hexdigest(), sha256):
                raise ValueError("Wheel SHA-256 mismatch")

            # Do not move a venv after creating it: script shebangs contain absolute paths.
            with (directory / "install.log").open("wb") as log:
                self._install_command([self.uv, "venv", "--python", sys.executable, str(directory / "venv")], log)
                self._install_command(
                    [
                        self.uv,
                        "pip",
                        "install",
                        "--python",
                        str(revision.python),
                        "--only-binary",
                        ":all:",
                        str(artifact),
                        "fastapi",
                        "uvicorn",
                        "nemo-helix-plugin",
                    ],
                    log,
                )
                with (directory / "resolved-requirements.txt").open("wb") as resolved:
                    self._install_command(
                        [self.uv, "pip", "freeze", "--python", str(revision.python)], log, output=resolved
                    )
            return revision
        except BaseException:
            shutil.rmtree(directory)
            raise

    def _install_command(self, command: list[str], log: BinaryIO, *, output: BinaryIO | None = None) -> None:
        # A separate session also allows cleanup of installer descendants on timeout.
        with subprocess.Popen(command, stdout=output or log, stderr=log, start_new_session=True) as process:
            try:
                returncode = process.wait(timeout=self.install_timeout)
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
            finally:
                self._terminate_group(process)

    def start(self, revision: PreparedPlugin) -> RunningPlugin:
        self._check_revision(revision)
        if revision.directory in self._running:
            raise ValueError("Revision is already running")
        # Keep the socket path short even when the artifact store has a long path.
        socket_directory = Path(tempfile.mkdtemp(prefix="nhx-plugin-"))
        try:
            with (revision.directory / "host.log").open("ab") as log:
                process = subprocess.Popen(
                    [
                        str(revision.python),
                        "-I",
                        str(Path(__file__).with_name("host.py")),
                        "--name",
                        revision.name,
                        "--socket",
                        str(socket_directory / "http.sock"),
                    ],
                    cwd=revision.directory,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except BaseException:
            shutil.rmtree(socket_directory)
            raise
        running = RunningPlugin(revision, process, socket_directory)
        self._running[revision.directory] = running
        try:
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Plugin host exited before readiness; inspect host.log")
                connection = _UnixHTTPConnection(running.socket_path)
                try:
                    connection.request("GET", "/_helix/ready")
                    if connection.getresponse().status == 200:
                        return running
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    connection.close()
                time.sleep(0.05)
            raise TimeoutError("Plugin host did not become ready; inspect host.log")
        except BaseException:
            self.stop(running)
            raise

    def _terminate_group(self, process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=self.stop_timeout)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # The leader may have exited while descendants ignored SIGTERM.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def stop(self, running: RunningPlugin) -> None:
        owned = self._running.get(running.revision.directory)
        if owned is not running:
            raise ValueError("Process handle is not owned by this runtime or was already stopped")
        self._terminate_group(running.process)
        self._running.pop(running.revision.directory)
        shutil.rmtree(running.socket_directory)

    def remove(self, revision: PreparedPlugin) -> None:
        self._check_revision(revision)
        if revision.directory in self._running:
            raise ValueError("Stop the revision before removing it")
        shutil.rmtree(revision.directory)


def create_runtime(kind: RuntimeKind, root: Path) -> PluginRuntime:
    if kind == "local-trusted":
        return LocalTrustedRuntime(root)
    return UnsupportedRuntime(kind)

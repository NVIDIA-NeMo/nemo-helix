# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate shared Rego using a service-owned native OPA child process."""

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Self

import httpx
from nhx.core.auth.app.embedded_pdp.opa_binary import resolve_opa_binary

POLICY_DIRECTORY = Path(__file__).parent.parent / "policies"
ENTRYPOINTS = ("allow", "has_permissions", "has_role")


class PolicyEngineError(Exception):
    """The policy runtime could not safely produce a decision."""


def _child_environment(directory: Path) -> dict[str, str]:
    # opa.runtime() exposes the child's environment. Do not inherit service keys
    # or OPA_* variables that could enable configuration or remote plugins.
    environment = {"PATH": os.defpath, "HOME": str(directory), "TMPDIR": str(directory)}
    for key in ("GOMAXPROCS", "GOMEMLIMIT"):
        if key in os.environ:
            environment[key] = os.environ[key]
    return environment


def _cleanup(
    owner_pid: int,
    process: subprocess.Popen[bytes] | None,
    client: httpx.Client | None,
    directory: Path,
    timeout: float,
) -> None:
    # A forked worker must never stop its parent's child or remove its socket.
    if os.getpid() != owner_pid:
        return
    try:
        if client is not None:
            client.close()
    finally:
        try:
            if process is not None:
                if process.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.terminate()
                    try:
                        process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(ProcessLookupError):
                            process.kill()
                        process.wait(timeout=timeout)
                else:
                    process.wait(timeout=timeout)
        finally:
            shutil.rmtree(directory, ignore_errors=True)


def _check_sources(binary: Path, directory: Path, timeout: float, environment: dict[str, str]) -> None:
    if not any(directory.glob("*.rego")):
        raise PolicyEngineError(f"No Rego policies found in {directory}")
    try:
        result = subprocess.run(
            [str(binary), "check", str(directory)],
            capture_output=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PolicyEngineError("Unable to validate native OPA policy sources") from exc
    if result.returncode:
        raise PolicyEngineError(f"OPA policy validation failed (exit {result.returncode})")


def check_policy(binary_path: str | None = None, timeout_seconds: float = 15.0) -> None:
    """Validate installed sources without creating a long-lived child."""
    try:
        binary = resolve_opa_binary(binary_path)
        _check_sources(binary, POLICY_DIRECTORY, timeout_seconds, {"PATH": os.defpath})
    except PolicyEngineError:
        raise
    except Exception as exc:
        raise PolicyEngineError(f"Unable to prepare native OPA: {exc}") from exc


class RegoPolicy:
    """Thread-safe OPA supervisor with replaceable data and fail-closed epochs.

    A single instance is shared by the auth service's request threads. OPA reads
    raw Rego sources; no WASM build or Python native extension is involved.
    """

    def __init__(
        self,
        binary_path: str | None = None,
        *,
        startup_timeout_seconds: float = 15.0,
        request_timeout_seconds: float = 5.0,
        shutdown_timeout_seconds: float = 5.0,
        policy_directory: Path | None = None,
    ) -> None:
        if os.name != "posix":
            raise PolicyEngineError("Managed OPA requires Unix sockets; use an external OPA provider on this host")
        if min(startup_timeout_seconds, request_timeout_seconds, shutdown_timeout_seconds) <= 0:
            raise ValueError("OPA timeouts must be positive")
        self._lock = threading.RLock()
        self._owner_pid = os.getpid()
        self._startup_timeout = startup_timeout_seconds
        self._request_timeout = request_timeout_seconds
        self._shutdown_timeout = shutdown_timeout_seconds
        self._policies = policy_directory or POLICY_DIRECTORY
        self._process: subprocess.Popen[bytes] | None = None
        self._client: httpx.Client | None = None
        self._directory: Path | None = None
        self._finalizer: weakref.finalize | None = None
        self._data_loaded = False
        self._generation = 0
        self._closed = False
        try:
            self._binary = resolve_opa_binary(binary_path or os.environ.get("NHX_AUTH_OPA_BINARY_PATH"))
            self._start()
        except Exception as exc:
            self.close()
            if isinstance(exc, PolicyEngineError):
                raise
            raise PolicyEngineError(f"Unable to start native OPA: {exc}") from exc

    def _assert_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise PolicyEngineError("Managed OPA instance cannot be used across a process fork")

    def _stop_resources(self) -> None:
        if self._directory is None:
            return
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        try:
            _cleanup(self._owner_pid, self._process, self._client, self._directory, self._shutdown_timeout)
        except BaseException:
            # Keep ownership if a bounded wait failed, so close()/shutdown can
            # retry rather than silently abandoning an unreaped child.
            self._finalizer = weakref.finalize(
                self,
                _cleanup,
                self._owner_pid,
                self._process,
                self._client,
                self._directory,
                self._shutdown_timeout,
            )
            raise
        self._process = None
        self._client = None
        self._directory = None

    def _start(self) -> None:
        self._assert_owner()
        if self._closed:
            raise PolicyEngineError("OPA runtime has been closed")
        self._data_loaded = False
        self._generation += 1
        self._stop_resources()
        # A short path also fits macOS's Unix socket pathname limit. mkdtemp,
        # unlike TemporaryDirectory, has no fork-unsafe automatic directory cleanup.
        directory = Path(tempfile.mkdtemp(prefix="nhx-opa-", dir="/tmp"))
        process = None
        client = None
        try:
            os.chmod(directory, 0o700)
            environment = _child_environment(directory)
            _check_sources(self._binary, self._policies, self._startup_timeout, environment)
            socket_path = directory / "opa.sock"
            command = [
                str(self._binary),
                "run",
                "--server",
                "--addr",
                f"unix://{socket_path}",
                "--unix-socket-perm",
                "600",
                "--skip-version-check",
                "--log-level",
                "error",
                "--log-level-request",
                "error",
                "--shutdown-grace-period",
                "1",
                str(self._policies),
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                close_fds=True,
            )
            transport = httpx.HTTPTransport(uds=str(socket_path), retries=0, trust_env=False)
            client = httpx.Client(
                base_url="http://opa",
                transport=transport,
                trust_env=False,
                timeout=self._request_timeout,
            )
            deadline = time.monotonic() + self._startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise PolicyEngineError(f"OPA exited during startup (exit {process.returncode})")
                try:
                    response = client.get("/health", timeout=min(0.2, self._startup_timeout))
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.02)
            else:
                raise PolicyEngineError("Timed out waiting for native OPA readiness")
            self._process = process
            self._client = client
            self._directory = directory
            self._finalizer = weakref.finalize(
                self,
                _cleanup,
                self._owner_pid,
                process,
                client,
                directory,
                self._shutdown_timeout,
            )
        except BaseException:
            self._process, self._client, self._directory = process, client, directory
            self._stop_resources()
            raise

    def _alive(self) -> bool:
        return not self._closed and self._process is not None and self._process.poll() is None

    def is_running(self) -> bool:
        if os.getpid() != self._owner_pid:
            return False
        with self._lock:
            return self._alive()

    def is_ready(self) -> bool:
        """Return liveness and data readiness without starting a process or doing I/O."""
        if os.getpid() != self._owner_pid:
            return False
        with self._lock:
            return self._data_loaded and self._alive()

    @property
    def pid(self) -> int | None:
        if os.getpid() != self._owner_pid:
            return None
        with self._lock:
            return self._process.pid if self._process is not None else None

    def invalidate_data(self) -> None:
        """Refuse new and in-flight decisions until a complete refresh succeeds."""
        self._assert_owner()
        with self._lock:
            self._data_loaded = False
            self._generation += 1

    def _request_json(
        self,
        client: httpx.Client,
        method: str,
        path: str,
        data: dict[str, Any],
    ) -> Any:
        content = json.dumps(data, allow_nan=False).encode("utf-8")
        response = client.request(method, path, content=content, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response.json() if response.content else None

    def set_data(self, data: dict[str, Any]) -> None:
        """Atomically replace data; restart a crashed child only on explicit refresh."""
        self._assert_owner()
        with self._lock:
            self._data_loaded = False
            self._generation += 1
            try:
                if self._closed:
                    raise PolicyEngineError("OPA runtime has been closed")
                if not self._alive():
                    self._start()
                if self._client is None:
                    raise PolicyEngineError("OPA HTTP client is unavailable")
                self._request_json(self._client, "PUT", "/v1/data", data)
                if not self._alive():
                    raise PolicyEngineError("OPA exited during data refresh")
            except Exception as exc:
                raise PolicyEngineError(f"Unable to load OPA policy data: {exc}") from exc
            self._data_loaded = True

    def evaluate(self, input_data: dict[str, Any], entrypoint: int = 0) -> list[dict[str, Any]]:
        """Evaluate concurrently, rejecting decisions that cross a data epoch."""
        self._assert_owner()
        if type(entrypoint) is not int or entrypoint not in range(len(ENTRYPOINTS)):
            raise PolicyEngineError(f"Invalid entrypoint: {entrypoint}")
        name = ENTRYPOINTS[entrypoint]
        for attempt in range(2):
            with self._lock:
                if not self._data_loaded:
                    raise PolicyEngineError("Policy data not loaded — refusing to evaluate (fail-closed)")
                if not self._alive() or self._client is None:
                    self._data_loaded = False
                    raise PolicyEngineError("OPA process is not running — refusing to evaluate")
                client, generation = self._client, self._generation
            try:
                document = self._request_json(client, "POST", f"/v1/data/authz/{name}", {"input": input_data})
            except Exception as exc:
                raise PolicyEngineError(f"OPA evaluation failed: {exc}") from exc
            with self._lock:
                if generation != self._generation or not self._data_loaded or not self._alive():
                    if attempt == 0 and self._data_loaded and self._alive():
                        continue
                    raise PolicyEngineError("OPA policy state changed during evaluation — refusing stale decision")
            result = document.get("result") if isinstance(document, dict) else None
            if not isinstance(result, dict):
                raise PolicyEngineError("OPA entrypoint returned an undefined or non-object decision")
            verdict = "has_role" if name == "has_role" else "allowed"
            if type(result.get(verdict)) is not bool:
                raise PolicyEngineError(f"OPA entrypoint returned an invalid '{verdict}' decision")
            return [{"result": result}]
        raise PolicyEngineError("OPA data changed repeatedly during evaluation")

    def close(self) -> None:
        """Stop and reap this owner's child; safe to call more than once."""
        if os.getpid() != self._owner_pid:
            return
        with self._lock:
            self._closed = True
            self._data_loaded = False
            self._generation += 1
            self._stop_resources()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

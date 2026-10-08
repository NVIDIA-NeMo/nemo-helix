# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker-backed sandbox provider.

Runs each sandbox as one persistent container (``docker run -d`` a keep-alive process),
execs commands with ``docker exec``, and moves files across the boundary with ``docker cp``
— the same transfer verb that maps to ``kubectl cp`` for the Kubernetes provider that
follows. Shells out to the ``docker`` CLI (stdlib ``subprocess``/asyncio only); no Python
Docker SDK dependency.

Isolation note: the container does **not** default to ``--network none``, because the agent
harness legitimately needs egress to reach its model endpoint. Network mode is a provider
option (``network``), defaulting to Docker's default bridge. Endpoint-scoped egress control
(allow the model API, deny everything else) is future work — it belongs to a policy-capable
backend (e.g. NVIDIA OpenShell), not this provider.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import posixpath
import shlex
import signal
import uuid
from dataclasses import dataclass
from pathlib import Path

from nhx_evals_sdk.agent_eval.runtimes.environment import _redact_for_logging
from nhx_evals_sdk.agent_eval.runtimes.sandbox.base import (
    SANDBOX_RUNTIME_RETURN_CODE,
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)

logger = logging.getLogger(__name__)

_CONTAINER_NAME_PREFIX = "nemo-eval-sbx-"
_KEEP_ALIVE_COMMAND = "sh -c 'exec sleep infinity'"
DEFAULT_EXEC_TIMEOUT_S = 180.0
DEFAULT_START_TIMEOUT_S = 120.0


@dataclass
class _DockerContainer:
    """Provider-private state stashed on ``SandboxHandle.raw``."""

    name: str
    image: str
    env: dict[str, str]


def _resource_flags(resources: SandboxResources) -> list[str]:
    """Translate neutral resources into ``docker run`` flags (unmappable fields ignored)."""
    flags: list[str] = []
    if resources.cpu is not None:
        flags += ["--cpus", str(resources.cpu)]
    if resources.memory_mib is not None:
        flags += ["--memory", f"{resources.memory_mib}m"]
    if resources.gpu:
        # Request all GPUs; a specific count/type is a future refinement.
        flags += ["--gpus", "all"]
    return flags


class DockerSandboxProvider:
    """Sandbox provider backed by the local Docker CLI."""

    name = "docker"

    def __init__(
        self,
        *,
        docker_bin: str = "docker",
        network: str | None = None,
        default_timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
        start_timeout_s: float | None = DEFAULT_START_TIMEOUT_S,
        extra_run_args: list[str] | None = None,
    ) -> None:
        self._docker = docker_bin
        self._network = network
        self._default_timeout_s = default_timeout_s
        self._start_timeout_s = start_timeout_s
        self._extra_run_args = list(extra_run_args or [])

    async def _run(
        self,
        argv: list[str],
        *,
        timeout_s: float | None,
        stdin: bytes | None = None,
    ) -> tuple[int, str, str]:
        """Run a ``docker`` CLI command. Returns (return_code, stdout, stderr).

        Single chokepoint every CLI call goes through (mocked at this boundary in tests).
        Enforces the timeout with ``asyncio.wait_for`` and kills the whole process group so
        no child lingers. Raises :class:`TimeoutError` on timeout.
        """
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(input=stdin), timeout=timeout_s)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            with contextlib.suppress(Exception):
                await proc.wait()
            # Redact argv: it can carry `-e KEY=<secret>` (e.g. NVIDIA_API_KEY), and this error is
            # caught and persisted into error.json, so a raw argv would leak credentials into artifacts.
            raise TimeoutError(f"docker command timed out after {timeout_s:g}s: {_redact_for_logging(argv)}") from exc
        code = proc.returncode if proc.returncode is not None else SANDBOX_RUNTIME_RETURN_CODE
        return code, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if spec.image is None:
            raise SandboxCreateError("spec.image is required for the docker provider")

        name = _CONTAINER_NAME_PREFIX + uuid.uuid4().hex
        argv: list[str] = [self._docker, "run", "-d", "--name", name]
        if self._network is not None:
            argv += ["--network", self._network]
        if spec.workdir is not None:
            argv += ["-w", spec.workdir]
        for key, value in spec.env.items():
            argv += ["-e", f"{key}={value}"]
        argv += _resource_flags(spec.resources)
        argv += self._extra_run_args
        # Keep the container alive so exec/cp can target it across its lifetime; the harness
        # is driven by exec, not by the container's entrypoint.
        argv += [spec.image, "sh", "-c", "exec sleep infinity"]

        try:
            code, _out, err = await self._run(argv, timeout_s=self._start_timeout_s)
        except TimeoutError as exc:
            # ``docker run -d`` may have created the container before we timed out (or SIGKILLed it);
            # best-effort remove by name so a slow start never leaks an orphan.
            await self._force_remove(name)
            raise SandboxCreateError(f"docker run timed out for image={spec.image!r}: {exc}") from exc
        if code != 0:
            await self._force_remove(name)  # clean up any partially-created container
            raise SandboxCreateError(f"docker run failed (code={code}) for image={spec.image!r}: {err.strip()}")

        return SandboxHandle(
            sandbox_id=name,
            provider_name=self.name,
            raw=_DockerContainer(name=name, image=spec.image, env=dict(spec.env)),
        )

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        stdin: bytes | None = None,
    ) -> SandboxExecResult:
        container = _container(handle)
        argv: list[str] = [self._docker, "exec"]
        if stdin is not None:
            argv.append("-i")
        if cwd is not None:
            argv += ["-w", cwd]
        if env:
            for key, value in env.items():
                argv += ["-e", f"{key}={value}"]
        argv += [container.name, "sh", "-c", command]

        effective_timeout = timeout_s if timeout_s is not None else self._default_timeout_s
        try:
            code, out, err = await self._run(argv, timeout_s=effective_timeout, stdin=stdin)
        except TimeoutError as exc:
            return SandboxExecResult(
                stdout=None, stderr=str(exc), return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="timeout"
            )
        return SandboxExecResult(stdout=out, stderr=err, return_code=code, error_type=None)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        container = _container(handle)
        parent = posixpath.dirname(target_path)
        if parent:
            result = await self.exec(handle, f"mkdir -p {shlex.quote(parent)}")
            if not result.ok:
                raise RuntimeError(f"docker upload: mkdir {parent!r} failed: {result.stderr}")
        await self._cp(f"{source_path}", f"{container.name}:{target_path}")

    async def upload_dir(self, handle: SandboxHandle, source_dir: Path, target_dir: str) -> None:
        container = _container(handle)
        result = await self.exec(handle, f"mkdir -p {shlex.quote(target_dir)}")
        if not result.ok:
            raise RuntimeError(f"docker upload_dir: mkdir {target_dir!r} failed: {result.stderr}")
        # A trailing "/." copies directory *contents* into target_dir (not nested under it).
        await self._cp(f"{source_dir}{os.sep}.", f"{container.name}:{target_dir}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        container = _container(handle)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        await self._cp(f"{container.name}:{source_path}", f"{target_path}")

    async def download_dir(self, handle: SandboxHandle, source_dir: str, target_dir: Path) -> None:
        container = _container(handle)
        target_dir.mkdir(parents=True, exist_ok=True)
        await self._cp(f"{container.name}:{posixpath.join(source_dir, '.')}", f"{target_dir}")

    async def _cp(self, source: str, dest: str) -> None:
        code, _out, err = await self._run([self._docker, "cp", source, dest], timeout_s=self._default_timeout_s)
        if code != 0:
            raise RuntimeError(f"docker cp {source!r} -> {dest!r} failed (code={code}): {err.strip()}")

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        container = _container(handle)
        try:
            code, out, _err = await self._run(
                [self._docker, "inspect", "-f", "{{.State.Status}}", container.name],
                timeout_s=self._default_timeout_s,
            )
        except TimeoutError:
            return SandboxStatus.UNKNOWN
        if code != 0:
            return SandboxStatus.STOPPED
        state = out.strip().lower()
        if state == "running":
            return SandboxStatus.RUNNING
        if state in {"created", "restarting"}:
            return SandboxStatus.STARTING
        if state in {"exited", "dead", "removing", "paused"}:
            return SandboxStatus.STOPPED
        return SandboxStatus.UNKNOWN

    async def close(self, handle: SandboxHandle) -> None:
        await self._force_remove(_container(handle).name)

    async def _force_remove(self, name: str) -> None:
        """Best-effort ``docker rm -f`` — teardown must never raise, or it leaks the container and can
        mask the in-block exception it runs alongside. Failures are logged, not propagated."""
        try:
            code, _out, err = await self._run([self._docker, "rm", "-f", name], timeout_s=self._default_timeout_s)
        except Exception as exc:  # noqa: BLE001 - teardown is best-effort
            logger.warning("docker rm -f %s errored during teardown: %s", name, exc)
            return
        if code != 0:
            logger.warning("docker rm -f %s failed during teardown (code=%d): %s", name, code, err.strip())

    async def aclose(self) -> None:
        return None


def _container(handle: SandboxHandle) -> _DockerContainer:
    raw = handle.raw
    if not isinstance(raw, _DockerContainer):
        raise TypeError(f"handle.raw is not a docker container handle: {type(raw).__name__}")
    return raw

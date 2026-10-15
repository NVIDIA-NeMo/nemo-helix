# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process/filesystem environment boundary for agent-eval runtimes.

Sits *below* :class:`AgentTaskRunner` so a runtime needn't know whether the
agent/verifier run under Docker, locally, or another filesystem-backed sandbox.
It is a process/filesystem abstraction: :class:`EnvRunSpec`'s ``mounts``/
``extra_args`` are filesystem hints that non-filesystem providers may ignore.
Handles route both roles through a single :meth:`AbstractEnvironmentHandle.run`.
"""

from __future__ import annotations

import abc
import asyncio
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask

EnvRole = Literal["agent", "verifier"]
_SENSITIVE_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _redact_for_logging(cmd: list[str]) -> str:
    """Scrub secret-looking values (``KEY=…`` tokens and ``--flag value`` pairs)."""
    out: list[str] = []
    redact_next = False
    for token in cmd:
        if redact_next:
            out.append("***REDACTED***")
            redact_next = False
        elif "=" in token:
            left, right = token.split("=", 1)
            sensitive = any(m in left.upper() for m in _SENSITIVE_MARKERS)
            out.append(f"{left}=***REDACTED***" if sensitive else f"{left}={right}")
        else:
            normalized = token.lstrip("-").replace("-", "_").upper()
            if token.startswith("-") and any(m in normalized for m in _SENSITIVE_MARKERS):
                redact_next = True
            out.append(token)
    return " ".join(out)


def default_image_tag(task_id: str) -> str:
    """Default task → image-tag mapping (callers may inject their own).

    Sanitizes ``task_id`` to a valid Docker image name so ids with spaces or
    other unsupported characters don't fail the build/run.
    """
    safe = re.sub(r"[^a-z0-9_.-]+", "-", task_id.lower()).strip(".-")
    return f"{safe or 'task'}:latest"


@dataclass(frozen=True)
class EnvCommandResult:
    """Outcome of running a single command inside a prepared environment."""

    exit_code: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class EnvRunSpec:
    """How to execute one command inside an environment handle.

    ``mounts``/``extra_args`` are filesystem-environment hints (e.g. Docker bind
    mounts and extra CLI args). Non-filesystem providers may ignore them.
    """

    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[tuple[str, str]] = field(default_factory=list)
    workdir: str | None = None
    timeout: int | None = None
    extra_args: list[str] = field(default_factory=list)


@runtime_checkable
class AgentEnvironmentHandle(Protocol):
    """A prepared, single-task environment that can run agent/verifier commands."""

    async def run_agent(self, spec: EnvRunSpec) -> EnvCommandResult: ...

    async def run_verifier(self, spec: EnvRunSpec) -> EnvCommandResult: ...

    async def close(self) -> None: ...


@runtime_checkable
class AgentEnvironmentProvider(Protocol):
    """Creates per-task environment handles. Pluggable: Docker now, others later."""

    async def prepare(
        self,
        task: AgentEvalTask,
        config: AgentEvalRunConfig | None = None,
    ) -> AgentEnvironmentHandle: ...


class AbstractEnvironmentHandle(abc.ABC):
    """Base handle that routes both roles through a single :meth:`run`.

    Concrete handles implement :meth:`run`; ``run_agent``/``run_verifier`` are
    role-specialized wrappers so the duplicated phase methods don't have to be
    reimplemented per backend.
    """

    @abc.abstractmethod
    async def run(self, spec: EnvRunSpec, role: EnvRole) -> EnvCommandResult: ...

    async def run_agent(self, spec: EnvRunSpec) -> EnvCommandResult:
        return await self.run(spec, "agent")

    async def run_verifier(self, spec: EnvRunSpec) -> EnvCommandResult:
        return await self.run(spec, "verifier")

    async def close(self) -> None:
        return None


def _docker_run(image: str, spec: EnvRunSpec) -> EnvCommandResult:
    """Run ``spec.command`` in a one-shot ``docker run --rm`` container.

    Shells out to the ``docker`` CLI (stdlib ``subprocess`` only), so no
    ``agent-runtimes`` extra is needed — just a ``docker`` binary at call time.
    """
    cmd = ["docker", "run", "--rm"]
    if spec.workdir:
        cmd += ["-w", spec.workdir]
    for key, value in spec.env.items():
        cmd += ["-e", f"{key}={value}"]
    for host_path, container_path in spec.mounts:
        cmd += ["-v", f"{host_path}:{container_path}"]
    cmd += spec.extra_args + os.environ.get("DOCKER_EXTRA_ARGS", "").split()
    cmd += [image, *spec.command]

    print(f"[agent-eval-runtime] $ {_redact_for_logging(cmd)}")
    try:
        result = subprocess.run(cmd, check=False, text=True, timeout=spec.timeout)
    except subprocess.TimeoutExpired:
        return EnvCommandResult(exit_code=124, timed_out=True)
    return EnvCommandResult(exit_code=result.returncode)


class DockerEnvironmentHandle(AbstractEnvironmentHandle):
    """Docker-backed environment handle bound to one task image."""

    def __init__(self, image: str) -> None:
        self.image = image

    async def run(self, spec: EnvRunSpec, role: EnvRole = "agent") -> EnvCommandResult:
        del role  # Docker runs both roles identically against the same image.
        return await asyncio.to_thread(_docker_run, self.image, spec)


class DockerEnvironmentProvider:
    """Default provider that maps each task to its built Docker image."""

    def __init__(self, *, image_tag_fn: Callable[[str], str] = default_image_tag) -> None:
        self._image_tag_fn = image_tag_fn

    async def prepare(
        self,
        task: AgentEvalTask,
        config: AgentEvalRunConfig | None = None,
    ) -> DockerEnvironmentHandle:
        del config
        return DockerEnvironmentHandle(self._image_tag_fn(task.id))

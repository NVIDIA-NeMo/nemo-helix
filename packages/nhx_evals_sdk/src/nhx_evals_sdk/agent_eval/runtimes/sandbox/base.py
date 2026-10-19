# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral sandbox seam for containerized agent-eval runtimes.

A minimal, boundary-crossing sandbox contract: a runtime describes a sandbox with
:class:`SandboxSpec`, a :class:`SandboxProvider` creates it and runs commands, and
context/artifacts move across the boundary by *file transfer* (``upload_*`` /
``download_*``) rather than shared mounts. That transfer model is what lets the same
runtime run on a local Docker backend today and a remote Kubernetes backend later:
bind mounts do not cross the Kubernetes API boundary, but ``docker cp`` / ``kubectl cp``
do.

The shape (exec + programmatic file I/O + the ``error_type`` sentinel convention) is
deliberately modeled on NeMo Gym's ``nemo_gym.sandbox`` provider protocol so a Gym
backend could be adapted later, but it is intentionally scoped to what the agent-eval
evidence contract needs — we own it, so it carries no heavyweight dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class SandboxStatus(str, Enum):
    """Provider-neutral sandbox lifecycle status."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"


#: Sentinel ``return_code`` a provider uses when the *sandbox runtime* (not the user's
#: command) failed to run the command — e.g. a timeout or a dead container. Distinguishes
#: "the sandbox broke" from "the command exited non-zero".
SANDBOX_RUNTIME_RETURN_CODE = 125


@dataclass(frozen=True)
class SandboxResources:
    """Provider-neutral resource request (providers map or ignore fields they can't honor)."""

    cpu: float | None = None
    memory_mib: int | None = None
    disk_gib: int | None = None
    gpu: int | None = None
    gpu_type: str | None = None

    @classmethod
    def from_mapping(cls, resources: Mapping[str, Any] | None) -> SandboxResources:
        if resources is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = set(resources) - allowed
        if unknown:
            raise ValueError(
                f"Unknown sandbox resource keys: {', '.join(sorted(unknown))}. "
                f"Expected keys: {', '.join(sorted(allowed))}"
            )
        return cls(
            cpu=float(resources["cpu"]) if resources.get("cpu") is not None else None,
            memory_mib=int(resources["memory_mib"]) if resources.get("memory_mib") is not None else None,
            disk_gib=int(resources["disk_gib"]) if resources.get("disk_gib") is not None else None,
            gpu=int(resources["gpu"]) if resources.get("gpu") is not None else None,
            gpu_type=str(resources["gpu_type"]) if resources.get("gpu_type") is not None else None,
        )


@dataclass(frozen=True)
class SandboxSpec:
    """A sandbox creation request.

    ``files`` are seed files written into the sandbox at ``start()`` as a
    ``{absolute_container_path: text_contents}`` map; larger or binary payloads use
    :meth:`SandboxProvider.upload_file` / ``upload_dir`` after start. ``provider_options``
    carries backend-specific knobs (e.g. the Docker network) the neutral spec doesn't model.

    ``resources`` is a typed :class:`SandboxResources`; build one from an untyped mapping with
    :meth:`SandboxResources.from_mapping` at the edge rather than passing a raw mapping here.
    """

    image: str | None = None
    workdir: str | None = None
    ttl_s: int | float | None = None
    env: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    resources: SandboxResources = field(default_factory=SandboxResources)
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxHandle:
    """Provider-neutral handle to a created sandbox.

    ``raw`` is provider-owned opaque state. Callers pass it back to the provider through
    this handle rather than inspecting it — it is typed ``object`` so no consumer depends
    on a provider's internal representation.
    """

    sandbox_id: str
    provider_name: str
    raw: object


@dataclass(frozen=True)
class SandboxExecResult:
    """Provider-neutral process-execution result.

    ``return_code`` is the process exit code when the sandbox actually ran the command.
    On a sandbox-runtime failure (timeout, dead sandbox) it is
    :data:`SANDBOX_RUNTIME_RETURN_CODE` and ``error_type`` names the failure.
    """

    stdout: str | None
    stderr: str | None
    return_code: int
    error_type: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the command exited 0 and the sandbox runtime did not fail."""
        return self.return_code == 0 and self.error_type is None


class SandboxCreateError(RuntimeError):
    """Raised when a provider cannot create a sandbox."""


class SandboxProvider(Protocol):
    """Runtime/infra provider contract used by the public sandbox facade.

    Concrete providers (Docker now, Kubernetes/agent-sandbox next) implement this
    structurally. File transfer is programmatic so it crosses a remote API boundary,
    not just a shared host filesystem.
    """

    name: str

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a ready sandbox and return a provider-neutral handle.

        Providers must return only once the sandbox can run commands and transfer files,
        raising :class:`SandboxCreateError` (or a subclass) if it cannot become ready.
        """
        ...

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
        """Run a shell command inside a sandbox; never raises for command failure."""
        ...

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one local file into a sandbox."""
        ...

    async def upload_dir(self, handle: SandboxHandle, source_dir: Path, target_dir: str) -> None:
        """Upload a local directory tree into a sandbox."""
        ...

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one sandbox file to the local filesystem."""
        ...

    async def download_dir(self, handle: SandboxHandle, source_dir: str, target_dir: Path) -> None:
        """Download a sandbox directory tree to the local filesystem."""
        ...

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Return the current sandbox lifecycle status."""
        ...

    async def close(self, handle: SandboxHandle) -> None:
        """End the sandbox lifecycle and release provider resources for it."""
        ...

    async def aclose(self) -> None:
        """Close provider-scoped resources (SDK clients, pools)."""
        ...

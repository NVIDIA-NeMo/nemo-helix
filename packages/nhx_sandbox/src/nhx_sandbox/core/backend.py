# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The interface every sandbox backend implements, so Harbor environments and Gym brokers can run on any of them without knowing which.

Callers describe what they want in a ``SandboxSpec`` and each backend translates it into its own API.
Callers hold opaque ``SandboxRef``s. Backends declare what they support in ``Capabilities`` and raise
``UnsupportedOperation`` instead of quietly doing less, so a task is never graded under conditions it
didn't ask for. Backends raise the errors in ``nhx_sandbox.core.errors``, never their SDK's own.

Some backends are managed services whose operator, not the caller, decides a sandbox's size, lifetime,
user and network. ``check`` lets a runner find out up front whether a task fits, and ``PlatformEgress``
lets a task accept network rules it can't set.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from nhx_sandbox.core.errors import UnsupportedOperation
from nhx_sandbox.egress import EgressPolicy

__all__ = [
    "AppliedEgress",
    "Capabilities",
    "EgressControl",
    "ExecRequest",
    "ExecResult",
    "Isolation",
    "PlatformEgress",
    "Resources",
    "SandboxBackend",
    "SandboxInfo",
    "SandboxPhase",
    "SandboxRef",
    "SandboxSpec",
    "WorkloadNeeds",
    "WorkloadUser",
    "capability_refusals",
    "require_absolute_path",
    "require_egress_policy",
    "require_own_ref",
]


class SandboxPhase(StrEnum):
    """Where a sandbox is in its life, in terms every backend can report."""

    STARTING = "starting"
    READY = "ready"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"


class Isolation(StrEnum):
    """How strongly a backend separates the workload from the machine it runs on, weakest first.

    Runners use this to refuse untrusted code on a backend that is too weak for it. Compare levels
    with ``at_least``.
    """

    NONE = "none"
    """Ordinary processes on the caller's machine. For tests only."""
    CONTAINER = "container"
    """Namespaces and cgroups on a kernel shared with other workloads, with or without extra
    confinement such as seccomp or Landlock (runc, OpenShell)."""
    USERSPACE_KERNEL = "userspace_kernel"
    """A user-space kernel handles the workload's system calls, so it never talks to the host kernel
    directly (gVisor)."""
    VM = "vm"
    """A lightweight virtual machine with its own guest kernel (Kata Containers, Firecracker)."""

    def at_least(self, other: Isolation) -> bool:
        """Whether this level is as strong as ``other`` or stronger."""
        order = list(Isolation)
        return order.index(self) >= order.index(other)


class WorkloadUser(StrEnum):
    """Who a sandbox's workload runs as, which decides what a task's image must allow.

    An image built for one of these can fail under another. For example, a file owned by the
    image's ``USER`` isn't writable by a uid the platform picks.
    """

    IMAGE = "image"
    """The image's ``USER`` decides, as in Docker."""
    FIXED = "fixed"
    """One known uid, whatever the image says (OpenShell's Kubernetes driver uses 10001)."""
    ASSIGNED = "assigned"
    """A non-root uid the platform picks for each sandbox, which the caller can't predict (OpenShift)."""


@dataclass(frozen=True)
class PlatformEgress:
    """Accepts the platform's own network rules in place of an ``EgressPolicy``.

    Some backends can't set egress per sandbox; the platform applies the same rules to all of them,
    and callers can't read those rules back. Those backends refuse a regular ``EgressPolicy``, and
    backends that do set egress per sandbox refuse this marker. It lives on the task rather than in
    backend config, so a task's unknown network access is visible where the task is defined.

    Attributes:
        reason: Why the task accepts the platform's rules, for logs and results.
    """

    reason: str


@dataclass(frozen=True)
class Resources:
    """Limits on what a sandbox may use. ``None`` leaves a value to the backend's default.

    ``disk_mib`` caps the sandbox's writable storage, so a task that fills its disk fails on its own
    instead of filling the node.
    """

    cpu: float | None = None
    memory_mib: int | None = None
    disk_mib: int | None = None
    gpus: int = 0


@dataclass(frozen=True)
class SandboxSpec:
    """What a caller wants from a new sandbox.

    Attributes:
        image: The container image, ideally pinned by digest.
        name_hint: A readable prefix for the sandbox's name. Backends shorten it and add a suffix to
            fit their naming rules, so two sandboxes never share a name.
        env: Environment for every command in the sandbox. Backends that don't apply the image's own
            ``ENV`` to commands (OpenShell) need the caller to include it here. Don't put secrets
            here. Depending on the backend, env ends up in pod specs, cluster objects or command
            records that others can read. The interface has no way to pass secrets yet.
        workdir: Default working directory for commands. ``None`` uses the backend's default, which
            isn't always the image's ``WORKDIR``.
        resources: CPU, memory, disk and GPU limits.
        egress: What the sandbox may reach on the network. An ``EgressPolicy`` is always applied, even
            an empty one, so no sandbox gets open network access by default. A backend that can't apply
            it refuses the spec. ``PlatformEgress`` accepts the platform's rules instead, on backends
            that can't set egress per sandbox.
        labels: Labels for finding the sandbox later, such as the job that owns it.
        ttl_s: How long the sandbox may live. ``None`` uses the backend's default lifetime, which
            can be short on a managed service, so set it whenever a task needs longer. On a backend
            without ``capabilities.server_ttl``, something outside the backend has to delete the
            sandbox once this runs out.
        ready_timeout_s: How long ``create`` waits for the sandbox to become ready.
        writable_paths: Paths the workload must be able to write, on backends that restrict the
            filesystem (OpenShell's Landlock policy, or a read-only image filesystem). A backend that
            can't make one writable refuses the spec rather than hand back a sandbox that fails later.
            Granting a path doesn't change who owns it, so the workload's user still needs
            permission to write there.
    """

    image: str
    name_hint: str = "sandbox"
    env: Mapping[str, str] = field(default_factory=dict)
    workdir: str | None = None
    resources: Resources = field(default_factory=Resources)
    egress: EgressPolicy | PlatformEgress = field(default_factory=EgressPolicy)
    labels: Mapping[str, str] = field(default_factory=dict)
    ttl_s: float | None = None
    ready_timeout_s: float = 600.0
    writable_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SandboxRef:
    """An opaque handle to one sandbox.

    Only the backend named in ``backend`` can act on it; any other raises ``ValueError``.
    """

    backend: str
    id: str
    name: str


@dataclass(frozen=True)
class SandboxInfo:
    """A sandbox's current state, as its backend reports it."""

    ref: SandboxRef
    phase: SandboxPhase
    labels: Mapping[str, str] = field(default_factory=dict)
    detail: str | None = None


@dataclass(frozen=True)
class ExecRequest:
    """One command to run in a sandbox.

    Attributes:
        command: A shell command line, run with ``sh -c``.
        cwd: Working directory. ``None`` uses the spec's ``workdir``, then the backend's default.
        env: Extra environment, layered over the spec's ``env``. The same caution about secrets applies.
        timeout_s: Seconds before the command is killed, along with anything it started. Backends
            must kill it rather than just stop waiting, so it can't keep changing the sandbox after
            the caller moves on. The sandbox stays usable afterwards, so a runner can still collect
            logs and grade the trial. A backend whose platform deletes a sandbox when a command
            overruns must enforce this timeout itself, inside the sandbox, before the platform's own
            limit fires. ``None`` runs the command until the backend's ceiling, minus whatever
            margin the backend needs to stop it in time. A timeout past that raises
            ``UnsupportedOperation`` instead of being cut short, since a command stopped early, such
            as an agent's whole run, would be graded as if it had finished.
        user: The user to run as. ``None`` uses the sandbox's default user. Backends that can't
            switch users raise ``UnsupportedOperation`` for anything else.
        stdin: Bytes for the command's standard input.
    """

    command: str
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_s: float | None = None
    user: str | int | None = None
    stdin: bytes | None = None


@dataclass(frozen=True)
class ExecResult:
    """How a command ended.

    A backend that keeps only part of a command's output returns what it kept and flags it, rather
    than failing the command. The exit code is still accurate, and a caller that needs all of the
    output, such as a grader parsing a test log, can tell what's missing and read a file instead.

    Attributes:
        exit_code: The command's exit code. For a timed-out command it depends on the backend
            (137 when it was killed with ``SIGKILL``, -1 on some), so check ``timed_out`` instead.
        stdout: Standard output, as much as the backend kept.
        stderr: Standard error, as much as the backend kept.
        timed_out: ``timeout_s`` ran out and the command was killed. The sandbox is still usable;
            if it went away, ``exec`` raises ``SandboxLost`` instead.
        stdout_truncated: The backend dropped part of ``stdout``.
        stderr_truncated: The backend dropped part of ``stderr``.
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass(frozen=True)
class WorkloadNeeds:
    """What a task will do in its sandbox that its ``SandboxSpec`` doesn't say, for ``check`` to weigh.

    These aren't settings a sandbox is created with, so they stay out of the spec. Keeping them in
    one type means a new need is a new field here, not a signature change in every backend.

    Attributes:
        longest_command_s: The longest any single command will run, such as an agent's whole budget.
        lifetime_s: How long the whole trial needs the sandbox, from ``create`` to its last call. A
            task that only sends short commands can still outlive a backend's maximum lifetime.
        runs_as_root: Some command has to run as uid 0.
        multi_container: The task needs several containers in one sandbox, as Compose services do.
    """

    longest_command_s: float | None = None
    lifetime_s: float | None = None
    runs_as_root: bool = False
    multi_container: bool = False


@dataclass(frozen=True)
class Capabilities:
    """What a backend supports, so a runner can refuse a task up front instead of failing it halfway.

    Attributes:
        isolation: How strongly the workload is separated from the machine it runs on.
        exec_as_root: Commands can run as uid 0.
        exec_as_other_users: Commands can run as a user other than the sandbox's default.
        workload_user: Who workloads run as: the image's ``USER``, one fixed uid, or a uid the
            platform assigns.
        fixed_uid: The uid every workload runs as when ``workload_user`` is ``FIXED``, else ``None``.
        server_ttl: The backend deletes a sandbox itself once its ``ttl_s`` runs out. Without this, a
            sandbox whose caller crashed stays until something else cleans it up.
        egress_live_update: Egress can change while the sandbox runs (``EgressControl``).
        egress_readback: The applied egress can be read back and confirmed.
        egress_tld_wildcards: A wildcard on a whole top-level domain, such as ``*.com``, can be allowed.
        gpus: Sandboxes can have GPUs.
        multi_container: Several containers can share one sandbox, as Compose services do.
        egress_per_sandbox: Each sandbox gets the ``EgressPolicy`` its spec asks for. Without this, the
            platform's rules apply to every sandbox and only ``PlatformEgress`` specs run.
        concurrent_exec: Several calls can run in one sandbox at once. Without this, calls on the same
            sandbox run one at a time, so a call can wait behind a long command.
    """

    isolation: Isolation
    exec_as_root: bool
    exec_as_other_users: bool
    workload_user: WorkloadUser
    fixed_uid: int | None
    server_ttl: bool
    egress_live_update: bool
    egress_readback: bool
    egress_tld_wildcards: bool
    gpus: bool
    multi_container: bool
    egress_per_sandbox: bool
    concurrent_exec: bool

    def __post_init__(self) -> None:
        if (self.workload_user is WorkloadUser.FIXED) != (self.fixed_uid is not None):
            raise ValueError("fixed_uid must be set when workload_user is FIXED, and only then")


@dataclass(frozen=True)
class AppliedEgress:
    """The egress a sandbox has right now, as its backend reports it.

    Attributes:
        revision: The backend's version of the policy. Pass it to ``set_egress`` to change the policy
            only if nothing else has changed it since.
        policy: The policy in the backend's own format, as read back from it.
    """

    revision: str
    policy: Mapping[str, Any]


@runtime_checkable
class SandboxBackend(Protocol):
    """Creates, uses and deletes sandboxes on one backend.

    Every call is safe to make concurrently. Calls on different sandboxes run in parallel. Calls on
    the same sandbox do too when ``capabilities.concurrent_exec`` is set, and otherwise wait their
    turn. Cancelling a call stops the command it started; see ``exec``. ``delete`` is idempotent.
    Paths inside a sandbox must be absolute; a relative one raises ``ValueError``.

    A backend that moves files using commands inside the sandbox may need ``sh``, ``tar`` and
    ``gzip`` in the image, and says so in its own docs. ``check`` doesn't inspect images.
    """

    name: str
    capabilities: Capabilities

    async def check(self, spec: SandboxSpec, needs: WorkloadNeeds = WorkloadNeeds()) -> list[str]:
        """Return the reasons a task won't run on this backend, without creating anything.

        An empty list means it fits. This lets a runner refuse a task at submission and name the
        limit it breaks, instead of failing it at ``create`` or partway through, where the mismatch
        can look like a score of 0. ``create`` refuses anything this reports for ``spec`` alone.

        Beyond what ``Capabilities`` shows, it reports limits only the backend knows: a lifetime
        (``spec.ttl_s`` or ``needs.lifetime_s``) or a command (``needs.longest_command_s``) longer
        than it allows, with room to stop the command itself, or an image that hasn't been prepared
        for it.
        """
        ...

    async def create(self, spec: SandboxSpec) -> SandboxRef:
        """Create a sandbox and return once it's ready to run commands.

        Raises:
            UnsupportedOperation: The spec asks for something this backend can't do; ``check`` says what.
            SandboxBackendError: The sandbox failed to start. Nothing is left behind.
        """
        ...

    async def get(self, ref: SandboxRef) -> SandboxInfo:
        """Return the sandbox's current state. Raises ``SandboxNotFound`` once it's gone."""
        ...

    async def list_sandboxes(self, labels: Mapping[str, str]) -> list[SandboxInfo]:
        """Return the sandboxes that carry all of ``labels``, including ones another process created.

        Backends without labels may store them another way, such as in the sandbox's name, so
        matching can be looser there. ``create`` refuses labels a backend can't store.
        """
        ...

    async def delete(self, ref: SandboxRef) -> None:
        """Delete the sandbox. Deleting one that's already gone isn't an error.

        Once this returns, nothing more runs in the sandbox. ``get`` reports it ``STOPPED`` until the
        backend finishes cleaning up, which can take minutes, and raises ``SandboxNotFound`` after that.
        """
        ...

    async def exec(self, ref: SandboxRef, request: ExecRequest) -> ExecResult:
        """Run one command and return how it ended.

        If the caller cancels this call, the backend makes sure the command stops. A backend that
        can't stop it marks the sandbox unusable, and later calls on that sandbox raise
        ``SandboxLost`` instead of waiting behind the command.

        Raises:
            UnsupportedOperation: The request asks for a user or timeout this backend can't give.
            ExecInterrupted: The command's outcome is unknown.
            SandboxLost: The sandbox went away while the command ran.
        """
        ...

    async def write_file(self, ref: SandboxRef, path: str, data: bytes, *, mode: int = 0o644) -> None:
        """Write one file, creating its parent directories."""
        ...

    async def read_file(self, ref: SandboxRef, path: str) -> bytes:
        """Read one file."""
        ...

    async def upload_dir(self, ref: SandboxRef, path: str, archive: bytes) -> None:
        """Unpack a gzipped tar into ``path``, creating it. Modes, symlinks and empty directories are kept."""
        ...

    async def download_dir(self, ref: SandboxRef, path: str) -> bytes:
        """Return the contents of ``path`` as a gzipped tar, with members relative to ``path``."""
        ...

    async def aclose(self) -> None:
        """Release the backend's clients. Sandboxes are left as they are."""
        ...


@runtime_checkable
class EgressControl(Protocol):
    """Changes a running sandbox's egress, on backends that set ``capabilities.egress_live_update``.

    When ``set_egress`` returns, the new policy is in force. A destination it no longer allows is
    unreachable, even over connections opened before the call. Other open connections may drop too,
    so only change egress while nothing in the sandbox is in the middle of a request.
    """

    async def get_egress(self, ref: SandboxRef) -> AppliedEgress:
        """Return the egress in force now."""
        ...

    async def set_egress(
        self, ref: SandboxRef, policy: EgressPolicy, *, expected_revision: str | None = None
    ) -> AppliedEgress:
        """Replace the sandbox's egress with ``policy`` and return it once it's in force.

        Raises:
            RevisionConflict: ``expected_revision`` is set and the sandbox's egress has changed since.
            UnsupportedOperation: The backend can't express ``policy``.
            SandboxBackendError: The backend refused or failed to apply the policy.
        """
        ...


def require_own_ref(ref: SandboxRef, backend: str) -> None:
    """Raise ``ValueError`` if ``ref`` came from another backend, whose ids could collide with this one's."""
    if ref.backend != backend:
        raise ValueError(f"sandbox {ref.name} belongs to the {ref.backend} backend, not {backend}")


def require_absolute_path(path: str) -> None:
    """Raise ``ValueError`` for a relative path, which each backend would resolve against a different directory."""
    if not path.startswith("/"):
        raise ValueError(f"sandbox paths must be absolute: {path!r}")


def require_egress_policy(spec: SandboxSpec, backend: str) -> EgressPolicy:
    """Return the spec's ``EgressPolicy``, for backends that set egress per sandbox; refuse ``PlatformEgress``."""
    if isinstance(spec.egress, PlatformEgress):
        raise UnsupportedOperation(
            f"{backend} applies each sandbox's own egress policy, so it can't accept the platform's rules "
            "instead; give the task an egress policy"
        )
    return spec.egress


def capability_refusals(capabilities: Capabilities, spec: SandboxSpec, needs: WorkloadNeeds) -> list[str]:
    """Return the reasons a task won't fit that ``Capabilities`` alone can tell. Backends add their own."""
    reasons: list[str] = []
    if spec.resources.gpus and not capabilities.gpus:
        reasons.append("the task needs GPUs, and this backend has none")
    if needs.runs_as_root and not capabilities.exec_as_root:
        reasons.append("the task runs commands as root, and this backend never runs anything as root")
    if needs.multi_container and not capabilities.multi_container:
        reasons.append("the task needs several containers, and this backend runs one per sandbox")
    if isinstance(spec.egress, PlatformEgress):
        if capabilities.egress_per_sandbox:
            reasons.append(
                "the task accepts the platform's network rules, but this backend applies an egress policy "
                "per sandbox; give the task one instead"
            )
    elif not capabilities.egress_per_sandbox:
        reasons.append(
            "this backend can't set egress per sandbox, so the task must accept the platform's rules "
            "(PlatformEgress) to run here"
        )
    return reasons

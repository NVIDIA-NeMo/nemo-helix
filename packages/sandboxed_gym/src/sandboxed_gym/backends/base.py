# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend seam for episode sandboxes.

The broker is the only OpenSandbox client for episodes, but it is not bound to OpenSandbox: it
talks to whatever satisfies :class:`EpisodeSandboxBackend`. Backends receive a
:class:`SanitizedEpisodeSpec` -- never a caller's request -- so a backend that grows a new
passthrough field cannot reopen an escalation path the broker closed.
"""

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from sandboxed_gym.egress import EgressPolicy
from sandboxed_gym.sandbox_types import SandboxExecResult, SandboxStatus
from sandboxed_gym.wire import EpisodeResources


class HelixMount(BaseModel):
    """A mount the platform owns and the broker injects. Callers can never supply one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_name: str
    mount_path: str
    sub_path: str | None = None
    read_only: bool = True


class SanitizedEpisodeSpec(BaseModel):
    """A create request rebuilt from only the fields the broker explicitly trusts.

    Constructed exclusively by :func:`sandboxed_gym.sanitize.sanitize_create_request`.
    Fields a caller could use to escalate -- volumes, platform, extensions, snapshot references,
    host namespaces, capabilities -- have no representation here, so they cannot be forwarded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    image: str
    ttl_s: float
    ready_timeout_s: float | None = None
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    resources: EpisodeResources = Field(default_factory=EpisodeResources)
    entrypoint: tuple[str, ...] | None = None
    mounts: tuple[HelixMount, ...] = ()
    # Egress is deliberately absent: it is not per-request. The policy is fixed for the life of
    # the broker and belongs to the backend that applies it (``EpisodeSandboxBackend.egress``).
    # A copy here was built separately from the backend's, and construction is not deterministic
    # -- it reads ``/etc/resolv.conf`` and resolves names at call time -- so the two could hold
    # different rules. Nothing read those rules, so nothing broke; one policy per broker means
    # nothing can.
    #
    # Staged by the broker after ``create`` returns, the same way NeMo-Gym's own sandbox API
    # uploads ``SandboxSpec.files``. Backends must not read this field.
    files: dict[str, bytes] = Field(default_factory=dict)


class EpisodeBackendError(RuntimeError):
    """A backend failed to service an otherwise-valid broker request."""


class UnsupportedEpisodeOperationError(EpisodeBackendError):
    """The backend cannot honour the request as written and will not silently downgrade it.

    Raised, for example, when a caller asks to ``exec`` as a user a backend cannot provide. The
    broker surfaces this as ``UNSUPPORTED_OPERATION`` so the environment fails fast during the
    rollout rather than grading under different conditions than it asked for.
    """


class DirectoryTransferError(EpisodeBackendError):
    """A directory transfer failed inside the episode rather than in the broker.

    Carries its own message to the caller, unlike a generic backend failure. The cause is almost
    always the episode image -- no ``tar``, or a ``/tmp`` the sandbox user cannot write -- which is
    the caller's to fix and discloses nothing about the backend.
    """


class EpisodeSandboxBackend(Protocol):
    """Lifecycle of episode sandboxes behind the broker.

    Implementations hold whatever credential their backend needs. That credential lives only in
    this trusted process; it is never passed to the job sandbox.

    Attributes:
        name: Backend identifier, matching the ``backend`` config value that selects it.
        egress: The policy this backend applies to every episode it creates, fixed for the life
            of the broker. It is the single source of truth for what an episode may reach: the
            broker reads it when auditing a create rather than rebuilding one to describe.
    """

    name: str
    egress: EgressPolicy

    async def create(self, spec: SanitizedEpisodeSpec) -> str:
        """Create one episode sandbox and return its backend-native id."""
        ...

    async def status(self, backend_id: str) -> SandboxStatus:
        """Return the current lifecycle status of an episode."""
        ...

    async def exec(
        self,
        backend_id: str,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        """Run one command inside an episode."""
        ...

    async def upload_file(self, backend_id: str, path: str, content: bytes) -> None:
        """Write one file into an episode."""
        ...

    async def download_file(self, backend_id: str, path: str) -> bytes:
        """Read one file out of an episode."""
        ...

    async def upload_dir(self, backend_id: str, path: str, archive: bytes) -> None:
        """Unpack a gzipped tar into ``path``, creating it if absent.

        Backends with a native directory transport should implement this directly; the rest can
        delegate to :func:`sandboxed_gym.backends.archive.upload_dir_via_archive`, which builds it
        from ``upload_file`` + ``exec``.
        """
        ...

    async def download_dir(self, backend_id: str, path: str) -> bytes:
        """Return ``path``'s contents as a gzipped tar, with members relative to ``path``."""
        ...

    async def close(self, backend_id: str) -> None:
        """Terminate one episode."""
        ...

    async def list_backend_ids(self, job_id: str) -> list[str]:
        """List backend ids this backend believes belong to ``job_id``.

        Used to reconcile after a broker restart, when the in-process handle map is gone but the
        episodes it created are not.
        """
        ...

    async def aclose(self) -> None:
        """Release backend-scoped resources such as SDK clients."""
        ...

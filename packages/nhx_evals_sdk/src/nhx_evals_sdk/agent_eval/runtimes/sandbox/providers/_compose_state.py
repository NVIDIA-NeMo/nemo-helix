# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State and ownership leases for Docker Compose sandbox lifecycles."""

from __future__ import annotations

import errno
import os
from dataclasses import dataclass
from pathlib import Path

from ..base import SandboxCreateError
from ._compose_contracts import ComposeServiceTopology


@dataclass
class _ComposeProjectLock:
    """Exclusive POSIX lock lease for one managed Compose project.

    Attributes:
        path: Host lock-file path.
        fd: Open file descriptor holding the lock, or ``None`` after release.
    """

    path: Path
    fd: int | None = None

    @classmethod
    def acquire(cls, path: Path) -> _ComposeProjectLock:
        """Acquire and return a nonblocking project lock lease.

        Args:
            path: Host lock-file path to create and lock.

        Returns:
            Lease whose open descriptor holds the exclusive lock.

        Raises:
            SandboxCreateError: If POSIX locking is unavailable or another process holds the lock.
            OSError: If the lock file cannot be created or locked for another reason.
        """
        try:
            import fcntl
        except ImportError as exc:
            raise SandboxCreateError("DockerComposeSandboxProvider requires POSIX fcntl file locking") from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise SandboxCreateError(f"Another Compose sandbox holds {path}") from exc
            raise
        return cls(path=path, fd=fd)

    def release(self) -> None:
        """Release and close the lease when it is still held."""
        if self.fd is None:
            return
        fd = self.fd
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(fd)
            finally:
                self.fd = None


@dataclass(frozen=True)
class _ComposeCommandScope:
    """Current project settings needed to construct Docker CLI commands.

    Attributes:
        docker_bin: Docker CLI executable name or path.
        project_directory: Host working directory and Compose project directory.
        compose_files: Ordered Compose configuration files.
        project_name: Explicit Compose project name.
        profiles: Ordered enabled Compose profiles.
    """

    docker_bin: str
    project_directory: Path
    compose_files: tuple[Path, ...]
    project_name: str
    profiles: tuple[str, ...]


@dataclass
class _ComposeSession:
    """State and ownership resources for one provider lifecycle.

    Attributes:
        session_id: Unique identifier used by the public sandbox handle.
        environment: Environment used for all commands in this lifecycle.
        lock: Exclusive project lock held until cleanup completes.
        command_scope: Immutable Docker and Compose project settings for this lifecycle.
        target_service: Service used for sandbox command execution and file transfer.
        service_topology: Service roles used for lifecycle readiness checks.
        owns_project: Whether startup reached the point requiring Compose teardown.
        target_identity: Cached ``UID:GID`` of the target service runtime user.
    """

    session_id: str
    environment: dict[str, str]
    lock: _ComposeProjectLock
    command_scope: _ComposeCommandScope
    target_service: str
    service_topology: ComposeServiceTopology
    owns_project: bool = False
    target_identity: str | None = None

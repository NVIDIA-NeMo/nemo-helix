# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts shared by the Docker Compose sandbox provider modules."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(f"{__package__}.compose")


class ComposeCleanupError(RuntimeError):
    """The managed Compose project could not be completely stopped."""


@dataclass(frozen=True)
class ComposeServiceTopology:
    """Expected active services for one Compose project.

    The rendered active service set must match these two groups exactly. Long-running
    services must be running and healthy when a health check is configured; one-shot
    services must have exited successfully.

    Attributes:
        target_service: Long-running service used for sandbox command execution and file transfer.
        long_running_services: Services that must remain running after startup.
        one_shot_services: Services that must finish successfully during startup.

    Example:
        ``ComposeServiceTopology("agent", frozenset({"agent", "redis"}), frozenset({"init"}))``
        targets ``agent``, requires ``agent`` and ``redis`` to stay up, and requires ``init`` to exit zero.
    """

    target_service: str
    long_running_services: frozenset[str]
    one_shot_services: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Normalize service collections and validate their lifecycle roles.

        Raises:
            ValueError: If a service has two roles or the target is not long-running.
        """
        object.__setattr__(self, "long_running_services", frozenset(self.long_running_services))
        object.__setattr__(self, "one_shot_services", frozenset(self.one_shot_services))
        overlap = self.long_running_services & self.one_shot_services
        if overlap:
            raise ValueError(f"Compose services cannot be both long-running and one-shot: {sorted(overlap)}")
        if self.target_service not in self.long_running_services:
            raise ValueError("target_service must be one of long_running_services")

    @property
    def active_services(self) -> frozenset[str]:
        """Return every service expected in the rendered Compose project."""
        return self.long_running_services | self.one_shot_services


@dataclass(frozen=True)
class ComposeCommandResult:
    """Result returned for a Docker or Compose command.

    Attributes:
        argv: Exact argument vector passed to the subprocess.
        return_code: Process return code, or the sandbox runtime code after a timeout.
        stdout: Captured standard output.
        stderr: Captured standard error.
        timed_out: Whether the provider terminated the command at its deadline.
    """

    argv: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """Return whether the command completed successfully before its deadline."""
        return self.return_code == 0 and not self.timed_out


PullPolicy = Literal["always", "missing", "never"]
ProgressCallback = Callable[[str], None]

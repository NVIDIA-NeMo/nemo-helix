# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The small cluster surface the controller needs. Tests supply a fake."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from config import PodSize


class ClusterWriteError(RuntimeError):
    """A fileset write failed. Local records stay the source of truth."""


@dataclass(frozen=True)
class JobView:
    name: str
    status: str


class Cluster(Protocol):
    def create_job(self, name: str, environment: dict[str, str], pod: PodSize, profile_name: str) -> bool:
        """Submit one job. Return False when the platform refuses it."""

    def jobs(self, prefix: str) -> list[JobView]:
        """Jobs whose names start with ``prefix``."""

    def cancel(self, name: str) -> None:
        """Ask the platform to stop a non-terminal job."""

    def delete(self, name: str) -> None:
        """Delete one terminal job."""

    def read_result(self, name: str) -> bytes | None:
        """Return the ``service-stress`` artifact, or None when it is absent."""

    def write_file(self, path: str, data: bytes) -> None:
        """Copy one object into the run fileset."""

    def remove_prefix(self, prefix: str) -> None:
        """Delete fileset objects under ``prefix`` and keep the fileset."""

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend interface.

A backend only declares: it never writes a row, reads a registry, or holds a credential.
"""

from __future__ import annotations

from typing import Protocol

from nemo_builder_plugin.plan import BuildPlan, Destination, PlannedImage
from nemo_helix_plugin.jobs.spec import HelixJobSpec


class BackendRejectedError(ValueError):
    """The backend cannot build this request, or cannot build at all (409)."""


class Backend(Protocol):
    #: What `builder.backend` calls it. Every row it builds records this, in `provenance.backend`.
    name: str
    #: Whether a row it builds is completed on its caller's word: the builder can check nothing the completion
    #: says against the build. Such a row records the job its submit created, and a job found under the build's
    #: name is the build's own only if the rows still waiting record it.
    binds_rows_to_job: bool

    def check(self, plan: BuildPlan) -> None:
        """Raise :class:`BackendRejectedError` for a request this backend cannot build, or if it is misconfigured."""
        ...

    def destination(self, image: PlannedImage) -> Destination:
        """Where ``image`` is published. Raises ``ValueError`` if it cannot be composed."""
        ...

    def compile(self, plan: BuildPlan) -> HelixJobSpec:
        """The build job for a checked and placed plan."""
        ...

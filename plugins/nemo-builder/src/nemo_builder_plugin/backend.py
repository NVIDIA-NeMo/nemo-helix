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
    def check(self, plan: BuildPlan) -> None:
        """Raise :class:`BackendRejectedError` for a request this backend cannot build, or if it is misconfigured."""
        ...

    def destination(self, image: PlannedImage) -> Destination:
        """Where ``image`` is published. Raises ``ValueError`` if it cannot be composed."""
        ...

    def compile(self, plan: BuildPlan) -> HelixJobSpec:
        """The build job for a checked and placed plan."""
        ...

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tune backend protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nemo_platform_plugin.client.adapter import SyncPlatformClient
from nemo_platform_plugin.job_context import JobContext


@runtime_checkable
class OptimizationBackend(Protocol):
    name: str

    def run_study(
        self,
        payload: dict[str, Any],
        *,
        ctx: JobContext,
        sdk: SyncPlatformClient | None = None,
    ) -> dict[str, Any]:
        """Execute one optimize study for the given Fabric-native payload."""

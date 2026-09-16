# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt GA backend stub."""

from __future__ import annotations

from typing import Any, ClassVar

from nemo_helix import NeMoHelix
from nemo_helix_plugin.job_context import JobContext


class GaBackendError(RuntimeError):
    """Raised when prompt GA is requested before the backend ships."""


class GaBackend:
    name: ClassVar[str] = "ga"

    def run_study(
        self,
        payload: dict[str, Any],
        *,
        ctx: JobContext,
        sdk: NeMoHelix | None = None,
    ) -> dict[str, Any]:
        del payload, ctx, sdk
        raise GaBackendError(
            "optimizer.prompt.enabled is not supported yet. "
            "Prompt GA is tracked separately; enable only optimizer.numeric for numeric HPO."
        )

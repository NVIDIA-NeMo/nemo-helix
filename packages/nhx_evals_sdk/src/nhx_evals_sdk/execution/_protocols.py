# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal optional capabilities used by the v4 evaluator implementation."""

from __future__ import annotations

from typing import runtime_checkable

from nhx_evals_sdk.values.params import RunConfig
from typing_extensions import Protocol


@runtime_checkable
class JobParamsConfigurableMetric(Protocol):
    """Optional metric capability for applying runtime job params."""

    def apply_evaluation_job_params(self, params: RunConfig) -> None:
        """Apply runtime execution params directly on the metric instance."""
        ...

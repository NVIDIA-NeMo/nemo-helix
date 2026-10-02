# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from nhx.common.observability import BaseContext


@dataclass
class JobContext(BaseContext):
    """
    Jobs context that will be enriched into logs and traces
    """

    otel_prefix: str = "job"

    id: str | None = None
    step_name: str | None = None
    task_id: str | None = None
    result_name: str | None = None


@dataclass
class JobBackendContext(BaseContext):
    """Job backend context that will allow us to enrich logs and traces with backend-specific information."""

    otel_prefix: str = "job.backend"

    provider: str | None = None
    profile: str | None = None
    name: str | None = None

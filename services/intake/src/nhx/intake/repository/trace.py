# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface for Intake traces."""

from abc import ABC, abstractmethod
from datetime import datetime

from nhx.common.api.common import PaginatedResult
from nhx.intake.spans.domain import (
    IntakeTrace,
    TraceListFilter,
    TraceMetricBucket,
    TraceMetricPoint,
    TraceMode,
)


class TraceRepository(ABC):
    """Domain-facing interface for trace reads."""

    @abstractmethod
    async def list_traces(
        self,
        *,
        filters: TraceListFilter,
        page: int,
        page_size: int,
        sort: str,
        mode: TraceMode,
    ) -> PaginatedResult[IntakeTrace]:
        pass

    @abstractmethod
    async def get_trace(self, *, workspace: str, trace_id: str, mode: TraceMode) -> IntakeTrace | None:
        pass

    @abstractmethod
    async def trace_metrics(
        self,
        *,
        filters: TraceListFilter,
        bucket: TraceMetricBucket,
        timezone_name: str,
    ) -> list[TraceMetricPoint]:
        pass

    @abstractmethod
    async def latest_trace_started_at_by_group(
        self,
        *,
        workspace: str,
        trace_refs_by_group: dict[str, list[str]],
    ) -> dict[str, datetime]:
        pass

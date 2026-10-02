# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface for Intake spans."""

from abc import ABC, abstractmethod

from nhx.common.api.common import PaginatedResult
from nhx.intake.spans.domain import IntakeResponseMode, IntakeSpan, SpanGroup, SpanListFilter


class SpanRepository(ABC):
    """Domain-facing interface for span persistence."""

    @abstractmethod
    async def save_spans(self, spans: list[IntakeSpan]) -> None:
        pass

    @abstractmethod
    async def list_spans(
        self,
        *,
        filters: SpanListFilter,
        page: int,
        page_size: int,
        sort: str,
        mode: IntakeResponseMode,
    ) -> PaginatedResult[IntakeSpan]:
        pass

    @abstractmethod
    async def list_span_groups(
        self,
        *,
        filters: SpanListFilter,
        group_by: list[str],
        page: int,
        page_size: int,
        sort: str,
    ) -> PaginatedResult[SpanGroup]:
        pass

    @abstractmethod
    async def get_span(self, *, workspace: str, span_id: str) -> IntakeSpan | None:
        pass

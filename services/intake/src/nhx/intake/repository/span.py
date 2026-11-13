# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface for Intake spans."""

from abc import ABC, abstractmethod
from typing import Any

from nhx.common.api.common import PaginatedResult
from nhx.intake.spans.domain import IntakeResponseMode, IntakeSpan, SpanGroup, SpanListFilter


class SpanAmbiguousError(Exception):
    """A workspace span ID belongs to more than one session."""


class SpanRepository(ABC):
    """Domain-facing interface for span persistence."""

    @abstractmethod
    async def save_spans(self, spans: list[IntakeSpan]) -> None:
        pass

    @abstractmethod
    async def insert_span_rows(self, rows: list[list[Any]]) -> None:
        """Insert rows prebuilt by ``span_rows_for_insert`` in the storage layer.

        Used by the ATIF ingest path, whose worker process builds the rows off
        the event loop; the column order is fixed by the storage implementation.
        """

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
    async def get_span(self, *, workspace: str, span_id: str, session_id: str | None = None) -> IntakeSpan | None:
        pass

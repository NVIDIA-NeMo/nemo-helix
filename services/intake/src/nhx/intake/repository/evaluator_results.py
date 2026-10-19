# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface for Intake evaluator results."""

from abc import ABC, abstractmethod

from nhx.common.api.common import PaginatedResult
from nhx.intake.spans.domain import EvaluatorResult, EvaluatorResultListFilter


class EvaluatorResultsRepository(ABC):
    """Domain-facing interface for evaluator-result persistence."""

    @abstractmethod
    async def save_evaluator_results(self, results: list[EvaluatorResult]) -> None:
        pass

    @abstractmethod
    async def get_evaluator_result(self, *, workspace: str, evaluator_result_id: str) -> EvaluatorResult | None:
        pass

    @abstractmethod
    async def list_evaluator_results(
        self,
        *,
        filters: EvaluatorResultListFilter,
        page: int,
        page_size: int,
        sort: str,
    ) -> PaginatedResult[EvaluatorResult]:
        pass

    @abstractmethod
    async def list_evaluator_results_for_span(self, *, workspace: str, span_id: str) -> list[EvaluatorResult]:
        pass

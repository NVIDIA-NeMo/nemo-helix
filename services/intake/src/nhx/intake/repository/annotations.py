# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository interface for Intake annotations."""

from abc import ABC, abstractmethod

from nhx.common.api.common import PaginatedResult
from nhx.intake.spans.domain import Annotation, AnnotationListFilter


class AnnotationsRepository(ABC):
    """Domain-facing interface for annotation persistence."""

    @abstractmethod
    async def save_annotations(self, annotations: list[Annotation]) -> None:
        pass

    @abstractmethod
    async def get_annotation(self, *, workspace: str, annotation_id: str) -> Annotation | None:
        pass

    @abstractmethod
    async def list_annotations(
        self,
        *,
        filters: AnnotationListFilter,
        page: int,
        page_size: int,
        sort: str,
    ) -> PaginatedResult[Annotation]:
        pass

    @abstractmethod
    async def soft_delete_annotation(self, *, annotation: Annotation) -> None:
        pass

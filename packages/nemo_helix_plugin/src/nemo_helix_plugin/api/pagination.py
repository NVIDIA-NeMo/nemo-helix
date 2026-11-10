# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Page-number pagination query parameters shared by list routes."""

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Query
from fastapi.exceptions import RequestValidationError

# Offsets are bound as signed 64-bit integers by every SQL backend we use.
MAX_OFFSET = 2**63 - 1


@dataclass(frozen=True)
class PageParams:
    """Validated ``page`` and ``page_size`` from a list request."""

    page: int
    page_size: int

    @property
    def offset(self) -> int:
        """Number of rows before the first row of this page."""
        return (self.page - 1) * self.page_size


def make_page_dep(*, default_size: int = 100, max_size: int = 1000) -> Callable[..., PageParams]:
    """Create a FastAPI dependency that parses the ``page`` and ``page_size`` query parameters.

    Both must be positive and ``page_size`` at most ``max_size``. A page whose offset does
    not fit a signed 64-bit integer is rejected with the same 422 shape as the other checks.
    """

    def _dependency(
        page: int = Query(default=1, ge=1, description="Page number."),
        page_size: int = Query(default=default_size, ge=1, le=max_size, description="Page size."),
    ) -> PageParams:
        params = PageParams(page=page, page_size=page_size)
        if params.offset > MAX_OFFSET:
            raise RequestValidationError(
                [
                    {
                        "type": "value_error",
                        "loc": ("query", "page"),
                        "msg": "Value error, page and page_size produce an offset beyond the signed 64-bit range",
                        "input": page,
                    }
                ]
            )
        return params

    return _dependency

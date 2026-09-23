# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Generic, TypeVar, Optional
from typing_extensions import override

from ._models import BaseModel
from ._base_client import BasePage, PageInfo, BaseSyncPage, BaseAsyncPage

__all__ = [
    "DefaultPaginationPagination",
    "SyncDefaultPagination",
    "AsyncDefaultPagination",
    "SyncLogsPagination",
    "AsyncLogsPagination",
]

_T = TypeVar("_T")


class DefaultPaginationPagination(BaseModel):
    current_page_size: int
    """The size for the current page."""

    page: int
    """The current page number."""

    page_size: int
    """The page size used for the query."""

    total_pages: int
    """The total number of pages."""

    total_results: int
    """The total number of results."""


class SyncDefaultPagination(BaseSyncPage[_T], BasePage[_T], Generic[_T]):
    data: List[_T]
    pagination: Optional[DefaultPaginationPagination] = None

    @override
    def _get_page_items(self) -> List[_T]:
        data = self.data
        if not data:
            return []
        return data

    @override
    def next_page_info(self) -> Optional[PageInfo]:
        current_page = None
        if self.pagination is not None:
            if self.pagination.page is not None:  # pyright: ignore[reportUnnecessaryComparison]
                current_page = self.pagination.page
        if current_page is None:
            current_page = 1

        total_pages = None
        if self.pagination is not None:
            if self.pagination.total_pages is not None:  # pyright: ignore[reportUnnecessaryComparison]
                total_pages = self.pagination.total_pages
        if total_pages is not None and current_page >= total_pages:
            return None

        return PageInfo(params={"page": current_page + 1})


class AsyncDefaultPagination(BaseAsyncPage[_T], BasePage[_T], Generic[_T]):
    data: List[_T]
    pagination: Optional[DefaultPaginationPagination] = None

    @override
    def _get_page_items(self) -> List[_T]:
        data = self.data
        if not data:
            return []
        return data

    @override
    def next_page_info(self) -> Optional[PageInfo]:
        current_page = None
        if self.pagination is not None:
            if self.pagination.page is not None:  # pyright: ignore[reportUnnecessaryComparison]
                current_page = self.pagination.page
        if current_page is None:
            current_page = 1

        total_pages = None
        if self.pagination is not None:
            if self.pagination.total_pages is not None:  # pyright: ignore[reportUnnecessaryComparison]
                total_pages = self.pagination.total_pages
        if total_pages is not None and current_page >= total_pages:
            return None

        return PageInfo(params={"page": current_page + 1})


class SyncLogsPagination(BaseSyncPage[_T], BasePage[_T], Generic[_T]):
    data: List[_T]
    next_page: Optional[str] = None

    @override
    def _get_page_items(self) -> List[_T]:
        data = self.data
        if not data:
            return []
        return data

    @override
    def next_page_info(self) -> Optional[PageInfo]:
        next_page = self.next_page
        if not next_page:
            return None

        return PageInfo(params={"page_cursor": next_page})


class AsyncLogsPagination(BaseAsyncPage[_T], BasePage[_T], Generic[_T]):
    data: List[_T]
    next_page: Optional[str] = None

    @override
    def _get_page_items(self) -> List[_T]:
        data = self.data
        if not data:
            return []
        return data

    @override
    def next_page_info(self) -> Optional[PageInfo]:
        next_page = self.next_page
        if not next_page:
            return None

        return PageInfo(params={"page_cursor": next_page})

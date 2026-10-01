# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility adapters for generated-SDK-style pagination surfaces."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterator
from typing import Any, Generic, TypeVar

import httpx
from nemo_helix_plugin.client.response import (
    AsyncNemoPaginatedCall,
    AsyncNemoPaginatedResponse,
    NemoPaginatedResponse,
)
from pydantic import BaseModel

LegacyItemT = TypeVar("LegacyItemT", bound=BaseModel)


class LegacyPageInfo:
    """Small page-info object compatible with Stainless pagination examples."""

    def __init__(self, *, params: dict[str, Any]) -> None:
        self.params = params

    def __repr__(self) -> str:
        return f"{type(self).__name__}(params={self.params!r})"


class SyncLegacyPage(Generic[LegacyItemT]):
    """Stainless-style page adapter for sync compatibility surfaces."""

    def __init__(
        self,
        *,
        data: list[LegacyItemT],
        metadata: object | None = None,
        next_page_params: dict[str, Any] | None = None,
        get_next_page: Callable[[], "SyncLegacyPage[LegacyItemT]"] | None = None,
    ) -> None:
        self.data = data
        self.metadata = metadata
        self._next_page_params = next_page_params
        self._get_next_page = get_next_page

    def __iter__(self) -> Iterator[LegacyItemT]:
        return iter(self.data)

    def has_next_page(self) -> bool:
        return self._next_page_params is not None

    def next_page_info(self) -> LegacyPageInfo | None:
        if self._next_page_params is None:
            return None
        return LegacyPageInfo(params=self._next_page_params)

    def get_next_page(self) -> "SyncLegacyPage[LegacyItemT]":
        if self._get_next_page is not None:
            return self._get_next_page()
        raise RuntimeError("No next page expected; please check `.has_next_page()` before calling `.get_next_page()`.")


class AsyncLegacyPage(Generic[LegacyItemT]):
    """Stainless-style page adapter for async compatibility surfaces."""

    def __init__(
        self,
        *,
        data: list[LegacyItemT],
        metadata: object | None = None,
        next_page_params: dict[str, Any] | None = None,
        get_next_page: Callable[[], Awaitable["AsyncLegacyPage[LegacyItemT]"]] | None = None,
    ) -> None:
        self.data = data
        self.metadata = metadata
        self._next_page_params = next_page_params
        self._get_next_page = get_next_page

    def __iter__(self) -> Iterator[LegacyItemT]:
        return iter(self.data)

    def has_next_page(self) -> bool:
        return self._next_page_params is not None

    def next_page_info(self) -> LegacyPageInfo | None:
        if self._next_page_params is None:
            return None
        return LegacyPageInfo(params=self._next_page_params)

    async def get_next_page(self) -> "AsyncLegacyPage[LegacyItemT]":
        if self._get_next_page is not None:
            return await self._get_next_page()
        raise RuntimeError("No next page expected; please check `.has_next_page()` before calling `.get_next_page()`.")


class LegacyPaginatedResponse(Generic[LegacyItemT]):
    """Compatibility adapter exposing ``.data`` and item iteration."""

    def __init__(self, response: NemoPaginatedResponse[LegacyItemT, Any]) -> None:
        self._response = response
        self._first_page: SyncLegacyPage[LegacyItemT] | None = None

    def _legacy_page(self, raw: httpx.Response) -> SyncLegacyPage[LegacyItemT]:
        items, body, metadata = self._response._parse_page(raw)
        next_page = self._response._strategy.next_page(body)
        if next_page is None:
            return SyncLegacyPage(data=items, metadata=metadata)
        return SyncLegacyPage(
            data=items,
            metadata=metadata,
            next_page_params=self._response._strategy.page_query_params(next_page),
            get_next_page=lambda: self._legacy_page(self._response._fetch_page(self._response.request, next_page)),
        )

    def _page(self) -> SyncLegacyPage[LegacyItemT]:
        if self._first_page is None:
            self._first_page = self._legacy_page(self._response.http_response)
        return self._first_page

    @property
    def data(self) -> list[LegacyItemT]:
        return self._page().data

    def __iter__(self) -> Iterator[LegacyItemT]:
        return self._response.items()

    def iter_pages(self) -> Iterator[SyncLegacyPage[LegacyItemT]]:
        page = self._page()
        while True:
            yield page
            if not page.has_next_page():
                return
            page = page.get_next_page()


class AsyncLegacyPaginatedResponse(Generic[LegacyItemT]):
    """Async compatibility adapter matching Stainless async paginator basics."""

    def __init__(self, response: Awaitable[AsyncNemoPaginatedResponse[LegacyItemT, Any]]) -> None:
        self._response = AsyncNemoPaginatedCall(lambda: response)
        self._first_page: AsyncLegacyPage[LegacyItemT] | None = None

    async def _get_response(self) -> AsyncNemoPaginatedResponse[LegacyItemT, Any]:
        return await self._response

    async def _legacy_page(
        self,
        response: AsyncNemoPaginatedResponse[LegacyItemT, Any],
        raw: httpx.Response,
    ) -> AsyncLegacyPage[LegacyItemT]:
        items, body, metadata = response._parse_page(raw)
        next_page = response._strategy.next_page(body)
        if next_page is None:
            return AsyncLegacyPage(data=items, metadata=metadata)

        async def get_next_page() -> AsyncLegacyPage[LegacyItemT]:
            raw = await response._fetch_page(response.request, next_page)
            return await self._legacy_page(response, raw)

        return AsyncLegacyPage(
            data=items,
            metadata=metadata,
            next_page_params=response._strategy.page_query_params(next_page),
            get_next_page=get_next_page,
        )

    async def _page(self) -> AsyncLegacyPage[LegacyItemT]:
        if self._first_page is not None:
            return self._first_page
        response = await self._get_response()
        self._first_page = await self._legacy_page(response, response.http_response)
        return self._first_page

    def __await__(self) -> Generator[object, None, AsyncLegacyPage[LegacyItemT]]:
        return self._page().__await__()

    async def __aiter__(self) -> AsyncIterator[LegacyItemT]:
        async for item in self._response:
            yield item

    async def iter_pages(self) -> AsyncIterator[AsyncLegacyPage[LegacyItemT]]:
        page = await self._page()
        while True:
            yield page
            if not page.has_next_page():
                return
            page = await page.get_next_page()

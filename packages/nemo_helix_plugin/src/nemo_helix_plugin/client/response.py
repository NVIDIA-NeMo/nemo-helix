# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP response wrappers for JSON, binary, and streaming endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Iterator
from contextlib import AbstractAsyncContextManager, AbstractContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload

import httpx
from nemo_helix_plugin.client.errors import NemoResponseValidationError, NemoTransportError, raise_for_status
from nemo_helix_plugin.client.types import OffsetPagination, PaginationStrategy, PreparedRequest
from pydantic import BaseModel, ValidationError
from typing_extensions import TypeVar as TypeVarExt

ModelT = TypeVar("ModelT", bound=BaseModel)


def _validated_page(
    response: httpx.Response,
    model_type: type[ModelT],
    strategy: type[PaginationStrategy[Any, Any]],
) -> tuple[list[ModelT], dict, Any]:
    """Decode one page and normalize response-contract failures."""
    try:
        body = response.json()
    except ValueError as exc:
        raise NemoResponseValidationError(response, exc) from exc

    if not isinstance(body, dict):
        exc = ValueError("Paginated responses must be JSON objects")
        raise NemoResponseValidationError(response, exc) from exc

    try:
        raw_items = strategy.extract_items(body)
        items = [model_type.model_validate(item) for item in raw_items]
        metadata = strategy.extract_metadata(body)
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise NemoResponseValidationError(response, exc) from exc
    return items, body, metadata


def _parse_stream_line(line: str, headers: httpx.Headers) -> str | None:
    """Extract a JSON payload from a stream line, or ``None`` to skip.

    Handles both NDJSON (pass-through) and SSE framing (strips ``data:``
    prefix, skips non-data fields like ``event:``, ``id:``, comments).
    """
    line = line.strip()
    if not line:
        return None
    if "text/event-stream" in headers.get("content-type", ""):
        if line.startswith("data:"):
            return line[5:].strip()
        return None
    return line


ResponseT = TypeVar("ResponseT")


def _raise_for_binary_status(response: httpx.Response, request: PreparedRequest) -> None:
    """Raise unless the status is successful for this endpoint contract."""
    if response.status_code not in request.additional_success_status_codes:
        raise_for_status(response)


@dataclass(frozen=True, slots=True)
class NemoResponse(Generic[ResponseT]):
    """Typed HTTP response for JSON endpoints.

    Example::

        resp = client.send(endpoints.get_user(workspace="default"))
        resp.body             # UserResponse
        resp.http_response    # full httpx.Response

        user = resp.data()    # raises on non-2xx, otherwise returns body

    When several 2xx codes share one typed body (e.g. a delete that returns 202
    Accepted for async teardown or 204 No Content when already gone, both typed
    ``None``), inspect ``resp.http_response.status_code`` to tell them apart.
    """

    http_response: httpx.Response
    body: ResponseT
    request: PreparedRequest[ResponseT]

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute access to the parsed body.

        Allows callers to use ``resp.name`` instead of ``resp.data().name``,
        smoothing the migration from the old SDK which returned models directly.
        Only triggered when normal attribute lookup fails — ``NemoResponse``'s
        own attributes (``body``, ``http_response``, ``request``, ``data``)
        always win.
        """
        body = object.__getattribute__(self, "body")
        if body is None:
            raise AttributeError(name)
        return getattr(body, name)

    def data(self) -> ResponseT:
        """Return the parsed response body.

        Since ``send()`` raises on non-2xx, this is a convenience accessor
        equivalent to ``.body``.
        """
        return self.body


# ---------------------------------------------------------------------------
# Sync streaming responses
# ---------------------------------------------------------------------------


class NemoBinaryResponse:
    """Sync response for binary download endpoints.

    For simple reads::

        resp = client.send(endpoints.download(...))
        data = resp.read()

    For streaming chunks::

        with resp.stream() as chunks:
            for chunk in chunks:
                f.write(chunk)
    """

    def __init__(self, stream_ctx: AbstractContextManager[httpx.Response], request: PreparedRequest) -> None:
        self._stream_ctx = stream_ctx
        self._http_response: httpx.Response | None = None
        self.request = request

    @property
    def http_response(self) -> httpx.Response:
        """The underlying response, available after ``read()`` or while streaming."""
        if self._http_response is None:
            raise RuntimeError("http_response is only available after read() or inside a stream() context")
        return self._http_response

    def read(self) -> bytes:
        """Read and return the entire response body as bytes."""
        try:
            with self._stream_ctx as raw:
                self._http_response = raw
                data = raw.read()
                _raise_for_binary_status(raw, self.request)
                return data
        except httpx.TransportError as exc:
            raise NemoTransportError(exc) from exc

    @contextmanager
    def stream(self, chunk_size: int | None = None) -> Iterator[Iterator[bytes]]:
        """Yield an iterator of raw byte chunks.

        Args:
            chunk_size: Maximum number of bytes per chunk. If None, uses the
                transport's default chunking.

        The underlying httpx response is available as ``http_response``
        after entering the context, e.g. for reading ``Content-Length``::

            with resp.stream() as chunks:
                size = resp.http_response.headers.get("content-length")
                for chunk in chunks:
                    ...
        """
        try:
            with self._stream_ctx as raw:
                self._http_response = raw
                _raise_for_binary_status(raw, self.request)
                yield raw.iter_raw(chunk_size) if chunk_size else raw.iter_raw()
        except httpx.TransportError as exc:
            raise NemoTransportError(exc) from exc


class NemoStreamResponse(Generic[ModelT]):
    """Sync response for SSE/NDJSON streaming endpoints.

    Handles both NDJSON (``application/x-ndjson``) and SSE
    (``text/event-stream``) framing automatically based on the
    response ``Content-Type``.  SSE ``data:`` prefixes are stripped
    before JSON parsing.

    Use via :meth:`stream`::

        with client.send(ChatEndpoint(...)).stream() as chunks:
            for chunk in chunks:
                print(chunk.text)
    """

    def __init__(
        self,
        stream_ctx: AbstractContextManager[httpx.Response],
        model_type: type[ModelT],
        request: PreparedRequest,
    ) -> None:
        self._stream_ctx = stream_ctx
        self._model_type = model_type
        self.request = request

    @contextmanager
    def stream(self) -> Iterator[Iterator[ModelT]]:
        """Yield an iterator of parsed model objects."""
        try:
            with self._stream_ctx as raw:
                raise_for_status(raw)

                def _iter() -> Iterator[ModelT]:
                    for line in raw.iter_lines():
                        payload = _parse_stream_line(line, raw.headers)
                        if payload is not None:
                            try:
                                yield self._model_type.model_validate_json(payload)
                            except (ValueError, ValidationError) as exc:
                                raise NemoResponseValidationError(raw, exc) from exc

                yield _iter()
        except httpx.TransportError as exc:
            raise NemoTransportError(exc) from exc


# ---------------------------------------------------------------------------
# Async streaming responses
# ---------------------------------------------------------------------------


class AsyncNemoBinaryResponse:
    """Async response for binary download endpoints.

    For simple reads::

        resp = await client.send(endpoints.download(...))
        data = await resp.read()

    For streaming chunks::

        async with resp.stream() as chunks:
            async for chunk in chunks:
                f.write(chunk)
    """

    def __init__(self, stream_ctx: AbstractAsyncContextManager[httpx.Response], request: PreparedRequest) -> None:
        self._stream_ctx = stream_ctx
        self._http_response: httpx.Response | None = None
        self.request = request

    @property
    def http_response(self) -> httpx.Response:
        """The underlying response, available after ``read()`` or while streaming."""
        if self._http_response is None:
            raise RuntimeError("http_response is only available after read() or inside a stream() context")
        return self._http_response

    async def read(self) -> bytes:
        """Read and return the entire response body as bytes."""
        try:
            async with self._stream_ctx as raw:
                self._http_response = raw
                data = await raw.aread()
                _raise_for_binary_status(raw, self.request)
                return data
        except httpx.TransportError as exc:
            raise NemoTransportError(exc) from exc

    @asynccontextmanager
    async def stream(self, chunk_size: int | None = None) -> AsyncIterator[AsyncIterator[bytes]]:
        """Yield an async iterator of raw byte chunks.

        Args:
            chunk_size: Maximum number of bytes per chunk. If None, uses the
                transport's default chunking.

        The underlying httpx response is available as ``http_response``
        after entering the context, e.g. for reading ``Content-Length``::

            async with resp.stream() as chunks:
                size = resp.http_response.headers.get("content-length")
                async for chunk in chunks:
                    ...
        """
        try:
            async with self._stream_ctx as raw:
                self._http_response = raw
                _raise_for_binary_status(raw, self.request)
                yield raw.aiter_raw(chunk_size) if chunk_size else raw.aiter_raw()
        except httpx.TransportError as exc:
            raise NemoTransportError(exc) from exc


class AsyncNemoStreamResponse(Generic[ModelT]):
    """Async response for SSE/NDJSON streaming endpoints.

    Handles both NDJSON and SSE framing automatically based on the
    response ``Content-Type``.  See :class:`NemoStreamResponse` for details.

    Use via :meth:`stream`::

        async with (await client.send(ChatEndpoint(...))).stream() as chunks:
            async for chunk in chunks:
                print(chunk.text)
    """

    def __init__(
        self,
        stream_ctx: AbstractAsyncContextManager[httpx.Response],
        model_type: type[ModelT],
        request: PreparedRequest,
    ) -> None:
        self._stream_ctx = stream_ctx
        self._model_type = model_type
        self.request = request

    @asynccontextmanager
    async def stream(self) -> AsyncIterator[AsyncIterator[ModelT]]:
        """Yield an async iterator of parsed model objects."""
        try:
            async with self._stream_ctx as raw:
                raise_for_status(raw)

                async def _iter() -> AsyncIterator[ModelT]:
                    async for line in raw.aiter_lines():
                        payload = _parse_stream_line(line, raw.headers)
                        if payload is not None:
                            try:
                                yield self._model_type.model_validate_json(payload)
                            except (ValueError, ValidationError) as exc:
                                raise NemoResponseValidationError(raw, exc) from exc

                yield _iter()
        except httpx.TransportError as exc:
            raise NemoTransportError(exc) from exc


# ---------------------------------------------------------------------------
# Paginated responses
# ---------------------------------------------------------------------------


# Type aliases for the page-fetching callbacks used by paginated responses.
# Fetchers receive the strategy-specific page token returned by ``next_page()``.
SyncPageFetcher = Callable[[PreparedRequest, Any], httpx.Response]
AsyncPageFetcher = Callable[[PreparedRequest, Any], Awaitable[httpx.Response]]


PageModelT = TypeVar("PageModelT", bound=BaseModel)
PageMetadataT = TypeVar("PageMetadataT")


@dataclass(frozen=True, slots=True)
class PageResult(Generic[PageModelT, PageMetadataT]):
    """A single page of results with pagination metadata.

    Returned by :meth:`NemoPaginatedResponse.page` for callers who want
    one page at a time rather than auto-iterating all pages::

        resp = client.send(list_items())
        page = resp.page()
        print(
            f"Page {page.metadata['page']} of {page.metadata['total_pages']} "
            f"({page.metadata['total_results']} total)"
        )
        for item in page.items:
            print(item.name)
    """

    items: list[PageModelT]
    metadata: PageMetadataT


PaginatedModelT = TypeVar("PaginatedModelT", bound=BaseModel)
PaginatedStrategyT_co = TypeVarExt(
    "PaginatedStrategyT_co",
    bound=PaginationStrategy[Any, Any],
    default=OffsetPagination,
    covariant=True,
)
PageTokenT = TypeVar("PageTokenT")
MetadataT = TypeVar("MetadataT")


class NemoPaginatedResponse(Generic[PaginatedModelT, PaginatedStrategyT_co]):
    """Sync paginated API response.

    Provides two iteration modes::

        # Iterate all items across all pages
        for item in response.items():
            print(item.name)

        # Iterate page by page with metadata
        for page in response.pages():
            print(f"Page {page.metadata['page']}/{page.metadata['total_pages']}")
            for item in page.items:
                process(item)

    For single-page access, use :meth:`page`::

        page = response.page()
        print(
            f"{page.metadata['total_results']} total across "
            f"{page.metadata['total_pages']} pages"
        )
    """

    def __init__(
        self,
        first_http_response: httpx.Response,
        model_type: type[PaginatedModelT],
        request: PreparedRequest,
        fetch_page: SyncPageFetcher,
        strategy: type[PaginationStrategy[Any, Any]] | None = None,
    ) -> None:
        self._first_response = first_http_response
        self._model_type = model_type
        self.request = request
        self._fetch_page = fetch_page
        self._strategy: type[PaginationStrategy[Any, Any]] = strategy or OffsetPagination

    @property
    def http_response(self) -> httpx.Response:
        return self._first_response

    @property
    def data(self) -> list[PaginatedModelT]:
        """First page items, matching legacy generated-SDK page objects."""
        return self.page().items

    def __iter__(self) -> Iterator[PaginatedModelT]:
        return self.items()

    def _parse_page(self, raw: httpx.Response) -> tuple[list[PaginatedModelT], dict, Any]:
        """Parse a page response into items, its raw body, and typed metadata."""
        raise_for_status(raw)
        return _validated_page(raw, self._model_type, self._strategy)

    @overload
    def page(
        self: NemoPaginatedResponse[PaginatedModelT, PaginationStrategy[PageTokenT, MetadataT]],
    ) -> PageResult[PaginatedModelT, MetadataT]: ...

    @overload
    def page(self) -> PageResult[PaginatedModelT, Any]: ...

    def page(self) -> PageResult[PaginatedModelT, Any]:
        """Return the first page as a :class:`PageResult` with metadata."""
        items, _, metadata = self._parse_page(self._first_response)
        return PageResult(items=items, metadata=metadata)

    def items(self) -> Iterator[PaginatedModelT]:
        """Iterate all items across all pages, fetching subsequent pages lazily."""
        items, body, _ = self._parse_page(self._first_response)
        yield from items

        next_page = self._strategy.next_page(body)
        while next_page is not None:
            items, body, _ = self._parse_page(self._fetch_page(self.request, next_page))
            yield from items
            next_page = self._strategy.next_page(body)

    @overload
    def pages(
        self: NemoPaginatedResponse[PaginatedModelT, PaginationStrategy[PageTokenT, MetadataT]],
    ) -> Iterator[PageResult[PaginatedModelT, MetadataT]]: ...

    @overload
    def pages(self) -> Iterator[PageResult[PaginatedModelT, Any]]: ...

    def pages(self) -> Iterator[PageResult[PaginatedModelT, Any]]:
        """Iterate page by page, yielding :class:`PageResult` objects with metadata."""
        items, body, metadata = self._parse_page(self._first_response)
        yield PageResult(items=items, metadata=metadata)

        next_page = self._strategy.next_page(body)
        while next_page is not None:
            items, body, metadata = self._parse_page(self._fetch_page(self.request, next_page))
            yield PageResult(items=items, metadata=metadata)
            next_page = self._strategy.next_page(body)


class AsyncNemoPaginatedResponse(Generic[PaginatedModelT, PaginatedStrategyT_co]):
    """Async paginated API response.

    Async twin of :class:`NemoPaginatedResponse`::

        async for item in response.items():
            print(item.name)

        async for page in response.pages():
            print(f"Page {page.metadata['page']}/{page.metadata['total_pages']}")
    """

    def __init__(
        self,
        first_http_response: httpx.Response,
        model_type: type[PaginatedModelT],
        request: PreparedRequest,
        fetch_page: AsyncPageFetcher,
        strategy: type[PaginationStrategy[Any, Any]] | None = None,
    ) -> None:
        self._first_response = first_http_response
        self._model_type = model_type
        self.request = request
        self._fetch_page = fetch_page
        self._strategy: type[PaginationStrategy[Any, Any]] = strategy or OffsetPagination

    @property
    def http_response(self) -> httpx.Response:
        return self._first_response

    @property
    def data(self) -> list[PaginatedModelT]:
        """First page items, matching legacy generated-SDK page objects."""
        return self.page().items

    def __aiter__(self) -> AsyncIterator[PaginatedModelT]:
        return self.items()

    def _parse_page(self, raw: httpx.Response) -> tuple[list[PaginatedModelT], dict, Any]:
        """Parse a page response into items, its raw body, and typed metadata."""
        raise_for_status(raw)
        return _validated_page(raw, self._model_type, self._strategy)

    @overload
    def page(
        self: AsyncNemoPaginatedResponse[PaginatedModelT, PaginationStrategy[PageTokenT, MetadataT]],
    ) -> PageResult[PaginatedModelT, MetadataT]: ...

    @overload
    def page(self) -> PageResult[PaginatedModelT, Any]: ...

    def page(self) -> PageResult[PaginatedModelT, Any]:
        """Return the first page as a :class:`PageResult` with metadata."""
        items, _, metadata = self._parse_page(self._first_response)
        return PageResult(items=items, metadata=metadata)

    async def items(self) -> AsyncIterator[PaginatedModelT]:
        """Iterate all items across all pages, fetching subsequent pages lazily."""
        items, body, _ = self._parse_page(self._first_response)
        for item in items:
            yield item

        next_page = self._strategy.next_page(body)
        while next_page is not None:
            raw = await self._fetch_page(self.request, next_page)
            items, body, _ = self._parse_page(raw)
            for item in items:
                yield item
            next_page = self._strategy.next_page(body)

    @overload
    def pages(
        self: AsyncNemoPaginatedResponse[PaginatedModelT, PaginationStrategy[PageTokenT, MetadataT]],
    ) -> AsyncIterator[PageResult[PaginatedModelT, MetadataT]]: ...

    @overload
    def pages(self) -> AsyncIterator[PageResult[PaginatedModelT, Any]]: ...

    async def pages(self) -> AsyncIterator[PageResult[PaginatedModelT, Any]]:
        """Iterate page by page, yielding :class:`PageResult` objects with metadata."""
        items, body, metadata = self._parse_page(self._first_response)
        yield PageResult(items=items, metadata=metadata)

        next_page = self._strategy.next_page(body)
        while next_page is not None:
            raw = await self._fetch_page(self.request, next_page)
            items, body, metadata = self._parse_page(raw)
            yield PageResult(items=items, metadata=metadata)
            next_page = self._strategy.next_page(body)


class AsyncNemoPaginatedCall(Generic[PaginatedModelT, PaginatedStrategyT_co]):
    """Awaitable and async-iterable handle for a pending paginated request.

    Nested resources use this when callers need both async styles:
    ``response = await client.items.list(...)`` and
    ``async for item in client.items.list(...): ...``.
    Pagination itself stays in :class:`AsyncNemoPaginatedResponse`.
    """

    def __init__(
        self,
        loader: Callable[[], Awaitable[AsyncNemoPaginatedResponse[PaginatedModelT, PaginatedStrategyT_co]]],
    ) -> None:
        self._loader = loader
        self._response: AsyncNemoPaginatedResponse[PaginatedModelT, PaginatedStrategyT_co] | None = None

    def __await__(
        self,
    ) -> Generator[object, None, AsyncNemoPaginatedResponse[PaginatedModelT, PaginatedStrategyT_co]]:
        return self._get_response().__await__()

    def __aiter__(self) -> AsyncIterator[PaginatedModelT]:
        return self.items()

    async def _get_response(self) -> AsyncNemoPaginatedResponse[PaginatedModelT, PaginatedStrategyT_co]:
        if self._response is None:
            self._response = await self._loader()
        return self._response

    async def page(self) -> PageResult[PaginatedModelT, Any]:
        return (await self._get_response()).page()

    async def items(self) -> AsyncIterator[PaginatedModelT]:
        async for item in (await self._get_response()).items():
            yield item

    async def pages(self) -> AsyncIterator[PageResult[PaginatedModelT, Any]]:
        async for page in (await self._get_response()).pages():
            yield page

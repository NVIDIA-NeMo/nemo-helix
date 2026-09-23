# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared types for the NeMo client infrastructure.

This module contains marker types, TypeVars, data classes, pagination
strategies, and client-side option definitions used across the client package.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Generic, ParamSpec, Protocol, TypeVar

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypedDict
from typing_extensions import TypeVar as TypeVarExt

P = ParamSpec("P")
ModelT = TypeVar("ModelT", bound=BaseModel)
ResponseT = TypeVar("ResponseT")


class BinaryContent:
    """Marker type: endpoint sends or receives raw bytes.

    Use ``content`` parameter for binary uploads::

        @put("/files/{path}")
        def UploadEndpoint(content: bytes, *, path: str) -> FileResponse: ...
    """


class Stream(Generic[ModelT]):
    """Marker type: endpoint returns a stream of ``ModelT`` objects (SSE/NDJSON).

    Used as return type in endpoint definitions::

        @post("/chat/{workspace}")
        def ChatEndpoint(body: ChatRequest, *, workspace: str) -> Stream[ChatChunk]: ...
    """


# ---------------------------------------------------------------------------
# Pagination strategies
# ---------------------------------------------------------------------------


PageTokenT = TypeVar("PageTokenT")
MetadataT = TypeVar("MetadataT")


class PaginationStrategy(Generic[PageTokenT, MetadataT]):
    """Base class associating a pagination strategy with its cursor and metadata types.

    Pagination strategies control how the client extracts items from a page
    response, determines the next page identifier, builds query params
    to fetch the next page, and extracts pagination metadata.
    """

    @classmethod
    def extract_items(cls, response_body: dict) -> list[dict]:
        raise NotImplementedError

    @classmethod
    def next_page(cls, response_body: dict) -> PageTokenT | None:
        raise NotImplementedError

    @classmethod
    def page_query_params(cls, page: PageTokenT) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def extract_metadata(cls, response_body: dict) -> MetadataT:
        raise NotImplementedError


class OffsetPaginationMetadata(TypedDict):
    page: int
    page_size: int
    current_page_size: int
    total_pages: int
    total_results: int


_OFFSET_PAGINATION_METADATA_ADAPTER = TypeAdapter(OffsetPaginationMetadata)


class OffsetPagination(PaginationStrategy[int, OffsetPaginationMetadata]):
    """Offset-based pagination using ``page`` query parameter.

    This is the default strategy, matching the standard ``NemoListResponse``
    envelope used by NeMo Helix services::

        {"data": [...], "pagination": {"page": 1, "total_pages": 5, ...}}

    Subclass to customise field names for non-standard envelopes::

        class MyPagination(OffsetPagination):
            items_field = "results"
            page_param = "offset"
    """

    items_field: ClassVar[str] = "data"
    page_param: ClassVar[str] = "page"
    pagination_field: ClassVar[str] = "pagination"
    page_field: ClassVar[str] = "page"
    page_size_field: ClassVar[str] = "page_size"
    current_page_size_field: ClassVar[str] = "current_page_size"
    total_pages_field: ClassVar[str] = "total_pages"
    total_results_field: ClassVar[str] = "total_results"

    @classmethod
    def extract_items(cls, response_body: dict) -> list[dict]:
        return response_body.get(cls.items_field, [])

    @classmethod
    def next_page(cls, response_body: dict) -> int | None:
        pagination = response_body.get(cls.pagination_field)
        if pagination is None:
            return None
        current_page = pagination.get(cls.page_field, 1)
        total = pagination.get(cls.total_pages_field, 1)
        if current_page < total:
            return current_page + 1
        return None

    @classmethod
    def page_query_params(cls, page: int) -> dict[str, int]:
        return {cls.page_param: page}

    @classmethod
    def extract_metadata(cls, response_body: dict) -> OffsetPaginationMetadata:
        pagination = response_body.get(cls.pagination_field)
        return _OFFSET_PAGINATION_METADATA_ADAPTER.validate_python(
            {
                "page": pagination[cls.page_field],
                "page_size": pagination[cls.page_size_field],
                "current_page_size": pagination[cls.current_page_size_field],
                "total_pages": pagination[cls.total_pages_field],
                "total_results": pagination[cls.total_results_field],
            }
            if isinstance(pagination, dict)
            else pagination
        )


class CursorPaginationMetadata(TypedDict):
    total: int
    next_page: str | None
    prev_page: str | None


class CursorPagination(PaginationStrategy[str, CursorPaginationMetadata]):
    """Cursor pagination matching the Jobs log response envelope.

    The response contains items and cursor metadata at the top level::

        {"data": [...], "total": 42, "next_page": "...", "prev_page": null}
    """

    items_field: ClassVar[str] = "data"
    cursor_param: ClassVar[str] = "page_cursor"
    total_field: ClassVar[str] = "total"
    next_page_field: ClassVar[str] = "next_page"
    prev_page_field: ClassVar[str] = "prev_page"

    @classmethod
    def extract_items(cls, response_body: dict) -> list[dict]:
        return response_body.get(cls.items_field, [])

    @classmethod
    def next_page(cls, response_body: dict) -> str | None:
        return response_body.get(cls.next_page_field)

    @classmethod
    def page_query_params(cls, page: str) -> dict[str, str]:
        return {cls.cursor_param: page}

    @classmethod
    def extract_metadata(cls, response_body: dict) -> CursorPaginationMetadata:
        return {
            "total": response_body.get(cls.total_field, 0),
            "next_page": response_body.get(cls.next_page_field),
            "prev_page": response_body.get(cls.prev_page_field),
        }


PaginatedModelT = TypeVar("PaginatedModelT", bound=BaseModel)
StrategyT = TypeVarExt(
    "StrategyT",
    bound=PaginationStrategy[Any, Any],
    default=OffsetPagination,
)


class Paginated(Generic[PaginatedModelT, StrategyT]):
    """Marker type: endpoint returns paginated results of ``ModelT``.

    The second type parameter selects the pagination strategy.  It defaults
    to :class:`OffsetPagination`, which matches the standard
    ``NemoListResponse`` envelope.

    Usage::

        # Default offset-based pagination
        @get("/apis/example/v2/workspaces/{workspace}/items")
        def list_items(...) -> Paginated[Item]: ...

        # Custom strategy
        class MyPagination(OffsetPagination):
            items_field = "results"
            page_param = "offset"

        @get("/apis/example/v2/workspaces/{workspace}/widgets")
        def list_widgets(...) -> Paginated[Widget, MyPagination]: ...

    Caller experience::

        for item in client.list_items():
            print(item.name)
    """


# ---------------------------------------------------------------------------
# Endpoint parameter registries
# ---------------------------------------------------------------------------

# Parameter names with special handling in the endpoint decorator.
# These are routed to specific fields on ``PreparedRequest`` (body, content,
# query_params) and are not treated as path parameters.
RESERVED_PARAM_NAMES: frozenset[str] = frozenset({"self", "body", "content", "query_params"})


class ConflictResolver(Protocol):
    """Builds the retrieve request to run when a create hits a 409 Conflict.

    Passed to ``@post(..., get_on_conflict=...)`` on a create endpoint. When the
    create is sent with ``exist_ok=True`` and the server responds 409, the client
    replays the request this resolver returns and returns *its* entity instead of
    raising :class:`ConflictError`.

    The resolver receives the create's request ``body`` model and the resolved
    ``workspace``, and returns a :class:`PreparedRequest` for the matching GET —
    typically by calling the linked GET endpoint::

        def _get_fileset_on_conflict(
            body: CreateFilesetRequest, workspace: str | None
        ) -> PreparedRequest[FilesetOutput]:
            return get_fileset(name=body.name, workspace=workspace)

    Because the resolver calls the real GET endpoint, its arguments are checked by
    the type checker at the call site — there is no runtime guessing.
    """

    def __call__(self, body: Any, workspace: str | None) -> PreparedRequest: ...


# Parameters with these names are recognised in endpoint signatures as
# client-side options.  They are stripped from the HTTP request and stashed
# in ``PreparedRequest.client_options`` for the client to act on.
#
# Each entry maps a parameter name to its expected Python type.
# Unknown parameters in an endpoint signature trigger a ``TypeError``
# at decoration time.
BLESSED_CLIENT_PARAMS: dict[str, type] = {
    # Declared but not yet acted on by the client — see AIRCORE-866.
    "exist_ok": bool,
}


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry transient failures with exponential backoff.

    Set as a client-level default via the ``retry`` constructor parameter,
    or override per-request via ``send()``'s ``retry`` keyword argument.

    This is an operational concern and does not belong in endpoint
    signatures. Response-header handling and broad server-error retries are
    opt-in so adapters can reproduce another client's retry contract without
    changing standalone client defaults.

    .. note::

        The retry loop replays the full request including the body.
        Callers are responsible for ensuring idempotency when retrying
        non-safe methods (POST, PATCH).  Consider using an
        ``Idempotency-Key`` header for create operations that must be
        safe to retry.

        A request whose ``content`` is a generator, file object or other
        one-shot iterable is only retried while its body is still
        untouched — that is, on a connection-establishment failure.  Once
        httpx has started reading the body it cannot be sent again, so a
        mid-body failure (and any retryable status code, which by
        definition arrives after the body) is raised to the caller
        instead.  Retry those where the body can be rebuilt from its
        source.

    Usage::

        # Client-level default
        client = MyClient(base_url="...", retry=RetryPolicy(max_retries=3))

        # Per-request override via send()
        client.send(endpoint_fn(...), retry=RetryPolicy(max_retries=10))
    """

    max_retries: int = 3
    backoff_base: float = 0.5
    retryable_status_codes: tuple[int, ...] = (502, 503, 504, 429)
    retry_all_server_errors: bool = False
    respect_retry_decision_headers: bool = False
    respect_retry_after_headers: bool = False


@dataclass(frozen=True, slots=True)
class PreparedRequest(Generic[ResponseT]):
    """A request ready to be sent — carries the endpoint metadata and payload.

    Path interpolation is deferred to the client's ``send()`` method, which
    merges client-level defaults (e.g. workspace) with the explicit path
    params before formatting.
    """

    path_template: str
    path_params: dict[str, str]
    method: str
    content: bytes | Iterable[bytes] | AsyncIterable[bytes] | None
    content_type: str | None
    response_type: type[ResponseT] | None
    query_params: dict[str, str | int | bool | None] | None = None
    extra_headers: dict[str, str] | None = None
    client_options: dict[str, object] | None = None
    additional_success_status_codes: tuple[int, ...] = ()
    # Prebuilt GET to replay on a 409 when ``exist_ok`` is set. Produced by a
    # ``get_on_conflict`` resolver at request-build time (the resolver needs the
    # live ``body`` model, which is serialised away by the time this request is
    # sent). ``send()`` replays it instead of raising ``ConflictError``.
    on_conflict_get: PreparedRequest[ResponseT] | None = None

    def with_headers(self, headers: dict[str, str]) -> PreparedRequest[ResponseT]:
        """Return a new PreparedRequest with additional headers merged in."""
        merged = {**(self.extra_headers or {}), **headers}
        return replace(self, extra_headers=merged)

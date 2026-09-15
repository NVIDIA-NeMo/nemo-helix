# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pagination helpers for CLI list commands.

Typed clients return a :class:`~nemo_platform_plugin.client.response.NemoPaginatedResponse`
for list endpoints. The helpers here turn that into the page objects the
formatters render: a single page keeps the server's pagination metadata, and
``--all-pages`` collects every page into one synthetic response with the same
shape.
"""

from __future__ import annotations

import logging
import typing
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from types import SimpleNamespace, TracebackType
from typing import Any, cast

from nemo_platform_plugin.client.response import NemoPaginatedResponse
from rich.progress import Progress, SpinnerColumn, TextColumn

from nemo_platform_ext.cli.core.help_formatter import add_warning

if typing.TYPE_CHECKING:
    from nemo_platform.pagination import SyncDefaultPagination, SyncLogsPagination

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_root_logging() -> Iterator[None]:
    """Temporarily suppress noisy SDK pagination logs without leaking state."""
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        root_logger.setLevel(previous_level)


class PaginationType(str, Enum):
    """Types of pagination supported by the API."""

    PAGE_NUMBER = "page_number"  # Uses page/page_size params (default_pagination)
    CURSOR = "cursor"  # Uses limit/page_cursor params (logs_pagination)
    NOT_PAGINATED = "not_paginated"  # List operation without pagination support


def _envelope_with_sort(envelope: dict[str, Any] | None) -> dict[str, Any]:
    """Return the envelope fields with ``sort`` always present.

    List output has always carried a ``sort`` key (``null`` when the server did
    not echo one); callers without an envelope keep that shape, and a server
    envelope keeps its own field order.
    """
    fields = dict(envelope or {})
    if "sort" not in fields:
        fields = {"sort": None, **fields}
    return fields


class AllPagesResponse:
    """
    A response object that mimics a single-page response but contains all items.

    This ensures --all-pages returns the same structure as a single page response.
    Used for page-number based pagination which has total_pages and total_results.
    """

    def __init__(
        self,
        data: list[Any],
        total_items: int,
        total_pages: int,
        page_size: int | None = None,
        envelope: dict[str, Any] | None = None,
    ):
        self.data = data
        self.envelope = _envelope_with_sort(envelope)
        self.sort = self.envelope.get("sort")

        # Create pagination info for all items
        self.pagination = type(
            "obj",
            (object,),
            {
                "page": 1,
                "page_size": page_size or len(data),
                "current_page_size": len(data),
                "total_pages": 1,  # All data is now in one "page"
                "total_results": total_items,
            },
        )()

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        """Make this compatible with Pydantic's model_dump."""
        # Convert data items to dicts if they're Pydantic models
        serialized_data = []
        for item in self.data:
            if hasattr(item, "model_dump"):
                serialized_data.append(item.model_dump(mode=mode))
            elif isinstance(item, dict):
                serialized_data.append(item)
            else:
                serialized_data.append(item)

        return {
            "data": serialized_data,
            **self.envelope,
            "pagination": {
                "page": self.pagination.page,
                "page_size": self.pagination.page_size,
                "current_page_size": self.pagination.current_page_size,
                "total_pages": self.pagination.total_pages,
                "total_results": self.pagination.total_results,
            },
        }


class AllCursorPagesResponse:
    """
    A response object for cursor-based pagination results.

    Cursor-based pagination doesn't have total_pages or total_results,
    so this response only contains the collected data and item count.
    """

    def __init__(self, data: list[Any], limit: int | None = None):
        self.data = data
        self.next_page = None  # All pages fetched, no next page

        # Store the original limit for reference
        self._limit = limit

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        """Make this compatible with Pydantic's model_dump."""
        # Convert data items to dicts if they're Pydantic models
        serialized_data = []
        for item in self.data:
            if hasattr(item, "model_dump"):
                serialized_data.append(item.model_dump(mode=mode))
            elif isinstance(item, dict):
                serialized_data.append(item)
            else:
                serialized_data.append(item)

        return {
            "data": serialized_data,
            "next_page": self.next_page,
        }


def _model_dump_item(item: Any, *, mode: str) -> Any:
    if hasattr(item, "model_dump"):
        return item.model_dump(mode=mode)
    if isinstance(item, list):
        return [_model_dump_item(child, mode=mode) for child in item]
    if isinstance(item, dict):
        return {key: _model_dump_item(value, mode=mode) for key, value in item.items()}
    return item


def _envelope_fields(response: NemoPaginatedResponse[Any, Any]) -> dict[str, Any]:
    """Return the first page's non-item envelope fields (``sort``, ``filter``, ``grouped_by``, ...) in wire order."""
    http_response = getattr(response, "http_response", None)
    if http_response is None:
        return {}
    try:
        body = http_response.json()
    except ValueError:
        return {}
    if not isinstance(body, dict):
        return {}
    return {key: value for key, value in body.items() if key not in {"data", "pagination"}}


class OffsetPageResponse:
    """One page of an offset-paginated list, keeping the server's envelope and pagination block."""

    def __init__(self, items: list[Any], metadata: dict[str, Any], envelope: dict[str, Any] | None = None) -> None:
        self.data = items
        self.envelope = _envelope_with_sort(envelope)
        self.sort = self.envelope.get("sort")
        self.pagination = SimpleNamespace(**metadata)

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "data": [_model_dump_item(item, mode=mode) for item in self.data],
            **self.envelope,
            "pagination": vars(self.pagination),
        }


class CursorPageResponse:
    """One page of a cursor-paginated list, keeping ``total`` and the page cursors."""

    def __init__(self, items: list[Any], metadata: dict[str, Any]) -> None:
        self.data = items
        self.total = metadata.get("total", len(items))
        self.next_page = metadata.get("next_page")
        self.prev_page = metadata.get("prev_page")

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "data": [_model_dump_item(item, mode=mode) for item in self.data],
            "total": self.total,
            "next_page": self.next_page,
            "prev_page": self.prev_page,
        }


def collect_offset_pages(
    response: NemoPaginatedResponse[Any, Any],
    *,
    all_pages: bool,
    show_progress: bool = True,
) -> OffsetPageResponse | AllPagesResponse:
    """Return the first page, or every page merged when *all_pages* is set."""
    envelope = _envelope_fields(response)
    if not all_pages:
        page = response.page()
        return OffsetPageResponse(list(page.items), dict(page.metadata), envelope)

    items: list[Any] = []
    total_results = 0
    total_pages = 0
    page_size: int | None = None
    with _progress(show_progress) as (progress, task):
        for page in response.pages():
            items.extend(page.items)
            metadata = dict(page.metadata)
            total_results = int(metadata.get("total_results") or total_results)
            total_pages = int(metadata.get("total_pages") or total_pages)
            page_size = cast(int | None, metadata.get("page_size") or page_size)
            current = int(metadata.get("page") or 0)
            progress.update(
                task,
                total=total_pages or None,
                completed=current,
                description=f"Fetching pages... (page {current}/{total_pages})",
            )

    return AllPagesResponse(
        data=items,
        total_items=total_results or len(items),
        total_pages=total_pages or 1,
        page_size=page_size,
        envelope=envelope,
    )


def collect_cursor_pages(
    response: NemoPaginatedResponse[Any, Any],
    *,
    all_pages: bool,
    limit: int | None = None,
    show_progress: bool = True,
) -> CursorPageResponse | AllCursorPagesResponse:
    """Return the first page, or every page merged when *all_pages* is set."""
    if not all_pages:
        page = response.page()
        return CursorPageResponse(list(page.items), dict(page.metadata))

    items: list[Any] = []
    with _progress(show_progress) as (progress, task):
        for page_num, page in enumerate(response.pages(), start=1):
            items.extend(page.items)
            progress.update(task, completed=page_num, description=f"Fetching pages... (page {page_num})")
    return AllCursorPagesResponse(data=items, limit=limit)


def collect_pages(
    response: NemoPaginatedResponse[Any, Any],
    *,
    all_pages: bool,
    pagination_type: PaginationType = PaginationType.PAGE_NUMBER,
    limit: int | None = None,
    show_progress: bool = True,
) -> OffsetPageResponse | CursorPageResponse | AllPagesResponse | AllCursorPagesResponse:
    """Dispatch to the offset or cursor collector based on *pagination_type*."""
    if pagination_type == PaginationType.CURSOR:
        return collect_cursor_pages(response, all_pages=all_pages, limit=limit, show_progress=show_progress)
    return collect_offset_pages(response, all_pages=all_pages, show_progress=show_progress)


class _progress:
    """Context manager yielding ``(progress, task)`` for page-fetch feedback."""

    def __init__(self, show_progress: bool) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            disable=not show_progress,
        )

    def __enter__(self) -> tuple[Progress, Any]:
        progress = self._progress.__enter__()
        return progress, progress.add_task("Fetching pages...", total=None)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._progress.__exit__(exc_type, exc_value, traceback)


def _fetch_all_pages_page_number(
    list_method: Callable[..., SyncDefaultPagination[Any]],
    progress: Progress,
    task: Any,
    path_args: tuple[Any, ...],
    body_args: dict[str, Any],
) -> AllPagesResponse:
    """
    Fetch all pages using page-number based pagination.

    This is used for endpoints with `default_pagination` in the stainless config.
    """
    all_items: list[Any] = []
    total_pages_count = 0
    total_results = 0
    original_page_size = None

    # Fetch the first page
    try:
        response = list_method(*path_args, **body_args)
    except Exception as e:
        progress.stop()
        raise e

    if (total_pages := response.pagination.total_pages) is not None:
        progress.update(task, total=total_pages)
        total_pages_count = total_pages

    with _suppress_root_logging():
        for page in response.iter_pages():
            page_num = page.pagination.page

            for item in page.data:
                all_items.append(item)

            progress.update(
                task, completed=page_num, description=f"Fetching pages... (page {page_num}/{total_pages_count})"
            )

    return AllPagesResponse(
        data=all_items,
        total_items=total_results or len(all_items),
        total_pages=total_pages_count,
        page_size=original_page_size,
    )


def _fetch_all_pages_cursor(
    list_method: Callable[..., SyncLogsPagination],
    progress: Progress,
    task: Any,
    path_args: tuple[Any, ...],
    body_args: dict[str, Any],
) -> AllCursorPagesResponse:
    """
    Fetch all pages using cursor-based pagination.

    This is used for endpoints with `logs_pagination` in the stainless config.
    """
    all_items: list[Any] = []
    # Fetch the first page
    try:
        response = list_method(*path_args, **body_args)
    except Exception as e:
        progress.stop()
        raise e

    with _suppress_root_logging():
        for page_num, page in enumerate(response.iter_pages(), start=1):
            for item in page.data:
                all_items.append(item)

            progress.update(task, completed=page_num, description=f"Fetching pages... (page {page_num})")

    return AllCursorPagesResponse(
        data=all_items,
        limit=body_args.get("limit", None),
    )


def fetch_all_pages(
    list_method: Callable[..., Any],
    path_args: tuple[Any, ...] = (),
    body_args: dict[str, Any] | None = None,
    show_progress: bool = True,
    pagination_type: PaginationType = PaginationType.PAGE_NUMBER,
) -> AllPagesResponse | AllCursorPagesResponse:
    """
    Fetch all pages from a paginated endpoint.

    Args:
        list_method: The SDK list method to call (e.g., client.namespaces.list)
        path_args: Positional arguments to pass to the list method (e.g., job_id)
        body_args: Keyword arguments to pass to the list method (e.g., filter, search)
        show_progress: Whether to show a progress indicator
        pagination_type: Type of pagination - PAGE_NUMBER (default) or CURSOR

    Returns:
        AllPagesResponse for page-number pagination, AllCursorPagesResponse for cursor pagination
    """
    if body_args is None:
        body_args = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        disable=not show_progress,
    ) as progress:
        task = progress.add_task("Fetching pages...", total=None)

        if pagination_type == PaginationType.CURSOR:
            return _fetch_all_pages_cursor(list_method, progress, task, path_args, body_args)
        else:
            return _fetch_all_pages_page_number(list_method, progress, task, path_args, body_args)


def warn_if_more_pages(
    response: Any,
    pagination_type: PaginationType,
) -> None:
    """
    Warn the user if there are more pages available.

    Args:
        response: The response object from the list method
        pagination_type: Type of pagination - PAGE_NUMBER (default) or CURSOR
    """
    if pagination_type == PaginationType.NOT_PAGINATED:
        return  # No pagination, no warning needed
    elif pagination_type == PaginationType.CURSOR:
        if getattr(response, "next_page", None):
            add_warning("More pages of results are available! Use --all-pages to fetch all results.")
    elif hasattr(response, "pagination") and getattr(response.pagination, "total_pages", 1) > 1:
        add_warning("More pages of results are available! Use --all-pages to fetch all results.")

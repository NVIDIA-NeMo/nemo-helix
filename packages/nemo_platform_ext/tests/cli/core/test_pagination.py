# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pagination utilities."""

from typing import Any
from unittest.mock import Mock
from urllib.parse import parse_qs

import httpx
import pytest
from nemo_platform_ext.cli.core.pagination import (
    AllCursorPagesResponse,
    AllPagesResponse,
    CursorPageResponse,
    OffsetPageResponse,
    PaginationType,
    collect_cursor_pages,
    collect_offset_pages,
    collect_pages,
    warn_if_more_pages,
)
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.client.endpoint import get
from nemo_platform_plugin.client.types import CursorPagination, Paginated
from pydantic import BaseModel

# =============================================================================
# Helpers: real typed paginated responses served by an in-memory transport
# =============================================================================


class Item(BaseModel):
    name: str


@get("/apis/test/v2/items")
def list_items(*, query_params: dict[str, Any] | None = None) -> Paginated[Item]:
    raise NotImplementedError


@get("/apis/test/v2/logs")
def list_logs(*, query_params: dict[str, Any] | None = None) -> Paginated[Item, CursorPagination]:
    raise NotImplementedError


def _offset_client(
    pages: dict[int, list[str]], *, page_size: int = 2, envelope: dict[str, Any] | None = None
) -> NemoClient:
    """Serve ``pages`` (page number -> names) with offset pagination metadata and optional envelope fields."""
    total_results = sum(len(v) for v in pages.values())
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(parse_qs(request.url.query.decode()).get("page", ["1"])[0])
        calls.append(page)
        data = pages.get(page, [])
        return httpx.Response(
            200,
            json={
                "data": [{"name": n} for n in data],
                **(envelope or {}),
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "current_page_size": len(data),
                    "total_pages": len(pages),
                    "total_results": total_results,
                },
            },
        )

    client = NemoClient(base_url="http://test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.calls = calls  # type: ignore[attr-defined]
    return client


def _cursor_client(pages: dict[str | None, tuple[list[str], str | None]]) -> NemoClient:
    """Serve ``pages`` (cursor -> (names, next_cursor)) with cursor pagination metadata."""
    total = sum(len(v[0]) for v in pages.values())

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = parse_qs(request.url.query.decode()).get("page_cursor", [None])[0]
        data, next_page = pages[cursor]
        return httpx.Response(
            200,
            json={"data": [{"name": n} for n in data], "total": total, "next_page": next_page, "prev_page": None},
        )

    return NemoClient(base_url="http://test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


# =============================================================================
# AllPagesResponse Tests
# =============================================================================


def test_all_pages_response_model_dump():
    """Test AllPagesResponse.model_dump() serialization."""
    # Create a mock pydantic model
    mock_item = Mock()
    mock_item.model_dump = Mock(return_value={"id": 1, "name": "test"})

    response = AllPagesResponse(
        data=[mock_item, {"id": 2}],
        total_items=2,
        total_pages=1,
        page_size=10,
    )

    result = response.model_dump()

    assert result["data"][0] == {"id": 1, "name": "test"}
    assert result["data"][1] == {"id": 2}
    assert result["pagination"]["total_results"] == 2
    assert result["pagination"]["page_size"] == 10


def test_all_pages_response_defaults():
    """Test AllPagesResponse with default values."""
    response = AllPagesResponse(
        data=[{"id": 1}, {"id": 2}],
        total_items=2,
        total_pages=1,
    )

    assert response.sort is None
    assert response.pagination.page == 1
    assert response.pagination.total_pages == 1
    assert response.pagination.current_page_size == 2


# =============================================================================
# AllCursorPagesResponse Tests
# =============================================================================


def test_all_cursor_pages_response_model_dump():
    """Test AllCursorPagesResponse.model_dump() serialization."""
    mock_item = Mock()
    mock_item.model_dump = Mock(return_value={"log": "line1"})

    response = AllCursorPagesResponse(
        data=[mock_item, {"log": "line2"}],
        limit=100,
    )

    result = response.model_dump()

    assert result["data"][0] == {"log": "line1"}
    assert result["data"][1] == {"log": "line2"}
    assert result["next_page"] is None


def test_all_cursor_pages_response_defaults():
    """Test AllCursorPagesResponse with default values."""
    response = AllCursorPagesResponse(
        data=[{"log": "line1"}],
    )

    assert response.next_page is None
    assert response._limit is None


# =============================================================================
# PaginationType Enum Tests
# =============================================================================


def test_pagination_type_values():
    """Test PaginationType enum values."""
    assert PaginationType.PAGE_NUMBER.value == "page_number"
    assert PaginationType.CURSOR.value == "cursor"


def test_pagination_type_is_string():
    """Test that PaginationType can be used as a string."""
    assert str(PaginationType.PAGE_NUMBER) == "PaginationType.PAGE_NUMBER"
    # Can compare with string value
    assert PaginationType.PAGE_NUMBER == "page_number"
    assert PaginationType.CURSOR == "cursor"


# =============================================================================
# collect_*_pages Tests (typed NemoPaginatedResponse)
# =============================================================================


def test_collect_offset_pages_single_page_keeps_server_metadata():
    client = _offset_client({1: ["a", "b"], 2: ["c"]})

    result = collect_offset_pages(client.send(list_items()), all_pages=False)

    assert isinstance(result, OffsetPageResponse)
    assert [item.name for item in result.data] == ["a", "b"]
    assert result.pagination.page == 1
    assert result.pagination.total_pages == 2
    assert result.pagination.total_results == 3
    assert client.calls == [1]
    dumped = result.model_dump()
    assert dumped["data"] == [{"name": "a"}, {"name": "b"}]
    assert dumped["pagination"]["total_pages"] == 2


def test_collect_offset_pages_all_pages_merges_every_page():
    client = _offset_client({1: ["a", "b"], 2: ["c", "d"], 3: ["e"]})

    result = collect_offset_pages(client.send(list_items()), all_pages=True, show_progress=False)

    assert isinstance(result, AllPagesResponse)
    assert [item.name for item in result.data] == ["a", "b", "c", "d", "e"]
    assert result.pagination.total_results == 5
    assert result.pagination.total_pages == 1
    assert result.pagination.page_size == 2
    assert client.calls == [1, 2, 3]


ENVELOPE = {"filter": {"role": "Viewer"}, "sort": "-created_at", "grouped_by": ["session_id"]}


def test_collect_offset_pages_single_page_keeps_envelope_fields_in_wire_order():
    """The server's sort/filter/grouped_by echo must survive into JSON output."""
    client = _offset_client({1: ["a"]}, envelope=ENVELOPE)

    dumped = collect_offset_pages(client.send(list_items()), all_pages=False).model_dump()

    assert list(dumped) == ["data", "filter", "sort", "grouped_by", "pagination"]
    assert dumped["filter"] == {"role": "Viewer"}
    assert dumped["sort"] == "-created_at"
    assert dumped["grouped_by"] == ["session_id"]


def test_collect_offset_pages_all_pages_keeps_first_page_envelope():
    client = _offset_client({1: ["a"], 2: ["b"]}, envelope=ENVELOPE)

    dumped = collect_offset_pages(client.send(list_items()), all_pages=True, show_progress=False).model_dump()

    assert dumped["data"] == [{"name": "a"}, {"name": "b"}]
    assert dumped["filter"] == {"role": "Viewer"}
    assert dumped["sort"] == "-created_at"
    assert dumped["pagination"]["total_results"] == 2


def test_collect_offset_pages_without_envelope_fields_keeps_a_null_sort():
    """List output always carries ``sort``; a server that echoes nothing yields null, as it always has."""
    client = _offset_client({1: ["a"]})

    dumped = collect_offset_pages(client.send(list_items()), all_pages=False).model_dump()

    assert list(dumped) == ["data", "sort", "pagination"]
    assert dumped["sort"] is None


def test_all_pages_response_without_envelope_keeps_legacy_shape():
    """Callers that build the merged response directly (fetch_all_pages, jobs) keep data/sort/pagination."""
    dumped = AllPagesResponse(data=[{"id": 1}], total_items=1, total_pages=1).model_dump()

    assert list(dumped) == ["data", "sort", "pagination"]
    assert dumped["sort"] is None


def test_collect_offset_pages_all_pages_empty():
    client = _offset_client({1: []})

    result = collect_offset_pages(client.send(list_items()), all_pages=True, show_progress=False)

    assert isinstance(result, AllPagesResponse)
    assert result.data == []
    assert result.pagination.total_results == 0


def test_collect_cursor_pages_single_page_keeps_cursors():
    client = _cursor_client({None: (["a", "b"], "cur-2"), "cur-2": (["c"], None)})

    result = collect_cursor_pages(client.send(list_logs()), all_pages=False)

    assert isinstance(result, CursorPageResponse)
    assert [item.name for item in result.data] == ["a", "b"]
    assert result.next_page == "cur-2"
    assert result.total == 3
    assert result.model_dump()["next_page"] == "cur-2"


def test_collect_cursor_pages_all_pages_follows_cursors():
    client = _cursor_client({None: (["a", "b"], "cur-2"), "cur-2": (["c"], "cur-3"), "cur-3": (["d"], None)})

    result = collect_cursor_pages(client.send(list_logs()), all_pages=True, limit=2, show_progress=False)

    assert isinstance(result, AllCursorPagesResponse)
    assert [item.name for item in result.data] == ["a", "b", "c", "d"]
    assert result.next_page is None
    assert result._limit == 2


def test_collect_pages_dispatches_on_pagination_type():
    offset = collect_pages(
        _offset_client({1: ["a"]}).send(list_items()),
        all_pages=False,
        pagination_type=PaginationType.PAGE_NUMBER,
    )
    cursor = collect_pages(
        _cursor_client({None: (["a"], None)}).send(list_logs()),
        all_pages=False,
        pagination_type=PaginationType.CURSOR,
    )

    assert isinstance(offset, OffsetPageResponse)
    assert isinstance(cursor, CursorPageResponse)


def test_collect_pages_propagates_http_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = NemoClient(base_url="http://test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(Exception):
        collect_offset_pages(client.send(list_items()), all_pages=False)


def test_model_dump_handles_plain_dict_items():
    result = OffsetPageResponse([{"name": "a"}, {"nested": [{"x": 1}]}], {"page": 1, "total_pages": 1})
    assert result.model_dump()["data"] == [{"name": "a"}, {"nested": [{"x": 1}]}]


# =============================================================================
# warn_if_more_pages Tests
# =============================================================================


def test_warn_if_more_pages_offset(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr("nemo_platform_ext.cli.core.pagination.add_warning", warnings.append)
    client = _offset_client({1: ["a"], 2: ["b"]})

    warn_if_more_pages(collect_offset_pages(client.send(list_items()), all_pages=False), PaginationType.PAGE_NUMBER)
    assert len(warnings) == 1

    warnings.clear()
    warn_if_more_pages(
        collect_offset_pages(client.send(list_items()), all_pages=True, show_progress=False),
        PaginationType.PAGE_NUMBER,
    )
    assert warnings == []


def test_warn_if_more_pages_cursor(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr("nemo_platform_ext.cli.core.pagination.add_warning", warnings.append)
    client = _cursor_client({None: (["a"], "cur-2"), "cur-2": (["b"], None)})

    warn_if_more_pages(collect_cursor_pages(client.send(list_logs()), all_pages=False), PaginationType.CURSOR)
    assert len(warnings) == 1

    warnings.clear()
    warn_if_more_pages(
        collect_cursor_pages(client.send(list_logs()), all_pages=True, show_progress=False), PaginationType.CURSOR
    )
    assert warnings == []


def test_warn_if_more_pages_not_paginated(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr("nemo_platform_ext.cli.core.pagination.add_warning", warnings.append)
    warn_if_more_pages(object(), PaginationType.NOT_PAGINATED)
    assert warnings == []

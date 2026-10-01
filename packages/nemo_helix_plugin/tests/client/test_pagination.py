# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Paginated[T] — automatic pagination via return type marker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.compat import AsyncLegacyPaginatedResponse, LegacyPaginatedResponse
from nemo_helix_plugin.client.endpoint import get
from nemo_helix_plugin.client.errors import NemoResponseValidationError
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.client.response import AsyncNemoPaginatedCall, AsyncNemoPaginatedResponse, NemoPaginatedResponse
from nemo_helix_plugin.client.types import CursorPagination, OffsetPagination, Paginated, RetryPolicy
from pydantic import BaseModel

BASE = "http://test:8000"


class Item(BaseModel):
    id: int
    name: str


@get("/apis/test/v2/workspaces/{workspace}/items")
def LIST_ITEMS(*, workspace: str | None = None) -> Paginated[Item]:
    raise NotImplementedError


def _page_response(items: list[dict], page: int, total_pages: int, page_size: int = 2) -> httpx.Response:
    """Helper to build a paginated response matching NemoListResponse format."""
    return httpx.Response(
        200,
        request=httpx.Request("GET", f"{BASE}/apis/test/v2/workspaces/default/items"),
        json={
            "data": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "current_page_size": len(items),
                "total_pages": total_pages,
                "total_results": total_pages * page_size,
            },
        },
    )


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


class TestPaginatedSync:
    def test_single_page_iteration(self) -> None:
        """When total_pages=1, iterating should yield items from only one page."""
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = _page_response(
            [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], page=1, total_pages=1
        )

        client = NemoClient(base_url=BASE, workspace="default", http_client=mock_http)
        resp = client.send(LIST_ITEMS())

        assert isinstance(resp, NemoPaginatedResponse)
        items = list(resp.items())
        assert len(items) == 2
        assert items[0].name == "a"
        assert items[1].name == "b"
        # Only one request made (no additional page fetches)
        assert mock_http.request.call_count == 1

    def test_response_iterates_items_directly_and_exposes_first_page_data(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = _page_response([{"id": 1, "name": "a"}], page=1, total_pages=1)

        response = NemoClient(base_url=BASE, workspace="default", http_client=mock_http).send(LIST_ITEMS())

        assert [item.name for item in response] == ["a"]
        assert [item.name for item in response.data] == ["a"]

    def test_legacy_paginated_response_reuses_common_page_helpers(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = _page_response([{"id": 1, "name": "a"}], page=1, total_pages=1)

        response = NemoClient(base_url=BASE, workspace="default", http_client=mock_http).send(LIST_ITEMS())
        legacy = LegacyPaginatedResponse(response)
        page = next(legacy.iter_pages())

        assert [item.name for item in legacy.data] == ["a"]
        assert [item.name for item in legacy] == ["a"]
        assert page.has_next_page() is False
        assert page.next_page_info() is None

    def test_multi_page_iteration(self) -> None:
        """Iterating should automatically fetch all pages."""
        mock_http = MagicMock(spec=httpx.Client)
        # First call (page 1) via send()
        mock_http.request.side_effect = [
            _page_response([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], page=1, total_pages=3),
            # Pages 2 and 3 fetched by _fetch_page
            _page_response([{"id": 3, "name": "c"}, {"id": 4, "name": "d"}], page=2, total_pages=3),
            _page_response([{"id": 5, "name": "e"}], page=3, total_pages=3),
        ]

        client = NemoClient(base_url=BASE, workspace="default", http_client=mock_http)
        resp = client.send(LIST_ITEMS())

        items = list(resp.items())
        assert len(items) == 5
        assert [i.name for i in items] == ["a", "b", "c", "d", "e"]
        assert mock_http.request.call_count == 3

    def test_data_returns_page_result_with_metadata(self) -> None:
        """data() returns a PageResult with items and pagination metadata."""
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = _page_response([{"id": 1, "name": "a"}], page=1, total_pages=5)

        client = NemoClient(base_url=BASE, workspace="default", http_client=mock_http)
        resp = client.send(LIST_ITEMS())

        page = resp.page()
        assert len(page.items) == 1
        assert page.items[0].name == "a"
        assert page.metadata["page"] == 1
        assert page.metadata["total_pages"] == 5
        assert page.metadata["total_results"] == 10
        assert page.metadata["page_size"] == 2
        assert page.metadata["current_page_size"] == 1
        # No additional requests for data()
        assert mock_http.request.call_count == 1

    def test_empty_page(self) -> None:
        """Empty data list should yield nothing."""
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = _page_response([], page=1, total_pages=1)

        client = NemoClient(base_url=BASE, workspace="default", http_client=mock_http)
        resp = client.send(LIST_ITEMS())

        items = list(resp.items())
        assert items == []

    @pytest.mark.parametrize("iteration", ["page", "items", "pages"])
    def test_missing_pagination_metadata_is_invalid(self, iteration: str) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/workspaces/default/items"),
            json={"data": [{"id": 1, "name": "a"}]},
        )

        client = NemoClient(base_url=BASE, workspace="default", http_client=mock_http)
        resp = client.send(LIST_ITEMS())

        with pytest.raises(NemoResponseValidationError):
            if iteration == "page":
                resp.page()
            elif iteration == "items":
                list(resp.items())
            else:
                list(resp.pages())

    def test_partial_pagination_metadata_is_invalid(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/workspaces/default/items"),
            json={"data": [{"id": 1, "name": "a"}], "pagination": {"page": 1}},
        )

        resp = NemoClient(base_url=BASE, workspace="default", http_client=mock_http).send(LIST_ITEMS())

        with pytest.raises(NemoResponseValidationError):
            resp.page()

    def test_page_query_param_passed_on_subsequent_pages(self) -> None:
        """Subsequent page fetches should include page=N in query params."""
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            _page_response([{"id": 1, "name": "a"}], page=1, total_pages=2),
            _page_response([{"id": 2, "name": "b"}], page=2, total_pages=2),
        ]

        client = NemoClient(base_url=BASE, workspace="default", http_client=mock_http)
        resp = client.send(LIST_ITEMS())
        list(resp.items())  # consume all pages

        # Second call should have page=2 in params
        second_call_params = mock_http.request.call_args_list[1][1]["params"]
        assert second_call_params["page"] == 2

    @pytest.mark.parametrize("iteration", ["items", "pages"])
    def test_iteration_continues_after_response_page(self, iteration: str) -> None:
        """A request beginning after page one must not fetch that page again."""
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            _page_response([{"id": 3, "name": "c"}], page=2, total_pages=3),
            _page_response([{"id": 4, "name": "d"}], page=3, total_pages=3),
        ]
        response = NemoClient(base_url=BASE, workspace="default", http_client=mock_http).send(LIST_ITEMS())

        if iteration == "items":
            names = [item.name for item in response.items()]
        else:
            names = [item.name for page in response.pages() for item in page.items]

        assert names == ["c", "d"]
        assert mock_http.request.call_count == 2
        assert mock_http.request.call_args_list[1].kwargs["params"]["page"] == 3


# ---------------------------------------------------------------------------
# Via method() descriptor
# ---------------------------------------------------------------------------


class TestPaginatedViaMethod:
    def test_method_descriptor_returns_paginated_response(self) -> None:
        """method() wrapping a Paginated endpoint should return NemoPaginatedResponse."""
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = _page_response([{"id": 1, "name": "a"}], page=1, total_pages=1)

        class _Methods:
            list_items = method(LIST_ITEMS)

        class TestClient(_Methods, NemoClient):
            pass

        client = TestClient(base_url=BASE, workspace="default", http_client=mock_http)
        resp = client.list_items()

        # Client options are applied but shouldn't break pagination
        items = list(resp.items())
        assert len(items) == 1
        assert items[0].name == "a"


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


class TestPaginatedAsync:
    @pytest.mark.asyncio
    async def test_async_multi_page_iteration(self) -> None:
        """Async iteration should automatically fetch all pages."""
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.side_effect = [
            _page_response([{"id": 1, "name": "a"}], page=1, total_pages=2),
            _page_response([{"id": 2, "name": "b"}], page=2, total_pages=2),
        ]

        client = AsyncNemoClient(base_url=BASE, workspace="default", http_client=mock_http)
        resp = await client.send(LIST_ITEMS())

        assert isinstance(resp, AsyncNemoPaginatedResponse)
        items = [item async for item in resp.items()]
        assert len(items) == 2
        assert items[0].name == "a"
        assert items[1].name == "b"

    @pytest.mark.asyncio
    async def test_async_response_iterates_items_directly_and_exposes_first_page_data(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.return_value = _page_response([{"id": 1, "name": "a"}], page=1, total_pages=1)

        response = await AsyncNemoClient(base_url=BASE, workspace="default", http_client=mock_http).send(LIST_ITEMS())

        assert [item.name async for item in response] == ["a"]
        assert [item.name for item in response.data] == ["a"]

    @pytest.mark.asyncio
    async def test_async_paginated_call_is_awaitable_iterable_and_cached(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.return_value = _page_response([{"id": 1, "name": "a"}], page=1, total_pages=1)
        client = AsyncNemoClient(base_url=BASE, workspace="default", http_client=mock_http)
        loader_calls = 0

        async def loader() -> AsyncNemoPaginatedResponse[Item, OffsetPagination]:
            nonlocal loader_calls
            loader_calls += 1
            return await client.send(LIST_ITEMS())

        call = AsyncNemoPaginatedCall(loader)

        response = await call
        assert [item.name for item in response.data] == ["a"]
        assert [item.name async for item in call] == ["a"]
        assert loader_calls == 1
        assert mock_http.request.call_count == 1

    @pytest.mark.asyncio
    async def test_async_legacy_paginated_response_reuses_common_page_helpers(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.return_value = _page_response([{"id": 1, "name": "a"}], page=1, total_pages=1)
        response = AsyncNemoClient(base_url=BASE, workspace="default", http_client=mock_http).send(LIST_ITEMS())
        legacy = AsyncLegacyPaginatedResponse(response)

        page = await legacy

        assert [item.name for item in page.data] == ["a"]
        assert [item.name async for item in legacy] == ["a"]
        assert page.has_next_page() is False
        assert page.next_page_info() is None

    @pytest.mark.asyncio
    async def test_async_data_returns_page_result(self) -> None:
        """Async data() returns a PageResult with metadata."""
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.return_value = _page_response([{"id": 1, "name": "a"}], page=1, total_pages=3)

        client = AsyncNemoClient(base_url=BASE, workspace="default", http_client=mock_http)
        resp = await client.send(LIST_ITEMS())

        page = resp.page()
        assert len(page.items) == 1
        assert page.metadata["page"] == 1
        assert page.metadata["total_pages"] == 3
        assert mock_http.request.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("iteration", ["items", "pages"])
    async def test_async_iteration_continues_after_response_page(self, iteration: str) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.side_effect = [
            _page_response([{"id": 3, "name": "c"}], page=2, total_pages=3),
            _page_response([{"id": 4, "name": "d"}], page=3, total_pages=3),
        ]
        response = await AsyncNemoClient(base_url=BASE, workspace="default", http_client=mock_http).send(LIST_ITEMS())

        if iteration == "items":
            names = [item.name async for item in response.items()]
        else:
            names = [item.name async for page in response.pages() for item in page.items]

        assert names == ["c", "d"]
        assert mock_http.request.call_count == 2
        assert mock_http.request.call_args_list[1].kwargs["params"]["page"] == 3


# ---------------------------------------------------------------------------
# Retry on subsequent pages
# ---------------------------------------------------------------------------


class TestPaginatedRetry:
    def test_retry_on_subsequent_page_503(self) -> None:
        """A 503 on page 2 should be retried when a retry policy is configured."""
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            # Page 1: success
            _page_response([{"id": 1, "name": "a"}], page=1, total_pages=2),
            # Page 2: first attempt 503, second attempt success
            httpx.Response(
                503,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/workspaces/default/items"),
                json={"detail": "unavailable"},
            ),
            _page_response([{"id": 2, "name": "b"}], page=2, total_pages=2),
        ]

        client = NemoClient(
            base_url=BASE,
            workspace="default",
            http_client=mock_http,
            retry=RetryPolicy(max_retries=2, backoff_base=0.0),
        )
        items = list(client.send(LIST_ITEMS()).items())

        assert len(items) == 2
        assert [i.name for i in items] == ["a", "b"]
        assert mock_http.request.call_count == 3  # page 1 + page 2 fail + page 2 retry


# ---------------------------------------------------------------------------
# Custom pagination strategy
# ---------------------------------------------------------------------------


class ResultsPagination(OffsetPagination):
    """Custom strategy: items in 'results', page param is 'offset'."""

    items_field = "results"
    page_param = "offset"


@get("/apis/test/v2/workspaces/{workspace}/things")
def LIST_THINGS(*, workspace: str | None = None) -> Paginated[Item, ResultsPagination]:
    raise NotImplementedError


class TestCustomStrategy:
    def test_custom_items_field(self) -> None:
        """Custom strategy should extract items from the configured field."""
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/workspaces/default/things"),
            json={
                "results": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
                "pagination": {
                    "page": 1,
                    "page_size": 10,
                    "current_page_size": 2,
                    "total_pages": 1,
                    "total_results": 2,
                },
            },
        )

        client = NemoClient(base_url=BASE, workspace="default", http_client=mock_http)
        resp = client.send(LIST_THINGS())

        items = list(resp.items())
        assert len(items) == 2
        assert items[0].name == "a"

    def test_custom_page_param(self) -> None:
        """Custom strategy should use the configured page query param."""
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/workspaces/default/things"),
                json={
                    "results": [{"id": 1, "name": "a"}],
                    "pagination": {
                        "page": 1,
                        "page_size": 1,
                        "current_page_size": 1,
                        "total_pages": 2,
                        "total_results": 2,
                    },
                },
            ),
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/workspaces/default/things"),
                json={
                    "results": [{"id": 2, "name": "b"}],
                    "pagination": {
                        "page": 2,
                        "page_size": 1,
                        "current_page_size": 1,
                        "total_pages": 2,
                        "total_results": 2,
                    },
                },
            ),
        ]

        client = NemoClient(base_url=BASE, workspace="default", http_client=mock_http)
        items = list(client.send(LIST_THINGS()).items())

        assert len(items) == 2
        # Verify the second call used "offset" not "page"
        second_call_params = mock_http.request.call_args_list[1][1]["params"]
        assert "offset" in second_call_params
        assert second_call_params["offset"] == 2


# ---------------------------------------------------------------------------
# Cursor pagination
# ---------------------------------------------------------------------------


@get("/apis/test/v2/workspaces/{workspace}/logs")
def LIST_LOGS(
    *, workspace: str | None = None, query_params: dict[str, str | int] | None = None
) -> Paginated[Item, CursorPagination]:
    raise NotImplementedError


def _cursor_response(
    items: list[dict], *, total: int, next_page: str | None, prev_page: str | None = None
) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("GET", f"{BASE}/apis/test/v2/workspaces/default/logs"),
        json={
            "data": items,
            "total": total,
            "next_page": next_page,
            "prev_page": prev_page,
        },
    )


class TestCursorPagination:
    def test_page_exposes_typed_cursor_metadata(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = _cursor_response([{"id": 1, "name": "a"}], total=2, next_page="cursor-2")

        page = NemoClient(base_url=BASE, workspace="default", http_client=mock_http).send(LIST_LOGS()).page()

        assert [item.name for item in page.items] == ["a"]
        assert page.metadata == {"total": 2, "next_page": "cursor-2", "prev_page": None}

    def test_items_follow_cursors_and_preserve_filters(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            _cursor_response([{"id": 1, "name": "a"}], total=2, next_page="cursor-2"),
            _cursor_response([{"id": 2, "name": "b"}], total=2, next_page=None, prev_page="cursor-1"),
        ]
        response = NemoClient(base_url=BASE, workspace="default", http_client=mock_http).send(
            LIST_LOGS(query_params={"limit": 1, "attempt_id": 3})
        )

        assert [item.name for item in response.items()] == ["a", "b"]
        assert mock_http.request.call_count == 2
        assert mock_http.request.call_args_list[1].kwargs["params"] == {
            "limit": 1,
            "attempt_id": 3,
            "page_cursor": "cursor-2",
        }

    def test_pages_follow_cursors_from_explicit_start(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            _cursor_response([{"id": 2, "name": "b"}], total=3, next_page="cursor-3", prev_page="cursor-1"),
            _cursor_response([{"id": 3, "name": "c"}], total=3, next_page=None, prev_page="cursor-2"),
        ]
        response = NemoClient(base_url=BASE, workspace="default", http_client=mock_http).send(
            LIST_LOGS(query_params={"page_cursor": "cursor-2"})
        )

        pages = list(response.pages())

        assert [[item.name for item in page.items] for page in pages] == [["b"], ["c"]]
        assert pages[0].metadata["prev_page"] == "cursor-1"
        assert pages[1].metadata["next_page"] is None
        assert mock_http.request.call_args_list[1].kwargs["params"]["page_cursor"] == "cursor-3"

    @pytest.mark.asyncio
    async def test_async_items_follow_cursors(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.side_effect = [
            _cursor_response([{"id": 1, "name": "a"}], total=2, next_page="cursor-2"),
            _cursor_response([{"id": 2, "name": "b"}], total=2, next_page=None, prev_page="cursor-1"),
        ]
        response = await AsyncNemoClient(base_url=BASE, workspace="default", http_client=mock_http).send(LIST_LOGS())

        assert [item.name async for item in response.items()] == ["a", "b"]
        assert mock_http.request.call_args_list[1].kwargs["params"]["page_cursor"] == "cursor-2"

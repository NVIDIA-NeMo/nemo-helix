# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for client-side options (exist_ok), RetryPolicy, and param validation."""

from __future__ import annotations

from datetime import timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient, _retry_after
from nemo_platform_plugin.client.endpoint import delete, get, post
from nemo_platform_plugin.client.errors import (
    ConflictError,
    NemoHTTPError,
    NemoResponseValidationError,
    NemoTransportError,
    NotFoundError,
)
from nemo_platform_plugin.client.method import method
from nemo_platform_plugin.client.types import BinaryContent, PreparedRequest, RetryPolicy, Stream
from pydantic import BaseModel

BASE = "http://test:8000"
STAINLESS_RETRY = RetryPolicy(
    max_retries=1,
    backoff_base=0.25,
    retryable_status_codes=(408, 409, 429),
    retry_all_server_errors=True,
    respect_retry_decision_headers=True,
    respect_retry_after_headers=True,
)


class ItemRequest(BaseModel):
    name: str


class ItemResponse(BaseModel):
    id: int
    name: str


@pytest.mark.parametrize("client_cls", [NemoClient, AsyncNemoClient])
def test_resolve_workspace_prefers_explicit_workspace(
    client_cls: type[NemoClient] | type[AsyncNemoClient],
) -> None:
    client = client_cls(base_url=BASE, workspace="client-ws")

    assert client.resolve_workspace("request-ws") == "request-ws"


@pytest.mark.parametrize("client_cls", [NemoClient, AsyncNemoClient])
def test_resolve_workspace_uses_client_workspace(
    client_cls: type[NemoClient] | type[AsyncNemoClient],
) -> None:
    client = client_cls(base_url=BASE, workspace="client-ws")

    assert client.resolve_workspace() == "client-ws"


@pytest.mark.parametrize("client_cls", [NemoClient, AsyncNemoClient])
def test_resolve_workspace_uses_default_workspace(
    client_cls: type[NemoClient] | type[AsyncNemoClient],
) -> None:
    client = client_cls(base_url=BASE, workspace=None)

    assert client.resolve_workspace() == "default"


@pytest.mark.parametrize("client_cls", [NemoClient, AsyncNemoClient])
def test_require_workspace_rejects_missing_workspace(
    client_cls: type[NemoClient] | type[AsyncNemoClient],
) -> None:
    client = client_cls(base_url=BASE, workspace=None)

    with pytest.raises(ValueError, match="workspace must be provided"):
        client.require_workspace()


# ---------------------------------------------------------------------------
# Endpoint definitions with client options
# ---------------------------------------------------------------------------


@get("/apis/test/v2/items/{name}")
def GET_ITEM(*, name: str) -> ItemResponse:
    raise NotImplementedError


@delete("/apis/test/v2/items/{name}")
def DELETE_ITEM(*, name: str) -> None:
    raise NotImplementedError


@get("/apis/test/v2/download")
def DOWNLOAD() -> BinaryContent:
    raise NotImplementedError


@get("/apis/test/v2/events")
def EVENTS() -> Stream[ItemResponse]:
    raise NotImplementedError


def _get_item_on_conflict(body: ItemRequest, workspace: str | None) -> PreparedRequest[ItemResponse]:
    """Resolver: on a create 409, retrieve the existing item by name."""
    return GET_ITEM(name=body.name)


@post("/apis/test/v2/items", get_on_conflict=_get_item_on_conflict)
def CREATE_ITEM(body: ItemRequest, *, exist_ok: bool = False) -> ItemResponse:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# exist_ok: stripped from request, stashed in client_options
# ---------------------------------------------------------------------------


class TestExistOkOption:
    def test_exist_ok_stripped_from_request(self) -> None:
        prepared = CREATE_ITEM(ItemRequest(name="alice"), exist_ok=True)

        assert isinstance(prepared, PreparedRequest)
        assert prepared.content is not None
        assert prepared.client_options is not None
        assert prepared.client_options["exist_ok"] is True

    def test_exist_ok_default_false(self) -> None:
        prepared = CREATE_ITEM(ItemRequest(name="alice"))

        assert prepared.client_options is not None
        assert prepared.client_options["exist_ok"] is False

    def test_endpoint_without_options_has_none(self) -> None:
        prepared = GET_ITEM(name="alice")
        assert prepared.client_options is None


# ---------------------------------------------------------------------------
# get_on_conflict: resolver wiring at request-build time
# ---------------------------------------------------------------------------


class TestConflictResolverWiring:
    def test_resolver_builds_get_at_request_time(self) -> None:
        """The create request carries a prebuilt GET derived from the body."""
        prepared = CREATE_ITEM(ItemRequest(name="alice"), exist_ok=True)

        assert prepared.on_conflict_get is not None
        get_req = prepared.on_conflict_get
        assert get_req.method == "GET"
        assert get_req.path_template == "/apis/test/v2/items/{name}"
        assert get_req.path_params == {"name": "alice"}

    def test_endpoint_without_resolver_has_no_on_conflict_get(self) -> None:
        # A create endpoint that declares no resolver (and no exist_ok) carries
        # no prebuilt GET.
        @post("/apis/test/v2/items")
        def create_plain(body: ItemRequest) -> ItemResponse:
            raise NotImplementedError

        prepared = create_plain(ItemRequest(name="alice"))
        assert prepared.on_conflict_get is None


# ---------------------------------------------------------------------------
# exist_ok: resolved via send() by replaying the linked GET on 409
# ---------------------------------------------------------------------------


def _mock_http(*responses: httpx.Response) -> MagicMock:
    """A sync httpx.Client whose .request() returns the given responses in order."""
    mock = MagicMock(spec=httpx.Client)
    mock.request.side_effect = list(responses)
    return mock


def _resp(
    status: int, body: dict, http_method: str = "POST", url: str = f"{BASE}/apis/test/v2/items"
) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request(http_method, url), json=body)


class TestExistOkViaSend:
    def test_409_with_exist_ok_replays_get_and_returns_entity(self) -> None:
        """409 (real error body) + exist_ok -> auto-GET returns the existing entity."""
        mock = _mock_http(
            _resp(409, {"detail": "Item 'alice' already exists"}),
            _resp(200, {"id": 7, "name": "alice"}, http_method="GET", url=f"{BASE}/apis/test/v2/items/alice"),
        )
        client = NemoClient(base_url=BASE, http_client=mock)

        resp = client.send(CREATE_ITEM(ItemRequest(name="alice"), exist_ok=True))

        assert resp.body is not None
        assert resp.body.id == 7
        assert resp.body.name == "alice"
        # POST then follow-up GET.
        assert mock.request.call_count == 2
        assert mock.request.call_args_list[1].args[0] == "GET"

    def test_409_with_exist_ok_is_not_retried_before_resolving(self) -> None:
        """A retry policy that lists 409 must not replay the POST when the caller opted into exist_ok."""
        mock = _mock_http(
            _resp(409, {"detail": "Item 'alice' already exists"}),
            _resp(200, {"id": 7, "name": "alice"}, http_method="GET", url=f"{BASE}/apis/test/v2/items/alice"),
        )
        client = NemoClient(
            base_url=BASE,
            http_client=mock,
            retry=RetryPolicy(max_retries=2, retryable_status_codes=(409,), backoff_base=0.0),
        )

        resp = client.send(CREATE_ITEM(ItemRequest(name="alice"), exist_ok=True))

        assert resp.body is not None and resp.body.id == 7
        assert [call.args[0] for call in mock.request.call_args_list] == ["POST", "GET"]

    def test_409_without_exist_ok_is_still_retried_when_the_policy_says_so(self) -> None:
        mock = _mock_http(
            _resp(409, {"detail": "Item 'alice' already exists"}),
            _resp(201, {"id": 1, "name": "alice"}),
        )
        client = NemoClient(
            base_url=BASE,
            http_client=mock,
            retry=RetryPolicy(max_retries=2, retryable_status_codes=(409,), backoff_base=0.0),
        )

        resp = client.send(CREATE_ITEM(ItemRequest(name="alice")))

        assert resp.http_response.status_code == 201
        assert [call.args[0] for call in mock.request.call_args_list] == ["POST", "POST"]

    def test_409_without_exist_ok_raises_conflict(self) -> None:
        mock = _mock_http(_resp(409, {"detail": "Item 'alice' already exists"}))
        client = NemoClient(base_url=BASE, http_client=mock)

        with pytest.raises(ConflictError) as exc_info:
            client.send(CREATE_ITEM(ItemRequest(name="alice")))

        assert exc_info.value.status_code == 409
        assert mock.request.call_count == 1  # no follow-up GET

    def test_non_409_with_exist_ok_passes_through(self) -> None:
        mock = _mock_http(_resp(201, {"id": 1, "name": "alice"}))
        client = NemoClient(base_url=BASE, http_client=mock)

        resp = client.send(CREATE_ITEM(ItemRequest(name="alice"), exist_ok=True))

        assert resp.http_response.status_code == 201
        assert resp.body is not None
        assert resp.body.name == "alice"
        assert mock.request.call_count == 1

    def test_get_404_after_409_is_surfaced(self) -> None:
        """Entity deleted between the 409 and the follow-up GET -> NotFoundError."""
        mock = _mock_http(
            _resp(409, {"detail": "Item 'alice' already exists"}),
            _resp(404, {"detail": "not found"}, http_method="GET", url=f"{BASE}/apis/test/v2/items/alice"),
        )
        client = NemoClient(base_url=BASE, http_client=mock)

        with pytest.raises(NotFoundError):
            client.send(CREATE_ITEM(ItemRequest(name="alice"), exist_ok=True))

    def test_caller_headers_are_preserved_on_conflict_replay(self) -> None:
        """Per-request headers passed to send() are carried onto the follow-up GET."""
        mock = _mock_http(
            _resp(409, {"detail": "Item 'alice' already exists"}),
            _resp(200, {"id": 7, "name": "alice"}, http_method="GET", url=f"{BASE}/apis/test/v2/items/alice"),
        )
        client = NemoClient(base_url=BASE, http_client=mock)

        client.send(CREATE_ITEM(ItemRequest(name="alice"), exist_ok=True), headers={"X-Trace": "abc"})

        get_headers = mock.request.call_args_list[1].kwargs["headers"]
        assert get_headers["X-Trace"] == "abc"


class TestExistOkViaMethod:
    def test_flat_method_call_resolves_conflict(self) -> None:
        """client.create_item(..., exist_ok=True) returns the entity on 409."""
        mock = _mock_http(
            _resp(409, {"detail": "Item 'alice' already exists"}),
            _resp(200, {"id": 7, "name": "alice"}, http_method="GET", url=f"{BASE}/apis/test/v2/items/alice"),
        )

        class _Methods:
            create_item = method(CREATE_ITEM)

        class TestClient(_Methods, NemoClient):
            pass

        client = TestClient(base_url=BASE, http_client=mock)
        resp = client.create_item(body=ItemRequest(name="alice"), exist_ok=True)

        assert resp.body is not None
        assert resp.body.id == 7


class TestAsyncExistOkViaSend:
    @pytest.mark.asyncio
    async def test_409_with_exist_ok_replays_get_and_returns_entity(self) -> None:
        mock = AsyncMock(spec=httpx.AsyncClient)
        mock.request.side_effect = [
            _resp(409, {"detail": "Item 'alice' already exists"}),
            _resp(200, {"id": 7, "name": "alice"}, http_method="GET", url=f"{BASE}/apis/test/v2/items/alice"),
        ]
        client = AsyncNemoClient(base_url=BASE, http_client=mock)

        resp = await client.send(CREATE_ITEM(ItemRequest(name="alice"), exist_ok=True))

        assert resp.body is not None
        assert resp.body.id == 7
        assert resp.body.name == "alice"
        assert mock.request.call_count == 2

    @pytest.mark.asyncio
    async def test_409_without_exist_ok_raises_conflict(self) -> None:
        mock = AsyncMock(spec=httpx.AsyncClient)
        mock.request.side_effect = [_resp(409, {"detail": "Item 'alice' already exists"})]
        client = AsyncNemoClient(base_url=BASE, http_client=mock)

        with pytest.raises(ConflictError):
            await client.send(CREATE_ITEM(ItemRequest(name="alice")))
        assert mock.request.call_count == 1


class TestAsyncExistOkViaMethod:
    @pytest.mark.asyncio
    async def test_flat_method_call_resolves_conflict(self) -> None:
        """await client.create_item(..., exist_ok=True) returns the entity on 409."""
        mock = AsyncMock(spec=httpx.AsyncClient)
        mock.request.side_effect = [
            _resp(409, {"detail": "Item 'alice' already exists"}),
            _resp(200, {"id": 7, "name": "alice"}, http_method="GET", url=f"{BASE}/apis/test/v2/items/alice"),
        ]

        class _Methods:
            create_item = method(CREATE_ITEM)

        class TestAsyncClient(_Methods, AsyncNemoClient):
            pass

        client = TestAsyncClient(base_url=BASE, http_client=mock)
        resp = await client.create_item(body=ItemRequest(name="alice"), exist_ok=True)

        assert resp.body is not None
        assert resp.body.id == 7
        assert mock.request.call_count == 2


# ---------------------------------------------------------------------------
# Param validation at decoration time
# ---------------------------------------------------------------------------


class TestParamValidation:
    def test_unknown_param_raises_at_decoration_time(self) -> None:
        with pytest.raises(TypeError, match="unrecognised parameters"):

            @post("/apis/test/v2/items")
            def bad_endpoint(body: ItemRequest, *, bogus: str = "oops") -> ItemResponse:
                raise NotImplementedError

    def test_blessed_param_is_allowed(self) -> None:
        @post("/apis/test/v2/items", get_on_conflict=_get_item_on_conflict)
        def ok_endpoint(body: ItemRequest, *, exist_ok: bool = False) -> ItemResponse:
            raise NotImplementedError

        prepared = ok_endpoint(ItemRequest(name="x"))
        assert isinstance(prepared, PreparedRequest)

    def test_exist_ok_without_resolver_raises_at_decoration_time(self) -> None:
        # exist_ok is inert without a resolver to fetch the entity on 409, so it
        # is rejected up front rather than failing on the first real conflict.
        with pytest.raises(TypeError, match="get_on_conflict"):

            @post("/apis/test/v2/items")
            def bad_endpoint(body: ItemRequest, *, exist_ok: bool = False) -> ItemResponse:
                raise NotImplementedError

    def test_path_params_are_allowed(self) -> None:
        @get("/items/{workspace}/{name}")
        def ok_endpoint(*, workspace: str, name: str) -> ItemResponse:
            raise NotImplementedError

        prepared = ok_endpoint(workspace="default", name="x")
        assert prepared.path_params == {"workspace": "default", "name": "x"}


# ---------------------------------------------------------------------------
# RetryPolicy: client-level default
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_retry_on_503(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            httpx.Response(
                503,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"detail": "Service Unavailable"},
            ),
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"id": 1, "name": "alice"},
            ),
        ]

        client = NemoClient(
            base_url=BASE,
            http_client=mock_http,
            retry=RetryPolicy(max_retries=2, backoff_base=0.0),
        )
        resp = client.send(GET_ITEM(name="alice"))

        assert resp.http_response.status_code == 200
        assert resp.body.name == "alice"
        assert mock_http.request.call_count == 2

    def test_retry_exhausted_raises(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = httpx.Response(
            503,
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
            json={"detail": "Service Unavailable"},
        )

        client = NemoClient(
            base_url=BASE,
            http_client=mock_http,
            retry=RetryPolicy(max_retries=2, backoff_base=0.0),
        )

        with pytest.raises(NemoHTTPError) as exc_info:
            client.send(GET_ITEM(name="alice"))

        assert exc_info.value.status_code == 503
        assert mock_http.request.call_count == 3

    def test_no_retry_on_non_retryable_status(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = httpx.Response(
            404,
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
            json={"detail": "Not found"},
        )

        client = NemoClient(
            base_url=BASE,
            http_client=mock_http,
            retry=RetryPolicy(max_retries=2, backoff_base=0.0),
        )

        with pytest.raises(NemoHTTPError) as exc_info:
            client.send(GET_ITEM(name="alice"))

        assert exc_info.value.status_code == 404
        assert mock_http.request.call_count == 1

    def test_retry_on_transport_error(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            httpx.ConnectError("Connection refused"),
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"id": 1, "name": "alice"},
            ),
        ]

        client = NemoClient(
            base_url=BASE,
            http_client=mock_http,
            retry=RetryPolicy(max_retries=2, backoff_base=0.0),
        )
        resp = client.send(GET_ITEM(name="alice"))

        assert resp.body.name == "alice"
        assert mock_http.request.call_count == 2

    def test_exhausted_transport_error_is_wrapped(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = httpx.ConnectError("Connection refused")
        client = NemoClient(
            base_url=BASE,
            http_client=mock_http,
            retry=RetryPolicy(max_retries=2, backoff_base=0.0),
        )

        with pytest.raises(NemoTransportError) as exc_info:
            client.send(GET_ITEM(name="alice"))

        assert isinstance(exc_info.value.error, httpx.ConnectError)
        assert exc_info.value.request is None
        assert mock_http.request.call_count == 3

    def test_per_request_retry_overrides_client_default(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = httpx.Response(
            503,
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
            json={"detail": "unavailable"},
        )

        client = NemoClient(
            base_url=BASE,
            http_client=mock_http,
            retry=RetryPolicy(max_retries=5, backoff_base=0.0),
        )

        with pytest.raises(NemoHTTPError):
            client.send(GET_ITEM(name="alice"), retry=RetryPolicy(max_retries=1, backoff_base=0.0))

        assert mock_http.request.call_count == 2

    def test_no_retry_without_policy(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = httpx.Response(
            503,
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
            json={"detail": "unavailable"},
        )

        client = NemoClient(base_url=BASE, http_client=mock_http)

        with pytest.raises(NemoHTTPError) as exc_info:
            client.send(GET_ITEM(name="alice"))

        assert exc_info.value.status_code == 503
        assert mock_http.request.call_count == 1

    def test_standalone_retry_policy_defaults_are_unchanged(self) -> None:
        policy = RetryPolicy()

        assert policy.retryable_status_codes == (502, 503, 504, 429)
        assert policy.retry_all_server_errors is False
        assert policy.respect_retry_decision_headers is False
        assert policy.respect_retry_after_headers is False

    @pytest.mark.parametrize("status_code", [408, 409, 500])
    def test_standalone_policy_does_not_add_stainless_statuses(self, status_code: int) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = httpx.Response(
            status_code,
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
            json={"detail": "error"},
        )
        client = NemoClient(
            base_url=BASE,
            http_client=mock_http,
            retry=RetryPolicy(max_retries=1, backoff_base=0.0),
        )

        with pytest.raises(NemoHTTPError):
            client.send(GET_ITEM(name="alice"))

        assert mock_http.request.call_count == 1

    @pytest.mark.parametrize("status_code", [408, 409, 500])
    def test_stainless_policy_retries_all_expected_statuses(self, status_code: int) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            httpx.Response(
                status_code,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"detail": "error"},
            ),
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"id": 1, "name": "alice"},
            ),
        ]
        client = NemoClient(base_url=BASE, http_client=mock_http, retry=STAINLESS_RETRY)

        with patch("nemo_platform_plugin.client.client.time.sleep"):
            response = client.send(GET_ITEM(name="alice"))

        assert response.body.name == "alice"
        assert mock_http.request.call_count == 2

    def test_stainless_true_header_forces_retry(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            httpx.Response(
                400,
                headers={"x-should-retry": "true"},
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"detail": "retry"},
            ),
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"id": 1, "name": "alice"},
            ),
        ]
        client = NemoClient(base_url=BASE, http_client=mock_http, retry=STAINLESS_RETRY)

        with patch("nemo_platform_plugin.client.client.time.sleep"):
            response = client.send(GET_ITEM(name="alice"))

        assert response.body.name == "alice"
        assert mock_http.request.call_count == 2

    def test_stainless_true_header_does_not_retry_success(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = httpx.Response(
            200,
            headers={"x-should-retry": "true"},
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
            json={"id": 1, "name": "alice"},
        )
        client = NemoClient(base_url=BASE, http_client=mock_http, retry=STAINLESS_RETRY)

        response = client.send(GET_ITEM(name="alice"))

        assert response.body.name == "alice"
        assert mock_http.request.call_count == 1

    def test_stainless_false_header_suppresses_retry(self) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.return_value = httpx.Response(
            500,
            headers={"x-should-retry": "false"},
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
            json={"detail": "do not retry"},
        )
        client = NemoClient(base_url=BASE, http_client=mock_http, retry=STAINLESS_RETRY)

        with pytest.raises(NemoHTTPError):
            client.send(GET_ITEM(name="alice"))

        assert mock_http.request.call_count == 1

    @pytest.mark.parametrize(
        ("headers", "expected_delay"),
        [
            ({"retry-after-ms": "1250"}, 1.25),
            ({"retry-after-ms": "invalid", "retry-after": "3"}, 3.0),
            ({"retry-after": "2.5"}, 2.5),
            ({"retry-after": "Mon, 12 Jan 1970 13:47:10 GMT"}, 30.0),
            ({"retry-after": "60"}, 60.0),
        ],
    )
    def test_stainless_policy_honors_retry_after(self, headers: dict[str, str], expected_delay: float) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            httpx.Response(
                500,
                headers=headers,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"detail": "retry"},
            ),
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"id": 1, "name": "alice"},
            ),
        ]
        client = NemoClient(base_url=BASE, http_client=mock_http, retry=STAINLESS_RETRY)

        with (
            patch("nemo_platform_plugin.client.client.time.time", return_value=1_000_000),
            patch("nemo_platform_plugin.client.client.time.sleep") as sleep,
        ):
            client.send(GET_ITEM(name="alice"))

        sleep.assert_called_once_with(expected_delay)

    @pytest.mark.parametrize(
        "headers",
        [
            {"retry-after-ms": "0"},
            {"retry-after-ms": "60001"},
            {"retry-after": "-1"},
            {"retry-after": "60.1"},
            {"retry-after": "not-a-delay"},
            {"retry-after": "Mon, 12 Jan 1970 13:46:39 GMT"},
        ],
    )
    def test_stainless_policy_falls_back_for_unreasonable_retry_after(self, headers: dict[str, str]) -> None:
        mock_http = MagicMock(spec=httpx.Client)
        mock_http.request.side_effect = [
            httpx.Response(
                500,
                headers=headers,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"detail": "retry"},
            ),
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"id": 1, "name": "alice"},
            ),
        ]
        client = NemoClient(base_url=BASE, http_client=mock_http, retry=STAINLESS_RETRY)

        with (
            patch("nemo_platform_plugin.client.client.time.time", return_value=1_000_000),
            patch("nemo_platform_plugin.client.client.time.sleep") as sleep,
        ):
            client.send(GET_ITEM(name="alice"))

        sleep.assert_called_once_with(STAINLESS_RETRY.backoff_base)

    def test_retry_after_swallows_overflow_from_parsing_far_future_date(self) -> None:
        response = httpx.Response(
            503,
            headers={"retry-after": "Fri, 31 Dec 9999999 23:59:59 GMT"},
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
        )
        with patch(
            "nemo_platform_plugin.client.client.email.utils.parsedate_to_datetime",
            side_effect=OverflowError("date value out of range"),
        ):
            assert _retry_after(response) is None

    def test_retry_after_swallows_overflow_from_timestamp_conversion(self) -> None:
        response = httpx.Response(
            503,
            headers={"retry-after": "Tue, 31 Dec 9999 23:59:59 GMT"},
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
        )
        overflowing = MagicMock()
        overflowing.tzinfo = timezone.utc
        overflowing.timestamp.side_effect = OverflowError("timestamp out of range")
        with patch(
            "nemo_platform_plugin.client.client.email.utils.parsedate_to_datetime",
            return_value=overflowing,
        ):
            assert _retry_after(response) is None

    def test_binary_stream_retries_before_returning_content(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(503, request=request, json={"detail": "unavailable"})
            return httpx.Response(200, request=request, content=b"artifact")

        client = NemoClient(
            base_url=BASE,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            retry=RetryPolicy(max_retries=1, backoff_base=0.0),
        )

        assert client.send(DOWNLOAD()).read() == b"artifact"
        assert attempts == 2

    def test_binary_stream_transport_failure_is_wrapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        client = NemoClient(
            base_url=BASE,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            retry=RetryPolicy(max_retries=1, backoff_base=0.0),
        )

        with pytest.raises(NemoTransportError):
            client.send(DOWNLOAD()).read()

    def test_stream_item_validation_failure_is_wrapped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "application/x-ndjson"},
                content=b'{"id":"bad","name":"alice"}\n',
            )

        client = NemoClient(base_url=BASE, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

        with pytest.raises(NemoResponseValidationError):
            with client.send(EVENTS()).stream() as events:
                list(events)


# ---------------------------------------------------------------------------
# Async: RetryPolicy
# ---------------------------------------------------------------------------


class TestAsyncRetryPolicy:
    @pytest.mark.asyncio
    async def test_retry_on_503_async(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.side_effect = [
            httpx.Response(
                503,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"detail": "unavailable"},
            ),
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"id": 1, "name": "alice"},
            ),
        ]

        client = AsyncNemoClient(
            base_url=BASE,
            http_client=mock_http,
            retry=RetryPolicy(max_retries=2, backoff_base=0.0),
        )
        resp = await client.send(GET_ITEM(name="alice"))

        assert resp.http_response.status_code == 200
        assert resp.body.name == "alice"
        assert mock_http.request.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [408, 409, 500])
    async def test_stainless_policy_retries_expected_statuses_async(self, status_code: int) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.side_effect = [
            httpx.Response(
                status_code,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"detail": "retry"},
            ),
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"id": 1, "name": "alice"},
            ),
        ]
        client = AsyncNemoClient(base_url=BASE, http_client=mock_http, retry=STAINLESS_RETRY)

        with patch("nemo_platform_plugin.client.client.asyncio.sleep", new_callable=AsyncMock):
            response = await client.send(GET_ITEM(name="alice"))

        assert response.body.name == "alice"
        assert mock_http.request.call_count == 2

    @pytest.mark.asyncio
    async def test_stainless_true_header_forces_retry_async(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.side_effect = [
            httpx.Response(
                400,
                headers={"x-should-retry": "true"},
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"detail": "retry"},
            ),
            httpx.Response(
                200,
                request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
                json={"id": 1, "name": "alice"},
            ),
        ]
        client = AsyncNemoClient(base_url=BASE, http_client=mock_http, retry=STAINLESS_RETRY)

        with patch("nemo_platform_plugin.client.client.asyncio.sleep", new_callable=AsyncMock):
            response = await client.send(GET_ITEM(name="alice"))

        assert response.body.name == "alice"
        assert mock_http.request.call_count == 2

    @pytest.mark.asyncio
    async def test_stainless_true_header_does_not_retry_success_async(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.return_value = httpx.Response(
            200,
            headers={"x-should-retry": "true"},
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
            json={"id": 1, "name": "alice"},
        )
        client = AsyncNemoClient(base_url=BASE, http_client=mock_http, retry=STAINLESS_RETRY)

        response = await client.send(GET_ITEM(name="alice"))

        assert response.body.name == "alice"
        assert mock_http.request.call_count == 1

    @pytest.mark.asyncio
    async def test_stainless_false_header_suppresses_retry_async(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.request.return_value = httpx.Response(
            500,
            headers={"x-should-retry": "false"},
            request=httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice"),
            json={"detail": "do not retry"},
        )
        client = AsyncNemoClient(base_url=BASE, http_client=mock_http, retry=STAINLESS_RETRY)

        with pytest.raises(NemoHTTPError):
            await client.send(GET_ITEM(name="alice"))

        assert mock_http.request.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_transport_error_is_wrapped_async(self) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        request = httpx.Request("GET", f"{BASE}/apis/test/v2/items/alice")
        mock_http.request.side_effect = httpx.ConnectError("Connection refused", request=request)
        client = AsyncNemoClient(
            base_url=BASE,
            http_client=mock_http,
            retry=RetryPolicy(max_retries=1, backoff_base=0.0),
        )

        with pytest.raises(NemoTransportError) as exc_info:
            await client.send(GET_ITEM(name="alice"))

        assert exc_info.value.request is request
        assert mock_http.request.call_count == 2

    @pytest.mark.asyncio
    async def test_binary_stream_retries_async(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("Connection refused", request=request)
            return httpx.Response(200, request=request, content=b"artifact")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = AsyncNemoClient(
                base_url=BASE,
                http_client=http_client,
                retry=RetryPolicy(max_retries=1, backoff_base=0.0),
            )
            response = await client.send(DOWNLOAD())
            assert await response.read() == b"artifact"

        assert attempts == 2

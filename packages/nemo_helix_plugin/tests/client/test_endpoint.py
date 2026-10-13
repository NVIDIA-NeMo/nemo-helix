# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import NotRequired, TypedDict

from nemo_helix_plugin.client.endpoint import delete, get, patch, post
from nemo_helix_plugin.client.types import PreparedRequest
from pydantic import BaseModel


class FakeRequest(BaseModel):
    name: str


class FakeResponse(BaseModel):
    id: int
    name: str


@post("/v2/workspaces/{workspace}/items")
def PostEndpoint(body: FakeRequest, *, workspace: str) -> FakeResponse:
    raise NotImplementedError


@get("/v2/workspaces/{workspace}/items/{name}")
def GetEndpoint(*, workspace: str, name: str) -> FakeResponse:
    raise NotImplementedError


@get("/v2/cacheable", additional_success_status_codes=(304,))
def CacheableGetEndpoint() -> FakeResponse:
    raise NotImplementedError


@delete("/v2/workspaces/{workspace}/items/{name}")
def DeleteEndpoint(*, workspace: str, name: str) -> None:
    raise NotImplementedError


@patch("/items/{id}")
def PatchEndpoint(body: FakeRequest, *, id: str) -> FakeResponse:
    raise NotImplementedError


class ListQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]


@get("/v2/workspaces/{workspace}/items")
def ListEndpoint(*, workspace: str, query_params: ListQueryParams | None = None) -> FakeResponse:
    raise NotImplementedError


def test_post_endpoint_produces_prepared_request() -> None:
    body = FakeRequest(name="alice")
    prepared = PostEndpoint(body, workspace="default")

    assert isinstance(prepared, PreparedRequest)
    assert prepared.path_template == "/v2/workspaces/{workspace}/items"
    assert prepared.path_params == {"workspace": "default"}
    assert prepared.method == "POST"
    assert prepared.content == body.model_dump_json().encode()
    assert prepared.content_type == "application/json"
    assert prepared.response_type is FakeResponse


def test_get_endpoint_no_body() -> None:
    prepared = GetEndpoint(workspace="default", name="item-1")

    assert prepared.path_template == "/v2/workspaces/{workspace}/items/{name}"
    assert prepared.path_params == {"workspace": "default", "name": "item-1"}
    assert prepared.method == "GET"
    assert prepared.content is None
    assert prepared.content_type is None
    assert prepared.response_type is FakeResponse


def test_get_endpoint_additional_success_status_codes() -> None:
    assert GetEndpoint(workspace="default", name="item-1").additional_success_status_codes == ()
    assert CacheableGetEndpoint().additional_success_status_codes == (304,)


def test_delete_endpoint() -> None:
    prepared = DeleteEndpoint(workspace="default", name="item-1")

    assert prepared.path_params == {"workspace": "default", "name": "item-1"}
    assert prepared.method == "DELETE"
    assert prepared.content is None


def test_patch_endpoint() -> None:
    body = FakeRequest(name="updated")
    prepared = PatchEndpoint(body, id="42")

    assert prepared.path_params == {"id": "42"}
    assert prepared.method == "PATCH"
    assert prepared.content == body.model_dump_json().encode()


def test_get_with_query_params() -> None:
    prepared = ListEndpoint(workspace="default", query_params={"page": 1, "page_size": 10})

    assert prepared.path_params == {"workspace": "default"}
    assert prepared.query_params == {"page": 1, "page_size": 10}
    assert prepared.method == "GET"


def test_query_params_default_none() -> None:
    prepared = ListEndpoint(workspace="default")

    assert prepared.query_params is None


def test_post_with_query_params() -> None:
    @post("/v2/items/{workspace}")
    def PostWithQuery(body: FakeRequest, *, workspace: str, query_params: dict | None = None) -> FakeResponse:
        raise NotImplementedError

    body = FakeRequest(name="alice")
    prepared = PostWithQuery(body, workspace="default", query_params={"dry_run": True})

    assert prepared.query_params == {"dry_run": True}
    assert prepared.content == body.model_dump_json().encode()


def test_delete_with_response_type() -> None:
    @delete("/v2/items/{id}")
    def DeleteWithResp(*, id: str) -> FakeResponse:
        raise NotImplementedError

    prepared = DeleteWithResp(id="42")

    assert prepared.method == "DELETE"
    assert prepared.response_type is FakeResponse
    assert prepared.content is None

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generic response policy is independent of service authorization middleware."""

import asyncio
from typing import Any, cast

import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import ResponseValidationError
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from nemo_helix_plugin.api import response_policy
from nemo_helix_plugin.api.response_policy import (
    ItemPolicyRoutes,
    ListPolicyRoutes,
    ResponsePolicy,
    nemo_page_policy_routes,
    page_policy_routes,
)
from nemo_helix_plugin.auth import RuntimeAuthClient
from nemo_helix_plugin.schema import NemoListResponse, Page
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from pydantic import BaseModel, Field, PrivateAttr, computed_field, field_serializer
from starlette.responses import StreamingResponse


class Item(BaseModel):
    name: str = Field(alias="n")
    secret: str | None = None
    description: str | None = None

    @field_serializer("name")
    def serialize_name(self, value: str) -> str:
        return f"serialized:{value}"


class Other(BaseModel):
    secret: str | None = None


class ComputedItem(BaseModel):
    name: str
    _secret: str = PrivateAttr(default="stored")

    @computed_field(alias="wire_secret")
    @property
    def secret(self) -> str | None:
        return self._secret


class SecretPolicy(ResponsePolicy[Item]):
    def can_read(self, auth_client: RuntimeAuthClient) -> bool:
        return auth_client.principal.id == "service:reader"


class ComputedPolicy(ResponsePolicy[ComputedItem]):
    def can_read(self, auth_client: RuntimeAuthClient) -> bool:
        return auth_client.principal.id == "service:reader"


def _app(monkeypatch: pytest.MonkeyPatch) -> tuple[FastAPI, Item]:
    def caller(request: Request) -> AuthClient:
        principal = Principal(id=request.headers.get("x-principal", "user@example.com"))
        return AuthClient(principal=principal, config=AuthConfig(enabled=True))

    monkeypatch.setattr(response_policy, "current_auth_client", caller)
    item = Item(n="alpha", secret="stored", description=None)
    policy = SecretPolicy(dto_type=Item, field="secret")
    router = APIRouter()
    items = ItemPolicyRoutes(router, policy)
    pages = page_policy_routes(router, policy, Page[Item])
    plugin_pages = nemo_page_policy_routes(router, policy, NemoListResponse[Item])
    lists = ListPolicyRoutes(router, policy)

    @items.get("/item", tags=["Items"])
    def get_item(workspace: str) -> Item:
        return item

    @pages.get("/page")
    async def get_page(workspace: str) -> Page[Item]:
        return Page(data=[item])

    @plugin_pages.get("/plugin-page")
    async def get_plugin_page(workspace: str) -> NemoListResponse[Item]:
        return NemoListResponse(data=[item])

    @lists.get("/list")
    async def get_list(workspace: str) -> list[Item]:
        return [item]

    @router.get("/unrelated", response_model=Other)
    async def unrelated(workspace: str) -> Other:
        return Other(secret="unrelated")

    @items.get("/error")
    async def error(workspace: str) -> Item:
        raise HTTPException(status_code=400, detail="bad")

    nested = APIRouter()
    nested.include_router(router, prefix="/v2/{workspace}")
    app = FastAPI()
    app.include_router(nested, prefix="/apis/test")
    return app, item


def _response_item(body: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(body, list):
        return body[0]
    if "data" in body:
        return body["data"][0]
    return body


@pytest.mark.parametrize("path", ["item", "page", "plugin-page", "list"])
def test_projection_uses_validated_serialization(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    app, item = _app(monkeypatch)
    client = TestClient(app)
    user = client.get(f"/apis/test/v2/default/{path}")
    service = client.get(f"/apis/test/v2/default/{path}", headers={"x-principal": "service:reader"})

    assert user.status_code == service.status_code == 200
    user_item = _response_item(user.json())
    service_item = _response_item(service.json())
    assert user_item == {"n": "serialized:alpha", "description": None}
    assert service_item == {"n": "serialized:alpha", "secret": "stored", "description": None}
    assert item.secret == "stored"
    assert client.get("/apis/test/v2/default/unrelated").json() == {"secret": "unrelated"}
    assert client.get("/apis/test/v2/default/error").status_code == 400
    assert app.openapi()["paths"]["/apis/test/v2/{workspace}/item"]["get"]["tags"] == ["Items"]
    assert app.openapi()["paths"]["/apis/test/v2/{workspace}/item"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/Item"}


def test_raw_response_cannot_bypass_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        response_policy,
        "current_auth_client",
        lambda request: AuthClient(principal=Principal(id="user@example.com"), config=AuthConfig(enabled=True)),
    )
    router = APIRouter()
    items = ItemPolicyRoutes(router, SecretPolicy(dto_type=Item, field="secret"))

    @items.get("/broken")
    async def broken() -> Item:
        return cast(Item, Response(content='{"secret":"leak"}', media_type="application/json"))

    app = FastAPI()
    app.include_router(router)
    with pytest.raises(ResponseValidationError):
        TestClient(app).get("/broken")


def test_computed_field_alias_is_projected_without_mutating_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def caller(request: Request) -> AuthClient:
        principal = Principal(id=request.headers.get("x-principal", "user@example.com"))
        return AuthClient(principal=principal, config=AuthConfig(enabled=True))

    monkeypatch.setattr(response_policy, "current_auth_client", caller)
    router = APIRouter()
    items = ItemPolicyRoutes(router, ComputedPolicy(dto_type=ComputedItem, field="secret"))
    item = ComputedItem(name="alpha")

    @items.get("/computed")
    def get_computed() -> ComputedItem:
        return item

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert client.get("/computed").json() == {"name": "alpha"}
    assert client.get("/computed", headers={"x-principal": "service:reader"}).json() == {
        "name": "alpha",
        "wire_secret": "stored",
    }
    assert item.secret == "stored"


@pytest.mark.asyncio
async def test_concurrent_callers_do_not_share_read_access(monkeypatch: pytest.MonkeyPatch) -> None:
    app, _ = _app(monkeypatch)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        responses = await asyncio.gather(
            *(
                client.get("/apis/test/v2/default/item", headers={"x-principal": principal})
                for principal in ["service:reader", "user@example.com", "service:reader", "user@example.com"]
            )
        )

    assert ["secret" in response.json() for response in responses] == [True, False, True, False]


@pytest.mark.parametrize("field", ["absent", "name"])
def test_rejects_invalid_field_at_registration(field: str) -> None:
    with pytest.raises(ValueError, match="optional and serializable"):
        ItemPolicyRoutes(APIRouter(), SecretPolicy(dto_type=Item, field=field))


def test_policy_can_protect_a_different_optional_field(monkeypatch: pytest.MonkeyPatch) -> None:
    def caller(request: Request) -> AuthClient:
        return AuthClient(principal=Principal(id="user@example.com"), config=AuthConfig(enabled=True))

    monkeypatch.setattr(response_policy, "current_auth_client", caller)
    router = APIRouter()
    items = ItemPolicyRoutes(router, SecretPolicy(dto_type=Item, field="description"))

    @items.get("/other-field")
    def get_other_field() -> Item:
        return Item(n="alpha", secret="public", description="private")

    app = FastAPI()
    app.include_router(router)
    assert TestClient(app).get("/other-field").json() == {"n": "serialized:alpha", "secret": "public"}


def test_rejects_wrong_dto_and_response_options_at_registration() -> None:
    router = APIRouter()
    items = ItemPolicyRoutes(router, SecretPolicy(dto_type=Item, field="secret"))

    with pytest.raises(ValueError, match="registered DTO"):

        @cast(Any, items.get("/wrong"))
        def wrong() -> Other:
            return Other()

    with pytest.raises(ValueError, match="registered DTO"):

        @cast(Any, items.get("/raw"))
        def raw() -> Response:
            return Response()

    with pytest.raises(ValueError, match="registered DTO"):

        @cast(Any, items.get("/stream"))
        def stream() -> StreamingResponse:
            return StreamingResponse(iter([b"body"]))

    def valid() -> Item:
        return Item(n="ok")

    for options in ({"status_code": 204}, {"response_model_by_alias": False}, {"response_class": Response}):
        with pytest.raises(ValueError, match="requires a JSON response"):
            items.get("/invalid", **options)(valid)

    non_json_router = APIRouter(default_response_class=Response)
    non_json_items = ItemPolicyRoutes(non_json_router, SecretPolicy(dto_type=Item, field="secret"))
    with pytest.raises(ValueError, match="requires a JSON response"):
        non_json_items.get("/invalid")(valid)

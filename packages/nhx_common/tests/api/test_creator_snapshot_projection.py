# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered creator projections run after FastAPI response serialization."""

import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Request
from httpx import ASGITransport, AsyncClient
from nemo_helix_plugin.api.creator_snapshot import creator_snapshot_route_class
from nhx.common.api.creator_snapshot_projection import (
    CreatorSnapshotProjectionMiddleware,
    collect_creator_snapshot_routes,
)
from nhx.common.auth.client import AuthClient
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from pydantic import BaseModel
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response


class SnapshotItem(BaseModel):
    name: str
    auth_context: dict[str, str] | None = None


class SnapshotPage(BaseModel):
    data: list[SnapshotItem]


class UnrelatedItem(BaseModel):
    auth_context: dict[str, str] | None = None


def _app(principal: Principal, *, auth_enabled: bool) -> FastAPI:
    auth_client = AuthClient(principal=principal, config=AuthConfig(enabled=auth_enabled))
    router = APIRouter(
        route_class=creator_snapshot_route_class(
            response_paths={
                SnapshotItem: ("auth_context",),
                SnapshotPage: ("data[].auth_context",),
                list[SnapshotItem]: ("[].auth_context",),
            },
            purpose="Test creator identity",
        )
    )

    @router.get("/item", response_model=SnapshotItem)
    async def item() -> SnapshotItem:
        return SnapshotItem(name="one", auth_context={"principal_id": "creator"})

    @router.get("/page", response_model=SnapshotPage)
    async def page() -> SnapshotPage:
        return SnapshotPage(
            data=[
                SnapshotItem(name="one", auth_context={"principal_id": "creator"}),
                SnapshotItem(name="two", auth_context=None),
            ]
        )

    @router.get("/list", response_model=list[SnapshotItem])
    async def listed() -> list[SnapshotItem]:
        return [SnapshotItem(name="one", auth_context={"principal_id": "creator"})]

    @router.get("/unrelated", response_model=UnrelatedItem)
    async def unrelated() -> UnrelatedItem:
        return UnrelatedItem(auth_context={"other_purpose": "visible"})

    @router.get("/error", response_model=SnapshotItem)
    async def error() -> SnapshotItem:
        raise HTTPException(status_code=400, detail="invalid")

    nested = FastAPI()
    nested.include_router(router, prefix="/v2")
    outer = FastAPI()
    outer.include_router(nested.router, prefix="/apis/test")
    outer.add_middleware(CreatorSnapshotProjectionMiddleware, registrations=collect_creator_snapshot_routes(outer))

    @outer.middleware("http")
    async def bind_auth(request: Request, call_next: RequestResponseEndpoint) -> Response:
        token = auth_client_context.set(auth_client)
        try:
            return await call_next(request)
        finally:
            auth_client_context.reset(token)

    return outer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("principal", "auth_enabled", "can_read"),
    [
        (Principal(id="service:models"), True, True),
        (Principal(id="service:models", on_behalf_of="user@example.com"), True, False),
        (Principal(id="user@example.com"), True, False),
        (Principal(), False, True),
    ],
)
async def test_registered_response_shapes(principal: Principal, auth_enabled: bool, can_read: bool) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=_app(principal, auth_enabled=auth_enabled)), base_url="http://test"
    ) as client:
        single = await client.get("/apis/test/v2/item")
        page = await client.get("/apis/test/v2/page")
        listed = await client.get("/apis/test/v2/list")
        unrelated = await client.get("/apis/test/v2/unrelated")
        error = await client.get("/apis/test/v2/error")

    assert single.status_code == page.status_code == listed.status_code == unrelated.status_code == 200
    assert ("auth_context" in single.json()) is can_read
    assert all(("auth_context" in item) is can_read for item in page.json()["data"])
    assert all(("auth_context" in item) is can_read for item in listed.json())
    assert unrelated.json()["auth_context"] == {"other_purpose": "visible"}
    assert error.status_code == 400 and error.json() == {"detail": "invalid"}
    assert single.headers["content-length"] == str(len(single.content))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for looking up a service by name from route code."""

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient
from nhx.common.service import RouterConfig, Service
from nhx.common.service.dependencies import find_service, get_service


class NamedService(Service):
    """Service whose route reports the name of the service it looks up."""

    def __init__(self, name: str):
        super().__init__(name=name, module_name="nhx.test")

    def get_routers(self) -> list[RouterConfig]:
        router = APIRouter()
        name = self.name

        @router.get("/whoami")
        async def whoami(request: Request) -> dict[str, str]:
            return {"name": get_service(request, name).name}

        return [RouterConfig(router, tag="Test", description="Test endpoints")]


def _request(app: FastAPI) -> Request:
    return Request({"type": "http", "app": app})


def test_standalone_app_registers_its_service() -> None:
    service = NamedService("alpha")
    app = service.create_app()

    assert app.state.services == {"alpha": service}
    assert get_service(_request(app), "alpha") is service
    assert TestClient(app).get("/whoami").json() == {"name": "alpha"}


def test_find_service_returns_none_when_missing() -> None:
    app = NamedService("alpha").create_app()

    assert find_service(_request(app), "beta") is None
    assert find_service(_request(FastAPI()), "alpha") is None


def test_get_service_raises_when_missing() -> None:
    app = NamedService("alpha").create_app()

    with pytest.raises(RuntimeError, match="'beta' is not registered"):
        get_service(_request(app), "beta")

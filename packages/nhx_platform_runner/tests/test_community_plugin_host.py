# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import EntryPoint
from unittest.mock import Mock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from nemo_helix_plugin.service import NemoService, RouterSpec
from nhx.platform_runner.community_plugins import host


class SampleService(NemoService):
    name = "sample"

    def __init__(self) -> None:
        self.events: list[str] = []

    def get_routers(self) -> list[RouterSpec]:
        router = APIRouter()

        @router.post("/echo")
        async def echo(body: dict) -> dict:
            return body

        @router.get("/error")
        async def error() -> None:
            raise ValueError("sample")

        return [RouterSpec(router, prefix="/v2/workspaces/{workspace}")]

    async def on_startup(self) -> None:
        self.events.append("start")

    async def on_shutdown(self) -> None:
        self.events.append("stop")


def test_host_routes_descriptor_and_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    service = SampleService()

    class Service(SampleService):
        def __new__(cls):
            return service

    ep = Mock(spec=EntryPoint, name="sample")
    ep.name = "sample"
    ep.load.return_value = Service
    monkeypatch.setattr(host, "entry_points", lambda **kwargs: [ep])
    app = host.create_app("sample")
    assert service.events == []
    with TestClient(app) as client:
        assert service.events == ["start"]
        assert client.get("/_helix/ready").json() == {"status": "ready"}
        assert client.post("/apis/sample/v2/workspaces/default/echo", json={"hello": "world"}).json() == {
            "hello": "world"
        }
        descriptor = client.get("/_helix/descriptor").json()
        assert descriptor["name"] == "sample"
        assert descriptor["protocol_version"] == 1
        assert "/apis/sample/v2/workspaces/{workspace}/echo" in descriptor["openapi"]["paths"]
        assert "/_helix/ready" not in descriptor["openapi"]["paths"]
    assert service.events == ["start", "stop"]


@pytest.mark.parametrize("count", [0, 2])
def test_host_requires_unambiguous_entry_point(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    ep = Mock()
    ep.name = "sample"
    monkeypatch.setattr(host, "entry_points", lambda **kwargs: [ep] * count)
    with pytest.raises(ValueError, match="exactly one"):
        host.create_app("sample")
    ep.load.assert_not_called()


def test_host_rejects_wrong_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = Mock()
    ep.name = "sample"
    ep.load.return_value = object
    monkeypatch.setattr(host, "entry_points", lambda **kwargs: [ep])
    with pytest.raises(TypeError, match="NemoService subclass"):
        host.create_app("sample")


def test_host_rejects_name_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = Mock()
    ep.name = "different"
    ep.load.return_value = SampleService
    monkeypatch.setattr(host, "entry_points", lambda **kwargs: [ep])
    with pytest.raises(ValueError, match="name does not match"):
        host.create_app("different")


def test_host_cleans_up_failed_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FailingService(SampleService):
        async def on_startup(self) -> None:
            raise RuntimeError("startup failed")

        async def on_shutdown(self) -> None:
            events.append("stop")

    ep = Mock()
    ep.name = "sample"
    ep.load.return_value = FailingService
    monkeypatch.setattr(host, "entry_points", lambda **kwargs: [ep])
    with pytest.raises(RuntimeError, match="startup failed"), TestClient(host.create_app("sample")):
        pass
    assert events == ["stop"]

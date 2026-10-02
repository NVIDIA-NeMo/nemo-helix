# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.controller.controller_manager import ControllerManager
from nhx.common.service import RouterConfig, Service
from nhx.platform_runner.health import create_platform_health_router


class ProbeService(Service):
    def __init__(self, name: str, *, ready: bool = True) -> None:
        super().__init__(name=name, module_name=f"nhx.{name}")
        self.ready = ready

    def get_routers(self) -> list[RouterConfig]:
        return []

    async def is_ready(self) -> bool:
        return self.ready


@pytest.fixture(autouse=True)
def reset_controller_manager() -> Iterator[None]:
    ControllerManager._instance = None
    yield
    ControllerManager._instance = None


def _client_for(services: list[Service]) -> TestClient:
    app = FastAPI()
    app.include_router(create_platform_health_router(services))
    return TestClient(app)


def test_status_and_ready_are_healthy_when_no_services_are_running() -> None:
    client = _client_for([])

    status_response = client.get("/status")
    ready_response = client.get("/health/ready")

    assert status_response.status_code == 200
    assert status_response.json() == {
        "status": "healthy",
        "services": {"ready": [], "not_ready": []},
        "controllers": {"healthy": True, "status": {}},
    }
    assert ready_response.status_code == 200
    assert ready_response.json() == {"status": "ready"}


def test_status_only_reports_services_registered_with_runner() -> None:
    registered = ProbeService("entities", ready=True)
    not_started = ProbeService("models", ready=False)
    client = _client_for([registered])

    response = client.get("/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["services"] == {"ready": ["entities"], "not_ready": []}
    assert not_started.name not in payload["services"]["ready"]
    assert not_started.name not in [service["name"] for service in payload["services"]["not_ready"]]
    assert client.get("/health/ready").status_code == 200


def test_status_remains_healthy_when_new_service_is_registered_after_it_is_ready() -> None:
    entities = ProbeService("entities", ready=True)
    models = ProbeService("models", ready=False)
    services: list[Service] = [entities]
    client = _client_for(services)

    assert client.get("/status").json()["status"] == "healthy"

    models.ready = True
    services.append(models)

    response = client.get("/status")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["services"] == {"ready": ["entities", "models"], "not_ready": []}
    assert client.get("/health/ready").status_code == 200


def test_registered_not_ready_service_degrades_status_and_blocks_readiness() -> None:
    entities = ProbeService("entities", ready=True)
    models = ProbeService("models", ready=False)
    client = _client_for([entities, models])

    status_response = client.get("/status")
    ready_response = client.get("/health/ready")

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "degraded"
    assert status_response.json()["services"] == {
        "ready": ["entities"],
        "not_ready": [{"name": "models", "message": ""}],
    }
    assert ready_response.status_code == 503
    assert ready_response.json() == {"detail": {"status": "not_ready"}}


def test_failed_controller_makes_top_level_status_unhealthy() -> None:
    manager = ControllerManager.get_instance()
    manager.mark_controller_failed("models", reason="startup failed")
    client = _client_for([ProbeService("entities", ready=True)])

    status_response = client.get("/status")

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "unhealthy"
    assert status_response.json()["controllers"] == {
        "healthy": False,
        "status": {"models": False},
    }
    assert client.get("/health/ready").status_code == 503

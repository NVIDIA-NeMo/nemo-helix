# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.service import NemoService, RouterSpec
from nhx.platform_runner import registry


def clear_registry_caches() -> None:
    registry.get_available_services.cache_clear()
    registry.get_available_controllers.cache_clear()


class AgentsService(NemoService):
    name = "agents"

    def get_routers(self) -> list[RouterSpec]:
        return [RouterSpec(router=APIRouter())]


class EvaluatorService(NemoService):
    name = "evaluator"

    def get_routers(self) -> list[RouterSpec]:
        return [RouterSpec(router=APIRouter())]


class AgentsDeploymentController(NemoController):
    name = "agents-deployment"

    async def list_objects(self) -> list:
        return []

    async def reconcile_one(self, obj: object) -> None:
        _ = obj
        return None


def test_service_groups_include_plugin_services(monkeypatch):
    clear_registry_caches()
    monkeypatch.setattr(registry, "discover_services", lambda: {"agents": AgentsService})
    monkeypatch.setattr(
        registry,
        "AVAILABLE_SERVICES",
        {"auth": "nhx.core.auth.main:service", "hello-world": "nhx.hello_world.main:service"},
    )
    monkeypatch.setattr(registry, "CORE_SERVICES", ["auth"])
    monkeypatch.setattr(registry, "API_SERVICES", ["hello-world"])

    available = registry.get_available_services()
    groups = registry.get_service_groups(available)

    assert "agents" not in groups["core"]
    assert "agents" in groups["api"]
    assert "agents" in groups["all"]


def test_service_groups_include_evaluator_plugin_service(monkeypatch):
    clear_registry_caches()
    monkeypatch.setattr(registry, "discover_services", lambda: {"evaluator": EvaluatorService})

    available = registry.get_available_services()
    groups = registry.get_service_groups(available)

    assert "evaluation" not in available
    assert "evaluation" not in groups["api"]
    assert "evaluator" not in groups["core"]
    assert "evaluator" in groups["api"]
    assert "evaluator" in groups["all"]


def test_controller_groups_include_plugin_controllers(monkeypatch):
    clear_registry_caches()
    monkeypatch.setattr(registry, "discover_controllers", lambda: {"agents-deployment": AgentsDeploymentController})
    monkeypatch.setattr(registry, "AVAILABLE_CONTROLLERS", {"jobs": "nhx.core.jobs.controllers.main:run"})

    available = registry.get_available_controllers()
    groups = registry.get_controller_groups(available)

    assert "agents-deployment" not in groups["core"]
    assert "agents-deployment" in groups["all"]


def test_default_controllers_include_plugin_controllers(monkeypatch):
    clear_registry_caches()
    monkeypatch.setattr(registry, "discover_controllers", lambda: {"agents-deployment": AgentsDeploymentController})
    monkeypatch.setattr(registry, "AVAILABLE_CONTROLLERS", {"jobs": "nhx.core.jobs.controllers.main:run"})

    available = registry.get_available_controllers()
    groups = registry.get_controller_groups(available)

    assert "agents-deployment" in registry.get_default_controllers(groups)


def test_openapi_services_are_explicit_and_do_not_auto_include_plugins(monkeypatch):
    clear_registry_caches()
    monkeypatch.setattr(
        registry,
        "AVAILABLE_SERVICES",
        {
            "auth": "nhx.core.auth.main:service",
            "hello-world": "nhx.hello_world.main:service",
        },
    )
    monkeypatch.setattr(registry, "OPENAPI_SERVICES", ["auth"])
    monkeypatch.setattr(registry, "discover_services", lambda: {"agents": AgentsService})

    available = registry.get_available_services()

    assert registry.get_openapi_service_names(available) == ["auth"]


def test_legacy_evaluation_service_is_not_registered_by_default():
    clear_registry_caches()
    available = registry.get_available_services()
    groups = registry.get_service_groups(available)

    assert "evaluation" not in available
    assert "evaluation" not in groups["api"]
    assert "evaluation" not in registry.get_openapi_service_names(available)


def test_customization_in_openapi_when_plugin_service_available(monkeypatch):
    clear_registry_caches()

    class CustomizationService(NemoService):
        name = "customization"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router=APIRouter())]

    monkeypatch.setattr(
        registry,
        "AVAILABLE_SERVICES",
        {"auth": "nhx.core.auth.main:service"},
    )
    monkeypatch.setattr(registry, "discover_services", lambda: {"customization": CustomizationService})

    available = registry.get_available_services()
    assert "customization" in registry.get_openapi_service_names(available)


def test_intake_is_registered_as_api_and_openapi_service():
    clear_registry_caches()
    available = registry.get_available_services()
    groups = registry.get_service_groups(available)

    assert available["intake"] == "nhx.intake.main:service"
    assert "intake" not in groups["core"]
    assert "intake" in groups["api"]
    assert "intake" in registry.get_openapi_service_names(available)

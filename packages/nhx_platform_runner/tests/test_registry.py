# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from fastapi import APIRouter
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.service import NemoService, RouterSpec
from nhx.platform_runner import registry
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter


def clear_registry_caches() -> None:
    registry.get_available_services.cache_clear()
    registry.get_available_controllers.cache_clear()


@pytest.fixture(autouse=True)
def _clear_registry_caches_after() -> Iterator[None]:
    """Drop registries cached from patched discovery so fake plugins never reach later tests."""
    yield
    clear_registry_caches()


class _FakeEntryPoint:
    """Stands in for ``importlib.metadata.EntryPoint``: a ``value`` string and a counted ``load``."""

    def __init__(self, loaded: object = None, *, error: Exception | None = None) -> None:
        self.value = f"fake.module:{getattr(loaded, '__qualname__', 'Broken')}"
        self._loaded = loaded
        self._error = error
        self.loads = 0

    def load(self) -> object:
        self.loads += 1
        if self._error is not None:
            raise self._error
        return self._loaded


def install_plugins(monkeypatch, group: str, entries: dict[str, _FakeEntryPoint]) -> dict[str, _FakeEntryPoint]:
    """Make *entries* the installed entry points of *group* (and no other group) for the registry."""
    monkeypatch.setattr(registry, "discover_entry_points", lambda name: entries if name == group else {})
    return entries


def install_plugin_services(monkeypatch, entries: dict[str, _FakeEntryPoint]) -> dict[str, _FakeEntryPoint]:
    return install_plugins(monkeypatch, "nemo.services", entries)


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
    install_plugin_services(monkeypatch, {"agents": _FakeEntryPoint(AgentsService)})
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
    install_plugin_services(monkeypatch, {"evaluator": _FakeEntryPoint(EvaluatorService)})

    available = registry.get_available_services()
    groups = registry.get_service_groups(available)

    assert "evaluation" not in available
    assert "evaluation" not in groups["api"]
    assert "evaluator" not in groups["core"]
    assert "evaluator" in groups["api"]
    assert "evaluator" in groups["all"]


def test_controller_groups_include_plugin_controllers(monkeypatch):
    clear_registry_caches()
    install_plugins(monkeypatch, "nemo.controllers", {"agents-deployment": _FakeEntryPoint(AgentsDeploymentController)})
    monkeypatch.setattr(registry, "AVAILABLE_CONTROLLERS", {"jobs": "nhx.core.jobs.controllers.main:run"})

    available = registry.get_available_controllers()
    groups = registry.get_controller_groups(available)

    assert "agents-deployment" not in groups["core"]
    assert "agents-deployment" in groups["all"]


def test_default_controllers_include_plugin_controllers(monkeypatch):
    clear_registry_caches()
    install_plugins(monkeypatch, "nemo.controllers", {"agents-deployment": _FakeEntryPoint(AgentsDeploymentController)})
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
    install_plugin_services(monkeypatch, {"agents": _FakeEntryPoint(AgentsService)})

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
    install_plugin_services(monkeypatch, {"customization": _FakeEntryPoint(CustomizationService)})

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


def test_names_and_groups_do_not_import_plugin_services(monkeypatch):
    clear_registry_caches()
    entries = install_plugin_services(
        monkeypatch,
        {
            "agents": _FakeEntryPoint(error=AssertionError("must not be imported")),
            "evaluator": _FakeEntryPoint(error=AssertionError("must not be imported")),
        },
    )

    available = registry.get_available_services()
    groups = registry.get_service_groups(available)

    assert "agents" in available
    assert {"agents", "evaluator"} <= set(available)
    assert {"agents", "evaluator"} <= set(groups["all"])
    assert all(entry.loads == 0 for entry in entries.values())


def test_lookup_loads_only_that_plugin_once(monkeypatch):
    clear_registry_caches()
    entries = install_plugin_services(
        monkeypatch,
        {"agents": _FakeEntryPoint(AgentsService), "evaluator": _FakeEntryPoint(EvaluatorService)},
    )

    available = registry.get_available_services()
    first = available["agents"]

    assert isinstance(first, NemoServiceAdapter)
    assert available["agents"] is first
    assert entries["agents"].loads == 1
    assert entries["evaluator"].loads == 0


def test_a_plugin_that_fails_to_load_is_reported_absent(monkeypatch):
    clear_registry_caches()
    install_plugin_services(monkeypatch, {"agents": _FakeEntryPoint(error=ImportError("boom"))})

    available = registry.get_available_services()

    assert available.get("agents") is None
    assert "agents" not in available
    assert "agents" not in set(available)


def test_a_plugin_that_fails_to_instantiate_is_reported_absent(monkeypatch):
    clear_registry_caches()

    class _Broken(NemoService):
        name = "agents"

        def __init__(self) -> None:
            raise RuntimeError("boom")

        def get_routers(self) -> list[RouterSpec]:
            return []

    install_plugin_services(monkeypatch, {"agents": _FakeEntryPoint(_Broken)})

    assert registry.get_available_services().get("agents") is None


def test_a_broken_plugin_falls_back_to_the_built_in_service_of_the_same_name(monkeypatch):
    clear_registry_caches()
    monkeypatch.setattr(registry, "AVAILABLE_SERVICES", {"agents": "nhx.agents.main:service"})
    install_plugin_services(monkeypatch, {"agents": _FakeEntryPoint(error=ImportError("boom"))})

    available = registry.get_available_services()

    assert available["agents"] == "nhx.agents.main:service"
    assert "agents" in available


def test_controller_names_do_not_import_plugin_controllers(monkeypatch):
    clear_registry_caches()
    entry = _FakeEntryPoint(error=AssertionError("must not be imported"))
    install_plugins(monkeypatch, "nemo.controllers", {"agents-deployment": entry})

    available = registry.get_available_controllers()
    groups = registry.get_controller_groups(available)

    assert "agents-deployment" in groups["all"]
    assert "agents-deployment" in registry.get_default_controllers(groups)
    assert entry.loads == 0


def test_controller_lookup_builds_a_run_function_once(monkeypatch):
    clear_registry_caches()
    entry = _FakeEntryPoint(AgentsDeploymentController)
    install_plugins(monkeypatch, "nemo.controllers", {"agents-deployment": entry})

    available = registry.get_available_controllers()
    run_func = available["agents-deployment"]

    assert callable(run_func)
    assert available["agents-deployment"] is run_func
    assert entry.loads == 1


def test_a_controller_that_fails_to_load_is_reported_absent(monkeypatch):
    clear_registry_caches()
    install_plugins(monkeypatch, "nemo.controllers", {"agents-deployment": _FakeEntryPoint(error=ImportError("boom"))})

    available = registry.get_available_controllers()

    assert available.get("agents-deployment") is None
    assert "agents-deployment" not in available


def test_an_entry_that_is_not_a_service_class_is_reported_absent(monkeypatch):
    clear_registry_caches()
    install_plugin_services(monkeypatch, {"agents": _FakeEntryPoint(object())})

    assert registry.get_available_services().get("agents") is None

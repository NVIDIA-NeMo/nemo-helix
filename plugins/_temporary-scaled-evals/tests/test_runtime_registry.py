# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("scaled_evals")

from scaled_evals.api.settings import settings
from scaled_evals.dispatch.gym.daytona import GymDaytonaBackend
from scaled_evals.dispatch.gym.sandbox_daytona import GymSandboxDaytonaBackend
from scaled_evals.dispatch.gym.sandbox_opensandbox import GymSandboxOpenSandboxBackend
from scaled_evals.dispatch.harbor_opensandbox import HarborOpenSandboxBackend
from scaled_evals.dispatch.registry import (
    RuntimeBackendRegistry,
    build_runtime_backend_registry,
    get_backend_capabilities,
    load_runtime_backend_plugins,
    registered_runtime_names,
)
from scaled_evals.dispatch.runtime_backend import (
    CallableRuntimeBackend,
    LaunchHandle,
    LaunchSpec,
    ResultSummary,
    RuntimeBackend,
    RuntimeBackendCapabilities,
    RuntimeBackendRegistration,
    RuntimeStatus,
)
from scaled_evals.dispatch.sandbox_k8s import SandboxK8sBackend


class FakeRuntimeBackend:
    name = "fake"

    def launch(self, spec: LaunchSpec) -> LaunchHandle:
        return LaunchHandle(backend=self.name, external_id=spec.evaluation_id)

    def status(self, handle: LaunchHandle) -> RuntimeStatus:
        return RuntimeStatus(phase="succeeded", raw={"id": handle.external_id})

    def teardown(self, handle: LaunchHandle) -> None:
        return None

    def summarize(self, result: Mapping[str, Any]) -> ResultSummary:
        return ResultSummary()


def _spec() -> LaunchSpec:
    return LaunchSpec(
        evaluation_id="ev_adapter",
        name="adapter",
        framework="test",
        image_ref="task:tag",
        parallelism=1,
    )


@pytest.mark.parametrize(
    ("backend_type", "backend_name"),
    [
        (SandboxK8sBackend, "sandbox_k8s"),
        (GymDaytonaBackend, "gym_daytona"),
        (GymSandboxDaytonaBackend, "gym_sandbox_daytona"),
        (GymSandboxOpenSandboxBackend, "gym_sandbox_opensandbox"),
        (HarborOpenSandboxBackend, "harbor_opensandbox"),
    ],
)
def test_builtin_backend_adapter_preserves_unwired_operation_errors(backend_type: type[Any], backend_name: str) -> None:
    backend = backend_type()
    handle = LaunchHandle(backend=backend_name, external_id="ev_adapter")

    with pytest.raises(NotImplementedError, match=backend_name):
        backend.launch(_spec())
    with pytest.raises(NotImplementedError, match=backend_name):
        backend.status(handle)
    with pytest.raises(NotImplementedError, match=backend_name):
        backend.teardown(handle)


def test_callable_adapter_registers_without_dispatcher_changes(tmp_path: Path) -> None:
    calls: list[str] = []
    backend = CallableRuntimeBackend(
        name="callable",
        submitter=lambda spec: LaunchHandle(backend="callable", external_id=spec.evaluation_id),
        status_reader=lambda handle: RuntimeStatus(phase="succeeded", raw={"id": handle.external_id}),
        terminator=lambda handle: calls.append(handle.external_id),
        summarizer=lambda result: ResultSummary(reward=result.get("reward")),
    )
    registry = RuntimeBackendRegistry(
        [
            RuntimeBackendRegistration(
                name="callable",
                factory=lambda: backend,
                capabilities=RuntimeBackendCapabilities(artifact_root=lambda evaluation_id: tmp_path / evaluation_id),
            )
        ]
    )

    handle = registry.build("callable").launch(_spec())
    assert backend.status(handle).phase == "succeeded"
    assert backend.summarize({"reward": 1.0}).reward == 1.0
    backend.teardown(handle)
    assert calls == ["ev_adapter"]


def test_structural_plugin_does_not_require_callable_adapter() -> None:
    backend = FakeRuntimeBackend()

    assert isinstance(backend, RuntimeBackend)
    assert not isinstance(backend, CallableRuntimeBackend)


def test_default_registry_contains_supported_runtimes() -> None:
    assert registered_runtime_names() == (
        "gym_daytona",
        "gym_sandbox_daytona",
        "gym_sandbox_opensandbox",
        "harbor_opensandbox",
        "sandbox_k8s",
    )


def test_empty_extra_plugin_list_still_registers_builtin_harbor_runtimes() -> None:
    registry = build_runtime_backend_registry(plugin_specs=())

    assert registry.names() == ("harbor_opensandbox", "sandbox_k8s")


def test_sandbox_k8s_plugin_registers_sandbox_runtime_directly() -> None:
    registry = RuntimeBackendRegistry()
    load_runtime_backend_plugins(registry, ("scaled_evals.dispatch.sandbox_k8s",))

    assert registry.names() == ("sandbox_k8s",)


def test_gym_plugin_registers_gym_runtimes_after_builtin_sandbox() -> None:
    registry = build_runtime_backend_registry(
        plugin_specs=("scaled_evals.dispatch.gym.plugin",),
    )

    assert registry.names() == (
        "gym_daytona",
        "gym_sandbox_daytona",
        "gym_sandbox_opensandbox",
        "harbor_opensandbox",
        "sandbox_k8s",
    )


def test_explicit_sandbox_plugin_is_ignored_because_it_is_builtin() -> None:
    registry = build_runtime_backend_registry(plugin_specs=("scaled_evals.dispatch.sandbox_k8s",))

    assert registry.names() == ("harbor_opensandbox", "sandbox_k8s")


def test_default_extra_plugin_set_registers_sandbox_and_gym_runtimes() -> None:
    registry = build_runtime_backend_registry(
        plugin_specs=("scaled_evals.dispatch.gym.plugin",),
    )

    assert registry.names() == (
        "gym_daytona",
        "gym_sandbox_daytona",
        "gym_sandbox_opensandbox",
        "harbor_opensandbox",
        "sandbox_k8s",
    )


def test_runtime_capabilities_resolve_sandbox_paths(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "harbor_dir", str(tmp_path / "harbor"))
    monkeypatch.setattr(settings, "sandbox_k8s_jobs_dir", "jobs/astra")
    monkeypatch.setattr(settings, "sandbox_k8s_artifact_root", None)
    monkeypatch.setattr(settings, "sandbox_k8s_work_dir", str(tmp_path / "work"))

    capabilities = get_backend_capabilities("sandbox_k8s")

    assert capabilities.artifact_root("ev_123") == tmp_path / "harbor" / "jobs/astra" / "ev_123"
    assert capabilities.dispatch_log_path("ev_123") == tmp_path / "work" / "ev_123" / "harbor.log"
    assert capabilities.runner_container_name("ev_123") == "harbor-ev_123"
    assert capabilities.supports_archive is True
    assert capabilities.supports_teardown is True


def test_runtime_capabilities_prefer_explicit_hosted_artifact_root(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    artifact_root = tmp_path / "harbor-jobs" / "astra"
    monkeypatch.setattr(settings, "sandbox_k8s_artifact_root", str(artifact_root))

    capabilities = get_backend_capabilities("sandbox_k8s")

    assert capabilities.artifact_root("ev_123") == artifact_root / "ev_123"


def test_runtime_capabilities_resolve_gym_paths(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setattr(settings, "gym_daytona_work_dir", str(tmp_path / "gym-daytona"))
    monkeypatch.setattr(
        settings,
        "gym_sandbox_daytona_work_dir",
        str(tmp_path / "gym-sandbox-daytona"),
    )
    monkeypatch.setattr(
        settings,
        "gym_sandbox_opensandbox_work_dir",
        str(tmp_path / "gym-sandbox-opensandbox"),
    )

    daytona = get_backend_capabilities("gym_daytona")
    sandbox_daytona = get_backend_capabilities("gym_sandbox_daytona")
    opensandbox = get_backend_capabilities("gym_sandbox_opensandbox")

    assert daytona.artifact_root("ev_123") == tmp_path / "gym-daytona" / "ev_123"
    assert sandbox_daytona.artifact_root("ev_123") == tmp_path / "gym-sandbox-daytona" / "ev_123"
    assert opensandbox.artifact_root("ev_123") == tmp_path / "gym-sandbox-opensandbox" / "ev_123"
    assert opensandbox.dispatch_log_path("ev_123") == (tmp_path / "gym-sandbox-opensandbox" / "ev_123" / "gym.log")
    assert opensandbox.runner_container_name("ev_123") == "gym-ev_123"


def test_registry_allows_fake_backend_without_dispatcher_edits(tmp_path: Path) -> None:
    backend = FakeRuntimeBackend()
    registry = RuntimeBackendRegistry(
        [
            RuntimeBackendRegistration(
                name=FakeRuntimeBackend.name,
                factory=lambda: backend,
                capabilities=RuntimeBackendCapabilities(
                    artifact_root=lambda evaluation_id: tmp_path / "artifacts" / evaluation_id,
                    dispatch_work_dir=lambda evaluation_id: tmp_path / "work" / evaluation_id,
                    dispatch_log_name="fake.log",
                    runner_container_prefix="fake",
                ),
            )
        ]
    )

    assert registry.build("fake") is backend
    assert registry.capabilities("fake").artifact_root("ev_123") == (tmp_path / "artifacts" / "ev_123")
    assert registry.capabilities("fake").dispatch_log_path("ev_123") == (tmp_path / "work" / "ev_123" / "fake.log")
    assert registry.capabilities("fake").runner_container_name("ev_123") == "fake-ev_123"


def test_registry_loads_runtime_plugin_module(monkeypatch, tmp_path: Path) -> None:
    module = types.ModuleType("scaled_evals_test_runtime_plugin")

    def register_runtime_backends(registry: RuntimeBackendRegistry) -> None:
        registry.register(
            RuntimeBackendRegistration(
                name=FakeRuntimeBackend.name,
                factory=FakeRuntimeBackend,
                capabilities=RuntimeBackendCapabilities(artifact_root=lambda evaluation_id: tmp_path / evaluation_id),
            )
        )

    setattr(module, "register_runtime_backends", register_runtime_backends)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    registry = RuntimeBackendRegistry()
    load_runtime_backend_plugins(registry, (module.__name__,))

    assert registry.names() == ("fake",)
    assert isinstance(registry.build("fake"), FakeRuntimeBackend)


def test_registry_loads_runtime_plugin_custom_function(monkeypatch, tmp_path: Path) -> None:
    module = types.ModuleType("scaled_evals_test_runtime_plugin_custom")

    def install(registry: RuntimeBackendRegistry) -> None:
        registry.register(
            RuntimeBackendRegistration(
                name=FakeRuntimeBackend.name,
                factory=FakeRuntimeBackend,
                capabilities=RuntimeBackendCapabilities(artifact_root=lambda evaluation_id: tmp_path / evaluation_id),
            )
        )

    setattr(module, "install", install)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    registry = RuntimeBackendRegistry()
    load_runtime_backend_plugins(registry, (f"{module.__name__}:install",))

    assert registry.names() == ("fake",)


def test_registry_rejects_unknown_and_duplicate_runtimes(tmp_path: Path) -> None:
    registration = RuntimeBackendRegistration(
        name="fake",
        factory=FakeRuntimeBackend,
        capabilities=RuntimeBackendCapabilities(artifact_root=lambda evaluation_id: tmp_path),
    )
    registry = RuntimeBackendRegistry([registration])

    with pytest.raises(ValueError, match="unknown runtime backend"):
        registry.build("missing")

    with pytest.raises(ValueError, match="already registered"):
        registry.register(registration)

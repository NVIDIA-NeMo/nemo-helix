# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission and controller dispatch for third-party backend profiles."""

import importlib.util
import sys
from typing import Literal
from unittest.mock import MagicMock

import pytest
from nemo_helix_plugin import discovery
from nemo_helix_plugin.jobs.spec import BaseExecutionProfile
from nhx.core.jobs.app.profiles import ExecutionProfileT
from nhx.core.jobs.controllers.backends import registry
from nhx.core.jobs.controllers.backends.base import JobBackend
from pydantic import Field, TypeAdapter, ValidationError


class ExampleProfile(BaseExecutionProfile):
    backend: Literal["example"] = "example"
    provider: Literal["cpu", "gpu"] = "gpu"
    config: dict[str, int] = Field(default_factory=dict)


@pytest.fixture
def contribution(monkeypatch):
    assert importlib.util.find_spec("nemo_helix_plugin.jobs.backends"), "Missing backend contribution surface"
    from nemo_helix_plugin.jobs.backends import JobBackendContribution

    item = JobBackendContribution("example", ExampleProfile, {"gpu": "example_runtime:Backend"})
    monkeypatch.setattr(discovery, "discover", lambda _: {"example-plugin": item})
    return item


def test_admission_uses_typed_config_and_provider_default(contribution):
    adapter = TypeAdapter(ExecutionProfileT)
    profile = adapter.validate_python({"backend": "example", "config": {"count": 2}})
    assert isinstance(profile, ExampleProfile)
    assert profile.provider == "gpu"
    assert adapter.dump_python(profile)["config"] == {"count": 2}
    assert "example_runtime" not in sys.modules
    with pytest.raises(ValidationError):
        adapter.validate_python({"backend": "example", "config": {"count": "bad"}})
    with pytest.raises(ValidationError, match="provider"):
        adapter.validate_python({"backend": "example", "provider": "cpu"})


@pytest.mark.parametrize("value", [{"backend": "missing", "config": {}}, ExampleProfile()])
def test_server_rejects_uninstalled_profiles_including_instances(monkeypatch, value):
    monkeypatch.setattr(discovery, "discover", lambda _: {})
    with pytest.raises(ValidationError, match="backend"):
        TypeAdapter(ExecutionProfileT).validate_python(value)


@pytest.mark.parametrize("value", [None, [], "backend", 7, {"backend": []}, {"backend": 1}])
def test_invalid_shapes_are_validation_errors(value):
    with pytest.raises(ValidationError):
        TypeAdapter(ExecutionProfileT).validate_python(value)


def test_server_preserves_builtin_behavior_and_schema():
    adapter = TypeAdapter(ExecutionProfileT)
    profile = adapter.validate_python({"backend": "kubernetes_job", "config": {}})
    assert type(profile.config.storage).__module__.startswith("nhx.core.jobs.")
    schema = adapter.json_schema(mode="serialization")
    assert "DockerJobExecutionProfile" in schema["$defs"]
    assert "GenericExecutionProfile" in schema["$defs"]


@pytest.fixture
def runtime(tmp_path, monkeypatch, contribution):
    (tmp_path / "example_runtime.py").write_text(
        "from nhx.core.jobs.controllers.backends.base import JobBackend\n"
        "class Backend(JobBackend):\n"
        "    def __init__(self, *args): self.args = args\n"
        "    def schedule(self, *args): pass\n"
        "    def sync(self, *args): pass\n"
        "    def cleanup_steps(self, *args): pass\n"
        "    def shutdown(self): pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield contribution
    sys.modules.pop("example_runtime", None)


def test_controller_loads_selected_backend_once(runtime, monkeypatch):
    # Unselected implementations may have unavailable dependencies.
    from dataclasses import replace

    other = replace(runtime, backend="unused", implementations={"gpu": "unavailable_runtime:Backend"})
    monkeypatch.setattr(discovery, "discover", lambda _: {"example-plugin": runtime, "other": other})

    # Use a matching Literal schema for the unrelated contribution.
    class UnusedProfile(BaseExecutionProfile):
        backend: Literal["unused"] = "unused"
        config: dict = Field(default_factory=dict)

    other = replace(other, profile_model=UnusedProfile)
    imports = []
    original = __import__("importlib").import_module

    def load(name):
        imports.append(name)
        return original(name)

    monkeypatch.setattr(registry, "import_module", load, raising=False)
    client = MagicMock()
    result = registry.BackendRegistry.from_config(client, [ExampleProfile(), ExampleProfile(profile="second")])
    backend = result.get_backend(provider="gpu", profile="default")
    assert isinstance(backend, JobBackend)
    assert getattr(backend, "args") == (client, {}, "default")
    assert imports == ["example_runtime"]
    assert result.registered_profile_keys() == frozenset({("gpu", "default"), ("gpu", "second")})
    assert "unavailable_runtime" not in sys.modules
    assert registry.BackendKey("gpu", "example") not in registry.backend_registry


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("missing_runtime:Backend", "example"),
        ("example_runtime:Missing", "example"),
        ("builtins:object", "JobBackend"),
        ("nhx.core.jobs.controllers.backends.base:JobBackend", "concrete"),
    ],
)
def test_selected_implementation_errors_have_context(runtime, monkeypatch, reference, message):
    runtime.implementations["gpu"] = reference
    with pytest.raises(ValueError, match=message):
        registry.BackendRegistry.from_config(MagicMock(), [ExampleProfile()])


def test_explicit_backend_mapping_keeps_existing_override_contract(contribution, monkeypatch):
    def unexpected_discovery():
        pytest.fail("Explicit registry must not discover plugins")

    monkeypatch.setattr(registry, "discover_job_backends", unexpected_discovery, raising=False)
    backend = MagicMock()
    client = MagicMock()
    result = registry.BackendRegistry.from_config(
        client, [ExampleProfile()], backends={registry.BackendKey("gpu", "example"): backend}
    )
    assert result.get_backend(provider="gpu") is backend.return_value
    backend.assert_called_once_with(client, {}, "default")


def test_missing_provider_fails_before_import(contribution):
    with pytest.raises((ValueError, KeyError), match="cpu"):
        registry.BackendRegistry.from_config(MagicMock(), [ExampleProfile(provider="cpu")])


def test_jobs_configuration_accepts_plugin_and_checks_unique_profiles(contribution):
    from nhx.core.jobs.config import JobsServiceConfig

    config = JobsServiceConfig(executors=[{"backend": "example", "config": {"count": 2}}])
    assert isinstance(config.executors[0], ExampleProfile)
    with pytest.raises(ValidationError, match="Duplicate executor profile"):
        JobsServiceConfig(executors=[ExampleProfile(), ExampleProfile()])


def test_builtin_default_backend_remains_compatible():
    assert TypeAdapter(ExecutionProfileT).validate_python({"config": {}}).backend == "docker"


def test_builtin_invalid_config_cannot_fall_back_to_generic():
    with pytest.raises(ValidationError):
        TypeAdapter(ExecutionProfileT).validate_python(
            {"backend": "docker", "config": {"env": {"NEMO_JOB_ID": "spoof"}}}
        )

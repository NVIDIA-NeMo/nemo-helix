# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend contributions load through entry points without controller dependencies."""

import importlib.util
import sys
from dataclasses import replace
from importlib.metadata import EntryPoint
from typing import Literal

import httpx
import pytest
from nemo_helix_plugin import discovery
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile
from nemo_helix_plugin.jobs.execution_profiles import SubprocessJobExecutionProfile
from nemo_helix_plugin.jobs.spec import BaseExecutionProfile
from pydantic import Field, TypeAdapter, ValidationError


class ExampleProfile(BaseExecutionProfile):
    backend: Literal["example"] = "example"
    provider: Literal["cpu", "gpu"] = "gpu"
    config: dict[str, int] = Field(default_factory=dict)


def surface():
    assert importlib.util.find_spec("nemo_helix_plugin.jobs.backends"), "Missing backend contribution surface"
    from nemo_helix_plugin.jobs import backends

    return backends


def contribution(**kwargs):
    return replace(
        surface().JobBackendContribution(
            backend="example", profile_model=ExampleProfile, implementations={"gpu": "example_runtime:Backend"}
        ),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def clear_discovery(monkeypatch):
    monkeypatch.delenv("NEMO_PLUGIN_ALLOWLIST", raising=False)
    monkeypatch.delenv("NEMO_PLUGIN_JOB_BACKENDS_ALLOWLIST", raising=False)
    cached_functions = (discovery.discover, discovery.discover_entry_points, discovery.discover_manifests)
    for func in cached_functions:
        func.cache_clear()
    yield
    for func in cached_functions:
        func.cache_clear()


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    # A real importable module: EntryPoint.load must import it, not a mocked loader.
    (tmp_path / "example_profile.py").write_text(
        "from typing import Literal\n"
        "from pydantic import Field\n"
        "from nemo_helix_plugin.jobs.spec import BaseExecutionProfile\n"
        "from nemo_helix_plugin.jobs.backends import JobBackendContribution\n"
        "class Profile(BaseExecutionProfile):\n"
        '    backend: Literal["example"] = "example"\n'
        '    provider: Literal["cpu", "gpu"] = "gpu"\n'
        "    config: dict[str, int] = Field(default_factory=dict)\n"
        'contribution = JobBackendContribution("example", Profile, {"gpu": "example_runtime:Backend"})\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    ep = EntryPoint(name="example-plugin", value="example_profile:contribution", group="nemo.job_backends")
    monkeypatch.setattr(discovery, "entry_points", lambda *, group: [ep] if group == ep.group else [])
    yield ep
    sys.modules.pop("example_profile", None)


def test_real_discovery_and_manifest_do_not_import_implementation(plugin):
    found = surface().discover_job_backends()["example"]
    assert found.profile_model.model_validate({"config": {"x": 2}}).config == {"x": 2}
    assert "example_runtime" not in sys.modules
    assert "example-plugin" in discovery.discover_manifests()


@pytest.mark.parametrize(
    ("global_value", "surface_value", "expected"),
    [
        (None, None, True),
        ("", None, False),
        ("example-plugin", None, True),
        ("*", "", False),
        ("", "example-plugin", True),
        (None, "other", False),
    ],
)
def test_allowlists_filter_before_import(plugin, monkeypatch, global_value, surface_value, expected):
    if global_value is not None:
        monkeypatch.setenv("NEMO_PLUGIN_ALLOWLIST", global_value)
    if surface_value is not None:
        monkeypatch.setenv("NEMO_PLUGIN_JOB_BACKENDS_ALLOWLIST", surface_value)
    assert bool(surface().discover_job_backends()) is expected
    assert ("example_profile" in sys.modules) is expected


def test_broken_import_is_isolated(plugin, monkeypatch, caplog):
    broken = EntryPoint(name="broken", value="missing_backend_module:contribution", group=plugin.group)
    monkeypatch.setattr(discovery, "entry_points", lambda **_: [broken, plugin])
    assert set(surface().discover_job_backends()) == {"example"}
    assert "Failed to load 'broken'" in caplog.text


@pytest.mark.parametrize("backend", ["docker", "kubernetes_job", "volcano_job", "subprocess", "e2e", "openshell"])
def test_builtin_names_are_reserved(monkeypatch, backend):
    monkeypatch.setattr(discovery, "discover", lambda _: {"plugin": contribution(backend=backend)})
    with pytest.raises(ValueError, match="built-in"):
        surface().discover_job_backends()


@pytest.mark.parametrize(
    "changes",
    [
        {"backend": ""},
        {"backend": " example"},
        {"profile_model": object},
        {"implementations": {}},
        {"implementations": {"": "module:Backend"}},
        {"implementations": {"gpu": ":Backend"}},
        {"implementations": {"gpu": "module:"}},
        {"implementations": {"gpu": "module:Backend:other"}},
        {"implementations": {"gpu": "module:bad-name"}},
        {"backend": "mismatch"},
    ],
)
def test_invalid_contributions_fail_with_entry_point_context(monkeypatch, changes):
    monkeypatch.setattr(discovery, "discover", lambda _: {"example-plugin": contribution(**changes)})
    with pytest.raises(ValueError, match="example-plugin"):
        surface().discover_job_backends()


def test_duplicate_backend_names_fail(monkeypatch):
    monkeypatch.setattr(discovery, "discover", lambda _: {"first": contribution(), "second": contribution()})
    with pytest.raises(ValueError, match="Duplicate"):
        surface().discover_job_backends()


def test_wrong_contribution_type_fails(monkeypatch):
    monkeypatch.setattr(discovery, "discover", lambda _: {"example-plugin": object()})
    with pytest.raises(ValueError, match="example-plugin"):
        surface().discover_job_backends()


def test_sdk_dispatches_installed_profile_and_preserves_config(plugin):
    parsed = TypeAdapter(ExecutionProfile).validate_python({"backend": "example", "config": {"count": 3}})
    assert type(parsed).__module__ == "example_profile"
    assert parsed.provider == "gpu"
    assert TypeAdapter(ExecutionProfile).dump_python(parsed)["config"] == {"count": 3}
    with pytest.raises(ValidationError):
        TypeAdapter(ExecutionProfile).validate_python({"backend": "example", "config": {"count": "bad"}})


def test_sdk_preserves_unknown_backend_config(monkeypatch):
    monkeypatch.setenv("NEMO_PLUGIN_JOB_BACKENDS_ALLOWLIST", "")
    payload = {"backend": "uninstalled", "provider": "gpu", "profile": "remote", "config": {"nested": {"x": 2}}}
    parsed = TypeAdapter(ExecutionProfile).validate_python(payload)
    assert TypeAdapter(ExecutionProfile).dump_python(parsed) == payload


@pytest.mark.parametrize("value", [None, [], "backend", 7, {"backend": []}, {"backend": 1}])
def test_invalid_profile_shape_is_validation_error(value):
    with pytest.raises(ValidationError):
        TypeAdapter(ExecutionProfile).validate_python(value)


def test_http_sdk_retains_profile_dispatch_metadata(plugin):
    payload = [{"backend": "example", "config": {"count": 3}}, {"backend": "subprocess", "config": {}}]
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as transport:
        with NemoClient(base_url="http://platform.test", http_client=transport) as client:
            profiles = JobsClient.from_client(client).get_execution_profiles().data()
    assert type(profiles[0]).__module__ == "example_profile"
    assert profiles[0].config == {"count": 3}
    assert isinstance(profiles[1], SubprocessJobExecutionProfile)


def test_builtin_schema_alternatives_are_retained():
    for mode in ("validation", "serialization"):
        schema = TypeAdapter(ExecutionProfile).json_schema(mode=mode)
        assert "DockerJobExecutionProfile" in schema["$defs"]
        assert "OpenShellJobExecutionProfile" in schema["$defs"]
        assert "GenericExecutionProfile" in schema["$defs"]


def test_contribution_surface_has_no_server_runtime_imports():
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; import nemo_helix_plugin.jobs.backends; "
            "assert not any(n.startswith(('nhx.', 'docker', 'kubernetes')) for n in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_builtin_default_backend_remains_compatible():
    assert TypeAdapter(ExecutionProfile).validate_python({"config": {}}).backend == "docker"


def test_builtin_invalid_config_cannot_fall_back_to_generic():
    with pytest.raises(ValidationError):
        TypeAdapter(ExecutionProfile).validate_python(
            {"backend": "docker", "config": {"env": {"NEMO_JOB_ID": "spoof"}}}
        )

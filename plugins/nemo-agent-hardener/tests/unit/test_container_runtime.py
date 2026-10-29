# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nemo_agent_hardener_plugin.config import AgentHardenerConfig
from nemo_agent_hardener_plugin.jobs import _common
from nemo_agent_hardener_plugin.jobs import openshell_gateway as gateway_module
from nemo_agent_hardener_plugin.jobs import run as run_module
from nemo_agent_hardener_plugin.jobs import synth_benign as synth_module
from nemo_agent_hardener_plugin.jobs.errors import AgentHardenerRunError
from nemo_helix_plugin.jobs import executor as executor_module
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    DockerJobExecutionProfileConfig,
    SubprocessJobExecutionProfile,
)
from packaging.requirements import Requirement

CPU_ONLY = [DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())]
MTLS_SECRETS = {"os-ca": "CA PEM", "os-cert": "CERT PEM", "os-key": "KEY PEM"}


@pytest.fixture(autouse=True)
def _qualified_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor_module, "get_qualified_image", lambda name: f"reg.example/{name}:t")


@pytest.fixture(autouse=True)
def _no_host_plugin_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [name for name in os.environ if name.startswith("NEMO_AGENT_HARDENER_")]:
        monkeypatch.delenv(name)


@pytest.fixture
def api_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentHardenerConfig:
    config = _config(
        tmp_path,
        container_jobs=True,
        openshell_gateway_endpoint="https://openshell.openshell.svc:8443",
        openshell_ca_secret="os-ca",
        openshell_client_cert_secret="os-cert",
        openshell_client_key_secret="os-key",
    )
    monkeypatch.setattr(_common.AgentHardenerConfig, "get", lambda: config)
    return config


def _env(step: Any) -> dict[str, str]:
    return {variable["name"]: variable["value"] for variable in step["environment"]}


@pytest.mark.usefixtures("api_config")
@pytest.mark.parametrize("execution_profiles", [CPU_ONLY])
async def test_war_game_runs_in_the_task_image_without_a_subprocess_profile() -> None:
    spec = run_module.WarGameSpec(config="agent-hardener.yaml")
    job_spec = await run_module.AgentHardenerRunJob.compile(
        workspace="default", spec=spec, entity_client=None, job_name=None, async_sdk=None
    )

    (step,) = list(job_spec["steps"])
    assert step["executor"]["provider"] == "cpu"
    assert step["executor"]["container"]["image"] == "reg.example/nhx-agent-hardener-tasks:t"
    assert step["executor"]["container"]["command"] == ["nemo_agent_hardener_plugin.tasks.war_game"]
    assert _env(step).keys().isdisjoint({"HOME", "DOCKER_HOST", "XDG_CONFIG_HOME"})


@pytest.mark.usefixtures("api_config")
@pytest.mark.parametrize("execution_profiles", [CPU_ONLY])
async def test_container_step_carries_the_config_the_task_pod_cannot_read() -> None:
    spec = run_module.WarGameSpec(config="agent-hardener.yaml")
    job_spec = await run_module.AgentHardenerRunJob.compile(
        workspace="default", spec=spec, entity_client=None, job_name=None, async_sdk=None
    )

    (step,) = list(job_spec["steps"])
    assert (
        _env(step).items()
        >= {
            "NEMO_AGENT_HARDENER_INFERENCE_SECRET_NAME": "agent-hardener-inference-key",
            "NEMO_AGENT_HARDENER_OPENSHELL_GATEWAY_ENDPOINT": "https://openshell.openshell.svc:8443",
            "NEMO_AGENT_HARDENER_OPENSHELL_CA_SECRET": "os-ca",
            "NEMO_AGENT_HARDENER_OPENSHELL_CLIENT_CERT_SECRET": "os-cert",
            "NEMO_AGENT_HARDENER_OPENSHELL_CLIENT_KEY_SECRET": "os-key",
        }.items()
    )


@pytest.mark.parametrize("execution_profiles", [CPU_ONLY])
async def test_container_jobs_are_refused_until_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common.AgentHardenerConfig, "get", lambda: _config(tmp_path))

    async def _fail(*_a: Any, **_k: Any) -> Any:
        pytest.fail("a refused war-game must not leave a run record behind")

    spec = run_module.WarGameSpec(manifest_id="m1", driver="service")
    with pytest.raises(HelixJobCompilationError, match="registry image"):
        await run_module.AgentHardenerRunJob.compile(
            workspace="default",
            spec=spec,
            entity_client=SimpleNamespace(get=_fail, create=_fail),
            job_name="job-1",
            async_sdk=None,
        )


@pytest.mark.usefixtures("api_config")
@pytest.mark.parametrize("execution_profiles", [CPU_ONLY])
async def test_synth_runs_in_the_task_image_without_a_subprocess_profile() -> None:
    spec = synth_module.SynthBenignSpec(manifest_id="m1")
    job_spec = await synth_module.AgentHardenerSynthBenignJob.compile(
        workspace="default", spec=spec, entity_client=None, job_name=None, async_sdk=None
    )

    (step,) = list(job_spec["steps"])
    assert step["executor"]["provider"] == "cpu"
    assert step["executor"]["container"]["command"] == ["nemo_agent_hardener_plugin.tasks.synth_benign"]


async def test_subprocess_step_forwards_the_hosts_openshell_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/op")
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/docker.sock")
    spec = run_module.WarGameSpec(config="agent-hardener.yaml")
    job_spec = await run_module.AgentHardenerRunJob.compile(
        workspace="default", spec=spec, entity_client=None, job_name=None, async_sdk=None
    )

    (step,) = list(job_spec["steps"])
    assert step["executor"]["provider"] == "subprocess"
    assert {"HOME", "DOCKER_HOST"} <= _env(step).keys()


@pytest.mark.parametrize("execution_profiles", [[SubprocessJobExecutionProfile(profile="other")]])
async def test_no_matching_profile_fails_before_precreating_a_run() -> None:
    async def _fail(*_a: Any, **_k: Any) -> Any:
        pytest.fail("an unschedulable war-game must not leave a run record behind")

    spec = run_module.WarGameSpec(manifest_id="m1", driver="service")
    with pytest.raises(HelixJobCompilationError, match="war-game step has nowhere to run"):
        await run_module.AgentHardenerRunJob.compile(
            workspace="default",
            spec=spec,
            entity_client=SimpleNamespace(get=_fail, create=_fail),
            job_name="job-1",
            async_sdk=None,
        )


def _config(tmp_path: Path, **overrides: Any) -> AgentHardenerConfig:
    return AgentHardenerConfig(
        venv_path=tmp_path / "venv",
        garak_venv_path=tmp_path / "garak-venv",
        operator_env_file=tmp_path / "absent.env",
        **overrides,
    )


def _runtime_env(config: AgentHardenerConfig, tmp_path: Path) -> dict[str, str]:
    return _common.job_runtime_env(config, sdk=object(), workspace="ws", openshell_home=tmp_path / "cfg")


def test_job_runtime_env_takes_the_inference_key_from_its_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    config = _config(tmp_path)
    monkeypatch.setattr(_common.AgentHardenerConfig, "get", lambda: config)
    monkeypatch.setattr(_common, "_resolve_secret", lambda _sdk, name, _ws: {"agent-hardener-inference-key": "k"}[name])

    assert _runtime_env(config, tmp_path) == {"INFERENCE_API_KEY": "k"}


def test_job_runtime_env_uses_the_key_the_preflight_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "from-env")
    config = _config(tmp_path)
    monkeypatch.setattr(_common.AgentHardenerConfig, "get", lambda: config)
    monkeypatch.setattr(_common, "_resolve_secret", lambda *_a: "from-secret")

    assert _runtime_env(config, tmp_path) == {"INFERENCE_API_KEY": "from-secret"}
    assert _common.resolve_model_key(object(), None, workspace="ws") == "from-secret"


class _OpenShell:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self._returncode = returncode
        self._stderr = stderr

    def __call__(self, args: list[str], *, env: dict[str, str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, env))
        return subprocess.CompletedProcess(args, self._returncode, stdout="", stderr=self._stderr)


@pytest.fixture
def openshell(monkeypatch: pytest.MonkeyPatch) -> _OpenShell:
    fake = _OpenShell()
    monkeypatch.setattr(gateway_module.shutil, "which", lambda _name: "/usr/local/bin/openshell")
    monkeypatch.setattr(gateway_module.subprocess, "run", fake)
    return fake


def test_an_unreadable_mtls_secret_reports_the_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, openshell: _OpenShell
) -> None:
    config = _config(
        tmp_path,
        openshell_gateway_endpoint="https://openshell.openshell.svc:8443",
        openshell_ca_secret="os-ca",
        openshell_client_cert_secret="os-cert",
        openshell_client_key_secret="os-key",
    )
    monkeypatch.setattr(_common.AgentHardenerConfig, "get", lambda: config)

    def _resolve(_sdk: Any, name: str, _ws: str) -> str:
        if name == "agent-hardener-inference-key":
            return "k"
        raise PermissionError(name)

    monkeypatch.setattr(_common, "_resolve_secret", _resolve)

    with pytest.raises(AgentHardenerRunError, match="could not read secret ws/os-ca .*PermissionError"):
        _runtime_env(config, tmp_path)
    assert openshell.calls == []


def test_no_endpoint_leaves_the_hosts_registration_alone(tmp_path: Path, openshell: _OpenShell) -> None:
    assert (
        gateway_module.register_openshell_gateway(
            _config(tmp_path), resolve_secret=MTLS_SECRETS.get, config_home=tmp_path / "cfg"
        )
        == {}
    )
    assert openshell.calls == []


def test_registers_a_plaintext_gateway_in_a_private_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, openshell: _OpenShell
) -> None:
    monkeypatch.setenv("NEMO_JOB_SECRETS", "job-secrets")
    monkeypatch.setenv("INFERENCE_API_KEY", "key")
    config = _config(tmp_path, openshell_gateway_endpoint="http://openshell.openshell.svc:8080")

    env = gateway_module.register_openshell_gateway(
        config, resolve_secret=MTLS_SECRETS.get, config_home=tmp_path / "cfg"
    )

    assert env == {"XDG_CONFIG_HOME": str(tmp_path / "cfg")}
    ((args, call_env),) = openshell.calls
    assert args == [
        "/usr/local/bin/openshell",
        "gateway",
        "add",
        "http://openshell.openshell.svc:8080",
        "--name",
        "auto-defender",
    ]
    assert call_env["XDG_CONFIG_HOME"] == str(tmp_path / "cfg")
    assert call_env.keys().isdisjoint({"NEMO_JOB_SECRETS", "INFERENCE_API_KEY"})


def test_registers_an_mtls_gateway_with_its_bundle_from_secrets(tmp_path: Path, openshell: _OpenShell) -> None:
    config = _config(
        tmp_path,
        openshell_gateway_endpoint="https://openshell.openshell.svc:8443",
        openshell_ca_secret="os-ca",
        openshell_client_cert_secret="os-cert",
        openshell_client_key_secret="os-key",
    )

    gateway_module.register_openshell_gateway(config, resolve_secret=MTLS_SECRETS.get, config_home=tmp_path / "cfg")

    mtls = tmp_path / "cfg" / "openshell" / "gateways" / "auto-defender" / "mtls"
    assert (mtls / "ca.crt").read_text() == "CA PEM"
    assert (mtls / "tls.crt").read_text() == "CERT PEM"
    assert (mtls / "tls.key").read_text() == "KEY PEM"
    assert stat.S_IMODE((mtls / "tls.key").stat().st_mode) == 0o600
    ((args, _env),) = openshell.calls
    assert args[-1] == "--local"


def test_mtls_key_is_private_even_over_an_existing_file(tmp_path: Path, openshell: _OpenShell) -> None:
    mtls = tmp_path / "cfg" / "openshell" / "gateways" / "auto-defender" / "mtls"
    mtls.mkdir(parents=True)
    (mtls / "tls.key").write_text("stale")
    (mtls / "tls.key").chmod(0o644)
    config = _config(
        tmp_path,
        openshell_gateway_endpoint="https://openshell.openshell.svc:8443",
        openshell_ca_secret="os-ca",
        openshell_client_cert_secret="os-cert",
        openshell_client_key_secret="os-key",
    )

    gateway_module.register_openshell_gateway(config, resolve_secret=MTLS_SECRETS.get, config_home=tmp_path / "cfg")

    assert stat.S_IMODE((mtls / "tls.key").stat().st_mode) == 0o600


def test_a_missing_mtls_secret_is_a_credential_error(tmp_path: Path, openshell: _OpenShell) -> None:
    config = _config(
        tmp_path,
        openshell_gateway_endpoint="https://openshell.openshell.svc:8443",
        openshell_ca_secret="os-ca",
        openshell_client_cert_secret="os-cert",
        openshell_client_key_secret="absent",
    )

    with pytest.raises(AgentHardenerRunError, match="'absent'") as raised:
        gateway_module.register_openshell_gateway(config, resolve_secret=MTLS_SECRETS.get, config_home=tmp_path / "cfg")
    assert raised.value.category == "missing_credential"
    assert openshell.calls == []


def test_a_failed_registration_surfaces_the_cli_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_module.shutil, "which", lambda _name: "/usr/local/bin/openshell")
    monkeypatch.setattr(gateway_module.subprocess, "run", _OpenShell(returncode=1, stderr="bad endpoint"))
    config = _config(tmp_path, openshell_gateway_endpoint="http://openshell.openshell.svc:8080")

    with pytest.raises(AgentHardenerRunError, match="bad endpoint"):
        gateway_module.register_openshell_gateway(config, resolve_secret=MTLS_SECRETS.get, config_home=tmp_path / "cfg")


def test_a_missing_openshell_cli_is_a_provisioning_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_module.shutil, "which", lambda _name: None)
    config = _config(tmp_path, openshell_gateway_endpoint="http://openshell.openshell.svc:8080")

    with pytest.raises(AgentHardenerRunError, match="not on PATH") as raised:
        gateway_module.register_openshell_gateway(config, resolve_secret=MTLS_SECRETS.get, config_home=tmp_path / "cfg")
    assert raised.value.category == "provisioning"
    assert "nemo agent-hardener setup" in (raised.value.remediation or "")


def test_the_task_image_pin_satisfies_the_setup_spec() -> None:
    lock_project = Path(__file__).parents[4] / "docker" / "locks" / "nhx-agent-hardener-tasks" / "pyproject.toml"
    (image_pin,) = tomllib.loads(lock_project.read_text())["project"]["dependencies"]
    setup_spec = Requirement(AgentHardenerConfig.model_fields["spec"].default)
    pinned = Requirement(image_pin)
    (pinned_version,) = (spec.version for spec in pinned.specifier)

    assert pinned.name == setup_spec.name
    assert setup_spec.specifier.contains(pinned_version, prereleases=True)


def test_an_https_gateway_requires_the_mtls_secrets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="uses mTLS"):
        _config(tmp_path, openshell_gateway_endpoint="https://openshell.openshell.svc:8443")


def test_the_gateway_endpoint_must_be_http_or_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="http:// or https://"):
        _config(tmp_path, openshell_gateway_endpoint="ssh://ops@gateway:8080")

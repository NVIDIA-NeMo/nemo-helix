# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import stat
import subprocess
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

CPU_ONLY = [DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())]
MTLS_SECRETS = {"os-ca": "CA PEM", "os-cert": "CERT PEM", "os-key": "KEY PEM"}


@pytest.fixture(autouse=True)
def _qualified_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor_module, "get_qualified_image", lambda name: f"reg.example/{name}:t")


def _env_names(step: Any) -> set[str]:
    return {variable["name"] for variable in step["environment"]}


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
    assert _env_names(step).isdisjoint({"HOME", "DOCKER_HOST", "XDG_CONFIG_HOME"})


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
    assert {"HOME", "DOCKER_HOST"} <= _env_names(step)


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


def test_job_runtime_env_takes_the_inference_key_from_its_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.setattr(_common, "_resolve_secret", lambda _sdk, name, _ws: {"agent-hardener-inference-key": "k"}[name])

    assert _common.job_runtime_env(_config(tmp_path), sdk=object(), workspace="ws") == {"INFERENCE_API_KEY": "k"}


def test_job_runtime_env_keeps_a_key_the_process_already_has(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_API_KEY", "from-env")
    monkeypatch.setattr(_common, "_resolve_secret", lambda *_a: pytest.fail("the Secret must not override the env"))

    assert _common.job_runtime_env(_config(tmp_path), sdk=object(), workspace="ws") == {}


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


def test_no_endpoint_leaves_the_hosts_registration_alone(tmp_path: Path, openshell: _OpenShell) -> None:
    assert gateway_module.register_openshell_gateway(_config(tmp_path), resolve_secret=MTLS_SECRETS.get) == {}
    assert openshell.calls == []


def test_registers_a_plaintext_gateway_in_a_private_config_dir(tmp_path: Path, openshell: _OpenShell) -> None:
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
        gateway_module.register_openshell_gateway(config, resolve_secret=MTLS_SECRETS.get)
    assert raised.value.category == "provisioning"


def test_an_https_gateway_requires_the_mtls_secrets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="uses mTLS"):
        _config(tmp_path, openshell_gateway_endpoint="https://openshell.openshell.svc:8443")


def test_the_gateway_endpoint_must_be_http_or_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="http:// or https://"):
        _config(tmp_path, openshell_gateway_endpoint="ssh://ops@gateway:8080")

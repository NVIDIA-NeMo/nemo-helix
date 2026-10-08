# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FabricAgentRuntime's two execution modes: chosen by ``sandbox=``, producing one trial shape.

The host-mode fakes come from ``test_fabric_runtime`` and the sandbox-mode fake provider from
``test_fabric_container_runtime``; this module checks the seam between them.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest
from nhx_evals_sdk.agent_eval.runtimes.fabric import _sandbox_execution
from nhx_evals_sdk.agent_eval.runtimes.fabric.container_runtime import FabricContainerRuntime
from nhx_evals_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nhx_evals_sdk.agent_eval.runtimes.sandbox.base import SandboxHandle
from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import AgentEvalTrialStatus
from nhx_evals_sdk.resolver_protocols import EnvSecretSource, MissingSecretError
from nhx_evals_sdk.values.common import SecretRef

from packages.nhx_evals_sdk.tests.agent_eval.test_fabric_container_runtime import _FakeProvider
from packages.nhx_evals_sdk.tests.agent_eval.test_fabric_runtime import (
    _FakeArtifact,
    _FakeResult,
    _install_fake_fabric,
)

_CONFIG = {"metadata": {"name": "eval"}, "harness": {"adapter_id": "nvidia.fabric.hermes"}}
_TASK = AgentEvalTask(
    id="fix-bug",
    intent="Fix the bug in fib.py",
    inputs={"instruction": "Fix fib.py.", "files": {"fib.py": "def fib(n): return n"}},
)


def _seeded_agent(provider: _FakeProvider) -> dict[str, Any]:
    return json.loads(provider.seeded["/in/agent.json"])


def test_image_without_a_sandbox_is_rejected() -> None:
    with pytest.raises(ValueError, match="sandbox="):
        FabricAgentRuntime(_CONFIG, image="doc-tools:1.0")


def test_host_accepts_env_secrets_and_records_only_references() -> None:
    runtime = FabricAgentRuntime(_CONFIG, env_secrets={"NVIDIA_API_KEY": SecretRef("ws/key")})
    assert runtime.runner_info().config["env_secrets"] == {"NVIDIA_API_KEY": "ws/key"}


@pytest.mark.parametrize("sandbox", [False, True])
def test_empty_env_secrets_do_not_require_an_env_source(sandbox: bool) -> None:
    runtime = FabricAgentRuntime(
        _CONFIG,
        sandbox=_FakeProvider() if sandbox else None,
        secret_resolver=cast(EnvSecretSource, object()),
    )
    assert runtime.runner_info().config["env_secrets"] == {}


@pytest.mark.parametrize("sandbox", [False, True])
def test_env_secrets_reject_value_only_sources(sandbox: bool) -> None:
    with pytest.raises(TypeError, match="can't name an env var"):
        FabricAgentRuntime(
            _CONFIG,
            sandbox=_FakeProvider() if sandbox else None,
            env_secrets={"KEY": SecretRef("ws/key")},
            secret_resolver=cast(EnvSecretSource, object()),
        )


@pytest.mark.parametrize("sandbox", [False, True])
def test_config_env_cannot_override_a_declared_secret(sandbox: bool) -> None:
    config = {**_CONFIG, "environment": {"env": {"KEY": "override"}}}
    with pytest.raises(ValueError, match="KEY"):
        FabricAgentRuntime(
            config, sandbox=_FakeProvider() if sandbox else None, env_secrets={"KEY": SecretRef("ws/key")}
        )


@pytest.mark.parametrize("sandbox", [False, True])
@pytest.mark.parametrize("role", ["default", "primary"])
async def test_model_override_preserves_connection_in_both_modes(
    sandbox: bool, role: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WS_NVIDIA_API_KEY", "first-value")
    selected = {
        "provider": "nvidia",
        "model": "nvidia/old",
        "api_key_env": "NVIDIA_API_KEY",
        "base_url": "https://provider.example/v1",
        "temperature": 0.2,
        "settings": {"nested": [1]},
    }
    config: dict[str, Any] = {**_CONFIG, "models": {role: selected}}
    original = copy.deepcopy(config)
    provider = _FakeProvider()
    client_cls = (
        None
        if sandbox
        else _install_fake_fabric(monkeypatch, lambda agent, kwargs: _FakeResult(status="succeeded", output="ok"))
    )
    runtime = FabricAgentRuntime(
        config,
        sandbox=provider if sandbox else None,
        image="img:test" if sandbox else None,
        model="nvidia/new",
        env_secrets={"NVIDIA_API_KEY": SecretRef("ws/nvidia-api-key")},
    )
    # The runner's snapshot must not share nested caller data.
    config["models"][role]["base_url"] = "https://changed.example"
    await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))
    if sandbox:
        actual = _seeded_agent(provider)["models"]["default"]
        assert provider.env == {"NVIDIA_API_KEY": "first-value"}
    else:
        assert client_cls is not None
        actual = client_cls.recorded[0]["agent"].models["default"].to_dict()
    expected = {
        **original["models"][role],
        "model": "nvidia/new",
        "api_key_env": "NVIDIA_API_KEY" if sandbox else "WS_NVIDIA_API_KEY",
    }
    assert actual == expected
    assert runtime._config == original


@pytest.mark.parametrize("failure", ["missing", "unusable"])
async def test_host_secret_failure_prevents_the_whole_task_batch(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WS_NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    if failure == "unusable":
        monkeypatch.setenv("WS_NVIDIA_API_KEY", "present")
    client_cls = _install_fake_fabric(monkeypatch, lambda agent, kwargs: _FakeResult(status="succeeded", output="ok"))
    runtime = FabricAgentRuntime(_CONFIG, env_secrets={"NVIDIA_API_KEY": SecretRef("ws/nvidia-api-key")})
    with pytest.raises(ValueError, match="env_secrets"):
        await runtime.run_tasks([_TASK, _TASK], AgentEvalRunConfig(work_dir=tmp_path))
    assert client_cls.recorded == []


async def test_reused_host_runner_resolves_each_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        **_CONFIG,
        "models": {"default": {"provider": "nvidia", "model": "nvidia/m", "api_key_env": "NVIDIA_API_KEY"}},
    }
    seen = []

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        import os

        key = agent.models["default"].extra["api_key_env"]
        seen.append((key, os.environ[key]))
        return _FakeResult(status="succeeded", output="ok")

    client_cls = _install_fake_fabric(monkeypatch, handler)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("WS_NVIDIA_API_KEY", "first")
    runtime = FabricAgentRuntime(config, env_secrets={"NVIDIA_API_KEY": SecretRef("ws/nvidia-api-key")})
    await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))
    monkeypatch.setenv("WS_NVIDIA_API_KEY", "rotated")
    await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))
    monkeypatch.delenv("WS_NVIDIA_API_KEY")
    monkeypatch.setenv("NVIDIA_API_KEY", "bare")
    await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))
    monkeypatch.delenv("NVIDIA_API_KEY")
    with pytest.raises(MissingSecretError):
        await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))
    assert seen == [("WS_NVIDIA_API_KEY", "first"), ("WS_NVIDIA_API_KEY", "rotated"), ("NVIDIA_API_KEY", "bare")]
    assert len(client_cls.recorded) == 3


@pytest.mark.parametrize("sandbox", [False, True])
async def test_provider_change_fails_before_tasks_or_image_build(
    sandbox: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nhx_evals_sdk.agent_eval.runtimes.fabric import runtime as runtime_module

    def unexpected_build() -> str:
        pytest.fail("provider change must fail before image build")

    monkeypatch.setattr(runtime_module, "ensure_fabric_image", unexpected_build)
    client_cls = _install_fake_fabric(monkeypatch, lambda agent, kwargs: _FakeResult(status="succeeded", output="ok"))
    config = {**_CONFIG, "models": {"default": {"provider": "nvidia", "model": "old", "api_key_env": "KEY"}}}
    runtime = FabricAgentRuntime(config, model="openai/new", sandbox=_FakeProvider() if sandbox else None)
    with pytest.raises(ValueError, match="changes provider"):
        await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))
    assert client_cls.recorded == []


async def test_missing_sandbox_secret_fails_before_image_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nhx_evals_sdk.agent_eval.runtimes.fabric import runtime as runtime_module

    def unexpected_build() -> str:
        pytest.fail("missing secret must fail before image build")

    monkeypatch.setattr(runtime_module, "ensure_fabric_image", unexpected_build)
    monkeypatch.delenv("WS_MISSING", raising=False)
    monkeypatch.delenv("MISSING", raising=False)
    provider = _FakeProvider()
    runtime = FabricAgentRuntime(_CONFIG, sandbox=provider, env_secrets={"KEY": SecretRef("ws/missing")})
    with pytest.raises(MissingSecretError, match="WS_MISSING"):
        await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))
    assert provider.execs == []


def test_host_only_settings_are_rejected_in_sandbox_mode() -> None:
    with pytest.raises(ValueError, match="base_dir is not supported in sandbox mode"):
        FabricAgentRuntime(_CONFIG, sandbox=_FakeProvider(), base_dir="/agents")  # type: ignore[arg-type]


async def test_no_sandbox_runs_on_the_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client_cls = _install_fake_fabric(monkeypatch, lambda agent, kwargs: _FakeResult(status="succeeded", output="ok"))
    (trial,) = await FabricAgentRuntime(_CONFIG).run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))

    assert trial.status == AgentEvalTrialStatus.COMPLETED
    assert len(client_cls.recorded) == 1  # Fabric.run was called in-process
    assert "sandbox_provider" not in trial.metadata
    assert trial.evidence is not None and "logs" not in trial.evidence.descriptors


async def test_a_sandbox_provider_runs_the_task_inside_it(tmp_path: Path) -> None:
    provider = _FakeProvider()
    runtime = FabricAgentRuntime(_CONFIG, sandbox=provider, image="img:test")  # type: ignore[arg-type]
    (trial,) = await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))

    assert trial.status == AgentEvalTrialStatus.COMPLETED
    assert len(provider.execs) == 1
    assert trial.metadata["sandbox_provider"] == "fake" and trial.metadata["image"] == "img:test"
    assert provider.aclosed == 1


async def test_both_modes_produce_the_same_trial_shape_for_the_same_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A metric written against one mode's trial must not have to know which mode ran it."""
    provider = _FakeProvider()
    sandboxed = FabricAgentRuntime(_CONFIG, sandbox=provider, image="img:test")  # type: ignore[arg-type]
    (sandbox_trial,) = await sandboxed.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path / "sandbox"))

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        # What Fabric + Relay leave behind on the host: the workspace the harness edited and the
        # promoted ATIF trajectory.
        Path(agent.environment.workspace, "fib.py").write_text("def fib(n): return n", encoding="utf-8")
        atif = Path(agent.relay["output_dir"]) / "trajectory-1.atif.json"
        atif.write_text('{"steps": []}', encoding="utf-8")
        return _FakeResult(
            status="succeeded",
            output={"response": "fixed the bug"},
            artifacts=[_FakeArtifact("trajectory", "atif", atif)],
        )

    _install_fake_fabric(monkeypatch, handler)
    (host_trial,) = await FabricAgentRuntime(_CONFIG).run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path / "host"))

    assert host_trial.status == sandbox_trial.status == AgentEvalTrialStatus.COMPLETED
    assert host_trial.id == sandbox_trial.id
    assert host_trial.output is not None and sandbox_trial.output is not None
    assert host_trial.output.response == sandbox_trial.output.response
    assert host_trial.output.output_text == sandbox_trial.output.output_text
    assert host_trial.evidence is not None and sandbox_trial.evidence is not None
    for key in ("result", "trace", "workspace"):
        host, sandbox = host_trial.evidence.require(key), sandbox_trial.evidence.require(key)
        assert (host.kind, host.format) == (sandbox.kind, sandbox.format), key
    # Same per-task layout on disk, so tooling that reads evidence dirs sees one tree shape.
    host_dir = Path(str(host_trial.output.metadata["evidence_dir"]))
    sandbox_dir = Path(str(sandbox_trial.output.metadata["evidence_dir"]))
    assert {"fabric_result.json", "workspace", "relay"} <= {p.name for p in host_dir.iterdir()}
    assert {"fabric_result.json", "workspace", "relay"} <= {p.name for p in sandbox_dir.iterdir()}
    # Trial metadata differs only by the keys that say where the harness ran.
    assert set(sandbox_trial.metadata) - set(host_trial.metadata) == {"image", "sandbox_provider"}
    assert set(host_trial.metadata) <= set(sandbox_trial.metadata)


async def test_sandbox_mode_applies_the_model_as_the_default(tmp_path: Path) -> None:
    provider = _FakeProvider()
    runtime = FabricAgentRuntime(_CONFIG, sandbox=provider, image="img:test", model="nvidia/some-model")  # type: ignore[arg-type]
    (trial,) = await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))

    assert _seeded_agent(provider)["models"]["default"] == {"provider": "nvidia", "model": "nvidia/some-model"}
    assert trial.metadata["agent_model"] == "nvidia/some-model"


async def test_sandbox_mode_honours_timeout_s(tmp_path: Path) -> None:
    class _Timing(_FakeProvider):
        timeouts: list[object] = []

        async def exec(self, handle: SandboxHandle, command: str, **kwargs: object) -> Any:
            _Timing.timeouts.append(kwargs.get("timeout_s"))
            return await super().exec(handle, command, **kwargs)

    provider = _Timing()
    runtime = FabricAgentRuntime(_CONFIG, sandbox=provider, image="img:test", timeout_s=42)  # type: ignore[arg-type]
    await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))
    assert _Timing.timeouts == [42]


async def test_sandbox_mode_stamps_trajectory_extra_onto_the_atif_config(tmp_path: Path) -> None:
    provider = _FakeProvider()
    runtime = FabricAgentRuntime(
        _CONFIG,
        sandbox=provider,  # type: ignore[arg-type]
        image="img:test",
        trajectory_extra={"nemo.optimizer.trial": 7},
    )
    await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))

    serialized = json.dumps(_seeded_agent(provider))
    assert '"nemo.optimizer.trial": 7' in serialized
    assert '"nemo.optimizer.row_id": "fix-bug"' in serialized


async def test_sandbox_mode_without_trajectory_capture_seeds_no_relay_or_receiver(tmp_path: Path) -> None:
    provider = _FakeProvider(atif=False)
    runtime = FabricAgentRuntime(_CONFIG, sandbox=provider, image="img:test", capture_trajectory=False)  # type: ignore[arg-type]
    (trial,) = await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))

    agent = _seeded_agent(provider)
    assert "relay" not in agent and "telemetry" not in agent
    assert _sandbox_execution._RECEIVER_PATH not in provider.seeded
    (command,) = provider.execs
    assert _sandbox_execution._RECEIVER_PATH not in command
    assert trial.status == AgentEvalTrialStatus.COMPLETED
    assert trial.evidence is not None and "trace" not in trial.evidence.descriptors


async def test_sandbox_artifacts_under_out_are_remapped_to_the_downloaded_tree(tmp_path: Path) -> None:
    class _WithArtifacts(_FakeProvider):
        async def download_dir(self, handle: SandboxHandle, source_dir: str, target_dir: Path) -> None:
            await super().download_dir(handle, source_dir, target_dir)
            (target_dir / "artifacts").mkdir()
            (target_dir / "artifacts" / "stdout.txt").write_text("hi", encoding="utf-8")
            (target_dir / "artifacts" / "escape-link").symlink_to("/etc")
            envelope = json.loads((target_dir / "fabric_result.json").read_text(encoding="utf-8"))
            envelope["artifacts"] = {
                "artifacts": [
                    {"name": "stdout", "kind": "log", "path": "/out/artifacts/stdout.txt", "media_type": "text/plain"},
                    {"name": "elsewhere", "kind": "log", "path": "/tmp/not-downloaded.txt"},
                    {"name": "escape", "kind": "log", "path": "/out/../../etc/passwd"},
                    {"name": "symlinked", "kind": "log", "path": "/out/artifacts/escape-link/hosts"},
                ]
            }
            (target_dir / "fabric_result.json").write_text(json.dumps(envelope), encoding="utf-8")

    provider = _WithArtifacts()
    runtime = FabricAgentRuntime(_CONFIG, sandbox=provider, image="img:test")  # type: ignore[arg-type]
    (trial,) = await runtime.run_tasks([_TASK], AgentEvalRunConfig(work_dir=tmp_path))

    assert trial.evidence is not None
    stdout = trial.evidence.require("stdout")
    assert Path(str(stdout.ref)).read_text(encoding="utf-8") == "hi"
    assert "elsewhere" not in trial.evidence.descriptors
    assert "escape" not in trial.evidence.descriptors  # a manifest path must not point outside the evidence dir
    assert "symlinked" not in trial.evidence.descriptors  # nor resolve outside it through a symlink in /out


def test_container_runtime_alias_warns_and_forwards_to_sandbox_mode() -> None:
    provider = _FakeProvider()
    with pytest.warns(DeprecationWarning, match="FabricAgentRuntime\\(config, sandbox=provider"):
        runtime = FabricContainerRuntime(_CONFIG, provider=provider, image="img:test")  # type: ignore[arg-type]

    assert isinstance(runtime, FabricAgentRuntime)
    info = runtime.runner_info()
    assert info.name == "fabric"
    assert info.config["sandbox"] == "fake" and info.config["image"] == "img:test"

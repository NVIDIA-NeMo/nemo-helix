# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FabricAgentRuntime's two execution modes: chosen by ``sandbox=``, producing one trial shape.

The host-mode fakes come from ``test_fabric_runtime`` and the sandbox-mode fake provider from
``test_fabric_container_runtime``; this module checks the seam between them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.fabric import _sandbox_execution
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.container_runtime import FabricContainerRuntime
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.runtime import FabricAgentRuntime
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.base import SandboxHandle
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrialStatus
from nemo_evaluator_sdk.values.common import SecretRef

from packages.nemo_evaluator_sdk.tests.agent_eval.test_fabric_container_runtime import _FakeProvider
from packages.nemo_evaluator_sdk.tests.agent_eval.test_fabric_runtime import (
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


def test_secrets_without_a_sandbox_are_rejected() -> None:
    with pytest.raises(ValueError, match="sandbox="):
        FabricAgentRuntime(_CONFIG, secrets={"NVIDIA_API_KEY": SecretRef(root="k")})


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

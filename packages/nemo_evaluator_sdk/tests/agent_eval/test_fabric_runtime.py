# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FabricAgentRuntime using a fake nemo_fabric SDK (the native package is optional)."""

from __future__ import annotations

import copy
import json
import sys
import types
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.fabric import runtime as fabric_runtime
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_evaluator_sdk.values.evidence import EVIDENCE_FORMAT_ATIF, EVIDENCE_TRACE


class _FakeEnvironment:
    """Stand-in for nemo_fabric.EnvironmentConfig (the runtime sets workspace/provider/artifacts)."""

    def __init__(self, *, provider: str = "local", workspace: str | None = None, artifacts: str | None = None) -> None:
        self.provider = provider
        self.workspace = workspace
        self.artifacts = artifacts


class _FakeRuntimeCfg:
    def __init__(self, artifacts: str | None = None) -> None:
        self.artifacts = artifacts


class _FakeHarness:
    """Stand-in for nemo_fabric FabricConfig.harness — the skill path reads ``adapter_id`` off it."""

    def __init__(self, adapter_id: str) -> None:
        self.adapter_id = adapter_id


class _FakeConfig:
    """Stand-in for nemo_fabric.FabricConfig with the config-first helpers the runtime composes onto."""

    def __init__(self, mapping: dict[str, Any]) -> None:
        self.mapping = mapping
        self.harness = _FakeHarness(mapping.get("harness", {}).get("adapter_id", ""))
        self.environment: _FakeEnvironment | None = None
        self.runtime = _FakeRuntimeCfg()
        self.models: dict[str, Any] = dict(mapping.get("models", {}))
        self.relay: dict[str, Any] | None = None  # records enable_relay(...)
        # Mirrors FabricConfig.skills.paths: seeded from the config, then appended to by
        # add_skill_path (which the capability-plan probe and native skill injection both use).
        self.skill_paths: list[str] = [str(p) for p in mapping.get("skills", {}).get("paths", [])]

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> _FakeConfig:
        return cls(mapping)

    def model_copy(self, *, deep: bool = False) -> _FakeConfig:
        clone = _FakeConfig(self.mapping)
        clone.environment = copy.deepcopy(self.environment)
        clone.runtime = _FakeRuntimeCfg(self.runtime.artifacts)
        clone.models = copy.deepcopy(self.models)
        clone.relay = copy.deepcopy(self.relay)
        clone.skill_paths = list(self.skill_paths)
        return clone

    def add_skill_path(self, path: Any) -> _FakeConfig:
        # Real SkillConfig.add_path appends only if absent, preserving order.
        value = str(path)
        if value not in self.skill_paths:
            self.skill_paths.append(value)
        return self

    def enable_relay(
        self,
        *,
        project: str | None = None,
        output_dir: str | None = None,
        observability: Any = None,
        components: Any = None,
        policy: Any = None,
    ) -> _FakeConfig:
        self.relay = {
            "project": project,
            "output_dir": output_dir,
            "observability": observability,
            "components": components,
            "policy": policy,
        }
        return self


class _FakeModelConfig:
    """Stand-in for nemo_fabric.ModelConfig — FabricConfig.models is dict[str, ModelConfig], so the
    runtime must build the typed model rather than assign a raw dict (which Fabric tolerates but
    does not validate, and warns about on serialization)."""

    def __init__(self, *, provider: str, model: str, **extra: Any) -> None:
        self.provider = provider
        self.model = model
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, **self.extra}


class _FakeRunRequest:
    """Stand-in for nemo_fabric.RunRequest (Fabric.run folds input + request id into it)."""

    def __init__(self, *, input: Any = None, request_id: str | None = None) -> None:
        self.input = input
        self.request_id = request_id


class _FakeRelayModel:
    """Stand-in for Fabric's typed relay models (RelayObservabilityConfig/RelayAtifConfig/RelayAtofConfig).

    The runtime constructs these to hand to ``enable_relay(observability=...)``; it never introspects
    them, so a plain kwargs bag suffices.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeArtifact:
    def __init__(self, name: str, kind: str, path: Path, media_type: str | None = None) -> None:
        self.name = name
        self.kind = kind
        self.path = path
        self.media_type = media_type


class _FakeManifest:
    def __init__(self, artifacts: list[_FakeArtifact]) -> None:
        self.root: Path | None = None
        self.artifacts = artifacts


class _FakeError:
    def __init__(self, stage: str, code: str, message: str) -> None:
        self.stage = stage
        self.code = code
        self.message = message


class _FakeTelemetry:
    def __init__(self, *, provider: str, kind: str, uri: str | None, trace_id: str | None) -> None:
        self.provider = provider
        self.kind = kind
        self.uri = uri
        self.trace_id = trace_id


class _FakeEvent:
    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        self.message = message


# Adapters the fake planner reports as accepting the native Fabric ``skills`` config, mirroring
# adapters/*/fabric-adapter.json. ``acme.custom.native`` stands in for an END-USER adapter the platform
# doesn't ship — the runtime learns it accepts skills purely from the plan, with no hardcoded list.
_NATIVE_SKILL_ADAPTERS = {
    "nvidia.fabric.hermes",
    "nvidia.fabric.claude",
    "acme.custom.native",
}
_KNOWN_HARNESSES = ("hermes", "codex", "claude", "deepagents")


def _harness_name(adapter_id: str) -> str:
    """Derive the Fabric harness name from an adapter id (mirrors what the planner reports)."""
    return next((h for h in _KNOWN_HARNESSES if h in adapter_id), "custom")


class _FakeAdapterInfo:
    """Stand-in for nemo_fabric AdapterInfo — the runtime reads ``harness`` off ``RunPlan.adapter``."""

    def __init__(self, harness: str) -> None:
        self.harness = harness


class _FakePlan:
    """Stand-in for nemo_fabric RunPlan (from Fabric.plan): capability routing + selected adapter."""

    def __init__(self, *, capability_plan: dict[str, Any], harness: str) -> None:
        self.capability_plan = capability_plan
        self.adapter = _FakeAdapterInfo(harness)


class _FakeResult:
    def __init__(
        self,
        *,
        status: str,
        output: Any = None,
        error: _FakeError | None = None,
        artifacts: list[_FakeArtifact] | None = None,
    ) -> None:
        self.status = status
        self.output = output
        self.error = error
        self.harness = "codex"
        self.adapter_id = "nvidia.fabric.codex"
        self.adapter_kind = "process"
        self.invocation_id = "inv-1"
        self.artifacts = _FakeManifest(artifacts or [])
        self.telemetry = [_FakeTelemetry(provider="relay", kind="trace", uri="file:///relay", trace_id="tid-1")]
        self.events = [_FakeEvent("runtime_start", "started")]

    def to_mapping(self) -> dict[str, Any]:
        return {"status": self.status, "output": self.output, "harness": self.harness}


def _install_fake_fabric(monkeypatch: pytest.MonkeyPatch, handler: Any) -> type:
    """Inject a fake ``nemo_fabric`` module (the runtime imports it lazily); return the client class."""

    class _FakeClient:
        # Fabric is a plain reusable facade (not an async context manager).
        recorded: list[dict[str, Any]] = []
        planned: list[dict[str, Any]] = []

        async def run(self, agent: Any, **kwargs: Any) -> Any:
            _FakeClient.recorded.append({"agent": agent, **kwargs})
            return handler(agent, kwargs)

        def plan(self, agent: Any, *, base_dir: Any = None) -> _FakePlan:
            # Mirror Fabric's capability planner: a ``skills`` route appears only when a skill path is
            # attached, and it routes ``harness_native`` iff the selected adapter accepts native skills.
            _FakeClient.planned.append({"agent": agent, "base_dir": base_dir})
            adapter_id = agent.harness.adapter_id
            has_skill_path = bool(getattr(agent, "skill_paths", None))
            native = has_skill_path and adapter_id in _NATIVE_SKILL_ADAPTERS
            routes = (
                [{"kind": "skills", "target": "harness_native" if native else "unsupported"}] if has_skill_path else []
            )
            return _FakePlan(capability_plan={"routes": routes}, harness=_harness_name(adapter_id))

    _FakeClient.recorded = []
    _FakeClient.planned = []
    module = types.ModuleType("nemo_fabric")
    module.Fabric = _FakeClient  # type: ignore[attr-defined]
    module.FabricConfig = _FakeConfig  # type: ignore[attr-defined]
    module.EnvironmentConfig = _FakeEnvironment  # type: ignore[attr-defined]
    module.ModelConfig = _FakeModelConfig  # type: ignore[attr-defined]
    module.RunRequest = _FakeRunRequest  # type: ignore[attr-defined]
    # The runtime builds the relay observability config from Fabric's own typed models (lazy import).
    module.RelayObservabilityConfig = _FakeRelayModel  # type: ignore[attr-defined]
    module.RelayAtifConfig = _FakeRelayModel  # type: ignore[attr-defined]
    module.RelayAtofConfig = _FakeRelayModel  # type: ignore[attr-defined]
    module.RelayAtofFileSinkConfig = _FakeRelayModel  # type: ignore[attr-defined]
    module.RelayOpenTelemetryConfig = _FakeRelayModel  # type: ignore[attr-defined]
    module.RelayOpenTelemetryEndpointConfig = _FakeRelayModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo_fabric", module)

    # ``run_tasks`` fails fast on ``import nemo_relay.observability`` when capture_trajectory is on
    # (the relay gateway is a runtime requirement); stub the optional package so that guard resolves
    # without the native dependency. The observability config itself is now built from nemo_fabric's
    # typed models (stubbed above), not from nemo_relay, so this stand-in only needs to be importable.
    relay_mod = types.ModuleType("nemo_relay")
    observability_mod = types.ModuleType("nemo_relay.observability")
    relay_mod.observability = observability_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo_relay", relay_mod)
    monkeypatch.setitem(sys.modules, "nemo_relay.observability", observability_mod)
    return _FakeClient


_TASK = AgentEvalTask(id="task/1", intent="Answer.", inputs={"instruction": "Ping?"})
_CONFIG = {"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.codex"}}


@pytest.mark.asyncio
async def test_fabric_runtime_maps_succeeded_result_to_completed_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _FakeArtifact("stdout", "log", tmp_path / "stdout.txt")

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(
            status="succeeded",
            output={"adapter": "cli", "response": "PONG", "returncode": 0},
            artifacts=[artifact],
        )

    client_cls = _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, model="openai/gpt-5.4", work_root=tmp_path / "fabric")

    trials = await runtime.run_tasks([_TASK])

    trial = trials[0]
    assert trial.status == "completed"
    assert trial.output is not None
    assert trial.output.output_text == "PONG"  # extracted from the adapter envelope's `response`
    assert trial.output.response == {"adapter": "cli", "response": "PONG", "returncode": 0}
    assert trial.metadata["harness"] == "codex"
    assert trial.metadata["adapter_id"] == "nvidia.fabric.codex"
    assert trial.metadata["generated"] is True
    # agent_ok mirrors the Codex runtime so AgentPhaseSuccessMetric scores the phase as clean.
    assert trial.metadata["agent_ok"] is True
    # Evidence: the persisted result envelope + each Fabric artifact by name.
    assert trial.evidence is not None
    assert trial.evidence.descriptors["result"].ref.endswith("fabric_result.json")
    assert trial.evidence.descriptors["stdout"].ref == str(tmp_path / "stdout.txt")
    # Evidence lands under a per-run id subdir (isolates A/B baseline vs. skilled runs sharing a root).
    result_file = next((tmp_path / "fabric").glob("*/000000-task-1/fabric_result.json"))
    assert json.loads(result_file.read_text(encoding="utf-8"))["status"] == "succeeded"
    # Config-first: the model is set on the config's default model and relay (ATIF trajectory) is
    # enabled on the config, rather than layered as profile overlays.
    composed = client_cls.recorded[0]["agent"]
    assert (composed.models["default"].provider, composed.models["default"].model) == ("openai", "openai/gpt-5.4")
    assert composed.relay is not None  # capture_trajectory defaults on -> enable_relay(...) called
    assert client_cls.recorded[0]["request"].request_id == "task/1"
    # Telemetry reference is preserved end-to-end (uri + trace_id), not just provider/kind.
    assert trial.evidence.metadata["telemetry"][0]["uri"] == "file:///relay"
    assert trial.evidence.metadata["telemetry"][0]["trace_id"] == "tid-1"


@pytest.mark.asyncio
async def test_fabric_runtime_prompt_excludes_intent_and_frames_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The prompt is exactly the task instruction — no runtime framing. `task.intent` is eval-side
    # "desired behavior" metadata and must never reach the agent (reward-hacking hole); other inputs
    # keys are not templated into the prompt either.
    task = AgentEvalTask(
        id="task/2",
        intent="SECRET_GRADER_INTENT",
        inputs={"instruction": "Ping?", "files": {"data.csv": "s3://seed"}, "hint": "be terse"},
    )

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output="ok")

    client_cls = _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")

    await runtime.run_tasks([task])

    prompt = client_cls.recorded[0]["request"].input
    assert prompt == "Ping?"  # the instruction verbatim, nothing else
    assert "SECRET_GRADER_INTENT" not in prompt  # intent stays eval-side
    assert "be terse" not in prompt  # non-instruction inputs are not templated in


@pytest.mark.asyncio
async def test_fabric_runtime_maps_atif_artifact_to_trace_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atif = _FakeArtifact("relay_atif", "atif", tmp_path / "trajectory.atif.json", "application/json")
    atif.path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "steps": [_step(prompt_tokens=8, completion_tokens=2)],
                "final_metrics": {"total_cost_usd": 0.25},
            }
        ),
        encoding="utf-8",
    )

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output={"response": "ok"}, artifacts=[atif])

    _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")

    trials = await runtime.run_tasks([_TASK])

    evidence = trials[0].evidence
    assert evidence is not None
    # The ATIF artifact is exposed both under its own name and the standard trace key.
    assert evidence.descriptors["relay_atif"].ref == str(tmp_path / "trajectory.atif.json")
    trace = evidence.descriptors[EVIDENCE_TRACE]
    assert trace.format == EVIDENCE_FORMAT_ATIF
    assert trace.ref == str(tmp_path / "trajectory.atif.json")
    assert trials[0].measurements.model_dump(exclude_none=True) == {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
        "cost_usd": 0.25,
    }
    assert trials[0].measurements.runtime_sec is None
    assert "prompt_tokens" not in trials[0].metadata


def _workspace_from_config(config: Any) -> Path:
    """Pull the staged workspace path out of the composed per-task config."""
    return Path(config.environment.workspace)


@pytest.mark.asyncio
async def test_supplied_config_cannot_override_evaluator_owned_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Since Fabric dropped profile overlays there is exactly one config, so the evaluator's per-task
    # workspace (isolation + `workspace` evidence integrity) and model-under-eval stay authoritative by
    # being composed on last. A caller config that pins these must NOT survive into the run.
    hijacked = {
        **_CONFIG,
        "environment": {"provider": "local", "workspace": "/caller/hijacked-workspace"},
        "models": {"default": {"provider": "openai", "model": "caller/rogue-model"}},
    }

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output="ok")

    client_cls = _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=hijacked, model="openai/gpt-5.4", work_root=tmp_path / "fabric")

    await runtime.run_tasks([_TASK])

    config = client_cls.recorded[0]["agent"]
    # The config handed to Fabric carries the evaluator's per-task workspace and model, not the
    # caller's — and there is no second layer that could put them back.
    assert config.environment.workspace != "/caller/hijacked-workspace"
    assert Path(config.environment.workspace).is_relative_to(tmp_path / "fabric")
    assert (config.models["default"].provider, config.models["default"].model) == ("openai", "openai/gpt-5.4")


@pytest.mark.asyncio
async def test_fabric_runtime_seeds_workspace_and_exposes_workspace_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = AgentEvalTask(
        id="fix/bug",
        intent="Fix the bug.",
        inputs={"instruction": "make the tests pass", "files": {"calc.py": "value = 1\n"}},
    )

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        # The harness runs in the staged workspace; simulate an edit it leaves behind.
        workspace = _workspace_from_config(agent)
        (workspace / "result.txt").write_text("done", encoding="utf-8")
        return _FakeResult(status="succeeded", output={"response": "ok"})

    client_cls = _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")

    trials = await runtime.run_tasks([task])

    trial = trials[0]
    assert trial.status == "completed"
    # The composed config carries environment.workspace (the harness's cwd) with provider=local.
    composed = client_cls.recorded[0]["agent"]
    assert composed.environment.provider == "local"
    workspace = _workspace_from_config(composed)
    # The seed file is staged and the agent's edit is present in the same dir.
    assert (workspace / "calc.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (workspace / "result.txt").read_text(encoding="utf-8") == "done"
    # The final workspace is exposed as filesystem evidence (same key/kind as the Codex runtime).
    assert trial.evidence is not None
    workspace_evidence = trial.evidence.descriptors["workspace"]
    assert workspace_evidence.kind == "filesystem"
    assert workspace_evidence.ref == str(workspace)
    # Seed-file contents are not inlined into the prompt (they are already on disk in the workspace).
    harness_input = client_cls.recorded[0]["request"].input
    assert "value = 1" not in harness_input


@pytest.mark.asyncio
async def test_fabric_runtime_stages_workspace_even_without_seed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every task gets a workspace, even with no inputs['files'] — the harness may still create files,
    # and the per-task dir is exposed as evidence uniformly (a from-scratch coding task, say).
    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output="ok")

    client_cls = _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")

    trials = await runtime.run_tasks([_TASK])  # _TASK has no 'files' input

    workspace = _workspace_from_config(client_cls.recorded[0]["agent"])
    assert workspace.is_dir()
    assert trials[0].evidence is not None
    assert trials[0].evidence.descriptors["workspace"].ref == str(workspace)


@pytest.mark.asyncio
async def test_fabric_runtime_bad_seed_fails_only_that_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A seed path escaping the workspace must fail just this task (as a failed trial), not abort the run.
    bad_task = AgentEvalTask(
        id="bad/seed",
        intent="unused",
        inputs={"files": {"../escape.py": "nope"}},
    )

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output="unreached")

    _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")

    trials = await runtime.run_tasks([bad_task])

    assert trials[0].status == "failed"
    assert trials[0].metadata["error_type"] == "WorkspaceSeedError"


@pytest.mark.asyncio
async def test_fabric_runtime_passes_trajectory_extra_to_atif_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output={"response": "ok"})

    client_cls = _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(
        config=_CONFIG,
        work_root=tmp_path / "fabric",
        trajectory_extra={"nemo.optimizer.experiment_id": "exp-1", "nemo.optimizer.trial_number": 2},
    )

    await runtime.run_tasks([_TASK])

    # Trajectory capture is composed onto the per-task config via enable_relay (no profile overlays).
    observability = client_cls.recorded[0]["agent"].relay["observability"]
    atif_extra = observability.kwargs["atif"].kwargs["extra"]
    assert atif_extra["nemo.optimizer.experiment_id"] == "exp-1"
    assert atif_extra["nemo.optimizer.trial_number"] == 2
    assert atif_extra["nemo.optimizer.row_id"] == "task/1"


@pytest.mark.asyncio
async def test_fabric_runtime_capture_trajectory_false_skips_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output="ok")

    client_cls = _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric", capture_trajectory=False)

    await runtime.run_tasks([_TASK])

    assert client_cls.recorded[0]["agent"].relay is None


@pytest.mark.asyncio
async def test_fabric_runtime_maps_failed_result_to_failed_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(
            status="failed",
            error=_FakeError(stage="invoke", code="process_exit_nonzero", message="boom"),
        )

    _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")

    trials = await runtime.run_tasks([_TASK])

    trial = trials[0]
    assert trial.status == "failed"
    assert trial.output is None
    assert trial.metadata["error_type"] == "process_exit_nonzero"
    assert trial.metadata["error"] == "boom"
    assert trial.metadata["agent_ok"] is False
    assert trial.evidence is not None
    assert "error" in trial.evidence.descriptors


@pytest.mark.asyncio
async def test_fabric_runtime_maps_timeout_to_failed_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output="never")

    _install_fake_fabric(monkeypatch, handler)

    async def fake_wait_for(awaitable: Any, timeout: float) -> Any:
        awaitable.close()
        evidence_dir = next((tmp_path / "fabric").glob("*/000000-task-1"))
        relay_dir = evidence_dir / "relay"
        relay_dir.mkdir(exist_ok=True)
        (relay_dir / "trajectory-timeout.atif.json").write_text(
            json.dumps({"schema_version": "ATIF-v1.7", "steps": [_step(prompt_tokens=5, completion_tokens=1)]}),
            encoding="utf-8",
        )
        raise TimeoutError

    monkeypatch.setattr(fabric_runtime.asyncio, "wait_for", fake_wait_for)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")

    trials = await runtime.run_tasks([_TASK])

    assert trials[0].status == "failed"
    assert trials[0].metadata["error_type"] == "TimeoutError"
    assert trials[0].measurements.total_tokens == 6


@pytest.mark.asyncio
async def test_fabric_runtime_captures_run_exception_as_failed_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        raise RuntimeError("client blew up")

    _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")

    trials = await runtime.run_tasks([_TASK])

    assert trials[0].status == "failed"
    assert trials[0].metadata["error_type"] == "RuntimeError"
    assert trials[0].metadata["error"] == "client blew up"


@pytest.mark.asyncio
async def test_fabric_runtime_raises_without_nemo_fabric(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # run_tasks surfaces a clear error when the optional native package can't be imported.
    monkeypatch.setitem(sys.modules, "nemo_fabric", None)  # forces ImportError on `import nemo_fabric`
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")
    with pytest.raises(RuntimeError, match="requires the `nemo-fabric` package"):
        await runtime.run_tasks([_TASK])


@pytest.mark.asyncio
async def test_fabric_runtime_trajectory_capture_requires_nemo_relay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Trajectory capture builds the profile from nemo_relay; surface a clear error when it is absent.
    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output="ok")

    _install_fake_fabric(monkeypatch, handler)  # installs fabric + relay stubs...
    monkeypatch.setitem(sys.modules, "nemo_relay.observability", None)  # ...then force ImportError on relay
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")
    with pytest.raises(RuntimeError, match="nemo-relay"):
        await runtime.run_tasks([_TASK])


@pytest.mark.asyncio
async def test_fabric_success_trial_scores_agent_phase_success_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression guard: AgentPhaseSuccessMetric reads candidate.metadata["agent_ok"]; a successful Fabric
    # trial must set it (via the same _trial_sample path the evaluator uses) so the metric scores True.
    from nemo_evaluator_sdk.agent_eval.evaluator import _metric_row, _trial_sample
    from nemo_evaluator_sdk.agent_eval.metrics import AgentPhaseSuccessMetric
    from nemo_evaluator_sdk.execution.samples import build_metric_input

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output={"response": "ok"})

    _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")

    trial = (await runtime.run_tasks([_TASK]))[0]
    metric_input = build_metric_input(_metric_row(_TASK, trial), _trial_sample(trial), 0)
    result = await AgentPhaseSuccessMetric().compute_scores(metric_input)

    assert result.outputs[0].name == "agent_phase_success"
    assert result.outputs[0].value is True


@pytest.mark.asyncio
async def test_fabric_runtime_normalizes_runoutput_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Newer Fabric wraps RunResult.output in a RunOutput Mapping (not a plain JSON value); the runtime
    # must copy it into a plain dict so it round-trips through the trial's JsonValue response field.
    class _FakeRunOutput(Mapping):
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = dict(data)

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output=_FakeRunOutput({"response": "PONG", "returncode": 0}))

    _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")

    trial = (await runtime.run_tasks([_TASK]))[0]

    assert trial.status == "completed"
    assert trial.output is not None
    assert trial.output.output_text == "PONG"  # extracted from the normalized mapping
    assert trial.output.response == {"response": "PONG", "returncode": 0}  # plain dict, not RunOutput


def _skill_bundle(base: Path, *, name: str = "code-review", body: str = "Be thorough.") -> Path:
    """Write a minimal agentskills bundle under ``base/skills/<name>/`` and return its path."""
    root = base / "skills" / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n\n{body}\n", encoding="utf-8")
    return root


_HERMES_CONFIG = {"metadata": {"name": "a"}, "harness": {"adapter_id": "nvidia.fabric.hermes"}}


@pytest.mark.asyncio
async def test_fabric_runtime_native_skill_adds_skill_path_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output={"response": "ok"})

    client_cls = _install_fake_fabric(monkeypatch, handler)
    skill = AgentSkill.from_directory(_skill_bundle(tmp_path, body="# Code Review\n\nBe thorough."))
    runtime = fabric_runtime.FabricAgentRuntime(config=_HERMES_CONFIG, work_root=tmp_path / "fabric", skills=[skill])

    trials = await runtime.run_tasks([_TASK])

    # The mode is resolved by probing Fabric's capability planner (with a skill path attached), not a
    # hardcoded adapter list.
    assert client_cls.planned, "expected the runtime to query Fabric.plan for skills routing"
    assert client_cls.planned[0]["agent"].skill_paths, "expected a probe skill path attached for planning"
    # The staged <name>/ skill dir is on the config handed to client.run.
    config = client_cls.recorded[0]["agent"]
    assert config.skill_paths[-1].endswith("/code-review")
    # Provenance is stamped into trial metadata for the A/B diff.
    prov = trials[0].metadata["skill"]
    assert prov["name"] == "code-review"
    assert prov["mode"] == "native"
    assert prov["hash"]


@pytest.mark.asyncio
async def test_fabric_runtime_native_skill_preserves_preconfigured_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Injection appends to the config's skills.paths, so skills the config already declares survive.
    # If they were dropped, the treated arm would differ from the baseline by more than the injected
    # skill and the A/B would be invalid.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output={"response": "ok"})

    client_cls = _install_fake_fabric(monkeypatch, handler)
    config = {**_HERMES_CONFIG, "skills": {"paths": ["/pre/existing-a", "/pre/existing-b"]}}
    skill = AgentSkill.from_directory(_skill_bundle(tmp_path))
    runtime = fabric_runtime.FabricAgentRuntime(config=config, work_root=tmp_path / "fabric", skills=[skill])

    await runtime.run_tasks([_TASK])

    paths = client_cls.recorded[0]["agent"].skill_paths
    # Config-declared skills are preserved, in order, ahead of the evaluated skill.
    assert paths[:2] == ["/pre/existing-a", "/pre/existing-b"]
    assert paths[-1].endswith("/code-review")


@pytest.mark.asyncio
async def test_fabric_runtime_native_skill_on_runtime_discovered_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An end-user adapter the platform doesn't ship (harness "custom", not codex) still gets native
    # injection purely because Fabric's planner routes its skills ``harness_native`` — nothing is
    # hardcoded, so runtime capability discovery is what makes this work.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output={"response": "ok"})

    client_cls = _install_fake_fabric(monkeypatch, handler)
    custom = {"metadata": {"name": "a"}, "harness": {"adapter_id": "acme.custom.native"}}
    skill = AgentSkill.from_directory(_skill_bundle(tmp_path))
    runtime = fabric_runtime.FabricAgentRuntime(config=custom, work_root=tmp_path / "fabric", skills=[skill])

    trials = await runtime.run_tasks([_TASK])

    assert any(p.endswith("/code-review") for p in client_cls.recorded[0]["agent"].skill_paths)
    assert trials[0].metadata["skill"]["mode"] == "native"


@pytest.mark.asyncio
async def test_fabric_runtime_codex_skill_staged_for_run_then_excluded_from_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    seen: dict[str, bool] = {}

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        # Codex discovers the bundle from .agents/skills/<name>/ in its workspace *during* the run.
        workspace = Path(agent.environment.workspace)
        seen["present_during_run"] = (workspace / ".agents" / "skills" / "code-review" / "SKILL.md").is_file()
        return _FakeResult(status="succeeded", output={"response": "ok"})

    client_cls = _install_fake_fabric(monkeypatch, handler)
    skill = AgentSkill.from_directory(_skill_bundle(tmp_path))
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric", skills=[skill])

    trials = await runtime.run_tasks([_TASK])

    # Staged into the workspace so the harness could discover it during the run...
    assert seen["present_during_run"] is True
    # ...then removed (with its emptied .agents parents) before the workspace is exposed as evidence, so
    # the injected files don't read as agent output to workspace-reading metrics.
    workspace = next((tmp_path / "fabric").glob("*/000000-task-1/workspace"))
    assert not (workspace / ".agents").exists()
    # No skills path added to the config; provenance still records the codex injection.
    assert client_cls.recorded[0]["agent"].skill_paths == []
    assert trials[0].metadata["skill"]["mode"] == "codex_skills_dir"


@pytest.mark.asyncio
async def test_fabric_runtime_skill_on_unsupported_adapter_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output={"response": "ok"})

    _install_fake_fabric(monkeypatch, handler)
    unsupported = {"metadata": {"name": "a"}, "harness": {"adapter_id": "some.other.adapter"}}
    runtime = fabric_runtime.FabricAgentRuntime(
        config=unsupported,
        work_root=tmp_path / "fabric",
        skills=[AgentSkill.from_directory(_skill_bundle(tmp_path, name="s"))],
    )

    with pytest.raises(RuntimeError, match="no known skill-injection strategy"):
        await runtime.run_tasks([_TASK])


@pytest.mark.asyncio
async def test_fabric_runtime_no_skill_leaves_metadata_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output={"response": "ok"})

    client_cls = _install_fake_fabric(monkeypatch, handler)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric")

    trials = await runtime.run_tasks([_TASK])

    assert trials[0].metadata["skill"] is None
    # No skill -> no planner probe (the no-skill path must not pay for a plan()).
    assert client_cls.planned == []


def test_with_skill_returns_independent_copy() -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    base = fabric_runtime.FabricAgentRuntime(config=_HERMES_CONFIG, model="m", work_root="/tmp/x")
    skill = AgentSkill(name="s", directory=Path("/skills/s"))

    treated = base.with_skill(skill)

    # A new instance is returned; the original is untouched.
    assert treated is not base
    assert base._skill_set.skills == ()
    assert treated._skill_set.skills == (skill,)


def test_with_skills_returns_independent_copy_of_the_set() -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    base = fabric_runtime.FabricAgentRuntime(config=_HERMES_CONFIG, model="m", work_root="/tmp/x")
    skills = [
        AgentSkill(name="docx", directory=Path("/skills/docx")),
        AgentSkill(name="pptx", directory=Path("/skills/pptx")),
    ]

    treated = base.with_skills(skills)

    # A new instance carries the added set; the original is untouched.
    assert treated is not base
    assert base._skill_set.skills == ()
    assert treated._skill_set.skills == tuple(skills)


def test_with_skill_is_additive_and_chainable() -> None:
    # Regression: with_skill must ADD, not replace. Chaining injects every skill, not just the last.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    base = fabric_runtime.FabricAgentRuntime(config=_HERMES_CONFIG, model="m", work_root="/tmp/x")
    a = AgentSkill(name="docx", directory=Path("/skills/docx"))
    b = AgentSkill(name="pptx", directory=Path("/skills/pptx"))

    chained = base.with_skill(a).with_skill(b)

    # Both skills are present, in order; each intermediate runtime is left untouched.
    assert chained._skill_set.skills == (a, b)
    assert base._skill_set.skills == ()
    assert base.with_skill(a)._skill_set.skills == (a,)
    # with_skills extends the same way (equivalent to chaining the single-skill calls).
    assert base.with_skills([a, b])._skill_set.skills == (a, b)
    assert base.with_skill(a).with_skills([b])._skill_set.skills == (a, b)


def test_with_skills_rejects_duplicate_names() -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    base = fabric_runtime.FabricAgentRuntime(config=_HERMES_CONFIG, work_root="/tmp/x")
    dupes = [AgentSkill(name="docx", directory=Path("/a/docx")), AgentSkill(name="docx", directory=Path("/b/docx"))]

    # Two bundles claiming the same <name>/ would collide when staged, so it is rejected up front.
    with pytest.raises(ValueError, match="duplicate skill name"):
        base.with_skills(dupes)


def test_with_skill_rejects_re_adding_same_name() -> None:
    # Adding a skill whose name is already present collides on the combined set — rejected, not silently
    # deduped, so a caller notices the double-add.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    base = fabric_runtime.FabricAgentRuntime(config=_HERMES_CONFIG, work_root="/tmp/x")
    first = base.with_skill(AgentSkill(name="docx", directory=Path("/a/docx")))

    with pytest.raises(ValueError, match="duplicate skill name"):
        first.with_skill(AgentSkill(name="docx", directory=Path("/b/docx")))


def test_constructor_rejects_duplicate_skill_names(tmp_path: Path) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    # Same name, distinct source directories — still a <name>/ bundle collision; fail before any run.
    dup_a = AgentSkill.from_directory(_skill_bundle(tmp_path / "a", name="docx"))
    dup_b = AgentSkill.from_directory(_skill_bundle(tmp_path / "b", name="docx"))

    with pytest.raises(ValueError, match="duplicate skill name"):
        fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric", skills=[dup_a, dup_b])


@pytest.mark.asyncio
async def test_fabric_runtime_multiple_native_skills_all_reach_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # LAB hands the agent all of its skills on every task. A native harness must stage each skill into its
    # own <name>/ bundle and every one of them must reach the config's skills.paths, in order.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output={"response": "ok"})

    client_cls = _install_fake_fabric(monkeypatch, handler)
    skills = [AgentSkill.from_directory(_skill_bundle(tmp_path, name=name)) for name in ("docx", "pptx", "xlsx")]
    runtime = fabric_runtime.FabricAgentRuntime(config=_HERMES_CONFIG, work_root=tmp_path / "fabric", skills=skills)

    trials = await runtime.run_tasks([_TASK])

    # Every staged <name>/ bundle is on the config handed to client.run, in order.
    paths = client_cls.recorded[0]["agent"].skill_paths
    assert [Path(p).name for p in paths] == ["docx", "pptx", "xlsx"]
    # Each bundle is staged on disk under its own <name>/ dir with its SKILL.md.
    for path in paths:
        assert (Path(path) / "SKILL.md").is_file()
    # One provenance per skill is stamped for the A/B diff; the single-skill `skill` field is None (multi).
    provs = trials[0].metadata["skills"]
    assert [prov["name"] for prov in provs] == ["docx", "pptx", "xlsx"]
    assert all(prov["mode"] == "native" for prov in provs)
    assert trials[0].metadata["skill"] is None


@pytest.mark.asyncio
async def test_fabric_runtime_multiple_codex_skills_each_staged_then_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codex self-discovers each bundle from .agents/skills/<name>/ in its workspace during the run; all are
    # then removed before the workspace is exposed as evidence so they don't read as agent output.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    names = ("docx", "pptx", "xlsx")
    seen: dict[str, bool] = {}

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        workspace = Path(agent.environment.workspace)
        seen["all_present_during_run"] = all(
            (workspace / ".agents" / "skills" / name / "SKILL.md").is_file() for name in names
        )
        return _FakeResult(status="succeeded", output={"response": "ok"})

    client_cls = _install_fake_fabric(monkeypatch, handler)
    skills = [AgentSkill.from_directory(_skill_bundle(tmp_path, name=name)) for name in names]
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric", skills=skills)

    trials = await runtime.run_tasks([_TASK])

    # Every bundle was discoverable at its own .agents/skills/<name>/ path during the run...
    assert seen["all_present_during_run"] is True
    # ...then all removed (with the emptied .agents parent) before the workspace is exposed as evidence.
    workspace = next((tmp_path / "fabric").glob("*/000000-task-1/workspace"))
    assert not (workspace / ".agents").exists()
    # Codex mode adds no skills path; one provenance per skill records the injection.
    assert client_cls.recorded[0]["agent"].skill_paths == []
    provs = trials[0].metadata["skills"]
    assert [prov["name"] for prov in provs] == list(names)
    assert all(prov["mode"] == "codex_skills_dir" for prov in provs)


@pytest.mark.asyncio
async def test_fabric_runtime_codex_skills_removed_even_when_run_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Cleanup runs in a `finally`, so a failed/errored run still removes the injected bundles from the
    # durable workspace — they must not linger and read as agent output on any path that exposes it.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    names = ("docx", "pptx")

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        # The bundles were staged before the run; blow up mid-run to hit the except/finally path.
        assert all((Path(agent.environment.workspace) / ".agents" / "skills" / n / "SKILL.md").is_file() for n in names)
        raise RuntimeError("harness blew up")

    _install_fake_fabric(monkeypatch, handler)
    skills = [AgentSkill.from_directory(_skill_bundle(tmp_path, name=name)) for name in names]
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric", skills=skills)

    trials = await runtime.run_tasks([_TASK])

    # Failed trial, but every injected bundle (and the emptied .agents parent) was still cleaned up.
    assert trials[0].status == "failed"
    workspace = next((tmp_path / "fabric").glob("*/000000-task-1/workspace"))
    assert not (workspace / ".agents").exists()
    # Provenance is still stamped on the failed trial for the A/B diff.
    assert [prov["name"] for prov in trials[0].metadata["skills"]] == list(names)


# --- ATIF token capture -----------------------------------------------------


def _atif(tmp_path: Path, payload: Mapping[str, Any]) -> Path:
    path = tmp_path / "trajectory-abc.atif.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _step(**metrics: int) -> dict[str, Any]:
    return {"source": "agent", "message": "", "metrics": metrics}


def _measurement_values(path: Path | None) -> dict[str, int | float]:
    return fabric_runtime._atif_measurements(path).model_dump(exclude_none=True)


def test_final_metrics_totals_are_projected_onto_the_token_keys(tmp_path: Path) -> None:
    path = _atif(
        tmp_path,
        {
            "schema_version": "ATIF-v1.7",
            "steps": [_step(prompt_tokens=1)],
            "final_metrics": {
                "total_prompt_tokens": 358,
                "total_completion_tokens": 19324,
                "total_cached_tokens": 3984621,
            },
        },
    )
    assert _measurement_values(path) == {
        "prompt_tokens": 358,
        "completion_tokens": 19324,
        "total_tokens": 19682,
        "cache_read_tokens": 3984621,
    }


def test_steps_are_summed_when_the_trajectory_has_no_aggregate_block(tmp_path: Path) -> None:
    path = _atif(
        tmp_path,
        {
            "schema_version": "ATIF-v1.7",
            "steps": [_step(prompt_tokens=100, completion_tokens=10), _step(prompt_tokens=58, cached_tokens=7)],
        },
    )
    assert _measurement_values(path) == {
        "prompt_tokens": 158,
        "completion_tokens": 10,
        "total_tokens": 168,
        "cache_read_tokens": 7,
    }


def test_a_partial_aggregate_does_not_suppress_the_fields_only_the_steps_report(tmp_path: Path) -> None:
    # Every final_metrics field is optional, so a block reporting only steps/cost validates cleanly.
    # Resolving per field is what keeps a mid-run flush (the timeout path) from publishing nothing.
    path = _atif(
        tmp_path,
        {
            "schema_version": "ATIF-v1.7",
            "steps": [_step(prompt_tokens=100, completion_tokens=10), _step(prompt_tokens=58)],
            "final_metrics": {"total_steps": 2, "total_cost_usd": 0.12},
        },
    )
    assert _measurement_values(path) == {
        "prompt_tokens": 158,
        "completion_tokens": 10,
        "total_tokens": 168,
        "cost_usd": 0.12,
    }


def test_a_reported_total_wins_over_the_step_sum_for_that_field_alone(tmp_path: Path) -> None:
    # The aggregate is authoritative where it speaks; the steps fill only the fields it omits.
    path = _atif(
        tmp_path,
        {
            "schema_version": "ATIF-v1.7",
            "steps": [_step(prompt_tokens=1, completion_tokens=10)],
            "final_metrics": {"total_prompt_tokens": 358},
        },
    )
    assert _measurement_values(path) == {"prompt_tokens": 358, "completion_tokens": 10, "total_tokens": 368}


def test_an_invalid_aggregate_field_is_omitted_instead_of_falling_back(tmp_path: Path) -> None:
    path = _atif(
        tmp_path,
        {
            "schema_version": "ATIF-v1.7",
            "steps": [_step(prompt_tokens=158)],
            "final_metrics": {"total_prompt_tokens": "not-an-int"},
        },
    )
    assert _measurement_values(path) == {}


def test_a_trajectory_with_no_counts_anywhere_records_nothing(tmp_path: Path) -> None:
    path = _atif(tmp_path, {"schema_version": "ATIF-v1.7", "steps": [_step()], "final_metrics": {}})
    assert _measurement_values(path) == {}


@pytest.mark.parametrize("payload", ["[]", "{ not json", '"a string"', '{"steps": 7}'])
def test_an_unreadable_trajectory_records_nothing_rather_than_failing(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "trajectory-abc.atif.json"
    path.write_text(payload, encoding="utf-8")
    assert _measurement_values(path) == {}


def test_a_missing_trajectory_records_nothing(tmp_path: Path) -> None:
    assert _measurement_values(None) == {}
    assert _measurement_values(tmp_path / "absent.atif.json") == {}


@pytest.mark.parametrize(
    ("value", "is_float", "expected"),
    [
        (0, False, True),
        (1.5, True, True),
        (1.5, False, False),
        (True, False, False),
        (-1, False, False),
        (float("nan"), True, False),
        (float("inf"), True, False),
        ("1", False, False),
    ],
)
def test_valid_atif_measurement(value: object, is_float: bool, expected: bool) -> None:
    assert fabric_runtime._valid_atif_measurement(value, is_float=is_float) is expected


def test_sum_step_metric_treats_missing_steps_as_unmeasured_and_non_list_as_invalid() -> None:
    assert fabric_runtime._sum_step_metric({}, "prompt_tokens", is_float=False) == (None, True)
    assert fabric_runtime._sum_step_metric({"steps": None}, "prompt_tokens", is_float=False) == (None, True)
    assert fabric_runtime._sum_step_metric({"steps": 7}, "prompt_tokens", is_float=False) == (None, False)


@pytest.mark.asyncio
async def test_a_failing_trace_fold_costs_the_trace_not_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fold that raises must not take the other tasks' trials with it.

    The fold runs in ``finally``, past the per-task exception handlers, so an unguarded raise
    propagates into the ``asyncio.gather`` over every task — turning one unwritable evidence
    directory into a run with no results at all.
    """

    def handler(agent: Any, kwargs: dict[str, Any]) -> _FakeResult:
        return _FakeResult(status="succeeded", output={"response": "ok"})

    _install_fake_fabric(monkeypatch, handler)

    def explode(directory: Path) -> int:
        raise OSError("no space left on device")

    monkeypatch.setattr(fabric_runtime, "fold_exports", explode)
    runtime = fabric_runtime.FabricAgentRuntime(config=_CONFIG, work_root=tmp_path / "fabric", capture_trajectory=True)

    first, second = await runtime.run_tasks([_TASK, _TASK])

    for trial in (first, second):
        assert trial.status == "completed"
        assert trial.evidence is not None
        assert EVIDENCE_TRACE not in trial.evidence.descriptors

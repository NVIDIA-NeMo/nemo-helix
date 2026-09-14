# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FabricContainerRuntime over a fake sandbox provider.

No real Docker or image: a fake provider records the marshaled inputs and simulates the
in-container ``/out`` layout on ``download_dir``, so we can assert the evidence contract
(keys/kinds) and the success/failure/isolation paths the metrics depend on.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes.fabric import container_runtime as crt
from nemo_evaluator_sdk.agent_eval.runtimes.fabric.container_runtime import FabricContainerRuntime
from nemo_evaluator_sdk.agent_eval.runtimes.sandbox.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus
from nemo_evaluator_sdk.values.common import SecretRef

_CONFIG = {"metadata": {"name": "eval"}, "harness": {"adapter_id": "nvidia.fabric.hermes"}}


@pytest.fixture(autouse=True)
def _stub_image_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never build a real image in unit tests; the runtime asks for one per run."""
    monkeypatch.setattr(crt, "ensure_fabric_image", lambda **_kwargs: "fabric-img:test")


class _FakeResolver:
    """Resolves any SecretRef to a fixed value."""

    def __init__(self, value: str = "resolved-secret") -> None:
        self._value = value

    async def resolve_secret(self, secret_ref: SecretRef) -> str:
        return self._value


class _FakeProvider:
    """Simulates a sandbox: records seeds/uploads/execs, materializes /out on download."""

    name = "fake"

    def __init__(
        self,
        *,
        status: str = "succeeded",
        error_type: str | None = None,
        return_code: int = 0,
        write_result: bool = True,
        result_bytes: bytes | None = None,
        atif: bool = True,
    ) -> None:
        self._status = status
        self._error_type = error_type  # exec sandbox-runtime failure (e.g. "timeout")
        self._return_code = return_code
        self._write_result = write_result  # False => the CLI crashed before writing a RunResult
        self._result_bytes = result_bytes  # raw override for fabric_result.json (non-object / binary)
        self._atif = atif
        self.seeded: dict[str, str] = {}
        self.env: dict[str, str] = {}
        self.image: str | None = None
        self.uploaded_dirs: list[tuple[Path, str]] = []
        self.execs: list[str] = []
        self.closed = 0
        self.aclosed = 0

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.seeded = dict(spec.files)
        self.env = dict(spec.env)
        self.image = spec.image
        return SandboxHandle(sandbox_id="fake-1", provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: object) -> SandboxExecResult:
        self.execs.append(command)
        stderr = "stderr-boom" if (self._return_code or self._error_type) else ""
        return SandboxExecResult(stdout="", stderr=stderr, return_code=self._return_code, error_type=self._error_type)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        return None

    async def upload_dir(self, handle: SandboxHandle, source_dir: Path, target_dir: str) -> None:
        self.uploaded_dirs.append((source_dir, target_dir))

    async def download_dir(self, handle: SandboxHandle, source_dir: str, target_dir: Path) -> None:
        # Materialize the /out layout the in-sandbox driver would have produced.
        out = target_dir
        (out / "workspace").mkdir(parents=True, exist_ok=True)
        (out / "logs").mkdir(parents=True, exist_ok=True)
        (out / "logs" / "fabric-stderr.txt").write_text("stderr-boom", encoding="utf-8")
        if self._result_bytes is not None:  # raw override: non-object JSON, or non-UTF-8 garbage
            (out / "fabric_result.json").write_bytes(self._result_bytes)
            return
        if not self._write_result:  # crashed CLI wrote no RunResult
            return
        (out / "workspace" / "fib.py").write_text("def fib(n): return n", encoding="utf-8")
        envelope = {
            "status": self._status,
            "output": {"response": "fixed the bug"},
            "error": None if self._status == "succeeded" else {"stage": "run", "code": "E", "message": "nope"},
        }
        (out / "fabric_result.json").write_text(json.dumps(envelope), encoding="utf-8")
        if self._atif:
            # Relay nests the trajectory under a per-run subdir, as the live gateway does.
            run_dir = out / "relay" / "runtime-123-4"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "trajectory-abc.atif.json").write_text('{"steps": []}', encoding="utf-8")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        return None

    async def status(self, handle: SandboxHandle) -> object:
        return None

    async def close(self, handle: SandboxHandle) -> None:
        self.closed += 1

    async def aclose(self) -> None:
        self.aclosed += 1


def _runtime(provider: _FakeProvider, **kwargs: object) -> FabricContainerRuntime:
    return FabricContainerRuntime(_CONFIG, provider=provider, **kwargs)  # type: ignore[arg-type]


async def _run(runtime: FabricContainerRuntime, tasks: list[AgentEvalTask], tmp_path: Path) -> Sequence[AgentEvalTrial]:
    return await runtime.run_tasks(tasks, AgentEvalRunConfig(work_dir=tmp_path))


def _task() -> AgentEvalTask:
    return AgentEvalTask(
        id="fix-bug",
        intent="Fix the bug in fib.py",  # eval-side metadata; must NOT reach the agent
        inputs={
            "instruction": "Fix fib.py so fib(n) returns the nth Fibonacci number.",
            "files": {"fib.py": "def fib(n): return n  # buggy"},
        },
    )


async def test_success_maps_evidence_contract(tmp_path: Path) -> None:
    provider = _FakeProvider()
    trials = await _run(_runtime(provider), [_task()], tmp_path)
    (trial,) = trials

    assert trial.status == AgentEvalTrialStatus.COMPLETED
    assert trial.measurements.model_dump(exclude_none=True) == {}
    assert trial.output is not None and trial.output.output_text == "fixed the bug"
    # Same evidence keys/kinds FabricAgentRuntime + Codex produce, so metrics work unchanged.
    ws = trial.evidence.require("workspace")
    assert ws.kind == "filesystem" and Path(ws.ref).is_dir()  # type: ignore[arg-type]
    trace = trial.evidence.require("trace")
    assert trace.kind == "trace" and trace.format == "atif"
    assert trial.evidence.require("result").kind == "json"
    assert provider.closed == 1
    assert provider.aclosed == 1  # the batch disposes the shared provider once, after all tasks


async def test_success_response_is_output_payload_not_full_envelope(tmp_path: Path) -> None:
    # The host FabricAgentRuntime sets AgentOutput.response to RunResult.output; the container path must
    # match so metrics reading `sample.response` see the same shape — the output payload, not the whole
    # normalized RunResult envelope (status/output/error).
    provider = _FakeProvider()
    (trial,) = await _run(_runtime(provider), [_task()], tmp_path)
    assert trial.output is not None
    assert trial.output.response == {"response": "fixed the bug"}  # output payload only, not the envelope


async def test_success_trial_stamps_agent_ok_true(tmp_path: Path) -> None:
    # AgentPhaseSuccessMetric reads metadata["agent_ok"] and treats a missing value as False, so a
    # successful container trial must set agent_ok=True (matching host Fabric + Codex) or every clean
    # container run is scored a failed phase.
    provider = _FakeProvider()
    (trial,) = await _run(_runtime(provider), [_task()], tmp_path)
    assert trial.metadata["agent_ok"] is True


async def test_failed_trial_stamps_agent_ok_false(tmp_path: Path) -> None:
    provider = _FakeProvider(status="failed")
    (trial,) = await _run(_runtime(provider), [_task()], tmp_path)
    assert trial.status == AgentEvalTrialStatus.FAILED
    assert trial.measurements.model_dump(exclude_none=True) == {}
    assert trial.metadata["agent_ok"] is False


async def test_seeds_composed_agent_config_and_execs_driver(tmp_path: Path) -> None:
    provider = _FakeProvider()
    await _run(_runtime(provider), [_task()], tmp_path)
    assert "/in/agent.json" in provider.seeded and "/in/input.txt" in provider.seeded
    # Everything rides in the single agent config: the caller's harness plus the runtime's workspace,
    # artifact roots, and trajectory telemetry.
    assert not [key for key in provider.seeded if key.startswith("/in/profile-")]
    agent = json.loads(provider.seeded["/in/agent.json"])
    assert agent["harness"]["adapter_id"] == _CONFIG["harness"]["adapter_id"]  # caller keys survive
    assert agent["environment"] == {"provider": "local", "workspace": "/out/workspace", "artifacts": "/out/artifacts"}
    assert agent["runtime"]["artifacts"] == "/out/artifacts"
    assert "relay" in agent["telemetry"]["providers"]
    # Workspace seed files were staged and uploaded across the boundary.
    assert provider.uploaded_dirs and provider.uploaded_dirs[0][1] == "/out/workspace"
    # The driver is seeded as source and the exec runs that file: naming a path nothing seeds would
    # leave the sandbox with no entrypoint at all.
    assert provider.seeded[crt._DRIVER_PATH] == crt._DRIVER_SOURCE.read_text(encoding="utf-8")
    (cmd,) = provider.execs
    assert f"python3 {crt._DRIVER_PATH} /in/agent.json /in/input.txt" in cmd
    assert "> /out/fabric_result.json" in cmd


async def test_agent_input_uses_instruction_not_intent(tmp_path: Path) -> None:
    provider = _FakeProvider()
    await _run(_runtime(provider), [_task()], tmp_path)
    agent_input = provider.seeded["/in/input.txt"]
    assert "Fix fib.py so fib(n) returns the nth Fibonacci number." in agent_input
    assert "Fix the bug in fib.py" not in agent_input  # the intent must not leak to the agent
    assert "fib.py" in agent_input  # seed file listed by name


async def test_secrets_are_resolved_and_injected_as_env(tmp_path: Path) -> None:
    provider = _FakeProvider()
    runtime = _runtime(provider, secrets={"NVIDIA_API_KEY": SecretRef(root="nvidia-build-api-key")})
    # The orchestrator (AgentEvaluator / backend) owns the resolver and resolves before running.
    await runtime.resolve_secrets(_FakeResolver("nvapi-xyz"))
    await _run(runtime, [_task()], tmp_path)
    assert provider.env == {"NVIDIA_API_KEY": "nvapi-xyz"}  # resolved value, keyed by the harness env var


async def test_supplied_image_is_used_verbatim_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A caller-supplied image (an escape hatch for sandboxes needing extra tooling, e.g. a
    # document-processing image) is used verbatim and short-circuits the build-if-missing path.
    builds: list[bool] = []
    monkeypatch.setattr(crt, "ensure_fabric_image", lambda **_kwargs: builds.append(True) or "unused")
    provider = _FakeProvider()
    (trial,) = await _run(_runtime(provider, image="doc-tools:1.0"), [_task()], tmp_path)

    assert trial.status == AgentEvalTrialStatus.COMPLETED
    assert builds == []  # the supplied tag is used; ensure_fabric_image is never called
    assert provider.image == "doc-tools:1.0"  # and it reaches the SandboxSpec
    assert trial.metadata["image"] == "doc-tools:1.0"  # surfaced on the trial metadata


async def test_default_provisions_build_if_missing_image(tmp_path: Path) -> None:
    # Without an override, the runtime provisions the harness-agnostic Fabric image (build-if-missing);
    # the autouse fixture stubs that build to "fabric-img:test".
    provider = _FakeProvider()
    (trial,) = await _run(_runtime(provider), [_task()], tmp_path)
    assert provider.image == "fabric-img:test"
    assert trial.metadata["image"] == "fabric-img:test"


async def test_early_failure_trial_carries_image_metadata(tmp_path: Path) -> None:
    # A failure BEFORE _to_trial (here: download_dir raises) must still stamp the runtime + selected image
    # on the trial, matching what _to_trial records on the success and late-failure paths.
    class _BrokenProvider(_FakeProvider):
        async def download_dir(self, handle: SandboxHandle, source_dir: str, target_dir: Path) -> None:
            raise RuntimeError("sandbox died mid-download")

    provider = _BrokenProvider()
    (trial,) = await _run(_runtime(provider, image="doc-tools:1.0"), [_task()], tmp_path)
    assert trial.status == AgentEvalTrialStatus.FAILED
    assert trial.metadata["image"] == "doc-tools:1.0"
    assert trial.metadata["runtime"] == "fabric_container"


async def test_no_run_result_fails_trial(tmp_path: Path) -> None:
    # The CLI crashed (non-zero exit) and produced no RunResult; the trial must fail (not silently complete).
    provider = _FakeProvider(return_code=1, write_result=False)
    (trial,) = await _run(_runtime(provider), [_task()], tmp_path)
    assert trial.status == AgentEvalTrialStatus.FAILED
    assert trial.evidence.require("error").kind == "error"
    assert "stderr-boom" in str(trial.metadata.get("error"))
    assert provider.closed == 1


async def test_fabric_error_status_fails_trial(tmp_path: Path) -> None:
    provider = _FakeProvider(status="failed")
    (trial,) = await _run(_runtime(provider), [_task()], tmp_path)
    assert trial.status == AgentEvalTrialStatus.FAILED
    assert trial.metadata["error"] == "nope"


async def test_exec_timeout_with_stale_success_file_fails(tmp_path: Path) -> None:
    # Finding #1: a timed-out / SIGKILLed exec that still left a status=succeeded file must NOT be COMPLETED.
    provider = _FakeProvider(error_type="timeout", return_code=125, status="succeeded")
    (trial,) = await _run(_runtime(provider), [_task()], tmp_path)
    assert trial.status == AgentEvalTrialStatus.FAILED
    assert provider.closed == 1


async def test_non_object_result_fails(tmp_path: Path) -> None:
    # Finding #2: fabric_result.json is valid JSON but not an object — must fail, not default to success.
    provider = _FakeProvider(result_bytes=b'"just a diagnostic string"')
    (trial,) = await _run(_runtime(provider), [_task()], tmp_path)
    assert trial.status == AgentEvalTrialStatus.FAILED


async def test_unreadable_result_is_isolated_not_batch_aborting(tmp_path: Path) -> None:
    # Finding #3: a non-UTF-8 / unparseable result must fail per-task, not raise and abort the whole batch.
    provider = _FakeProvider(result_bytes=b"\xff\xfe not utf-8 \x00")
    tasks = [_task(), AgentEvalTask(id="other", intent="x", inputs={"instruction": "do a thing"})]
    trials = await _run(_runtime(provider), tasks, tmp_path)  # must not raise
    assert len(trials) == 2
    assert all(trial.status == AgentEvalTrialStatus.FAILED for trial in trials)
    assert provider.closed == 2


def test_empty_instruction_is_rejected() -> None:
    # The container prompt is task.agent_prompt(): an empty/absent instruction cannot be evaluated, so
    # it raises here and the runtime turns that into a failed trial for just that task (see _run_task).
    with pytest.raises(ValueError, match="no instruction"):
        AgentEvalTask(id="x", intent="ignored", inputs={"instruction": ""}).agent_prompt()


# --------------------------------------------------------------------------------------------------
# Agent-skill injection (containerized) — mirrors the host-runtime skill tests in test_fabric_runtime.py.
# --------------------------------------------------------------------------------------------------

# Adapters the fake planner reports as accepting the native Fabric ``skills`` config. ``acme.custom.native``
# stands in for an END-USER adapter the platform doesn't ship — the runtime learns it accepts skills purely
# from the plan, with no hardcoded list.
_NATIVE_SKILL_ADAPTERS = {"nvidia.fabric.hermes", "acme.custom.native"}
_KNOWN_HARNESSES = ("hermes", "codex", "claude")
_CODEX_CONFIG = {"metadata": {"name": "eval"}, "harness": {"adapter_id": "nvidia.fabric.codex"}}


def _harness_name(adapter_id: str) -> str:
    return next((harness for harness in _KNOWN_HARNESSES if harness in adapter_id), "custom")


class _FakeAdapterInfo:
    def __init__(self, harness: str) -> None:
        self.harness = harness


class _FakePlan:
    def __init__(self, *, capability_plan: dict[str, object], harness: str) -> None:
        self.capability_plan = capability_plan
        self.adapter = _FakeAdapterInfo(harness)


class _FakeFabric:
    planned: list[dict[str, Any]] = []

    def plan(self, agent: Any, *, base_dir: object = None) -> _FakePlan:
        # Mirror Fabric's planner: a ``skills`` route appears only when a skill path is attached, and it
        # routes ``harness_native`` iff the selected adapter accepts native skills.
        _FakeFabric.planned.append({"agent": agent})
        adapter_id = agent.harness.adapter_id
        has_skill_path = bool(agent.skills is not None and agent.skills.paths)
        native = has_skill_path and adapter_id in _NATIVE_SKILL_ADAPTERS
        routes = [{"kind": "skills", "target": "harness_native" if native else "unsupported"}] if has_skill_path else []
        return _FakePlan(capability_plan={"routes": routes}, harness=_harness_name(adapter_id))


def _install_fake_fabric(monkeypatch: pytest.MonkeyPatch) -> type[_FakeFabric]:
    """Inject a ``nemo_fabric`` module whose planner is fake, so skills routing is ours to decide."""
    _FakeFabric.planned = []
    import nemo_fabric  # ty: ignore[unresolved-import]

    # Only the planner is faked; every other symbol stays the installed one. The runtime also builds
    # its config and telemetry out of this module, and a stand-in for those would happily accept a
    # config Fabric itself rejects — which is the drift these tests exist to notice.
    module = types.ModuleType("nemo_fabric")
    module.__dict__.update(nemo_fabric.__dict__)
    module.Fabric = _FakeFabric  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo_fabric", module)
    return _FakeFabric


def _skill_bundle(base: Path, *, name: str = "code-review", extra: dict[str, str] | None = None) -> Path:
    """Write a minimal agentskills bundle under ``base/<name>/`` and return its path."""
    root = base / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n\nBe thorough.\n", encoding="utf-8")
    for rel, content in (extra or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def _seeded_skill_paths(provider: _FakeProvider) -> list[str]:
    """``skills.paths`` on the composed agent config the runtime seeded into /in."""
    agent = json.loads(provider.seeded["/in/agent.json"])
    return list(agent.get("skills", {}).get("paths", []))


async def test_native_skill_seeds_bundle_into_seed_set_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    fabric = _install_fake_fabric(monkeypatch)
    skill = AgentSkill.from_directory(_skill_bundle(tmp_path / "src", extra={"references/r.md": "material"}))
    provider = _FakeProvider()  # module _CONFIG is the hermes.sdk adapter -> native routing
    (trial,) = await _run(_runtime(provider, skills=[skill]), [_task()], tmp_path)

    assert trial.status == AgentEvalTrialStatus.COMPLETED
    # The mode is resolved by probing Fabric's capability planner (with a probe skill path attached).
    assert fabric.planned and fabric.planned[0]["agent"].skills.paths
    # The bundle is rendered INTO the sandbox seed set at the native in-/in discovery path (not /out, so it
    # never lands in the downloaded workspace evidence).
    assert provider.seeded["/in/skills/code-review/SKILL.md"].startswith("---")
    assert provider.seeded["/in/skills/code-review/references/r.md"] == "material"
    # The composed agent config's skills.paths points at the staged bundle dir.
    assert _seeded_skill_paths(provider)[-1] == "/in/skills/code-review"
    # Provenance is stamped into trial metadata for the A/B diff.
    prov = trial.metadata["skill"]
    assert prov["name"] == "code-review" and prov["mode"] == "native" and prov["hash"]
    assert prov["location"] == "/in/skills/code-review"


async def test_native_skill_preserves_preconfigured_skill_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Injected paths are APPENDED to the config's own skills.paths (order-preserved), so preconfigured
    # skills survive injection and the treated arm differs by exactly the injected skill.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    _install_fake_fabric(monkeypatch)
    config = {**_CONFIG, "skills": {"paths": ["/pre/existing-a", "/pre/existing-b"]}}
    skill = AgentSkill.from_directory(_skill_bundle(tmp_path / "src"))
    provider = _FakeProvider()
    runtime = FabricContainerRuntime(config, provider=provider, skills=[skill])  # type: ignore[arg-type]
    await runtime.run_tasks([_task()], AgentEvalRunConfig(work_dir=tmp_path))

    paths = _seeded_skill_paths(provider)
    assert paths[:2] == ["/pre/existing-a", "/pre/existing-b"]
    assert paths[-1] == "/in/skills/code-review"


async def test_native_skill_on_runtime_discovered_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An end-user adapter the platform doesn't ship (harness "custom", not codex) still gets native injection
    # purely because Fabric's planner routes its skills ``harness_native`` — nothing is hardcoded.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    _install_fake_fabric(monkeypatch)
    custom = {"metadata": {"name": "eval"}, "harness": {"adapter_id": "acme.custom.native"}}
    skill = AgentSkill.from_directory(_skill_bundle(tmp_path / "src"))
    provider = _FakeProvider()
    runtime = FabricContainerRuntime(custom, provider=provider, skills=[skill])  # type: ignore[arg-type]
    (trial,) = await runtime.run_tasks([_task()], AgentEvalRunConfig(work_dir=tmp_path))

    assert "/in/skills/code-review" in _seeded_skill_paths(provider)
    assert trial.metadata["skill"]["mode"] == "native"


async def test_codex_skill_seeds_workspace_and_is_excluded_from_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    class _CodexWorkspaceProvider(_FakeProvider):
        # Simulate the codex-seeded bundle landing in the workspace that gets downloaded as /out evidence.
        async def download_dir(self, handle: SandboxHandle, source_dir: str, target_dir: Path) -> None:
            await super().download_dir(handle, source_dir, target_dir)
            skill_md = target_dir / "workspace" / ".agents" / "skills" / "code-review" / "SKILL.md"
            skill_md.parent.mkdir(parents=True, exist_ok=True)
            skill_md.write_text("---\nname: code-review\n---\n", encoding="utf-8")

    _install_fake_fabric(monkeypatch)
    skill = AgentSkill.from_directory(_skill_bundle(tmp_path / "src"))
    provider = _CodexWorkspaceProvider()
    runtime = FabricContainerRuntime(_CODEX_CONFIG, provider=provider, skills=[skill])  # type: ignore[arg-type]
    (trial,) = await runtime.run_tasks([_task()], AgentEvalRunConfig(work_dir=tmp_path))

    # Codex discovers agentskills from .agents/skills/ in its working dir, so the bundle is seeded there in
    # the workspace (not /in), for the harness to self-discover during the run.
    assert provider.seeded["/out/workspace/.agents/skills/code-review/SKILL.md"].startswith("---")
    # No skills path added: placement in the workspace is the delivery mechanism.
    assert _seeded_skill_paths(provider) == []
    prov = trial.metadata["skill"]
    assert prov["mode"] == "codex_skills_dir"
    assert prov["location"] == ".agents/skills/code-review"
    # ...then removed from the downloaded evidence (with its emptied .agents parents) before the workspace is
    # exposed, so the injected files don't read as agent output to workspace-reading metrics.
    workspace = Path(trial.evidence.require("workspace").ref)  # type: ignore[arg-type]
    assert not (workspace / ".agents").exists()


async def test_skill_on_unsupported_adapter_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    _install_fake_fabric(monkeypatch)
    unsupported = {"metadata": {"name": "eval"}, "harness": {"adapter_id": "some.other.adapter"}}
    skill = AgentSkill.from_directory(_skill_bundle(tmp_path / "src", name="s"))
    runtime = FabricContainerRuntime(unsupported, provider=_FakeProvider(), skills=[skill])  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="no known skill-injection strategy"):
        await runtime.run_tasks([_task()], AgentEvalRunConfig(work_dir=tmp_path))


async def test_no_skill_leaves_metadata_none_and_skips_planner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fabric = _install_fake_fabric(monkeypatch)
    provider = _FakeProvider()
    (trial,) = await _run(_runtime(provider), [_task()], tmp_path)

    assert trial.metadata["skill"] is None and trial.metadata["skills"] == []
    # No skill -> no planner probe (the no-skill path must not import nemo_fabric or pay for a plan()).
    assert fabric.planned == []
    # And nothing is seeded under a skills discovery path.
    assert not any("/skills/" in key or "/.agents/" in key for key in provider.seeded)


async def test_multiple_native_skills_each_staged_and_all_in_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A set of skills: each bundle stages under its own /in/skills/<name>/, and all ride in ONE merged
    # config's skills.paths (all of them must be listed, or the treated arm would drop all
    # but the last). Trial metadata carries one provenance per skill; the lone `skill` field is None.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    _install_fake_fabric(monkeypatch)
    skills = [
        AgentSkill.from_directory(_skill_bundle(tmp_path / "a", name="docx")),
        AgentSkill.from_directory(_skill_bundle(tmp_path / "b", name="pptx")),
    ]
    provider = _FakeProvider()  # hermes.sdk -> native
    (trial,) = await _run(_runtime(provider, skills=skills), [_task()], tmp_path)

    assert provider.seeded["/in/skills/docx/SKILL.md"].startswith("---")
    assert provider.seeded["/in/skills/pptx/SKILL.md"].startswith("---")
    # Both bundle roots land on the composed config's skills.paths, in order.
    assert _seeded_skill_paths(provider) == ["/in/skills/docx", "/in/skills/pptx"]
    # One provenance per skill; the historical lone `skill` field is None for a multi-skill run.
    names = [prov["name"] for prov in trial.metadata["skills"]]
    assert names == ["docx", "pptx"]
    assert trial.metadata["skill"] is None


async def test_multiple_codex_skills_all_removed_from_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    class _CodexWorkspaceProvider(_FakeProvider):
        # Simulate every codex-seeded bundle landing in the downloaded /out workspace.
        async def download_dir(self, handle: SandboxHandle, source_dir: str, target_dir: Path) -> None:
            await super().download_dir(handle, source_dir, target_dir)
            for name in ("docx", "pptx"):
                md = target_dir / "workspace" / ".agents" / "skills" / name / "SKILL.md"
                md.parent.mkdir(parents=True, exist_ok=True)
                md.write_text("---\n---\n", encoding="utf-8")

    _install_fake_fabric(monkeypatch)
    skills = [
        AgentSkill.from_directory(_skill_bundle(tmp_path / "a", name="docx")),
        AgentSkill.from_directory(_skill_bundle(tmp_path / "b", name="pptx")),
    ]
    provider = _CodexWorkspaceProvider()
    runtime = FabricContainerRuntime(_CODEX_CONFIG, provider=provider, skills=skills)  # type: ignore[arg-type]
    (trial,) = await runtime.run_tasks([_task()], AgentEvalRunConfig(work_dir=tmp_path))

    # Both bundles seeded under the codex discovery dir, no skills path, all scrubbed from evidence.
    assert provider.seeded["/out/workspace/.agents/skills/docx/SKILL.md"].startswith("---")
    assert provider.seeded["/out/workspace/.agents/skills/pptx/SKILL.md"].startswith("---")
    assert _seeded_skill_paths(provider) == []
    assert [prov["name"] for prov in trial.metadata["skills"]] == ["docx", "pptx"]
    workspace = Path(trial.evidence.require("workspace").ref)  # type: ignore[arg-type]
    assert not (workspace / ".agents").exists()


def test_duplicate_skill_names_rejected_at_construction_and_with_skills() -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill, SkillInjectionError

    a = AgentSkill(name="dup", directory=Path("/skills/a"))
    b = AgentSkill(name="dup", directory=Path("/skills/b"))
    # Two bundles claiming the same <name>/ would collide — rejected up front, before any task runs.
    with pytest.raises(SkillInjectionError, match="duplicate skill name"):
        _runtime(_FakeProvider(), skills=[a, b])
    with pytest.raises(SkillInjectionError, match="duplicate skill name"):
        _runtime(_FakeProvider(), skills=[a]).with_skills([b])


def test_with_skills_is_additive_and_independent() -> None:
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    base = _runtime(_FakeProvider())
    a = AgentSkill(name="docx", directory=Path("/skills/docx"))
    b = AgentSkill(name="pptx", directory=Path("/skills/pptx"))

    # with_skill is a thin, additive, chainable wrapper over with_skills; the original is untouched.
    chained = base.with_skill(a).with_skill(b)
    assert chained is not base
    assert base._skill_set.skills == ()
    assert chained._skill_set.skills == (a, b)
    assert base.with_skills([a, b])._skill_set.skills == (a, b)


async def test_same_skill_from_both_injection_and_task_files_fails_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Seeding a skill via task files is legitimate; doing it for a skill the runtime ALSO injects is
    # not. The task upload lands after the pre-start seed, so it would overwrite the injected bundle
    # and leave the stamped provenance hash describing content the agent never saw.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    _install_fake_fabric(monkeypatch)
    skill = AgentSkill.from_directory(_skill_bundle(tmp_path / "src"))
    provider = _FakeProvider()
    runtime = FabricContainerRuntime(_CODEX_CONFIG, provider=provider, skills=[skill])  # type: ignore[arg-type]
    task = AgentEvalTask(
        id="collision",
        intent="...",
        inputs={
            "instruction": "Do something.",
            "files": {".agents/skills/code-review/SKILL.md": "# override"},
        },
    )
    (trial,) = await runtime.run_tasks([task], AgentEvalRunConfig(work_dir=tmp_path))

    assert trial.status == AgentEvalTrialStatus.FAILED
    error = json.loads(Path(trial.evidence.require("error").ref).read_text())  # type: ignore[arg-type]
    assert error["error_type"] == "SkillInjectionError"
    assert "also injected as the runtime skill 'code-review'" in error["error"]


async def test_task_seeded_skill_coexists_with_a_different_injected_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both mechanisms may populate .agents/skills/ in the same run: the A/B-injected skill and a
    # skill the task definition always ships as a file input. They only conflict when they target
    # the same <name>/ bundle, so different names go through their separate paths untouched.
    from nemo_evaluator_sdk.agent_eval.runtimes.fabric.skills import AgentSkill

    _install_fake_fabric(monkeypatch)
    skill = AgentSkill.from_directory(_skill_bundle(tmp_path / "src", name="code-review"))
    provider = _FakeProvider()
    runtime = FabricContainerRuntime(_CODEX_CONFIG, provider=provider, skills=[skill])  # type: ignore[arg-type]
    task = AgentEvalTask(
        id="coexist",
        intent="...",
        inputs={
            "instruction": "Do something.",
            # A skill the task always ships as a file input (different name, not A/B-injected).
            "files": {".agents/skills/style-guide/SKILL.md": "---\nname: style-guide\n---\n"},
        },
    )
    (trial,) = await runtime.run_tasks([task], AgentEvalRunConfig(work_dir=tmp_path))

    assert trial.status == AgentEvalTrialStatus.COMPLETED
    assert "/out/workspace/.agents/skills/code-review/SKILL.md" in provider.seeded
    assert any(target == "/out/workspace" for _, target in provider.uploaded_dirs)


async def test_sandbox_exception_is_isolated_per_task(tmp_path: Path) -> None:
    # A sandbox that blows up mid-run must yield a FAILED trial per task, not abort the gather.
    class _BrokenProvider(_FakeProvider):
        async def download_dir(self, handle: SandboxHandle, source_dir: str, target_dir: Path) -> None:
            raise RuntimeError("sandbox died")

    provider = _BrokenProvider()
    other = AgentEvalTask(id="other", intent="x", inputs={"instruction": "do a thing"})
    trials = await _run(_runtime(provider), [_task(), other], tmp_path)
    assert len(trials) == 2
    assert all(trial.status == AgentEvalTrialStatus.FAILED for trial in trials)
    assert {trial.task_id for trial in trials} == {"fix-bug", "other"}
    assert provider.closed == 2  # both sandboxes were torn down despite the mid-run failures

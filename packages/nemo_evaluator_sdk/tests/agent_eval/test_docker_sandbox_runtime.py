# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import io
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from nemo_evaluator_sdk.agent_eval.runtimes import docker_sandbox
from nemo_evaluator_sdk.agent_eval.runtimes.docker_sandbox import (
    DockerSandboxAgentRuntime,
    SandboxSDK,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask


@dataclass
class _FakeFile:
    content: bytes


@dataclass
class _FakeDir:
    pass


@dataclass
class _FakeLocalDir:
    src: Path


@dataclass
class _FakeManifest:
    root: str
    entries: dict[str, Any]


@dataclass
class _FakeDockerOptions:
    image: str


@dataclass
class _FakeRunConfig:
    sandbox: Any


@dataclass
class _FakeSandboxRunConfig:
    session: Any


@dataclass
class _FakeResult:
    final_output: str = "Runtime answer"
    new_items: list[dict[str, Any]] | None = None
    raw_responses: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.new_items is None:
            self.new_items = [{"type": "message", "text": self.final_output}]
        if self.raw_responses is None:
            self.raw_responses = [{"id": "response-1"}]


class _FakeSandboxAgent:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeDockerClient:
    def __init__(self) -> None:
        self.created: list[_FakeSandbox] = []
        self.deleted: list[_FakeSandbox] = []

    async def create(self, *, manifest: _FakeManifest, options: _FakeDockerOptions) -> "_FakeSandbox":
        sandbox = _FakeSandbox(manifest=manifest, options=options)
        self.created.append(sandbox)
        return sandbox

    async def delete(self, session: "_FakeSandbox") -> "_FakeSandbox":
        self.deleted.append(session)
        return session


class _FakeSandbox:
    def __init__(self, *, manifest: _FakeManifest, options: _FakeDockerOptions) -> None:
        self.manifest = manifest
        self.options = options
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> "_FakeSandbox":
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        self.exited = True

    async def persist_workspace(self) -> io.BytesIO:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            payload = b"done\n"
            info = tarfile.TarInfo("output/result.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        archive.seek(0)
        return archive


class _FakeRunner:
    def __init__(self, *, delay_s: float = 0) -> None:
        self.prompts: list[str] = []
        self.active = 0
        self.max_active = 0
        self.delay_s = delay_s

    async def run(self, agent: _FakeSandboxAgent, prompt: str, *, run_config: _FakeRunConfig) -> _FakeResult:
        del agent, run_config
        self.prompts.append(prompt)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay_s:
                import asyncio

                await asyncio.sleep(self.delay_s)
            return _FakeResult()
        finally:
            self.active -= 1


class _FailingRunner:
    async def run(self, agent: _FakeSandboxAgent, prompt: str, *, run_config: _FakeRunConfig) -> _FakeResult:
        del agent, prompt, run_config
        raise RuntimeError("sandbox run failed")


def _fake_sdk() -> SandboxSDK:
    return SandboxSDK(
        Runner=_FakeRunner(),
        RunConfig=_FakeRunConfig,
        SandboxRunConfig=_FakeSandboxRunConfig,
        Manifest=_FakeManifest,
        SandboxAgent=_FakeSandboxAgent,
        DockerSandboxClient=lambda docker_client: _FakeDockerClient(),
        DockerSandboxClientOptions=_FakeDockerOptions,
        File=_FakeFile,
        Dir=_FakeDir,
        LocalDir=_FakeLocalDir,
        DEFAULT_PYTHON_SANDBOX_IMAGE="python:fake",
        docker_from_env=lambda: object(),
    )


def _task(
    *,
    task_id: str = "task-1",
    instruction: str | None = "Instruction text.",
    workspace_dir: Path | None = None,
) -> AgentEvalTask:
    inputs: dict[str, Any] = {}
    if instruction is not None:
        inputs["instruction"] = instruction
    if workspace_dir is not None:
        inputs["workspace_dir"] = str(workspace_dir)
    return AgentEvalTask(id=task_id, intent="Intent text.", inputs=inputs)


def test_module_imports_without_agents_extra() -> None:
    assert DockerSandboxAgentRuntime.__name__ == "DockerSandboxAgentRuntime"


def test_missing_optional_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name == "agents" or name.startswith("agents.") or name == "docker":
            raise ImportError("missing optional dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match=r"nemo-evaluator-sdk\[agent-runtimes\]") as exc_info:
        docker_sandbox._load_agents_sdk()

    message = str(exc_info.value)
    assert "pip install 'openai-agents[docker]'" in message
    assert "openai-agents[docker]>=" not in message


def test_manifest_uses_instruction_and_never_leaks_intent() -> None:
    # The instruction is surfaced verbatim; `task.intent` is eval-side metadata and must never leak to
    # the agent (reward-hacking hole), so it is never a fallback.
    runtime = DockerSandboxAgentRuntime()
    manifest = runtime._build_manifest(_task(instruction="Instruction text."), _fake_sdk())

    content = manifest.entries["instruction.md"].content.decode("utf-8")
    assert content == "Instruction text."
    assert "Intent text." not in content


def test_manifest_raises_when_task_has_no_instruction() -> None:
    # A task with no instruction cannot be evaluated; building its manifest raises rather than
    # producing an empty prompt (and `task.intent` must never leak as a fallback).
    runtime = DockerSandboxAgentRuntime()
    with pytest.raises(ValueError, match="no instruction"):
        runtime._build_manifest(_task(instruction=None), _fake_sdk())


def test_manifest_maps_workspace_dir_to_local_dir(tmp_path: Path) -> None:
    runtime = DockerSandboxAgentRuntime()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    manifest = runtime._build_manifest(_task(workspace_dir=workspace), _fake_sdk())

    assert manifest.entries["workspace"].src == workspace.resolve()


def test_manifest_omits_serialized_task_to_avoid_leaking_grader_fields() -> None:
    # The workspace is seeded only with the agent-facing projection (prompt + declared files); the
    # task object is never serialized in, so grader-only fields like ``reference`` cannot leak.
    runtime = DockerSandboxAgentRuntime()
    task = AgentEvalTask(
        id="task-1",
        intent="Intent text.",
        inputs={"instruction": "Instruction text."},
        reference={"test_calculator.py": "def test_add(): assert add(2, 3) == 5"},
    )

    manifest = runtime._build_manifest(task, _fake_sdk())

    assert "task.json" not in manifest.entries
    seeded_files = [entry.content for entry in manifest.entries.values() if isinstance(entry, _FakeFile)]
    seeded = b"".join(seeded_files).decode("utf-8")
    assert "reference" not in seeded
    assert "test_calculator.py" not in seeded


def test_manifest_rejects_relative_or_missing_workspace_dir(tmp_path: Path) -> None:
    runtime = DockerSandboxAgentRuntime()

    with pytest.raises(ValueError, match="absolute path"):
        runtime._build_manifest(_task(workspace_dir=Path("relative/workspace")), _fake_sdk())

    with pytest.raises(ValueError, match="does not exist"):
        runtime._build_manifest(_task(workspace_dir=tmp_path / "missing"), _fake_sdk())


@pytest.mark.asyncio
async def test_completed_run_writes_artifacts_and_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = _FakeDockerClient()
    runner = _FakeRunner()
    monkeypatch.setattr(docker_sandbox, "_load_agents_sdk", _fake_sdk)
    runtime = DockerSandboxAgentRuntime(
        image="python:test",
        sandbox_client_factory=lambda: client,
        runner=runner,
    )

    trials = await runtime.run_tasks(
        [_task()],
        config=AgentEvalRunConfig(work_dir=tmp_path, run_id="run-1", parallelism=1),
    )

    evidence_dir = tmp_path / "agent-runtime" / "run-1" / "000000-task-1"
    assert len(client.created) == 1
    assert client.created[0].options.image == "python:test"
    assert client.deleted == client.created
    assert trials[0].status == "completed"
    assert trials[0].measurements.model_dump(exclude_none=True) == {}
    assert trials[0].output is not None
    assert trials[0].output.output_text == "Runtime answer"
    assert (evidence_dir / "final_output.txt").read_text(encoding="utf-8") == "Runtime answer"
    assert (evidence_dir / "run_items.json").exists()
    assert (evidence_dir / "raw_responses.json").exists()
    assert (evidence_dir / "workspace.tar").exists()
    assert (evidence_dir / "final_state" / "output" / "result.txt").read_text(encoding="utf-8") == "done\n"

    assert trials[0].evidence is not None
    handle = await trials[0].evidence.filesystem("final_state")
    assert await handle.exists("output/result.txt") is True


@pytest.mark.asyncio
async def test_runtime_creates_and_deletes_one_sandbox_per_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeDockerClient()
    monkeypatch.setattr(docker_sandbox, "_load_agents_sdk", _fake_sdk)
    runtime = DockerSandboxAgentRuntime(sandbox_client_factory=lambda: client, runner=_FakeRunner())

    await runtime.run_tasks(
        [_task(task_id="task-1"), _task(task_id="task-2")],
        config=AgentEvalRunConfig(work_dir=tmp_path, run_id="run-1", parallelism=2),
    )

    assert len(client.created) == 2
    assert client.deleted == client.created


@pytest.mark.asyncio
async def test_direct_runtime_call_uses_one_generated_run_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeDockerClient()
    monkeypatch.setattr(docker_sandbox, "_load_agents_sdk", _fake_sdk)
    runtime = DockerSandboxAgentRuntime(sandbox_client_factory=lambda: client, runner=_FakeRunner())

    await runtime.run_tasks(
        [_task(task_id="task-1"), _task(task_id="task-2")],
        config=AgentEvalRunConfig(work_dir=tmp_path, parallelism=2),
    )

    run_dirs = list((tmp_path / "agent-runtime").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "000000-task-1").exists()
    assert (run_dirs[0] / "000001-task-2").exists()


@pytest.mark.asyncio
async def test_parallelism_limits_concurrent_task_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeDockerClient()
    runner = _FakeRunner(delay_s=0.01)
    monkeypatch.setattr(docker_sandbox, "_load_agents_sdk", _fake_sdk)
    runtime = DockerSandboxAgentRuntime(sandbox_client_factory=lambda: client, runner=runner)

    await runtime.run_tasks(
        [_task(task_id=f"task-{index}") for index in range(4)],
        config=AgentEvalRunConfig(work_dir=tmp_path, run_id="run-1", parallelism=2),
    )

    assert runner.max_active == 2


@pytest.mark.asyncio
async def test_runtime_exception_returns_failed_trial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeDockerClient()
    monkeypatch.setattr(docker_sandbox, "_load_agents_sdk", _fake_sdk)
    runtime = DockerSandboxAgentRuntime(sandbox_client_factory=lambda: client, runner=_FailingRunner())

    trials = await runtime.run_tasks(
        [_task()],
        config=AgentEvalRunConfig(work_dir=tmp_path, run_id="run-1", parallelism=1),
    )

    error_path = tmp_path / "agent-runtime" / "run-1" / "000000-task-1" / "error.json"
    assert trials[0].status == "failed"
    assert trials[0].measurements.model_dump(exclude_none=True) == {}
    assert trials[0].output is None
    assert trials[0].metadata["error"] == "sandbox run failed"
    assert trials[0].evidence is not None
    assert trials[0].evidence.require("error", kind="error").ref == str(error_path)
    assert error_path.exists()
    assert client.deleted == client.created

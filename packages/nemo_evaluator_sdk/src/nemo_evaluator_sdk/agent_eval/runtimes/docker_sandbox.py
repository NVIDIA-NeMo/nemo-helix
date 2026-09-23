# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker-backed sandbox runtime for agent-eval trials."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
import shutil
import tarfile
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import AgentEvalTrial, AgentEvalTrialStatus, AgentOutput, RunnerInfo
from nemo_evaluator_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from pydantic_core import to_jsonable_python

DEFAULT_INSTRUCTIONS = (
    "Complete the task inside the sandbox workspace. Inspect the provided task files, "
    "write any durable artifacts under output/, and return a concise final answer."
)
_RUNTIME_NAME = "docker_sandbox"
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class SandboxSDK:
    """Loaded OpenAI Agents SDK symbols used by the runtime."""

    Runner: Any
    RunConfig: Any
    SandboxRunConfig: Any
    Manifest: Any
    SandboxAgent: Any
    DockerSandboxClient: Any
    DockerSandboxClientOptions: Any
    File: Any
    Dir: Any
    LocalDir: Any
    DEFAULT_PYTHON_SANDBOX_IMAGE: str
    docker_from_env: Callable[[], Any]


def _load_agents_sdk() -> SandboxSDK:
    try:
        # The OpenAI Agents SDK is imported only when this Docker runtime is actually used, so it
        # is absent from the default type-checking environment.
        from agents.run import RunConfig  # ty: ignore[unresolved-import]
        from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig  # ty: ignore[unresolved-import]
        from agents.sandbox.config import DEFAULT_PYTHON_SANDBOX_IMAGE  # ty: ignore[unresolved-import]
        from agents.sandbox.entries import Dir, File, LocalDir  # ty: ignore[unresolved-import]
        from agents.sandbox.sandboxes.docker import (  # ty: ignore[unresolved-import]
            DockerSandboxClient,
            DockerSandboxClientOptions,
        )

        from agents import Runner  # ty: ignore[unresolved-import]
        from docker import from_env as docker_from_env
    except ImportError as exc:
        # Audience split is in the error text: SDK extras are not propagated into the
        # vendored nemo_helix.beta.evaluator mirror.
        raise RuntimeError(
            "DockerSandboxAgentRuntime requires the openai-agents[docker] Python packages. "
            "Standalone SDK: pip install 'nemo-evaluator-sdk[agent-runtimes]'. "
            "Vendored nemo-helix.beta.evaluator (no SDK extras): "
            "pip install 'openai-agents[docker]'"
        ) from exc

    return SandboxSDK(
        Runner=Runner,
        RunConfig=RunConfig,
        SandboxRunConfig=SandboxRunConfig,
        Manifest=Manifest,
        SandboxAgent=SandboxAgent,
        DockerSandboxClient=DockerSandboxClient,
        DockerSandboxClientOptions=DockerSandboxClientOptions,
        File=File,
        Dir=Dir,
        LocalDir=LocalDir,
        DEFAULT_PYTHON_SANDBOX_IMAGE=DEFAULT_PYTHON_SANDBOX_IMAGE,
        docker_from_env=docker_from_env,
    )


class DockerSandboxAgentRuntime:
    """Generate agent-eval trials by running a SandboxAgent in Docker per task."""

    def __init__(
        self,
        *,
        model: str | None = None,
        instructions: str | None = None,
        image: str | None = None,
        work_root: Path | None = None,
        timeout_s: float | None = None,
        agent_factory: Callable[..., Any] | None = None,
        sandbox_client_factory: Callable[[], Any] | None = None,
        runner: Any | None = None,
    ) -> None:
        self._model = model
        self._instructions = instructions or DEFAULT_INSTRUCTIONS
        self._image = image
        self._work_root = work_root
        self._timeout_s = timeout_s
        self._agent_factory = agent_factory
        self._sandbox_client_factory = sandbox_client_factory
        self._runner = runner

    def runner_info(self) -> RunnerInfo:
        """Identify this runner and the sandbox settings that shape its results."""
        return RunnerInfo(
            name="docker_sandbox",
            kind="runner",
            config={
                "model": self._model,
                "image": self._image,
                "timeout_s": self._timeout_s,
                "instructions": self._instructions,
            },
        )

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> Sequence[AgentEvalTrial]:
        resolved_config = config or AgentEvalRunConfig()
        if resolved_config.run_id is None:
            resolved_config = resolved_config.model_copy(update={"run_id": _new_runtime_run_id()})
        sdk = _load_agents_sdk()
        semaphore = asyncio.Semaphore(resolved_config.parallelism)

        async def run_one(index: int, task: AgentEvalTask) -> AgentEvalTrial:
            async with semaphore:
                return await self._run_task(index, task, resolved_config, sdk)

        return await asyncio.gather(*(run_one(index, task) for index, task in enumerate(tasks)))

    async def _run_task(
        self,
        index: int,
        task: AgentEvalTask,
        config: AgentEvalRunConfig,
        sdk: SandboxSDK,
    ) -> AgentEvalTrial:
        evidence_dir = self._evidence_dir(index, task, config)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        client = self._build_client(sdk)
        sandbox = None

        try:
            # Build the prompt inside the guarded block: an instruction-less task raises here and fails
            # just this task rather than aborting the whole run.
            prompt = task.agent_prompt()
            manifest = self._build_manifest(task, sdk)
            agent = self._build_agent(manifest, sdk)
            sandbox = await client.create(
                manifest=manifest,
                options=sdk.DockerSandboxClientOptions(image=self._image or sdk.DEFAULT_PYTHON_SANDBOX_IMAGE),
            )
            async with sandbox:
                result = await self._run_agent(agent, prompt, sandbox, sdk)
                return await self._completed_trial(task, result, sandbox, evidence_dir)
        except Exception as exc:
            return self._failed_trial(task, exc, evidence_dir)
        finally:
            if sandbox is not None:
                with contextlib.suppress(Exception):
                    await client.delete(sandbox)

    def _build_manifest(self, task: AgentEvalTask, sdk: SandboxSDK) -> Any:
        # Seed only the agent-facing projection of the task: the prompt (its instruction) plus any
        # declared workspace files. We deliberately do NOT serialize the task object into the
        # workspace — nothing in the runtime consumes it, and dumping the whole DTO would expose
        # grader-only fields (e.g. ``reference`` held-out ground truth) to the agent.
        entries: dict[str, Any] = {
            "instruction.md": sdk.File(content=task.agent_prompt().encode("utf-8")),
            "output": sdk.Dir(),
        }
        workspace_dir = task.inputs.get("workspace_dir")
        if workspace_dir is not None:
            entries["workspace"] = sdk.LocalDir(src=_validated_workspace_dir(workspace_dir))
        return sdk.Manifest(root="/workspace", entries=entries)

    def _build_agent(self, manifest: Any, sdk: SandboxSDK) -> Any:
        agent_factory = self._agent_factory or sdk.SandboxAgent
        kwargs = {
            "name": "NeMo Agent Eval Docker Sandbox Runtime",
            "instructions": self._instructions,
            "default_manifest": manifest,
        }
        if self._model is not None:
            kwargs["model"] = self._model
        return agent_factory(**kwargs)

    def _build_client(self, sdk: SandboxSDK) -> Any:
        if self._sandbox_client_factory is not None:
            return self._sandbox_client_factory()
        return sdk.DockerSandboxClient(sdk.docker_from_env())

    async def _run_agent(self, agent: Any, prompt: str, sandbox: Any, sdk: SandboxSDK) -> Any:
        runner = self._runner or sdk.Runner
        run = runner.run(
            agent,
            prompt,
            run_config=sdk.RunConfig(sandbox=sdk.SandboxRunConfig(session=sandbox)),
        )
        if self._timeout_s is not None:
            return await asyncio.wait_for(_maybe_await(run), timeout=self._timeout_s)
        return await _maybe_await(run)

    async def _completed_trial(
        self,
        task: AgentEvalTask,
        result: Any,
        sandbox: Any,
        evidence_dir: Path,
    ) -> AgentEvalTrial:
        final_output = getattr(result, "final_output", None)
        final_output_text = "" if final_output is None else str(final_output)

        final_output_path = evidence_dir / "final_output.txt"
        run_items_path = evidence_dir / "run_items.json"
        raw_responses_path = evidence_dir / "raw_responses.json"
        workspace_tar_path = evidence_dir / "workspace.tar"
        final_state_dir = evidence_dir / "final_state"

        final_output_path.write_text(final_output_text, encoding="utf-8")
        _write_json(run_items_path, _jsonable(getattr(result, "new_items", [])))
        _write_json(raw_responses_path, _jsonable(getattr(result, "raw_responses", [])))
        await _persist_workspace(sandbox, workspace_tar_path, final_state_dir)

        return AgentEvalTrial(
            id=f"{task.id}:docker-sandbox",
            task_id=task.id,
            status=AgentEvalTrialStatus.COMPLETED,
            output=AgentOutput(
                output_text=final_output_text,
                response={"final_output": final_output_text},
                metadata={
                    "runtime": _RUNTIME_NAME,
                    "evidence_dir": str(evidence_dir),
                },
            ),
            evidence=CandidateEvidence(
                descriptors={
                    "final_state": EvidenceDescriptor(kind="filesystem", ref=str(final_state_dir)),
                    "workspace_archive": EvidenceDescriptor(kind="archive", format="tar", ref=str(workspace_tar_path)),
                    "run_items": EvidenceDescriptor(kind="run_items", format="json", ref=str(run_items_path)),
                    "raw_responses": EvidenceDescriptor(
                        kind="raw_responses", format="json", ref=str(raw_responses_path)
                    ),
                    "final_output": EvidenceDescriptor(kind="text", format="txt", ref=str(final_output_path)),
                },
                metadata={"runtime": _RUNTIME_NAME, "sandbox_backend": "docker"},
            ),
            metadata={"runtime": _RUNTIME_NAME, "generated": True},
        )

    def _failed_trial(self, task: AgentEvalTask, exc: Exception, evidence_dir: Path) -> AgentEvalTrial:
        error_path = evidence_dir / "error.json"
        _write_json(
            error_path,
            {
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )
        return AgentEvalTrial(
            id=f"{task.id}:docker-sandbox",
            task_id=task.id,
            status=AgentEvalTrialStatus.FAILED,
            output=None,
            evidence=CandidateEvidence(
                descriptors={
                    "error": EvidenceDescriptor(kind="error", format="json", ref=str(error_path)),
                },
                metadata={"runtime": _RUNTIME_NAME, "sandbox_backend": "docker"},
            ),
            metadata={
                "runtime": _RUNTIME_NAME,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )

    def _evidence_dir(self, index: int, task: AgentEvalTask, config: AgentEvalRunConfig) -> Path:
        root = config.work_dir if config.work_dir is not None else self._work_root
        if root is None:
            root = Path(tempfile.gettempdir()) / "nemo-evaluator-agent-runtime"
        run_id = config.run_id or _new_runtime_run_id()
        safe_task_id = _safe_path_name(task.id)
        task_name = f"{index:06d}-{safe_task_id}" if safe_task_id else f"task-{index:06d}"
        return Path(root) / "agent-runtime" / run_id / task_name


def _validated_workspace_dir(workspace_dir: Any) -> Path:
    if not isinstance(workspace_dir, (str, Path)):
        raise ValueError(f"workspace_dir must be a path, got {type(workspace_dir).__name__}")
    path = Path(workspace_dir).expanduser()
    if not path.is_absolute():
        raise ValueError(f"workspace_dir must be an absolute path; got {workspace_dir!r}")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"workspace_dir does not exist or is not a directory: {resolved}")
    return resolved


async def _maybe_await(value: Awaitable[Any] | Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _persist_workspace(sandbox: Any, workspace_tar_path: Path, final_state_dir: Path) -> None:
    archive = await sandbox.persist_workspace()
    try:
        with workspace_tar_path.open("wb") as out:
            shutil.copyfileobj(archive, out)
    finally:
        close = getattr(archive, "close", None)
        if close is not None:
            close()

    _extract_tar_safely(workspace_tar_path, final_state_dir)


def _extract_tar_safely(archive_path: Path, destination_root: Path) -> None:
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    # The stdlib `data` filter (Python 3.12+) rejects absolute paths, parent-directory
    # traversal, links, and special files, so we do not hand-roll those guards.
    with tarfile.open(archive_path, "r:*") as archive:
        archive.extractall(destination_root, filter="data")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    # Normalize pydantic models, dataclasses, Paths, sets, etc. into JSON-safe values;
    # `repr` is the last-resort fallback for anything still not serializable.
    return to_jsonable_python(value, fallback=repr)


def _safe_path_name(value: str) -> str:
    sanitized = _SAFE_NAME_PATTERN.sub("-", value).strip(".-")
    return sanitized[:120]


def _new_runtime_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return f"agent-runtime-{timestamp}-{uuid4().hex[:8]}"

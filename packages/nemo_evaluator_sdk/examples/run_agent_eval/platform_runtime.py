# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo-Platform glue that lets this example run a real ``tests/agentic-use`` task.

Generic logic lives in ``nemo_evaluator_sdk.agent_eval``; this module holds only
the agentic-use-specific pieces: :func:`agentic_task_from_dir` (load a task from
``instruction.md`` + ``task.toml``), :func:`ensure_task_image` (BUILD),
:class:`NatWorkflowRuntime` (AGENT via ``nat run`` + optional pytest VERIFY,
shaped through the shared :func:`run_agent_then_verify`), and
:class:`VerifierRewardMetric` (scores the pytest reward).

Running a real task requires Docker, the ``nmp-agentic-base:latest`` image, a
running NeMo Platform, and ``NVIDIA_API_KEY`` — see this example's README.
"""

from __future__ import annotations

import subprocess
import textwrap
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from nemo_evaluator_sdk.agent_eval.runtimes.environment import (
    AgentEnvironmentHandle,
    AgentEnvironmentProvider,
    DockerEnvironmentProvider,
    EnvRunSpec,
)
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nemo_evaluator_sdk.agent_eval.trials import (
    AgentEvalTrial,
    AgentEvalTrialStatus,
    AgentOutput,
    RunnerInfo,
    TrialMeasurements,
    resolve_trial_status,
    standard_evidence_descriptors,
)
from nemo_evaluator_sdk.metrics.protocol import MetricInput, MetricOutput, MetricOutputSpec, MetricResult
from nemo_evaluator_sdk.values.evidence import (
    EVIDENCE_FORMAT_ATIF,
    CandidateEvidence,
    EvidenceDescriptor,
    read_atif,
)

from .build_spec import execute_build_plan, plan_task_build
from .layout import prepare_run_layout, resolve_run_dir
from .usage import agent_log_has_workflow_error, extract_usage_metrics
from .verify import (
    VerifierOutcome,
    apply_verify_to_metadata,
    collect_verifier_outcome,
    skipped_outcome,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
AGENTIC_USE_DIR = REPO_ROOT / "tests" / "agentic-use"
SHARED_DIR = AGENTIC_USE_DIR / "shared"
EVALUATOR_SDK_SRC = REPO_ROOT / "packages" / "nemo_evaluator_sdk" / "src"

RUNTIME_NAME = "workflow"
DEFAULT_TIMEOUT_SEC = 600
DEFAULT_LOCAL_NMP_BASE_URL = "http://localhost:8080"
FILES_STORAGE_CONFIG = '{"type":"local","path":"/data/files_storage"}'
PLATFORM_CONFIG_PATH = "/app/packages/nmp_platform/config/local.yaml"
NAT_TRACE_EXPORT_SCRIPT_CONTAINER_PATH = "/app/tests/agentic-use/scripts/nat_trace_export.py"
INSTRUCTION_CONTAINER_PATH = "/tmp/nat_instruction.md"
WORKFLOW_CONTAINER_PATH = "/tmp/nat_workflow.yml"
DOCKER_SOCKET_HOST_PATH = Path("/var/run/docker.sock")
DOCKER_SOCKET_CONTAINER_PATH = "/var/run/docker.sock"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NatWorkflowConfig:
    """Configuration for :class:`NatWorkflowRuntime`."""

    nmp_base_url: str = DEFAULT_LOCAL_NMP_BASE_URL
    nvidia_api_key: str | None = None
    agent_model: str | None = None
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    run_verify: bool = False
    docker_extra_args: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Run layout + image tag
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgenticRunLayout:
    """Run layout extending the SDK's generic one with a platform ``state_dir``."""

    run_dir: Path
    agent_log_dir: Path
    workspace_dir: Path
    state_dir: Path
    instruction_path: Path


def task_image_tag(task_id: str) -> str:
    return f"nmp-nat-{task_id}:latest"


def resolve_run_layout(task: AgentEvalTask, config: AgentEvalRunConfig | None) -> AgenticRunLayout:
    """Resolve/create the on-disk layout for one task run."""
    output_dir = config.work_dir if config is not None else None
    run_dir = resolve_run_dir(output_dir, lambda: Path.cwd() / "nat-jobs" / task.id) / task.id
    base = prepare_run_layout(run_dir, str(task.inputs.get("instruction") or task.intent))
    state_dir = base.run_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return AgenticRunLayout(
        run_dir=base.run_dir,
        agent_log_dir=base.agent_log_dir,
        workspace_dir=base.workspace_dir,
        state_dir=state_dir,
        instruction_path=base.instruction_path,
    )


class PlatformDockerEnvironmentProvider(DockerEnvironmentProvider):
    """Docker provider defaulting each task to ``nmp-nat-<id>:latest``."""

    def __init__(self, *, image_tag_fn: Callable[[str], str] = task_image_tag) -> None:
        super().__init__(image_tag_fn=image_tag_fn)


# --------------------------------------------------------------------------- #
# Verifier reward metric (compatibility shim for the pytest reward)
# --------------------------------------------------------------------------- #
class VerifierRewardMetric:
    """Score the pytest verifier reward stamped on trial metadata."""

    @property
    def type(self) -> str:
        return "agentic_use_verifier_reward"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.continuous_score("verifier_reward")]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        metadata = input.candidate.metadata
        reward = metadata.get("reward")
        if reward is None:
            reward = 1.0 if metadata.get("passed") else 0.0
        return MetricResult(outputs=[MetricOutput(name="verifier_reward", value=float(reward))])


# --------------------------------------------------------------------------- #
# Task loader
# --------------------------------------------------------------------------- #
def load_task_toml(task_dir: Path) -> dict[str, Any]:
    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return {}
    try:
        with task_toml.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def task_agent_timeout_sec(task_dir: Path) -> int | None:
    agent = load_task_toml(task_dir).get("agent")
    if not isinstance(agent, dict):
        return None
    timeout_value = agent.get("timeout_sec")
    if isinstance(timeout_value, int | float) and timeout_value > 0:
        return int(timeout_value)
    return None


def agentic_task_from_dir(task_dir: str | Path, *, tasks_root: Path | None = None) -> AgentEvalTask:
    """Build an ``AgentEvalTask`` from an agentic-use task directory.

    ``inputs`` carries only agent-facing material (``instruction``); runtime
    materialization (``task_dir``) lives in ``metadata`` so it can't leak into a
    metric scoring row. Metrics default to ``[AgentPhaseSuccessMetric()]``.
    """
    from nemo_evaluator_sdk.agent_eval.metrics import AgentPhaseSuccessMetric

    root = Path(tasks_root or AGENTIC_USE_DIR)
    task_path = Path(task_dir)
    if not task_path.is_absolute():
        task_path = (root / task_path).resolve()

    instruction_path = task_path / "instruction.md"
    if not instruction_path.exists():
        raise FileNotFoundError(f"instruction.md not found in {task_path}")
    instruction = instruction_path.read_text(encoding="utf-8")

    return AgentEvalTask(
        id=task_path.name,
        intent=instruction,
        inputs={"instruction": instruction},
        metrics=[AgentPhaseSuccessMetric()],
        metadata={
            "benchmark": "agentic-use",
            "task_toml": load_task_toml(task_path),
            "instruction_path": str(instruction_path),
            "task_dir": str(task_path),
        },
    )


# --------------------------------------------------------------------------- #
# BUILD phase
# --------------------------------------------------------------------------- #
def ensure_task_image(task: AgentEvalTask, *, skip_build: bool = False) -> str:
    """Build (or verify the presence of) the task's Docker image; return its tag."""
    image_tag = task_image_tag(task.id)
    task_dir = Path(str(task.metadata["task_dir"]))
    if skip_build:
        exists = (
            subprocess.run(["docker", "image", "inspect", image_tag], capture_output=True, check=False).returncode == 0
        )
        if not exists:
            raise RuntimeError(f"--skip-build set but image {image_tag!r} is not available locally; build it first.")
        return image_tag
    execute_build_plan(plan_task_build(task_dir, image_tag))
    return image_tag


# --------------------------------------------------------------------------- #
# AGENT phase: command + workflow prep + container env
# --------------------------------------------------------------------------- #
def build_workflow_agent_cmd(workflow_container: str, instruction_container: str) -> list[str]:
    """``bash -c`` command that runs ``nat run`` and exports the trajectory."""
    return [
        "bash",
        "-c",
        textwrap.dedent(f"""\
            /app/.venv/bin/nat run \\
              --config_file {workflow_container} \\
              --input "$(cat {instruction_container})" \\
              2>&1 | tee /tmp/nat_agent.log
            EXIT=${{PIPESTATUS[0]}}
            cp /tmp/nat_agent.log /logs/agent/nat_agent.log 2>/dev/null || true
            if [ -f /logs/agent/intermediate_steps.jsonl ]; then
              /app/.venv/bin/python {NAT_TRACE_EXPORT_SCRIPT_CONTAINER_PATH} convert-jsonl \\
                --input /logs/agent/intermediate_steps.jsonl \\
                --output /logs/agent/trajectory.json \\
                >> /tmp/nat_agent.log 2>&1
              cp /tmp/nat_agent.log /logs/agent/nat_agent.log 2>/dev/null || true
            fi
            exit $EXIT
        """),
    ]


def prepare_workflow_for_runtime(
    workflow_path: Path,
    output_dir: Path,
    nmp_base_url: str,
    *,
    nat_model: str | None = None,
) -> Path:
    """Rewrite a task ``workflow.yml`` for container execution + trajectory export."""
    text = workflow_path.read_text(encoding="utf-8")
    text = text.replace("http://localhost:8080", nmp_base_url)
    if nat_model:
        text = text.replace(
            "model_name: nvidia/llama-3.1-nemotron-70b-instruct",
            f"model_name: {nat_model}",
            1,
        )
    if ("_type: mcp_client" in text or "_type: per_user_mcp_client" in text) and (
        "\nfunction_groups:\n" not in text and "\nfunctions:\n" in text
    ):
        text = text.replace("\nfunctions:\n", "\nfunction_groups:\n", 1)

    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError(f"Workflow config must be a mapping: {workflow_path}")
    general = config.setdefault("general", {})
    telemetry = general.setdefault("telemetry", {})
    tracing = telemetry.setdefault("tracing", {})
    tracing["agentic_use_file_trace"] = {
        "_type": "file",
        "output_path": "/logs/agent/intermediate_steps.jsonl",
        "project": "agentic-use",
        "mode": "overwrite",
        "cleanup_on_init": True,
    }

    rewritten = output_dir / "workflow.runtime.yml"
    rewritten.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return rewritten


def base_container_env(nmp_base_url: str, *, timeout_sec: int) -> dict[str, str]:
    env = {
        "NMP_BASE_URL": nmp_base_url,
        "AGENTIC_USE_WORKSPACE_DIR": "/app/workspace",
        "DATABASE_DIALECT": "sqlite",
        "DATABASE_PATH": "/data/nmp-platform.db",
        "NMP_FILES_DEFAULT_STORAGE_CONFIG": FILES_STORAGE_CONFIG,
        "NMP_CONFIG_FILE_PATH": PLATFORM_CONFIG_PATH,
        "NEMO_AGENTS_GATEWAY_READ_TIMEOUT": str(timeout_sec),
        "NEMO_AGENTS_INVOKE_TIMEOUT": str(timeout_sec),
    }
    if DOCKER_SOCKET_HOST_PATH.exists():
        env["DOCKER_HOST"] = f"unix://{DOCKER_SOCKET_CONTAINER_PATH}"
    return env


def docker_socket_mounts() -> list[tuple[str, str]]:
    """Bind-mount the host Docker socket into the container when it exists."""
    if DOCKER_SOCKET_HOST_PATH.exists():
        return [(str(DOCKER_SOCKET_HOST_PATH), DOCKER_SOCKET_CONTAINER_PATH)]
    return []


# --------------------------------------------------------------------------- #
# VERIFY phase
# --------------------------------------------------------------------------- #
def verifier_log_dir(layout: AgenticRunLayout) -> Path:
    return layout.run_dir / "verifier"


def build_verify_run_spec(
    task_dir: Path,
    layout: AgenticRunLayout,
    *,
    nmp_base_url: str,
    agent_model: str,
    agent_backend: str = RUNTIME_NAME,
    timeout_sec: int | None = None,
    extra_args: list[str] | None = None,
) -> EnvRunSpec | None:
    """Build the verifier ``EnvRunSpec`` (pytest ``test_outputs.py``), or ``None``."""
    tests_dir = task_dir / "tests"
    if not (tests_dir / "test_outputs.py").exists():
        return None

    log_dir = verifier_log_dir(layout)
    log_dir.mkdir(parents=True, exist_ok=True)
    layout.workspace_dir.mkdir(parents=True, exist_ok=True)

    verify_cmd = [
        "bash",
        "-c",
        textwrap.dedent("""\
            export PYTHONPATH="/app/tests/agentic-use/shared:/app/packages/nemo_evaluator_sdk/src:${PYTHONPATH}"
            export NAT_AGENT=1
            /app/.venv/bin/python -m pytest /tests/test_outputs.py -rA -v 2>&1 | tee /logs/verifier/test-stdout.txt
            EXIT=${PIPESTATUS[0]}
            if [ $EXIT -eq 0 ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
            exit $EXIT
        """),
    ]

    env = base_container_env(nmp_base_url, timeout_sec=timeout_sec or DEFAULT_TIMEOUT_SEC)
    env.update(
        {
            "NAT_AGENT": "1",
            "NAT_AGENT_BACKEND": agent_backend,
            "NAT_AGENT_MODEL": agent_model,
            "AGENTIC_USE_TASK_DIR": "/task",
        }
    )

    mounts = [
        (str(tests_dir), "/tests"),
        (str(task_dir), "/task"),
        (str(layout.workspace_dir), "/app/workspace"),
        (str(SHARED_DIR), "/app/tests/agentic-use/shared:ro"),
        (str(EVALUATOR_SDK_SRC), "/app/packages/nemo_evaluator_sdk/src:ro"),
        (str(layout.agent_log_dir), "/logs/agent"),
        (str(log_dir), "/logs/verifier"),
        (str(layout.state_dir), "/data"),
        *docker_socket_mounts(),
    ]

    return EnvRunSpec(
        command=verify_cmd, env=env, mounts=mounts, timeout=timeout_sec, extra_args=list(extra_args or [])
    )


async def maybe_run_verify(
    handle: AgentEnvironmentHandle,
    *,
    enabled: bool,
    task_dir: Path,
    layout: AgenticRunLayout,
    nmp_base_url: str,
    agent_model: str,
    agent_backend: str = RUNTIME_NAME,
    timeout_sec: int | None = None,
    extra_args: list[str] | None = None,
) -> VerifierOutcome:
    """Run the verifier through ``handle`` when enabled and a verifier exists."""
    if not enabled:
        return skipped_outcome()
    spec = build_verify_run_spec(
        task_dir,
        layout,
        nmp_base_url=nmp_base_url,
        agent_model=agent_model,
        agent_backend=agent_backend,
        timeout_sec=timeout_sec,
        extra_args=extra_args,
    )
    if spec is None:
        return skipped_outcome()
    result = await handle.run_verifier(spec)
    return collect_verifier_outcome(ok=result.ok, exit_code=result.exit_code, log_dir=verifier_log_dir(layout))


# --------------------------------------------------------------------------- #
# Trial construction from live artifacts
# --------------------------------------------------------------------------- #
def build_trial_from_artifacts(
    *,
    task: AgentEvalTask,
    layout: AgenticRunLayout,
    runtime_name: str,
    agent_model: str,
    exit_code: int,
    agent_ok: bool,
    runtime_sec: float,
) -> AgentEvalTrial:
    """Shape an ``AgentEvalTrial`` from on-disk agent artifacts.

    Token usage is parsed from the agent log with the ``nat_runner`` extractor so
    the SDK summary's token/runtime aggregates populate exactly as they do in
    ``result.json["metrics"]``.
    """
    log_text = _read_agent_log(layout.agent_log_dir)
    usage = extract_usage_metrics(log_text)
    trace_path = layout.agent_log_dir / "trajectory.json"
    descriptors = standard_evidence_descriptors(
        logs_dir=layout.agent_log_dir,
        final_state_dir=layout.workspace_dir,
        trace_path=trace_path if trace_path.exists() else None,
        trace_format=EVIDENCE_FORMAT_ATIF if read_atif(trace_path) is not None else None,
        verifier_logs_dir=verifier_log_dir(layout),
        primary_log="nat_agent.log",
    )
    descriptors["state"] = EvidenceDescriptor(
        kind="filesystem",
        format="dir",
        ref=str(layout.state_dir),
        metadata={"role": "platform_state", "extension": "nemo-platform"},
    )

    output_text = log_text.strip() or ("" if agent_ok else "(agent phase failed)")
    measurements = TrialMeasurements(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        cache_creation_tokens=usage["cache_creation_tokens"],
        cache_read_tokens=usage["cache_read_tokens"],
        runtime_sec=runtime_sec,
    )
    metadata: dict[str, Any] = {
        "agent_runtime": runtime_name,
        "agent_model": agent_model,
        "agent_ok": agent_ok,
        "exit_code": exit_code,
        "run_dir": str(layout.run_dir),
        "agent_log_dir": str(layout.agent_log_dir),
        "workspace_dir": str(layout.workspace_dir),
        "state_dir": str(layout.state_dir),
        "generated": True,
    }
    return AgentEvalTrial(
        id=f"{task.id}:{runtime_name}",
        task_id=task.id,
        status=resolve_trial_status(agent_ok),
        output=AgentOutput(
            output_text=output_text,
            metadata={"runtime": runtime_name, "agent_model": agent_model},
        ),
        evidence=CandidateEvidence(descriptors=descriptors, metadata={"runtime": runtime_name}),
        measurements=measurements,
        metadata=metadata,
    )


def _read_agent_log(agent_log_dir: Path) -> str:
    log_path = agent_log_dir / "nat_agent.log"
    if log_path.is_file():
        return log_path.read_text(encoding="utf-8", errors="replace")
    return ""


async def run_agent_then_verify(
    handle: AgentEnvironmentHandle,
    *,
    task: AgentEvalTask,
    layout: AgenticRunLayout,
    spec: EnvRunSpec,
    runtime_name: str,
    agent_model: str,
    run_verify: bool,
    nmp_base_url: str,
    verify_timeout_sec: int,
    docker_extra_args: list[str],
) -> AgentEvalTrial:
    """Shared AGENT → VERIFY → trial flow for the Docker-backed runtimes.

    Runs the agent ``spec`` through ``handle``, flips success on a logged
    ``workflow_error``, optionally runs the pytest verifier, then shapes a trial
    (promoting it to ``COMPLETED`` when the verifier passes).
    """
    started = time.monotonic()
    try:
        result = await handle.run_agent(spec)
        agent_ok = result.ok
        log_text = _read_agent_log(layout.agent_log_dir)
        if agent_ok and log_text and agent_log_has_workflow_error(log_text):
            agent_ok = False
        verify_outcome = await maybe_run_verify(
            handle,
            enabled=run_verify and agent_ok,
            task_dir=Path(str(task.metadata["task_dir"])),
            layout=layout,
            nmp_base_url=nmp_base_url,
            agent_model=agent_model,
            agent_backend=runtime_name,
            timeout_sec=verify_timeout_sec,
            extra_args=docker_extra_args,
        )
    finally:
        await handle.close()
    runtime_sec = time.monotonic() - started

    trial = build_trial_from_artifacts(
        task=task,
        layout=layout,
        runtime_name=runtime_name,
        agent_model=agent_model,
        exit_code=result.exit_code,
        agent_ok=agent_ok,
        runtime_sec=runtime_sec,
    )
    apply_verify_to_metadata(trial.metadata, verify_outcome)
    if verify_outcome.ran and verify_outcome.passed and trial.status != AgentEvalTrialStatus.COMPLETED:
        trial = trial.model_copy(update={"status": AgentEvalTrialStatus.COMPLETED})
    return trial


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
class NatWorkflowRuntime:
    """Run agentic-use tasks via ``nat run`` inside the task image (an ``AgentTaskRunner``)."""

    def __init__(
        self,
        config: NatWorkflowConfig | None = None,
        *,
        environment: AgentEnvironmentProvider | None = None,
    ) -> None:
        self.config = config or NatWorkflowConfig()
        self.environment = environment or PlatformDockerEnvironmentProvider()

    def runner_info(self) -> RunnerInfo:
        """Identify this runtime and the settings that shape its results (the AgentTaskRunner contract).

        Enumerates fields rather than dumping ``NatWorkflowConfig``: it also carries ``nvidia_api_key``,
        and this is persisted with the run.
        """
        return RunnerInfo(
            name="example_nat_workflow",
            kind="runner",
            config={
                "nmp_base_url": self.config.nmp_base_url,
                "agent_model": self.config.agent_model,
                "timeout_sec": self.config.timeout_sec,
                "run_verify": self.config.run_verify,
                "docker_extra_args": list(self.config.docker_extra_args),
            },
        )

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> Sequence[AgentEvalTrial]:
        trials: list[AgentEvalTrial] = []
        for task in tasks:
            trials.append(await self._run_task(task, config))
        return trials

    async def _run_task(self, task: AgentEvalTask, config: AgentEvalRunConfig | None) -> AgentEvalTrial:
        layout = resolve_run_layout(task, config)
        task_dir = Path(str(task.metadata["task_dir"]))
        agent_model = self.config.agent_model or "unknown"
        handle = await self.environment.prepare(task, config)
        return await run_agent_then_verify(
            handle,
            task=task,
            layout=layout,
            spec=self._agent_run_spec(task_dir, layout),
            runtime_name=RUNTIME_NAME,
            agent_model=agent_model,
            run_verify=self.config.run_verify,
            nmp_base_url=self.config.nmp_base_url,
            verify_timeout_sec=self.config.timeout_sec + 120,
            docker_extra_args=list(self.config.docker_extra_args),
        )

    def _agent_run_spec(self, task_dir: Path, layout: AgenticRunLayout) -> EnvRunSpec:
        workflow_path = task_dir / "workflow.yml"
        if not workflow_path.exists():
            raise FileNotFoundError(f"workflow.yml not found in {task_dir}")

        task_timeout = task_agent_timeout_sec(task_dir) or 0
        timeout_sec = max(self.config.timeout_sec, task_timeout)
        workflow_host = prepare_workflow_for_runtime(
            workflow_path,
            layout.agent_log_dir,
            self.config.nmp_base_url,
            nat_model=self.config.agent_model,
        )

        env = base_container_env(self.config.nmp_base_url, timeout_sec=timeout_sec)
        if self.config.nvidia_api_key:
            env["NVIDIA_API_KEY"] = self.config.nvidia_api_key
        if self.config.agent_model:
            env["NAT_MODEL"] = self.config.agent_model

        mounts = [
            (str(layout.instruction_path), INSTRUCTION_CONTAINER_PATH),
            (str(layout.agent_log_dir), "/logs/agent"),
            (str(layout.workspace_dir), "/app/workspace"),
            (str(workflow_host), WORKFLOW_CONTAINER_PATH),
            (str(layout.state_dir), "/data"),
            *docker_socket_mounts(),
        ]

        return EnvRunSpec(
            command=build_workflow_agent_cmd(WORKFLOW_CONTAINER_PATH, INSTRUCTION_CONTAINER_PATH),
            env=env,
            mounts=mounts,
            timeout=timeout_sec,
            extra_args=list(self.config.docker_extra_args),
        )


__all__ = [
    "AGENTIC_USE_DIR",
    "AgenticRunLayout",
    "NatWorkflowConfig",
    "NatWorkflowRuntime",
    "PlatformDockerEnvironmentProvider",
    "VerifierRewardMetric",
    "agentic_task_from_dir",
    "ensure_task_image",
    "run_agent_then_verify",
    "task_image_tag",
]

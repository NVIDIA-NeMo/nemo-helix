# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow-runtime adapter over the trials-based agent-eval SDK.

The three pieces an integrator supplies on top of the promoted SDK generics:
:class:`WorkflowAgentRuntime` (an ``AgentTaskRunner`` that launches a per-task
command and shapes an :class:`AgentEvalTrial`), :class:`TrialJsonSerde` (the
``AgentTrialSerde`` for offline rescoring), and :func:`example_tasks` (self-
contained tasks wired to reusable + task-authored metrics).

The default command runs the bundled :mod:`mini_agent` so the example runs
end-to-end with no external infrastructure; point ``command`` at a real agent
(``nat run``, ``codex exec``, ...) honoring the same CLI contract to swap it in.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

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
from nemo_evaluator_sdk.values.evidence import CandidateEvidence

from .layout import prepare_run_layout

RUNTIME_NAME = "workflow"
MINI_AGENT = Path(__file__).resolve().parent / "mini_agent.py"
DEFAULT_TIMEOUT_S = 120

# Command tokens substituted per task before launch.
_INSTRUCTION_TOKEN = "{instruction}"
_WORKSPACE_TOKEN = "{workspace}"
_INPUT_JSON_TOKEN = "{input_json}"


def _default_workflow_command() -> list[str]:
    """Run the bundled toy agent via the current interpreter (no external deps)."""
    return [
        sys.executable,
        str(MINI_AGENT),
        "--instruction",
        _INSTRUCTION_TOKEN,
        "--workspace",
        _WORKSPACE_TOKEN,
        "--input-json",
        _INPUT_JSON_TOKEN,
    ]


@dataclass(frozen=True)
class WorkflowRuntimeConfig:
    """Configuration for :class:`WorkflowAgentRuntime`."""

    command: list[str] = field(default_factory=_default_workflow_command)
    timeout_s: int = DEFAULT_TIMEOUT_S
    agent_model: str = "mini-agent"


class WorkflowAgentRuntime:
    """Run agent-eval tasks via a per-task workflow command (an ``AgentTaskRunner``)."""

    def __init__(self, config: WorkflowRuntimeConfig | None = None) -> None:
        self.config = config or WorkflowRuntimeConfig()

    def runner_info(self) -> RunnerInfo:
        """Identify this runtime and the settings that shape its results (the AgentTaskRunner contract)."""
        return RunnerInfo(
            name="example_workflow",
            kind="runner",
            config={
                "command": list(self.config.command),
                "timeout_s": self.config.timeout_s,
                "agent_model": self.config.agent_model,
            },
        )

    async def run_tasks(
        self,
        tasks: Sequence[AgentEvalTask],
        config: AgentEvalRunConfig | None = None,
    ) -> Sequence[AgentEvalTrial]:
        resolved = config or AgentEvalRunConfig()
        semaphore = asyncio.Semaphore(resolved.parallelism)

        async def run_one(index: int, task: AgentEvalTask) -> AgentEvalTrial:
            async with semaphore:
                return await self._run_task(index, task, resolved)

        return await asyncio.gather(*(run_one(index, task) for index, task in enumerate(tasks)))

    async def _run_task(self, index: int, task: AgentEvalTask, config: AgentEvalRunConfig) -> AgentEvalTrial:
        run_dir = self._run_dir(index, task, config)
        instruction = str(task.inputs.get("instruction") or task.intent)
        layout = prepare_run_layout(run_dir, instruction)

        input_json_path = layout.agent_log_dir / "task_input.json"
        input_json_path.write_text(json.dumps(task.inputs, indent=2), encoding="utf-8")

        command = self._format_command(layout.instruction_path, layout.workspace_dir, input_json_path)
        started = time.monotonic()
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.config.timeout_s)
        except Exception as exc:  # noqa: BLE001 - any launch/timeout failure is a trial-production failure.
            # wait_for cancels communicate() on timeout but leaves the child running; kill it.
            if process is not None:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
            return self._failed_trial(task, layout.run_dir, exc)
        runtime_sec = round(time.monotonic() - started, 3)

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        (layout.agent_log_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")
        (layout.agent_log_dir / "stderr.txt").write_text(stderr_text, encoding="utf-8")

        agent_ok = process.returncode == 0
        descriptors = standard_evidence_descriptors(
            logs_dir=layout.agent_log_dir,
            final_state_dir=layout.workspace_dir,
        )
        trial = AgentEvalTrial(
            id=f"{task.id}:{RUNTIME_NAME}",
            task_id=task.id,
            status=resolve_trial_status(agent_ok),
            output=AgentOutput(
                output_text=stdout_text.strip(),
                metadata={"runtime": RUNTIME_NAME, "agent_model": self.config.agent_model},
            ),
            evidence=CandidateEvidence(descriptors=descriptors, metadata={"runtime": RUNTIME_NAME}),
            measurements=TrialMeasurements(runtime_sec=runtime_sec),
            metadata={
                "runtime": RUNTIME_NAME,
                "agent_model": self.config.agent_model,
                "agent_ok": agent_ok,
                "exit_code": process.returncode,
                "run_dir": str(layout.run_dir),
                "generated": True,
            },
        )
        # Persist the trial next to its evidence so a single run dir can be
        # re-scored offline via TrialJsonSerde.
        TrialJsonSerde(layout.run_dir).write(trial)
        return trial

    def _failed_trial(self, task: AgentEvalTask, run_dir: Path, exc: Exception) -> AgentEvalTrial:
        return AgentEvalTrial(
            id=f"{task.id}:{RUNTIME_NAME}",
            task_id=task.id,
            status=AgentEvalTrialStatus.FAILED,
            output=None,
            metadata={
                "runtime": RUNTIME_NAME,
                "agent_ok": False,
                "run_dir": str(run_dir),
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "generated": True,
            },
        )

    def _format_command(self, instruction_path: Path, workspace_dir: Path, input_json: Path) -> list[str]:
        substitutions = {
            _INSTRUCTION_TOKEN: str(instruction_path),
            _WORKSPACE_TOKEN: str(workspace_dir),
            _INPUT_JSON_TOKEN: str(input_json),
        }
        return [substitutions.get(token, token) for token in self.config.command]

    def _run_dir(self, index: int, task: AgentEvalTask, config: AgentEvalRunConfig) -> Path:
        root = (config.work_dir or Path.cwd()) / "evidence" / RUNTIME_NAME
        return root / (_safe_name(task.id) or f"task-{index}")


class TrialJsonSerde:
    """Read/write one stored trial as ``<run_dir>/trial.json`` (an ``AgentTrialSerde``)."""

    def __init__(self, run_dir: str | Path) -> None:
        self._path = Path(run_dir) / "trial.json"

    def read(self) -> AgentEvalTrial:
        return AgentEvalTrial.model_validate_json(self._path.read_text(encoding="utf-8"))

    def write(self, trial: AgentEvalTrial) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(trial.model_dump_json(indent=2), encoding="utf-8")


def load_stored_trials(run_dir: str | Path) -> list[AgentEvalTrial]:
    """Load stored trial(s) from a run dir for offline rescoring.

    Accepts either a full run bundle (``trials.jsonl``) or a single runtime run
    dir holding one ``trial.json`` (read via :class:`TrialJsonSerde`).
    """
    run_dir = Path(run_dir)
    jsonl = run_dir / "trials.jsonl"
    if jsonl.exists():
        trials = [
            AgentEvalTrial.model_validate_json(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line
        ]
        if not trials:
            raise ValueError(f"trials.jsonl under {run_dir!r} is empty")
        return trials
    if (run_dir / "trial.json").exists():
        return [TrialJsonSerde(run_dir).read()]
    raise FileNotFoundError(f"no trials.jsonl or trial.json found under {run_dir!r}")


class OutputContainsMetric:
    """Task-authored metric: emit ``True`` when the agent output contains ``expected``."""

    def __init__(self, expected: str, *, output_name: str = "output_contains") -> None:
        self._expected = expected
        self._output_name = output_name

    @property
    def type(self) -> str:
        return "output_contains"

    def output_spec(self) -> list[MetricOutputSpec]:
        return [MetricOutputSpec.boolean(self._output_name)]

    async def compute_scores(self, input: MetricInput) -> MetricResult:
        text = input.candidate.output_text or ""
        present = self._expected in text
        return MetricResult(outputs=[MetricOutput(name=self._output_name, value=present)])


def example_tasks() -> list[AgentEvalTask]:
    """Two self-contained file-writing tasks wired to reusable + authored metrics."""
    from nemo_evaluator_sdk.agent_eval.metrics import AgentPhaseSuccessMetric, EvidencePresenceMetric

    def build(task_id: str, intent: str, create_file: str, content: str) -> AgentEvalTask:
        return AgentEvalTask(
            id=task_id,
            intent=intent,
            inputs={
                "instruction": f"{intent}\n\nWrite the file {create_file!r} into your workspace, then report.",
                "create_file": create_file,
                "content": content,
            },
            metrics=[
                AgentPhaseSuccessMetric(),
                EvidencePresenceMetric(),
                OutputContainsMetric(create_file),
            ],
        )

    return [
        build(
            "write-report",
            "Produce a status report file.",
            "report.txt",
            "status: green\nsummary: all systems nominal\n",
        ),
        build(
            "write-notes",
            "Capture a short notes file.",
            "notes.md",
            "# Notes\n\n- agent-eval trials example\n",
        ),
    ]


def tasks_by_id(task_names: Sequence[str]) -> list[AgentEvalTask]:
    """Select example tasks by id, preserving the requested order."""
    catalog = {task.id: task for task in example_tasks()}
    unknown = [name for name in task_names if name not in catalog]
    if unknown:
        raise ValueError(f"unknown task(s): {sorted(unknown)}; available: {sorted(catalog)}")
    return [catalog[name] for name in task_names]


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value).strip(".-")[:120]


__all__ = [
    "OutputContainsMetric",
    "TrialJsonSerde",
    "WorkflowAgentRuntime",
    "WorkflowRuntimeConfig",
    "example_tasks",
    "load_stored_trials",
    "tasks_by_id",
]

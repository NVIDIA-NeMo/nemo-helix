# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PipelineJob — run a fixed sequence of agent jobs, feeding each step's output to the next.

Registered under ``nemo.jobs`` as ``agents.pipeline``, serving ``/jobs/pipeline``.

This is a **stub**. It runs three hardcoded steps, each a real ``agents.execute`` job:

1. ``plan``: a PR planner agent turns the task description into a plan.
2. ``write``: a PR writer agent implements the planner's plan.
3. ``review``: a PR reviewer agent reviews the writer's output.

The pipeline job itself is a single-step orchestrator. Its step submits each execute job in
turn, waits for it to finish, and passes the previous step's text response to the next step.
It stops at the first step that does not complete.

TODO(pipeline owners): the step list, prompts and data flow below are placeholders. Replace
``PIPELINE_STEPS`` and ``_STEP_PROMPTS`` with the real definition when it lands.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from nemo_agents_plugin.jobs.execute import FABRIC_RUN_RESULT_NAME, ExecuteAgentJob
from nemo_agents_plugin.sdk import _ExecuteJobsResource
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
)
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

PIPELINE_RESULT_NAME = "pipeline_result"
PIPELINE_RESULT_FILENAME = "pipeline_result.json"

# Hardcoded for now; these become configuration once the pipeline config lands.
DEFAULT_PLANNER_AGENT = "pr-planner"
DEFAULT_WRITER_AGENT = "pr-writer"
DEFAULT_REVIEWER_AGENT = "pr-reviewer"

#: Job statuses (see ``nemo_helix_plugin.jobs.watch``) that end a step job's life.
_SUCCESS_STATUSES = frozenset({"completed"})
_FAILURE_STATUSES = frozenset({"cancelled", "error", "failed", "paused"})
_POLL_INTERVAL_SECONDS = 5.0
_STEP_TIMEOUT_SECONDS = 2 * 60 * 60


@dataclass(frozen=True, slots=True)
class PipelineStep:
    """One hardcoded stage: its role and the Agent entity that plays it."""

    role: str
    agent: str


PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    PipelineStep("plan", DEFAULT_PLANNER_AGENT),
    PipelineStep("write", DEFAULT_WRITER_AGENT),
    PipelineStep("review", DEFAULT_REVIEWER_AGENT),
)

# Prompt per role. ``{task}`` is the original task description; ``{previous}`` is the previous
# step's response (empty for the first step).
_STEP_PROMPTS: dict[str, str] = {
    "plan": "{task}",
    "write": ("Implement the following plan for this programming task.\n\nTask:\n{task}\n\nPlan:\n{previous}"),
    "review": (
        "Review the following change for this programming task. Point out bugs, missing cases and "
        "style problems.\n\nTask:\n{task}\n\nChange:\n{previous}"
    ),
}


class PipelineJobConfig(BaseModel):
    """Input for ``agents.pipeline``: the task description (the agents per role are hardcoded for now)."""

    model_config = {"json_schema_mode_override": "validation"}

    task: str = Field(description="Programming task description. It is the planner's input.")
    environment: str | None = Field(
        default=None,
        description='AgentEnvironment ("workspace/name" ref) every step runs under. Omit for none.',
    )

    @field_validator("task")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class PipelineStepFailedError(RuntimeError):
    """A step's execute job did not complete, or completed without a usable response."""


def build_step_input(role: str, *, task: str, previous: str) -> str:
    """Render the prompt for *role* from the task and the previous step's response."""
    return _STEP_PROMPTS[role].format(task=task, previous=previous)


def _response_text(run_result: Mapping[str, Any]) -> str:
    """Pull the agent's text answer out of a ``fabric_run_result`` payload."""
    response = run_result.get("response")
    if not isinstance(response, str):
        output = run_result.get("output")
        response = output.get("output") if isinstance(output, Mapping) else None
    return response.strip() if isinstance(response, str) else ""


class PipelineJob(NemoJob):
    name: ClassVar[str] = "pipeline"
    description: ClassVar[str] = "Run a PR planner, writer and reviewer agent in sequence as one platform job."
    container: ClassVar[str] = "cpu-tasks"
    # No separate input/spec split: the submitted input is already the stored spec.
    spec_schema: ClassVar[type[BaseModel] | None] = PipelineJobConfig

    @classmethod
    async def compile(
        cls,
        *,
        workspace: str,
        spec: BaseModel,
        entity_client: object,
        job_name: str | None,
        async_sdk: object,
        profile: str | None = None,
        options: dict | None = None,
    ) -> HelixJobSpec:
        """Single-step HelixJobSpec running ``nemo_agents_plugin.tasks.pipeline`` in a CPU container."""
        del workspace, entity_client, job_name, async_sdk, options
        if not isinstance(spec, PipelineJobConfig):
            raise HelixJobCompilationError(f"PipelineJob.compile received unexpected spec type: {type(spec).__name__}")
        executor = CPUExecutionProviderSpec(
            profile=profile or "default",
            provider="cpu",
            container=ContainerSpec(
                image=ExecuteAgentJob._execution_image(""),
                entrypoint=["python", "-m"],
                command=["nemo_agents_plugin.tasks.pipeline"],
            ),
        )
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="pipeline",
                    executor=executor,
                    config=spec.model_dump(mode="json"),
                )
            ],
        )

    def run(
        self,
        config: dict,
        *,
        ctx: JobContext,
        sdk: NemoClient | None = None,
        poll_interval: float = _POLL_INTERVAL_SECONDS,
        step_timeout: float = _STEP_TIMEOUT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict:
        """Submit each step's execute job in order, feeding each response to the next step."""
        if sdk is None:
            raise ValueError("PipelineJob requires a platform client to submit its agent jobs.")
        cfg = PipelineJobConfig.model_validate(config)
        execute_jobs = _ExecuteJobsResource(AgentsClient.from_client(sdk))

        steps: list[dict[str, Any]] = []
        previous = ""
        failure: str | None = None
        for step in PIPELINE_STEPS:
            agent = step.agent
            prompt = build_step_input(step.role, task=cfg.task, previous=previous)
            record: dict[str, Any] = {"role": step.role, "agent": agent, "job": None, "status": "pending"}
            steps.append(record)
            try:
                previous = self._run_step(
                    execute_jobs,
                    record,
                    workspace=ctx.workspace,
                    spec=self._execute_spec(agent, prompt, cfg.environment),
                    poll_interval=poll_interval,
                    step_timeout=step_timeout,
                    sleep=sleep,
                )
            except PipelineStepFailedError as exc:
                failure = f"Step '{step.role}' failed: {exc}"
                logger.error(failure)
                break
            record["response"] = previous

        status = "failed" if failure else "completed"
        payload: dict[str, Any] = {"status": status, "steps": steps}
        if failure:
            payload["error"] = failure
        self._save_result(ctx, payload)
        if failure:
            raise PipelineStepFailedError(failure)
        return payload

    @staticmethod
    def _execute_spec(agent: str, prompt: str, environment: str | None) -> dict[str, Any]:
        spec: dict[str, Any] = {"agent": agent, "input": prompt}
        if environment is not None:
            spec["environment"] = environment
        return spec

    @staticmethod
    def _run_step(
        execute_jobs: _ExecuteJobsResource,
        record: dict[str, Any],
        *,
        workspace: str,
        spec: dict[str, Any],
        poll_interval: float,
        step_timeout: float,
        sleep: Callable[[float], None],
    ) -> str:
        """Submit one execute job, wait for it, and return its response text."""
        created = execute_jobs.create(spec=spec, workspace=workspace)
        job_name = created.get("name")
        if not isinstance(job_name, str):
            raise PipelineStepFailedError("the Jobs service did not return a job name.")
        record["job"] = job_name
        logger.info("Pipeline step '%s' submitted execute job %s.", record["role"], job_name)

        deadline = time.monotonic() + step_timeout
        while True:
            job = execute_jobs.get(job_name, workspace=workspace)
            status = str(job.get("status", "")).lower()
            record["status"] = status
            if status in _SUCCESS_STATUSES:
                break
            if status in _FAILURE_STATUSES:
                raise PipelineStepFailedError(f"execute job {job_name} ended with status '{status}'.")
            if time.monotonic() >= deadline:
                raise PipelineStepFailedError(f"execute job {job_name} did not finish within {step_timeout:g}s.")
            sleep(poll_interval)

        raw = execute_jobs.download_result(FABRIC_RUN_RESULT_NAME, job=job_name, workspace=workspace)
        try:
            run_result = json.loads(raw)
        except ValueError as exc:
            raise PipelineStepFailedError(f"execute job {job_name} saved an unreadable run result.") from exc
        response = _response_text(run_result) if isinstance(run_result, Mapping) else ""
        if not response:
            raise PipelineStepFailedError(f"execute job {job_name} completed without a text response.")
        return response

    @staticmethod
    def _save_result(ctx: JobContext, payload: dict[str, Any]) -> None:
        path: Path = ctx.storage.ephemeral / PIPELINE_RESULT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        ctx.results.save(PIPELINE_RESULT_NAME, path)

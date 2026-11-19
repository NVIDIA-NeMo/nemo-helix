# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run configured agent pipelines as sequential execute jobs, with a legacy task-only fallback."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

from filesets import parse_fileset_ref
from nemo_agents_plugin.entities import AgentEnvironmentInline, AgentPipelineConfig
from nemo_agents_plugin.jobs.execute import FABRIC_RUN_RESULT_NAME, ExecuteAgentJob
from nemo_agents_plugin.sdk import _ExecuteJobsResource
from nemo_helix_plugin.agents import pipeline_template as tpl
from nemo_helix_plugin.agents.client import AgentsClient
from nemo_helix_plugin.agents.pipelines import (
    PipelineInline,
    parse_step_response,
    render_step,
    resolve_run_inputs,
    should_run,
)
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.entity_client import NemoEntitiesClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
)
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError
from nemo_helix_plugin.refs import parse_entity_ref
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

PIPELINE_RESULT_NAME = "pipeline_result"
PIPELINE_RESULT_FILENAME = "pipeline_result.json"

# Hardcoded for now; these become configuration once the pipeline config lands.
DEFAULT_PLANNER_AGENT = "pr-planner"
DEFAULT_WRITER_AGENT = "pr-writer"
DEFAULT_REVIEWER_AGENT = "pr-reviewer"

#: Job statuses (see ``nemo_helix_plugin.jobs.watch``) that end a step job's life.
_SUCCESS_STATUSES = frozenset({"completed"})
_FAILURE_STATUSES = frozenset({"cancelled", "error", "failed", "paused", "timed_out", "timeout"})
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


class PipelineStepRunOptions(BaseModel):
    """Execution settings supplied per run, separately from the reusable pipeline definition."""

    model_config = {"extra": "forbid"}
    image: str = ""
    environment: str | AgentEnvironmentInline | None = None
    profile: str | None = None
    timeout_seconds: float = Field(default=600, gt=0)
    wait_timeout_seconds: float = Field(
        default=1800, gt=0, description="Wait budget including scheduling and artifact upload."
    )


class PipelineJobConfig(BaseModel):
    """A stored or inline pipeline, run inputs, and independently configurable child jobs."""

    model_config = {"json_schema_mode_override": "validation", "extra": "forbid"}
    pipeline: str | PipelineInline | None = Field(
        default=None, description="Stored pipeline name/workspace ref, or inline definition."
    )
    inputs: dict[str, Any] | str = Field(default_factory=dict)
    image: str = Field(default="", description="Image for the pipeline orchestrator, containing this runner.")
    step_defaults: PipelineStepRunOptions = Field(default_factory=PipelineStepRunOptions)
    step_overrides: dict[str, PipelineStepRunOptions] = Field(default_factory=dict)
    task: str | None = Field(
        default=None, description="Legacy task-only PR pipeline; mutually exclusive with pipeline."
    )
    environment: str | None = Field(default=None, description="Legacy shared AgentEnvironment reference.")

    @model_validator(mode="after")
    def _validate_mode(self) -> PipelineJobConfig:
        if self.pipeline is None:
            if self.task is None or not self.task.strip():
                raise ValueError("Set pipeline, or a nonblank legacy task.")
            if self.inputs or self.step_overrides or self.step_defaults.model_fields_set:
                raise ValueError("Run inputs and step settings require pipeline.")
        else:
            if self.task is not None or self.environment is not None:
                raise ValueError(
                    "pipeline cannot be combined with legacy task/environment; use step_defaults.environment."
                )
            if isinstance(self.pipeline, str) and not self.pipeline.strip():
                raise ValueError("pipeline must not be blank")
            if isinstance(self.pipeline, PipelineInline):
                self.validate_pipeline(self.pipeline)
        return self

    def validate_pipeline(self, pipeline: PipelineInline) -> None:
        resolve_run_inputs(pipeline, self.inputs)
        unknown = set(self.step_overrides) - {step.name for step in pipeline.steps}
        if unknown:
            raise ValueError(f"Unknown step overrides: {', '.join(sorted(unknown))}")

    def options_for(self, name: str) -> PipelineStepRunOptions:
        values = self.step_defaults.model_dump()
        if name in self.step_overrides:
            values.update(self.step_overrides[name].model_dump(exclude_unset=True))
        return PipelineStepRunOptions.model_validate(values)


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
    description: ClassVar[str] = "Run a configured sequence of independent agent jobs with output handoffs."
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
        del job_name, async_sdk, options
        if not isinstance(spec, PipelineJobConfig):
            raise HelixJobCompilationError(f"PipelineJob.compile received unexpected spec type: {type(spec).__name__}")
        # Snapshot stored definitions at submission, while the submitting identity can read them.
        # Qualify agent refs against the pipeline's workspace, not the caller's workspace.
        compiled = spec.model_dump(mode="json", exclude_unset=True)
        if spec.pipeline is not None:
            pipeline_workspace = workspace
            if isinstance(spec.pipeline, str):
                ref = parse_entity_ref(spec.pipeline, default_workspace=workspace)
                entity = await cast(NemoEntitiesClient, entity_client).get(
                    AgentPipelineConfig, name=ref.name, workspace=ref.workspace
                )
                definition = {key: getattr(entity, key) for key in PipelineInline.model_fields}
                pipeline = PipelineInline.model_validate(definition).model_copy(deep=True)
                pipeline_workspace = ref.workspace
            else:
                pipeline = spec.pipeline.model_copy(deep=True)
            spec.validate_pipeline(pipeline)
            for step in pipeline.steps:
                ref = parse_entity_ref(step.agent, default_workspace=pipeline_workspace)
                step.agent = f"{ref.workspace}/{ref.name}"
            compiled["pipeline"] = pipeline.model_dump(mode="json", exclude_unset=True)
        executor = CPUExecutionProviderSpec(
            profile=profile or "default",
            provider="cpu",
            container=ContainerSpec(
                image=ExecuteAgentJob._execution_image(spec.image),
                entrypoint=["python", "-m"],
                command=["nemo_agents_plugin.tasks.pipeline"],
            ),
        )
        return HelixJobSpec(
            steps=[
                HelixJobStep(
                    name="pipeline",
                    executor=executor,
                    config=compiled,
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
        if cfg.pipeline is not None:
            return self._run_configured(
                cfg, ctx=ctx, sdk=sdk, execute_jobs=execute_jobs, poll_interval=poll_interval, sleep=sleep
            )

        steps: list[dict[str, Any]] = []
        previous = ""
        failure: str | None = None
        for step in PIPELINE_STEPS:
            agent = step.agent
            prompt = build_step_input(step.role, task=cast(str, cfg.task), previous=previous)
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
            record.pop("_output", None)
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
        profile: str | None = None,
        response_artifact: str | None = None,
        files_client: FilesClient | None = None,
    ) -> str:
        """Submit one execute job, wait for it, and return its response text."""
        kwargs = {"profile": profile} if profile is not None else {}
        created = execute_jobs.create(spec=spec, workspace=workspace, **kwargs)
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
                record["status"] = "timed_out"
                try:
                    execute_jobs.cancel(job_name, workspace=workspace)
                except Exception:
                    logger.exception("Could not cancel timed-out child job %s.", job_name)
                raise PipelineStepFailedError(f"execute job {job_name} did not finish within {step_timeout:g}s.")
            sleep(poll_interval)

        raw = execute_jobs.download_result(FABRIC_RUN_RESULT_NAME, job=job_name, workspace=workspace)
        try:
            run_result = json.loads(raw)
        except ValueError as exc:
            raise PipelineStepFailedError(f"execute job {job_name} saved an unreadable run result.") from exc
        if not isinstance(run_result, Mapping):
            raise PipelineStepFailedError(f"execute job {job_name} saved a non-object run result.")
        if run_result.get("status") in {"failed", "error", "cancelled"}:
            raise PipelineStepFailedError(f"execute job {job_name} returned a failed Fabric result.")
        record["_output"] = dict(run_result)
        if response_artifact is not None:
            if files_client is None:
                raise ValueError("Artifact responses require a Files client.")
            return PipelineJob._artifact_response(execute_jobs, files_client, job_name, workspace, response_artifact)
        response = _response_text(run_result)
        if not response:
            raise PipelineStepFailedError(f"execute job {job_name} completed without a text response.")
        return response

    def _run_configured(
        self,
        cfg: PipelineJobConfig,
        *,
        ctx: JobContext,
        sdk: NemoClient,
        execute_jobs: _ExecuteJobsResource,
        poll_interval: float,
        sleep: Callable[[float], None],
    ) -> dict:
        if not isinstance(cfg.pipeline, PipelineInline):
            raise ValueError("Stored pipeline references must be resolved by compile before task execution.")
        pipeline = cfg.pipeline
        context = resolve_run_inputs(pipeline, cfg.inputs)
        context["steps"] = {}
        records: list[dict[str, Any]] = []
        files_client = FilesClient.from_client(sdk)
        failure = None
        try:
            for index, step in enumerate(pipeline.steps):
                record: dict[str, Any] = {
                    "role": step.name,
                    "agent": step.agent,
                    "job": None,
                    "status": "pending",
                    "attempts": [],
                }
                records.append(record)
                if not should_run(step, context):
                    record["status"] = "skipped"
                    continue
                prompt, agent_context = render_step(pipeline, index, context)
                options = cfg.options_for(step.name)
                spec = options.model_dump(mode="json", exclude={"profile", "wait_timeout_seconds"}, exclude_none=True)
                spec.update(agent=step.agent, input=prompt, context=agent_context)
                for attempt in range(step.retries + 1):
                    current: dict[str, Any] = {"role": step.name, "job": None, "status": "pending"}
                    record["attempts"].append(current)
                    try:
                        text = self._run_step(
                            execute_jobs,
                            current,
                            workspace=ctx.workspace,
                            spec=spec,
                            poll_interval=poll_interval,
                            step_timeout=options.wait_timeout_seconds,
                            sleep=sleep,
                            profile=options.profile,
                            response_artifact=step.response_artifact,
                            files_client=files_client,
                        )
                    except PipelineStepFailedError as exc:
                        current["error"] = str(exc)
                        # Retry only known terminal errors. Do not duplicate publication after
                        # an ambiguous submission, a timeout, or a malformed successful output.
                        if attempt < step.retries and current["status"] in {"error", "failed"}:
                            continue
                        raise
                    finally:
                        record["job"] = current["job"]
                        record["status"] = current["status"]
                    response = parse_step_response(step, text)
                    output = current.pop("_output", {})
                    record["response"] = response
                    if step.expect is not None and not step.expect.matches(
                        tpl.resolve(tpl.parse_path(step.expect.path), {"response": response})
                    ):
                        raise PipelineStepFailedError(f"output check {step.expect.path!r} did not match")
                    value = {"response": response, "output": output}
                    context["steps"][step.name] = value
                    context["previous"] = value
                    record["status"] = "completed"
                    break
        except BaseException as exc:
            if (
                records
                and records[-1]["job"]
                and records[-1]["status"] not in _SUCCESS_STATUSES | _FAILURE_STATUSES | {"timed_out"}
            ):
                try:
                    execute_jobs.cancel(records[-1]["job"], workspace=ctx.workspace)
                except Exception:
                    logger.exception("Could not cancel interrupted pipeline child.")
            name = records[-1]["role"] if records else "setup"
            failure = f"Step '{name}' failed: {exc}"
            if records:
                records[-1]["status"] = "failed"
                records[-1]["error"] = str(exc)
            logger.error(failure)
            if not isinstance(exc, Exception):
                raise
        finally:
            # Persist a useful partial result even on task cancellation; never persist the
            # raw Fabric result twice (it remains in the child job's named results).
            for record in records:
                for attempt in record["attempts"]:
                    attempt.pop("_output", None)
            payload = {"status": "failed" if failure else "completed", "steps": records}
            if failure:
                payload["error"] = failure
            self._save_result(ctx, payload)
        if failure:
            raise PipelineStepFailedError(failure)
        return payload

    @staticmethod
    def _artifact_response(
        execute_jobs: _ExecuteJobsResource,
        files_client: FilesClient,
        job: str,
        workspace: str,
        path: str,
    ) -> str:
        # Download the selected file, not the potentially large repository workdir archive.
        results = execute_jobs.list_results(job, workspace=workspace)
        result = next((item for item in results.get("data", []) if item.get("name") == "output_workdir"), None)
        if result is None or result.get("artifact_storage_type") != "fileset":
            raise PipelineStepFailedError(f"execute job {job} has no fileset output_workdir")
        ws, fileset, prefix = parse_fileset_ref(result["artifact_url"], workspace_fallback=workspace)
        response = files_client.download_file(workspace=ws, name=fileset, path=f"{prefix.rstrip('/')}/{path}")
        content = bytearray()
        with response.stream(chunk_size=65536) as chunks:
            for chunk in chunks:
                content.extend(chunk)
                if len(content) > 4 * 1024 * 1024:
                    raise PipelineStepFailedError(f"Response artifact {path!r} exceeds 4 MiB")
        try:
            return content.decode("utf-8")
        except UnicodeError as exc:
            raise PipelineStepFailedError(f"Response artifact {path!r} is not UTF-8 text") from exc

    @staticmethod
    def _save_result(ctx: JobContext, payload: dict[str, Any]) -> None:
        path: Path = ctx.storage.ephemeral / PIPELINE_RESULT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        ctx.results.save(PIPELINE_RESULT_NAME, path)

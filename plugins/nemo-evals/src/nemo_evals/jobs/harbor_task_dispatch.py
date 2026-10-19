# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One Harbor task of a taskset run, as one Jobs Service Job running on OpenSandbox.

``HarborTaskDispatchJob.compile`` (M2 story 6a) turns a ``HarborTaskJobSpec`` into the Job;
``HarborTaskAgentEvalJob`` is what the dispatcher entrypoint runs inside it: today's agent evaluation
over the one task, with its images bound to the prepared digests and Harbor's environment set to
``NemoOpenSandboxEnvironment``. M2 story 6b replaces the run with its own dispatcher.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from nemo_evals.jobs.agent_compiler import _environment_stage_step, _secret_refs, _staged_fileset
from nemo_evals.jobs.agent_evaluate import AsyncAgentEvalJob
from nemo_evals.jobs.harbor_task_spec import HarborTaskJobSpec
from nemo_evals.jobs.secret_env import build_task_environment
from nemo_evals.taskset_runs.identifiers import task_job_name
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import ContainerSpec, CPUExecutionProviderSpec, HelixJobSpec, HelixJobStep
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborEnvironmentConfig, HarborRuntimeConfig
from nhx_evals_sdk.agent_eval.runtimes.harbor.tasks import HARBOR_TASK_DIR_KEY
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from pydantic import BaseModel

TASK_JOB_SOURCE = "evals.harbor-task"
"""Jobs ``source`` of every task Job; with the name, spec and platform spec it marks a Job as ours."""

DISPATCH_STEP_NAME = "harbor-task-dispatch"
DISPATCH_ENTRYPOINT = ["python", "-m"]
DISPATCH_COMMAND = ["nemo_evals.tasks.harbor_task_dispatch"]
OPENSANDBOX_ENVIRONMENT = "nemo_evals.harbor.opensandbox_environment:AgentChannelOpenSandboxEnvironment"
"""Harbor environment the task Job runs trials in: scaled-evals' environment plus the agent channel."""

SANDBOX_READY_TIMEOUT_S = 600
COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml")


class ExecutionDeadlinePassedError(RuntimeError):
    """The run's execution deadline passed before this task Job started."""


class SandboxServerChangedError(RuntimeError):
    """The platform now points at another OpenSandbox server than the run was accepted for."""


class HarborTaskDispatchJob(NemoJob):
    """Compile one task of a taskset run into its Job. Not registered under ``nemo.jobs``: no route, no CLI."""

    name: ClassVar[str] = "harbor-task-dispatch"
    description: ClassVar[str] = "Run one Harbor task of a taskset run on OpenSandbox."
    spec_schema: ClassVar[type[BaseModel] | None] = HarborTaskJobSpec
    input_spec_schema: ClassVar[type[BaseModel] | None] = HarborTaskJobSpec

    @classmethod
    async def compile(cls, *, spec: HarborTaskJobSpec, profile: str, **_: Any) -> HelixJobSpec:  # ty: ignore[invalid-method-override]
        """A staging step for a registered agent's files, if any, then the dispatcher step; both on the dispatcher image.

        No OpenSandbox variables: Jobs injects OPEN_SANDBOX_DOMAIN and OPEN_SANDBOX_API_KEY into every pod.
        """
        steps: list[HelixJobStep] = []
        staged = _staged_fileset(spec.spec)
        if staged is not None:
            steps.append(_environment_stage_step(staged, profile, use_subprocess=False, image=spec.dispatcher_image))
        steps.append(
            HelixJobStep(
                name=DISPATCH_STEP_NAME,
                executor=CPUExecutionProviderSpec(
                    profile=profile,
                    provider="cpu",
                    container=ContainerSpec(
                        image=spec.dispatcher_image, entrypoint=DISPATCH_ENTRYPOINT, command=DISPATCH_COMMAND
                    ),
                ),
                config=spec.model_dump(mode="json"),
                environment=build_task_environment(_secret_refs(spec.spec)),
            )
        )
        return HelixJobSpec(steps=steps)


class HarborTaskAgentEvalJob(AsyncAgentEvalJob):
    """Today's agent evaluation over one task of a run, in OpenSandbox sandboxes.

    - Fails before any sandbox if the run's execution deadline has passed: the controller only cancels the
      Jobs it sees when it expires the run, so a Job a submit retry creates after that must stop itself.
    - Fails before any sandbox if the platform's OpenSandbox server isn't the one the run was accepted for.
    - Binds the materialized task to its prepared image digests.
    - Never deletes the agent-files snapshot: sibling task Jobs and Jobs retries still stage it.
    - Runs under its own name as the SDK run id, so trials it publishes to the run's Intake Evaluation
      have stable sessions (``<run_id>-task-<i>:<trial>``) and a re-publish replaces them.
    - Never writes the shared Evaluation's metadata: the run controller stamps the run's duration once.
    """

    discard_agent_files: ClassVar[bool] = False
    record_publication_durations: ClassVar[bool] = False
    _task_spec: HarborTaskJobSpec

    def run(self, config: dict, *, ctx: JobContext, async_client: AsyncNemoClient) -> dict:
        """Run the task Job's one task and persist its result bundle."""
        spec = HarborTaskJobSpec.model_validate(config)
        check_execution_deadline(spec)
        check_sandbox_server(spec)
        self._task_spec = spec
        return self._run_with_client(
            spec.spec.model_dump(mode="json"), ctx=ctx, platform_client=async_client, async_client=async_client
        )

    def _run_id(self) -> str | None:
        return task_job_name(self._task_spec.run_id, self._task_spec.task_index)

    def _prepared_tasks(self, tasks: list[AgentEvalTask]) -> None:
        environment = self._task_spec.environment
        verifier = environment.verifier.image_ref if environment.verifier is not None else None
        for task in tasks:
            task_dir = Path(str(task.metadata[HARBOR_TASK_DIR_KEY]))
            bind_task_image(task_dir, environment.agent.image_ref, verifier_image_ref=verifier)

    def _configure_harbor_runtime(self, config: HarborRuntimeConfig, ctx: JobContext) -> HarborRuntimeConfig:
        return configure_runtime(config, ctx, self._task_spec)


def check_execution_deadline(spec: HarborTaskJobSpec, now: datetime | None = None) -> None:
    """Fail with ``execution_deadline_passed`` when ``now`` (default: current UTC) is past the run's deadline."""
    now = now or datetime.now(timezone.utc)
    deadline = spec.execution_deadline_at
    if deadline.utcoffset() is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    if now >= deadline:
        raise ExecutionDeadlinePassedError(
            f"execution_deadline_passed: run {spec.run_id} passed its execution deadline "
            f"{deadline.isoformat()} before task {spec.task_index} started"
        )


def check_sandbox_server(spec: HarborTaskJobSpec) -> None:
    """Fail with ``sandbox_server_changed`` when the injected server isn't the plan's.

    Jobs injects the API key of the live platform config; calling the plan's server with another
    server's key would fail later and less clearly.
    """
    live = os.environ.get("OPEN_SANDBOX_DOMAIN") or os.environ.get("OPENSANDBOX_DOMAIN")
    planned = spec.sandbox_policy.connection.domain
    if live != planned:
        raise SandboxServerChangedError(
            f"sandbox_server_changed: the run was accepted for OpenSandbox server {planned!r}, "
            f"but this Job was given {live!r}; resubmit the run"
        )


def configure_runtime(config: HarborRuntimeConfig, ctx: JobContext, spec: HarborTaskJobSpec) -> HarborRuntimeConfig:
    """Select ``NemoOpenSandboxEnvironment`` with the run's frozen policy, labels and one trial at a time."""
    policy = spec.sandbox_policy
    connection = policy.connection
    environment = HarborEnvironmentConfig(
        import_path=OPENSANDBOX_ENVIRONMENT,
        kwargs={
            "domain": connection.domain,
            "protocol": connection.protocol,
            "use_server_proxy": connection.use_server_proxy,
            "ready_timeout_sec": SANDBOX_READY_TIMEOUT_S,
            "request_timeout_sec": connection.request_timeout_s,
            "sandbox_timeout_sec": policy.max_sandbox_lifetime_s,
            "trusted_allowed_hosts": [*policy.egress_allow, *policy.request_egress_allow],
            "agent_channel_hosts": list(policy.agent_channel),
            "egress_verification": policy.egress_verification,
            "metadata": {
                "nemo-run": spec.run_id,
                "nemo-task-index": str(spec.task_index),
                "nemo-job": ctx.job_id or spec.run_id,
            },
        },
    )
    return config.model_copy(update={"environment": environment, "n_concurrent_trials": 1})


def bind_task_image(task_dir: Path, image_ref: str, *, verifier_image_ref: str | None = None) -> bool:
    """Point a materialized task at its prepared images; OpenSandbox never builds images.

    ``verifier_image_ref`` is bound only when the task runs its verifier separately; any image the
    task names for the verifier is replaced. Returns whether it was bound. Ported from scaled-evals.

    Raises:
        ValueError: The task is a compose task, or its task.toml tables are malformed.
    """
    import tomlkit  # the cpu-tasks group installs it in the dispatcher image
    from tomlkit.items import InlineTable, Table

    if any((task_dir / "environment" / name).is_file() for name in COMPOSE_FILENAMES):
        raise ValueError("compose tasks are not supported in the MVP")
    task_toml = task_dir / "task.toml"
    document = tomlkit.parse(task_toml.read_text(encoding="utf-8"))

    environment = document.get("environment")
    if environment is None:
        environment = tomlkit.table()
        document["environment"] = environment
    if not isinstance(environment, MutableMapping):
        raise ValueError(f"task [environment] must be a TOML table: {task_toml}")
    environment["docker_image"] = image_ref

    verifier_bound = verifier_image_ref is not None and uses_separate_verifier(document)
    if verifier_bound:
        verifier = document.get("verifier")
        if not isinstance(verifier, MutableMapping):
            raise ValueError(f"task [verifier] must be a TOML table: {task_toml}")
        verifier_environment = verifier.get("environment")
        if verifier_environment is None:
            # Without [verifier.environment], Harbor verifies in a copy of [environment]; make it explicit.
            verifier_environment = tomlkit.table()
            plain = environment.unwrap() if isinstance(environment, Table | InlineTable) else dict(environment)
            for key, value in plain.items():
                verifier_environment[key] = value
            verifier["environment"] = verifier_environment
        if not isinstance(verifier_environment, MutableMapping):
            raise ValueError(f"task [verifier.environment] must be a TOML table: {task_toml}")
        verifier_environment["docker_image"] = verifier_image_ref

    temporary = task_toml.with_suffix(".toml.tmp")
    temporary.write_text(tomlkit.dumps(document), encoding="utf-8")
    os.replace(temporary, task_toml)
    return verifier_bound


def uses_separate_verifier(document: Mapping[str, Any]) -> bool:
    """True when any verify pass of the task runs in its own sandbox, as Harbor 0.20 resolves it."""
    task_mode = _verifier_mode(document.get("verifier")) or "shared"
    steps = document.get("steps") or []
    if not steps:
        return task_mode == "separate"
    for step in steps:
        step_verifier = step.get("verifier") if isinstance(step, Mapping) else None
        if (_verifier_mode(step_verifier) or task_mode) == "separate":
            return True
    return False


def _verifier_mode(verifier: object) -> str | None:
    if not isinstance(verifier, Mapping):
        return None
    mode = verifier.get("environment_mode")
    if mode is not None:
        return str(mode)
    return "separate" if verifier.get("environment") is not None else None

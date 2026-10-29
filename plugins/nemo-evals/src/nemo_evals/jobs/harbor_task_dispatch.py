# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One Harbor task of a taskset run, as one Jobs Service Job running on OpenSandbox.

``HarborTaskDispatchJob.compile`` (M2 story 6a) turns a ``HarborTaskJobSpec`` into the Job;
``HarborTaskAgentEvalJob`` is what the dispatcher entrypoint runs inside it: today's agent evaluation
over the one task, with its images bound to the prepared digests and Harbor's environment set to
``NemoOpenSandboxEnvironment``.

TODO(M2 story 6b): replace the agent evaluation with a dedicated dispatcher.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from nemo_evals.api.task_definitions.harbor import ResolvedHarborTaskDefinition
from nemo_evals.harbor.agent_setup_egress import (
    INNER_AGENT_KWARG,
    SETUP_EGRESS_AGENT,
    SETUP_HOSTS_KWARG,
    setup_timeout_multiplier,
)
from nemo_evals.harbor.compose.models import ComposeError
from nemo_evals.harbor.compose.spec import compose_services_kwarg
from nemo_evals.harbor.remote.profile import resource_defaults
from nemo_evals.jobs.agent_compiler import _environment_stage_step, _secret_refs, _staged_fileset
from nemo_evals.jobs.agent_evaluate import AsyncAgentEvalJob
from nemo_evals.jobs.gateway_models import (
    AGENT_CHANNEL_HOSTS_KWARG,
    add_gateway_key_placeholder,
    check_tasks_reach_the_gateway,
)
from nemo_evals.jobs.harbor_task_spec import HarborTaskJobSpec, PreparedHarborEnvironment
from nemo_evals.jobs.secret_env import build_task_environment
from nemo_evals.taskset_runs.identifiers import task_job_name
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from nemo_helix_plugin.jobs.api_factory import ContainerSpec, CPUExecutionProviderSpec, HelixJobSpec, HelixJobStep
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborEnvironmentConfig, HarborRuntimeConfig
from nhx_evals_sdk.agent_eval.runtimes.harbor.tasks import HARBOR_TASK_DIR_KEY
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask
from nhx_sandbox.compose_env import externalize_env
from pydantic import BaseModel

logger = logging.getLogger(__name__)

TASK_JOB_SOURCE = "evals.harbor-task"
"""Jobs ``source`` of every task Job; with the name, spec and platform spec it marks a Job as ours."""

DISPATCH_STEP_NAME = "harbor-task-dispatch"
DISPATCH_ENTRYPOINT = ["python", "-m"]
DISPATCH_COMMAND = ["nemo_evals.tasks.harbor_task_dispatch"]
OPENSANDBOX_ENVIRONMENT = "nemo_evals.harbor.opensandbox_environment:AgentChannelOpenSandboxEnvironment"
UNWRAPPED_AGENTS = frozenset({"oracle", "nop"})
"""Agents that install nothing, and that Harbor constructs or scores differently by name, so they stay unwrapped."""
"""Harbor environment the task Job runs trials in: nhx-sandbox's environment plus the agent channel."""

SANDBOX_READY_TIMEOUT_S = 600


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
                environment=add_gateway_key_placeholder(
                    build_task_environment(_secret_refs(spec.spec)), spec.spec.target
                ),
            )
        )
        return HelixJobSpec(steps=steps)


class HarborTaskAgentEvalJob(AsyncAgentEvalJob):
    """Today's agent evaluation over one task of a run, in OpenSandbox sandboxes.

    - Fails before any sandbox if the run's execution deadline has passed: the controller only cancels the
      Jobs it sees when it expires the run, so a Job a submit retry creates after that must stop itself.
    - Fails before any sandbox if the platform's OpenSandbox server isn't the one the run was accepted for.
    - Fails before any sandbox if a no-network task's sandbox couldn't reach the gateway model its agent calls.
    - Binds the materialized task to its prepared image digests.
    - Never deletes the agent-files snapshot: sibling task Jobs and Jobs retries still stage it.
    - Runs under its own name as the SDK run id, so trials it publishes to the run's Intake Evaluation
      have stable sessions (``<run_id>-task-<i>:<trial>``) and a re-publish replaces them.
    - Never writes the shared Evaluation's metadata: the run controller stamps the run's duration once.
    """

    discard_agent_files: ClassVar[bool] = False
    record_publication_durations: ClassVar[bool] = False
    _task_spec: HarborTaskJobSpec
    # The task's ``compose_services`` with its env values as templates, set by ``_prepared_tasks`` for
    # ``_configure_harbor_runtime``.
    _compose_services: dict[str, Any] | None = None

    def run(self, config: dict, *, ctx: JobContext, async_client: AsyncNemoClient) -> dict:
        """Run the task Job's one task and persist its result bundle."""
        spec = HarborTaskJobSpec.model_validate(config)
        check_execution_deadline(spec)
        check_sandbox_server(spec)
        check_tasks_reach_the_gateway(
            spec.spec.target,
            [(t.id, t.spec.config) for t in spec.spec.tasks if isinstance(t.spec, ResolvedHarborTaskDefinition)],
            spec.sandbox_policy.agent_channel,
        )
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
            # Derive the services before binding: binding sets docker_image, which would make the parser drop
            # main's ENTRYPOINT when it has to guess it from the Dockerfile.
            services = prepared_compose_services(task_dir, environment)
            # The services' variables go into this process's environment and the config keeps templates: Harbor
            # persists the config, and the values are the task's own, not credentials the caller passed.
            self._compose_services = externalize_env(services, os.environ) if services is not None else None
            bind_task_image(task_dir, environment.agent.image_ref, verifier_image_ref=verifier)

    def _configure_harbor_runtime(self, config: HarborRuntimeConfig, ctx: JobContext) -> HarborRuntimeConfig:
        return configure_runtime(config, ctx, self._task_spec, self._compose_services)


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


def prepared_compose_services(task_dir: Path, environment: PreparedHarborEnvironment) -> dict[str, Any] | None:
    """The task's ``compose_services`` kwarg with each service on the image the run prepared for it; None without any.

    When Builder recorded how ``main``'s image starts, the sandbox restores its ``ENTRYPOINT`` from that (see
    :func:`image_entrypoints`), so the spec only carries an ``entrypoint`` the compose file sets. Otherwise the
    parser guesses it from ``main``'s Dockerfile, so call this before :func:`bind_task_image`: binding sets
    ``docker_image``, which makes the parser skip the Dockerfile.

    Raises:
        ComposeError: The task's compose file can't run as a pod, or ``main`` sets a command whose image's
            ``ENTRYPOINT`` isn't known.
        ValueError: The prepared images are not for the task's services.
    """
    images = {name: image.image_ref for name, image in environment.compose_images.items()}
    spec = compose_services_kwarg(task_dir, images, read_entrypoint=environment.agent.runtime_config is None)

    main = spec.get("main", {}) if spec is not None else {}
    if main.get("command") and "entrypoint" not in main and environment.agent.runtime_config is None:
        # The command runs after the image's ENTRYPOINT, which isn't known for an image Builder didn't record.
        raise ComposeError(
            "compose_entrypoint_unknown",
            "`main` sets a command, which runs after its image's ENTRYPOINT, but Builder built the image before it "
            "recorded image ENTRYPOINTs, so it can't be started as Harbor would start it",
            compose_path=f"{task_dir.name}/environment/docker-compose.yaml",
        )

    services = {service["name"] for service in spec.get("services", [])} if spec is not None else set()
    if services != set(images):
        raise ValueError(
            f"the task's compose services {sorted(services)} are not the services the run prepared images for "
            f"{sorted(images)}"
        )
    return spec


def image_entrypoints(environment: PreparedHarborEnvironment) -> dict[str, list[str]]:
    """Each prepared sandbox image's own ``ENTRYPOINT``, by reference, for the images Builder recorded it for.

    OpenSandbox replaces a container's entrypoint, so the sandbox restores its image's from here. An image that has
    none maps to an empty list; one Builder didn't record is left out, and starts as it did before.
    """
    images = [environment.agent, *([environment.verifier] if environment.verifier is not None else [])]
    return {
        image.image_ref: list(image.runtime_config.entrypoint or [])
        for image in images
        if image.runtime_config is not None
    }


def configure_runtime(
    config: HarborRuntimeConfig,
    ctx: JobContext,
    spec: HarborTaskJobSpec,
    compose_services: dict[str, Any] | None = None,
) -> HarborRuntimeConfig:
    """Select ``NemoOpenSandboxEnvironment`` with the run's frozen policy, labels and one trial at a time.

    The request's resource overrides size the agent sandbox only; a separate verifier keeps task.toml's
    resources. Every sandbox gets ``resource_defaults(policy)`` for unset cpus and memory_mb, the values
    submit checked against the profile's ceilings. No Harbor resource mode is set, so every value is a limit.

    No sandbox outlives the run's execution deadline: the environment cuts each sandbox's lifetime to the time
    left when it creates it, so a later one (a retry, a separate verifier) ends then too. A dispatcher that is
    cancelled, OOM-killed or evicted leaks its sandboxes only until the deadline, not for the profile's whole
    ``max_sandbox_lifetime_s``. TODO: the controller deletes them on stop.

    A compose task's services go to the environment as ``compose_services``, and each image's own ``ENTRYPOINT``
    as ``image_entrypoints``.
    """
    policy = spec.sandbox_policy
    deadline = spec.execution_deadline_at
    if deadline.utcoffset() is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    connection = policy.connection
    kwargs: dict[str, Any] = {
        "domain": connection.domain,
        "protocol": connection.protocol,
        "use_server_proxy": connection.use_server_proxy,
        "ready_timeout_sec": SANDBOX_READY_TIMEOUT_S,
        "request_timeout_sec": connection.request_timeout_s,
        "sandbox_timeout_sec": policy.max_sandbox_lifetime_s,
        "sandbox_expires_at": deadline.isoformat(),
        "trusted_allowed_hosts": [*policy.egress_allow, *policy.request_egress_allow],
        AGENT_CHANNEL_HOSTS_KWARG: list(policy.agent_channel),
        "agent_setup_hosts": list(policy.agent_setup_egress),
        "egress_verification": policy.egress_verification,
        "enforce_disk_limit": policy.enforce_disk_limit,
        "max_disk_mb": policy.max_disk_mb,
        "agent_resource_overrides": policy.resource_overrides.model_dump(exclude_none=True),
        "sandbox_resource_defaults": resource_defaults(policy),
        "metadata": {
            "nemo-run": spec.run_id,
            "nemo-task-index": str(spec.task_index),
            "nemo-job": ctx.job_id or spec.run_id,
        },
    }
    if compose_services is not None:
        kwargs["compose_services"] = compose_services
    if entrypoints := image_entrypoints(spec.environment):
        kwargs["image_entrypoints"] = entrypoints
    environment = HarborEnvironmentConfig(import_path=OPENSANDBOX_ENVIRONMENT, kwargs=kwargs)
    update: dict[str, Any] = {"environment": environment, "n_concurrent_trials": 1}
    agent = config.agent_import_path or config.agent_name or "oracle"
    if policy.agent_setup_egress and (config.agent_dir is not None or agent in UNWRAPPED_AGENTS):
        reason = "a loose agent file (agent_dir)" if config.agent_dir is not None else "it installs nothing"
        logger.info("Agent %s runs setup without the agent setup egress window: %s", agent, reason)
    elif policy.agent_setup_egress:
        update["agent_import_path"] = SETUP_EGRESS_AGENT
        update["agent_kwargs"] = {
            **config.agent_kwargs,
            INNER_AGENT_KWARG: agent,
            SETUP_HOSTS_KWARG: list(policy.agent_setup_egress),
        }
        # Harbor falls back to the global multiplier, then 1.0, when the setup one is unset.
        multiplier = config.agent_setup_timeout_multiplier
        if multiplier is None:
            multiplier = config.timeout_multiplier if config.timeout_multiplier is not None else 1.0
        update["agent_setup_timeout_multiplier"] = setup_timeout_multiplier(agent, multiplier)
    return config.model_copy(update=update)


def bind_task_image(task_dir: Path, image_ref: str, *, verifier_image_ref: str | None = None) -> bool:
    """Point a materialized task at its prepared images; OpenSandbox never builds images.

    ``verifier_image_ref`` is bound only when the task runs its verifier separately; any image the
    task names for the verifier is replaced. Returns whether it was bound. Ported from scaled-evals.

    Raises:
        ValueError: The task's task.toml tables are malformed.
    """
    import tomlkit  # the cpu-tasks group installs it in the dispatcher image
    from tomlkit.items import InlineTable, Table

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

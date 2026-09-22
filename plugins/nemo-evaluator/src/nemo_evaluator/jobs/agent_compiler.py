# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-native agent-evaluation job compiler.

Parallels :mod:`nemo_evaluator.jobs.compiler` (row/model eval), emitting an
``agent-evaluate`` step in the platform task environment and, for FileSet-backed
Gym targets, a preceding ``stage-environment`` step. Standard and sandboxed
targets use ``cpu-tasks``; colocated Gym targets use ``gym-tasks``. Metric/endpoint
secrets are surfaced as ``from_secret`` environment variables; an agent *runner*
target (e.g. Fabric) carries no endpoint secret of its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from nemo_evaluator.config import config, platform_config
from nemo_evaluator.jobs.agent_spec import AgentEvalSpec, AgentTarget, GymRunnerTarget, HarborRunnerTarget, ModelTarget
from nemo_evaluator.jobs.environment_stage import EnvironmentStageSpec
from nemo_evaluator.jobs.gym_sandbox import GYM_SANDBOX_PLAN_ENVVAR, SandboxPlan, resolve_sandbox_plan
from nemo_evaluator.jobs.secret_env import build_task_environment
from nemo_platform_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    EnvironmentVariable,
    PlatformJobSpec,
    PlatformJobStep,
    SubprocessExecutionProviderSpec,
)
from nemo_platform_plugin.jobs.image import get_qualified_image

AGENT_EVAL_STEP_NAME = "agent-evaluate"

#: Container wiring for agent-evaluate steps, run via ``python -m``. Colocated Gym targets use a
#: dedicated image because NeMo Gym requires Ray. Sandboxed Gym targets only orchestrate the separate
#: Gym host, so they use the shared CPU task image.
AGENT_EVAL_IMAGE = "nmp-cpu-tasks"
GYM_AGENT_EVAL_IMAGE = "nmp-gym-tasks"
AGENT_EVAL_ENTRYPOINT = ["python", "-m"]
GYM_AGENT_EVAL_ENTRYPOINT = ["/app/.venv/bin/python", "-m"]
AGENT_EVAL_COMMAND = ["nemo_evaluator.tasks.agent_evaluate"]
ENVIRONMENT_STAGE_STEP_NAME = "stage-environment"
ENVIRONMENT_STAGE_COMMAND = ["nemo_evaluator.tasks.stage_environment"]


@dataclass(frozen=True)
class _AgentEvalCPUStep:
    step: PlatformJobStep
    executor: CPUExecutionProviderSpec


@dataclass(frozen=True)
class _AgentEvalCPUCompilation:
    platform_spec: PlatformJobSpec
    eval_step: PlatformJobStep
    executor: CPUExecutionProviderSpec


def compile_agent_eval_job(
    spec: AgentEvalSpec,
    *,
    profile: str | None = None,
    use_subprocess: bool = False,
) -> PlatformJobSpec:
    """Compile a canonical agent-evaluation spec into a plugin-native platform job."""
    if not use_subprocess:
        return _compile_agent_eval_cpu_job(spec, profile=profile).platform_spec

    sandbox_plan = _sandbox_plan(spec)
    steps = []
    # FileSet environments are downloaded onto job storage; that step must finish before the
    # Gym host mounts the same tree read-only.
    if isinstance(spec.target, GymRunnerTarget) and spec.target.environment is not None:
        steps.append(_environment_stage_step(spec.target, profile, use_subprocess=use_subprocess))

    steps.append(
        _agent_eval_step(
            spec,
            profile,
            use_subprocess=use_subprocess,
            sandbox_plan=sandbox_plan,
        )
    )

    return PlatformJobSpec(steps=steps)


def _compile_agent_eval_cpu_job(
    spec: AgentEvalSpec,
    *,
    profile: str | None = None,
) -> _AgentEvalCPUCompilation:
    """Compile agent evaluation with a CPU executor and return the final step's executor."""
    sandbox_plan = _sandbox_plan(spec)
    steps = []
    if isinstance(spec.target, GymRunnerTarget) and spec.target.environment is not None:
        steps.append(_environment_stage_step(spec.target, profile, use_subprocess=False))

    eval_step = _agent_eval_cpu_step(spec, profile, sandbox_plan=sandbox_plan)
    steps.append(eval_step.step)
    return _AgentEvalCPUCompilation(
        platform_spec=PlatformJobSpec(steps=steps),
        eval_step=eval_step.step,
        executor=eval_step.executor,
    )


def _secret_refs(spec: AgentEvalSpec) -> Iterator[tuple[str, str]]:
    """Yield ``(env_name, secret_name)`` for each metric secret and the endpoint target's api key."""
    for task in spec.tasks:
        for bundle in task.metrics:
            for env_name, secret_ref in bundle.secrets.items():
                yield env_name, secret_ref.root

    # A runner target may need credentials of its own -- a Gym environment's or Harbor agent's model
    # API key reaches it through the OS environment, not through an endpoint spec.
    if isinstance(spec.target, (GymRunnerTarget, HarborRunnerTarget)):
        for env_name, secret_ref in spec.target.env_secrets.items():
            yield env_name, secret_ref.root

    endpoint = None
    if isinstance(spec.target, ModelTarget):
        endpoint = spec.target.model
    elif isinstance(spec.target, AgentTarget):
        endpoint = spec.target.agent
    if endpoint is not None and endpoint.api_key_secret is not None and endpoint.api_key_env:
        yield endpoint.api_key_env, endpoint.api_key_secret.root


def _agent_eval_step(
    spec: AgentEvalSpec,
    profile: str | None,
    *,
    use_subprocess: bool,
    sandbox_plan: SandboxPlan | None,
) -> PlatformJobStep:
    """Build the evaluation step, including the sandbox plan when Gym runs sandboxed."""
    if not use_subprocess:
        return _agent_eval_cpu_step(spec, profile, sandbox_plan=sandbox_plan).step

    is_gym_target = isinstance(spec.target, GymRunnerTarget)
    is_colocated_gym = is_gym_target and sandbox_plan is None
    image = (
        config.gym_tasks_image
        if is_colocated_gym and config.gym_tasks_image is not None
        else get_qualified_image(GYM_AGENT_EVAL_IMAGE if is_colocated_gym else AGENT_EVAL_IMAGE)
    )
    executor = _executor(
        profile=profile,
        image=image,
        entrypoint=GYM_AGENT_EVAL_ENTRYPOINT if is_colocated_gym else AGENT_EVAL_ENTRYPOINT,
        command=AGENT_EVAL_COMMAND,
        use_subprocess=use_subprocess,
    )
    return PlatformJobStep(
        name=AGENT_EVAL_STEP_NAME,
        executor=executor,
        config=spec.model_dump(mode="json"),
        environment=_environment(spec, sandbox_plan=sandbox_plan),
    )


def _agent_eval_cpu_step(
    spec: AgentEvalSpec,
    profile: str | None,
    *,
    sandbox_plan: SandboxPlan | None,
) -> _AgentEvalCPUStep:
    """Build the CPU-backed evaluation step and keep its concrete executor type."""
    is_gym_target = isinstance(spec.target, GymRunnerTarget)
    is_colocated_gym = is_gym_target and sandbox_plan is None
    image = (
        config.gym_tasks_image
        if is_colocated_gym and config.gym_tasks_image is not None
        else get_qualified_image(GYM_AGENT_EVAL_IMAGE if is_colocated_gym else AGENT_EVAL_IMAGE)
    )
    executor = CPUExecutionProviderSpec(
        profile=profile or "default",
        provider="cpu",
        container=ContainerSpec(
            image=image,
            entrypoint=GYM_AGENT_EVAL_ENTRYPOINT if is_colocated_gym else AGENT_EVAL_ENTRYPOINT,
            command=AGENT_EVAL_COMMAND,
        ),
    )
    step = PlatformJobStep(
        name=AGENT_EVAL_STEP_NAME,
        executor=executor,
        config=spec.model_dump(mode="json"),
        environment=_environment(spec, sandbox_plan=sandbox_plan),
    )
    return _AgentEvalCPUStep(step=step, executor=executor)


def _sandbox_plan(spec: AgentEvalSpec) -> SandboxPlan | None:
    """Resolve the deployment-selected Gym sandbox plan once during compilation."""
    if not isinstance(spec.target, GymRunnerTarget):
        return None
    return resolve_sandbox_plan(
        config,
        spec.target,
        sandbox_server_protocol=platform_config.sandbox_server_protocol,
    )


def _environment(spec: AgentEvalSpec, *, sandbox_plan: SandboxPlan | None) -> list[EnvironmentVariable]:
    """The step's environment: secret refs, plus the sandbox plan when Gym runs sandboxed.

    Resolving the plan here rather than in the job is what makes the operator's configuration reach
    the run at all -- see :class:`nemo_evaluator.jobs.gym_sandbox.SandboxPlan`. It also moves the
    sandbox-capability gates to submit time, so a deployment that cannot sandbox is refused with a
    message naming the setting instead of failing partway through an evaluation.
    """
    environment = build_task_environment(_secret_refs(spec))
    if sandbox_plan is None:
        return environment
    environment.append(EnvironmentVariable(name=GYM_SANDBOX_PLAN_ENVVAR, value=sandbox_plan.model_dump_json()))
    return environment


def _environment_stage_step(
    target: GymRunnerTarget,
    profile: str | None,
    *,
    use_subprocess: bool,
) -> PlatformJobStep:
    """Download a custom Gym environment into the job PVC before evaluation."""
    assert target.environment is not None
    image = get_qualified_image(AGENT_EVAL_IMAGE)
    stage_spec = EnvironmentStageSpec(environment=target.environment)
    return PlatformJobStep(
        name=ENVIRONMENT_STAGE_STEP_NAME,
        executor=_executor(
            profile=profile,
            image=image,
            entrypoint=AGENT_EVAL_ENTRYPOINT,
            command=ENVIRONMENT_STAGE_COMMAND,
            use_subprocess=use_subprocess,
        ),
        config=stage_spec.model_dump(mode="json"),
        environment=build_task_environment(()),
    )


def _executor(
    *,
    profile: str | None,
    image: str,
    entrypoint: list[str],
    command: list[str],
    use_subprocess: bool,
) -> CPUExecutionProviderSpec | SubprocessExecutionProviderSpec:
    """Select the command form appropriate for the resolved execution profile."""
    resolved_profile = profile or "default"
    if use_subprocess:
        return SubprocessExecutionProviderSpec(
            profile=resolved_profile,
            provider="subprocess",
            command=[*AGENT_EVAL_ENTRYPOINT, *command],
        )

    return CPUExecutionProviderSpec(
        profile=resolved_profile,
        provider="cpu",
        container=ContainerSpec(image=image, entrypoint=entrypoint, command=command),
    )

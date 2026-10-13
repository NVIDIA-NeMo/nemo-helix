# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe a live agent runner as the target spec that reproduces it job-side.

The SDK's runners are Python objects built for the process they run in; the plugin's targets are
wire DTOs built to survive being sent somewhere else. This module is the one-way bridge: take a
runner someone got working locally with ``AgentEvaluator().run()``, and produce the spec that runs
the same evaluation as a governed job — without asking them to retype its configuration and get it
subtly wrong.

**One-way on purpose.** The reverse direction (spec to runner) already exists in
:mod:`nemo_evals.jobs.agent_evaluate`, where the job resolves its own target and can inject the
runtime-only values a spec has no business carrying — the way Harbor's ``jobs_dir`` comes from the
job's storage. Keeping construction there and description here means neither side has to know the
other's local details.

**Refuses rather than quietly drops.** A runner can hold state with no wire form: a callable, a
handle to something in this process, a path that means nothing on another machine. Silently
dropping it would submit a job that runs something *different* from what was tested locally, which
is worse than not submitting at all. Those cases raise :class:`UnsubmittableRunnerError` naming what
could not travel.

Gym runners and native Harbor runners are supported. Harbor's local storage is replaced by
job-owned storage; settings and execution overrides with no target representation are refused.
"""

from __future__ import annotations

from nemo_evals.jobs.agent_spec import (
    AgentRunnerTarget,
    GymAgentSource,
    GymPlacement,
    GymRunnerTarget,
    HarborBuiltinAgentSource,
    HarborImportedAgentSource,
    HarborRunnerTarget,
)
from nhx_evals_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborAgentTaskRunner
from nhx_evals_sdk.agent_eval.trials import AgentTaskRunner
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError


class UnsubmittableRunnerError(TypeError):
    """A live runner cannot be described as a submittable target spec.

    Raised both for runner types with no wire form at all and for a supported runner configured with
    state the wire cannot carry.
    """


def runner_to_target(runner: AgentTaskRunner, placement: GymPlacement | None = None) -> AgentRunnerTarget:
    """The target spec that reproduces ``runner`` as a job, placed by ``placement``.

    ``placement`` carries what the deployment decides rather than what the evaluation is — a staged
    environment FileSet, the agent instance a sandboxed host routes to. It is runner-specific, so
    supplying one for a runner that cannot be placed is refused rather than ignored.

    Raises:
        UnsubmittableRunnerError: If the runner has no wire form, carries state that would be lost in
            translation, or was given a placement it cannot use.
    """
    if isinstance(runner, GymAgentTaskRunner):
        return _gym_target(runner, placement or GymPlacement())
    if placement is not None:
        raise UnsubmittableRunnerError(
            f"a GymPlacement cannot place a {type(runner).__name__}; placement is per runner kind."
        )
    if isinstance(runner, HarborAgentTaskRunner):
        return _harbor_target(runner)
    raise UnsubmittableRunnerError(
        f"{type(runner).__name__} has no target spec, so it cannot be submitted as a job. Run it "
        "in-process with AgentEvaluator(), or pass a runner target spec directly."
    )


def _harbor_target(runner: HarborAgentTaskRunner) -> HarborRunnerTarget:
    """Describe a native Harbor runner without carrying local execution state.

    Algorithm:
        - Reject offline runners and explicit local execution overrides.
        - Require defaults for runtime settings that the target cannot represent.
        - Copy supported run settings and validate the target's JSON representation.

    ``jobs_dir`` is optional at construction and deliberately omitted from submission: workers
    supply job-owned storage. Standalone execution requires an explicit directory.
    Only ``env_secrets`` references travel; the platform resolves them with its own resolver.
    """
    config = runner._config
    if config is None:
        raise UnsubmittableRunnerError(
            "Harbor submission requires a native config; use saved-trial rescoring for an offline runner."
        )
    for field in ("dataset_path", "task_names", "job_dir", "run_job"):
        if getattr(runner, f"_{field}") is not None:
            raise UnsubmittableRunnerError(
                f"Harbor {field} cannot travel with a platform-based stored-taskset submission."
            )
    required_defaults = {
        "job_name": None,
        "force_rerun": False,
        "quiet": True,
        "agent_dir": None,
        "timeout_multiplier": None,
        "verifier_timeout_multiplier": None,
        "environment_build_timeout_multiplier": None,
    }
    for field, default in required_defaults.items():
        if getattr(config, field) != default:
            raise UnsubmittableRunnerError(f"Harbor {field} must retain its default for job submission.")
    source: HarborBuiltinAgentSource | HarborImportedAgentSource
    if config.agent_import_path is not None:
        source = HarborImportedAgentSource(import_path=config.agent_import_path, model_name=config.agent_model_name)
    elif config.agent_name is not None:
        source = HarborBuiltinAgentSource(name=config.agent_name, model_name=config.agent_model_name)
    else:
        raise UnsubmittableRunnerError("Harbor config selects no agent: set agent_name or agent_import_path.")
    carried_fields = (
        "agent_kwargs",
        "env_secrets",
        "env_vars",
        "n_attempts",
        "n_concurrent_trials",
        "max_retries",
        "artifacts",
        "trace_dir",
        "reward_key",
        "agent_setup_timeout_multiplier",
        "agent_timeout_multiplier",
    )
    try:
        target = HarborRunnerTarget(source=source, **{name: getattr(config, name) for name in carried_fields})
        target.model_dump(mode="json")
    except (ValidationError, PydanticSerializationError) as error:
        # Both models set `hide_input_in_errors`, so the text names the problem without echoing a value.
        raise UnsubmittableRunnerError(
            f"Harbor config cannot be represented as a valid JSON target spec: {error}"
        ) from error
    return target


def _gym_target(runner: GymAgentTaskRunner, placement: GymPlacement) -> GymRunnerTarget:
    """``GymAgentTaskRunner`` + ``GymPlacement`` -> ``GymRunnerTarget``.

    Every config field carries, and nothing is rejected — unusual among the runners, and worth saying
    plainly rather than inventing rejections to look careful. The placement only adds to that; it
    overrides nothing. Gym's settings are all *behaviour*
    (which environment, which agent, how many attempts, how long to wait) rather than *location*:
    there is no work root or local base directory. Only ``env_secrets`` references travel; a custom
    local secret resolver is replaced by the platform's resolver. The one thing
    that reads like a local path, ``agent_config``, is resolved relative to the Gym installation
    rather than the caller's filesystem, and the job container is required to have Gym installed
    regardless.

    ``env_vars`` carries verbatim, including absolute paths. A value like a model-cache root may
    well be wrong job-side — but a container path is absolute too, so nothing here can tell a local
    path from one that is correct in the job's image, and guessing would break the legitimate case
    to protect against a mistake the submitter is better placed to catch.

    The one rejection is a value with no JSON form. ``hydra_params`` and ``env_vars`` are typed
    loosely enough to hold a callable or an arbitrary object, which survives construction here and
    then fails inside ``model_dump(mode="json")`` at submit time — a ``PydanticSerializationError``
    raised from the transport, naming neither the runner nor the field. Checking here turns that
    into the refusal this module promises.
    """
    config = runner.config
    target = GymRunnerTarget(
        **config.model_dump(exclude={"agent", "agent_config"}),
        source=GymAgentSource(component=config.agent, config=config.agent_config, instance=placement.agent_ref_name),
        environment=placement.environment,
    )
    try:
        target.model_dump(mode="json")
    except PydanticSerializationError as error:
        raise UnsubmittableRunnerError(
            "this GymAgentTaskRunner holds configuration with no JSON form, so it cannot be sent to "
            f"the service: {error}. `hydra_params` and `env_vars` are free-form, but their values "
            "must be JSON-compatible — Gym receives them as Hydra overrides and environment "
            "variables, so a callable or a live object could not have travelled anyway."
        ) from error
    return target

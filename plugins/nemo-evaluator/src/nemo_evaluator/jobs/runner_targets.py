# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Describe a live agent runner as the target spec that reproduces it job-side.

The SDK's runners are Python objects built for the process they run in; the plugin's targets are
wire DTOs built to survive being sent somewhere else. This module is the one-way bridge: take a
runner someone got working locally with ``AgentEvaluator().run()``, and produce the spec that runs
the same evaluation as a governed job — without asking them to retype its configuration and get it
subtly wrong.

**One-way on purpose.** The reverse direction (spec to runner) already exists in
:mod:`nemo_evaluator.jobs.agent_evaluate`, where the job resolves its own target and can inject the
runtime-only values a spec has no business carrying — the way Harbor's ``jobs_dir`` comes from the
job's storage. Keeping construction there and description here means neither side has to know the
other's local details.

**Refuses rather than quietly drops.** A runner can hold state with no wire form: a callable, a
handle to something in this process, a path that means nothing on another machine. Silently
dropping it would submit a job that runs something *different* from what was tested locally, which
is worse than not submitting at all. Those cases raise :class:`UnsubmittableRunnerError` naming what
could not travel.

Native Harbor and Gym runners are supported. Other runners require an explicit translation contract.
"""

from __future__ import annotations

from nemo_evaluator.jobs.agent_spec import AgentRunnerTarget, GymRunnerTarget, HarborRunnerTarget
from nemo_evaluator_sdk.agent_eval.runtimes.gym import GymAgentTaskRunner
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_runtime import HarborAgentTaskRunner
from nemo_evaluator_sdk.agent_eval.trials import AgentTaskRunner
from nemo_evaluator_sdk.values import SecretRef
from pydantic_core import PydanticSerializationError


class UnsubmittableRunnerError(TypeError):
    """A live runner cannot be described as a submittable target spec.

    Raised both for runner types with no wire form at all and for a supported runner configured with
    state the wire cannot carry.
    """


def runner_to_target(runner: AgentTaskRunner, *, env_secrets: dict[str, SecretRef] | None = None) -> AgentRunnerTarget:
    """The target spec that reproduces ``runner`` as a job.

    Raises:
        UnsubmittableRunnerError: If the runner has no wire form, or carries state that would be
            lost in translation.
    """
    if isinstance(runner, HarborAgentTaskRunner):
        try:
            config = runner.export_submission_config()
        except ValueError as exc:
            raise UnsubmittableRunnerError(str(exc)) from exc
        unsupported = {"job_name": config.job_name is not None, "force_rerun": config.force_rerun}
        for field, supplied in unsupported.items():
            if supplied:
                raise UnsubmittableRunnerError(f"{field} cannot be submitted; remove local resume/rerun settings")
        missing = set(config.agent_env_from_host) - (env_secrets or {}).keys()
        if missing:
            raise UnsubmittableRunnerError(
                f"agent_env_from_host needs explicit env_secrets mappings for {sorted(missing)}"
            )
        omitted = {"jobs_dir", "agent_dir", "quiet", "job_name", "force_rerun", "agent_env_from_host"}
        values = config.model_dump(exclude=omitted)
        for field in values.keys() - HarborRunnerTarget.model_fields.keys():
            if values[field] != type(config).model_fields[field].get_default(call_default_factory=True):
                raise UnsubmittableRunnerError(f"Unsupported Harbor runtime field {field}; restore its default")
        return HarborRunnerTarget(
            **{key: value for key, value in values.items() if key in HarborRunnerTarget.model_fields},
            env_secrets=env_secrets or {},
        )
    if env_secrets is not None:
        raise UnsubmittableRunnerError("env_secrets is supported only for Harbor taskset submission")
    if isinstance(runner, GymAgentTaskRunner):
        return _gym_target(runner)
    raise UnsubmittableRunnerError(
        f"{type(runner).__name__} has no target spec, so it cannot be submitted as a job. Run it "
        "in-process with AgentEvaluator(), or pass a runner target spec directly."
    )


def _gym_target(runner: GymAgentTaskRunner) -> GymRunnerTarget:
    """``GymAgentTaskRunner`` -> ``GymRunnerTarget``.

    Every field carries, and nothing is rejected — unusual among the runners, and worth saying
    plainly rather than inventing rejections to look careful. Gym's settings are all *behaviour*
    (which environment, which agent, how many attempts, how long to wait) rather than *location*:
    there is no work root, no local base directory, and no injected callable to lose. The one thing
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
    target = GymRunnerTarget(**runner.config.model_dump())
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

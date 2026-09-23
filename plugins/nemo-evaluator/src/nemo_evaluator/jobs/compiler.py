# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-native evaluator job compiler."""

from __future__ import annotations

from collections.abc import Iterator

from nemo_evaluator.jobs.evaluate import EvaluateSpec
from nemo_evaluator.jobs.secret_env import build_task_environment
from nemo_evaluator_sdk.values import AgentBase, Model, RunConfig, RunConfigOnline, RunConfigOnlineModel
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
)
from nemo_helix_plugin.jobs.image import get_qualified_image

EVALUATE_STEP_NAME = "evaluate"


def compile_evaluate_job(spec: EvaluateSpec, *, profile: str | None = None) -> HelixJobSpec:
    """Compile a bundle-native evaluator plugin job."""
    _validate_evaluate_spec(spec)
    return HelixJobSpec(steps=[_evaluate_step(spec, profile)])


def _validate_evaluate_spec(spec: EvaluateSpec) -> None:
    if isinstance(spec.target, Model):
        if spec.prompt_template is None:
            raise ValueError("prompt_template is required when EvaluateSpec.target is a model")
        if not isinstance(spec.params, RunConfigOnlineModel):
            raise TypeError("model target requires RunConfigOnlineModel")
    elif isinstance(spec.target, AgentBase):
        if spec.prompt_template is None:
            raise ValueError("prompt_template is required when EvaluateSpec.target is an agent")
        if not isinstance(spec.params, RunConfigOnline):
            raise TypeError("agent target requires RunConfigOnline")
    elif not isinstance(spec.params, RunConfig):
        raise TypeError("offline evaluation requires RunConfig")


def _secret_refs(spec: EvaluateSpec) -> Iterator[tuple[str, str]]:
    """Yield ``(env_name, secret_name)`` for each metric secret and the endpoint target's api key."""
    for bundle in spec.metrics:
        for env_name, secret_ref in bundle.secrets.items():
            yield env_name, secret_ref.root

    if (
        isinstance(spec.target, (Model, AgentBase))
        and spec.target.api_key_secret is not None
        and spec.target.api_key_env
    ):
        yield spec.target.api_key_env, spec.target.api_key_secret.root


def _evaluate_step(spec: EvaluateSpec, profile: str | None) -> HelixJobStep:
    return HelixJobStep(
        name=EVALUATE_STEP_NAME,
        executor=CPUExecutionProviderSpec(
            profile=profile or "default",
            provider="cpu",
            container=ContainerSpec(
                image=get_qualified_image("nhx-cpu-tasks"),
                entrypoint=["python", "-m"],
                command=["nemo_evaluator.tasks.evaluate"],
            ),
        ),
        config=spec.model_dump(mode="json"),
        environment=build_task_environment(_secret_refs(spec)),
    )

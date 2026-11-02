# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-native evaluator job compiler."""

from __future__ import annotations

from nemo_evals.config import get_config
from nemo_evals.jobs.evaluate import EvaluateSpec, evaluate_secret_refs
from nemo_evals.jobs.secret_env import build_task_environment
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    HelixJobSpec,
    HelixJobStep,
)
from nemo_helix_plugin.jobs.image import get_qualified_image
from nhx_evals_sdk.values import AgentBase, Model, RunConfig, RunConfigOnline, RunConfigOnlineModel

EVALUATE_STEP_NAME = "evaluate"


def compile_evaluate_job(spec: EvaluateSpec, *, profile: str | None = None) -> HelixJobSpec:
    """Compile a bundle-native evals plugin job."""
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


def _evaluate_step(spec: EvaluateSpec, profile: str | None) -> HelixJobStep:
    return HelixJobStep(
        name=EVALUATE_STEP_NAME,
        executor=CPUExecutionProviderSpec(
            profile=profile or "default",
            provider="cpu",
            container=ContainerSpec(
                image=get_config().container_image_ref or get_qualified_image("nhx-tasks"),
                entrypoint=["python", "-m"],
                command=["nemo_evals.tasks.evaluate"],
            ),
        ),
        config=spec.model_dump(mode="json"),
        environment=build_task_environment(evaluate_secret_refs(spec)),
    )

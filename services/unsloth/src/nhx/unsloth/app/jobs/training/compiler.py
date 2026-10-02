# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile the GPU training step of an unsloth container job.

This is the second step in the 4-step ``HelixJobSpec`` built by
:func:`~nhx.unsloth.compile.platform_job_config_compiler`. The container
entrypoint is ``python -m nhx.unsloth.tasks.training``.
"""

from __future__ import annotations

import logging

from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    EnvironmentVariable,
    GPUExecutionProviderSpec,
    HelixJobStep,
    ResourcesSpec,
)
from nhx.customization_common.integrations import (
    collect_integration_secret_envs,
    warn_incomplete_integrations,
)
from nhx.unsloth.app.constants import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_MODEL_PATH,
)
from nhx.unsloth.app.jobs.training.schemas import TrainingStepConfig
from nhx.unsloth.images import UNSLOTH_PYTHON_ENTRYPOINT, get_training_image
from nhx.unsloth.schemas import UnslothJobOutput

logger = logging.getLogger(__name__)


def compile_training_step(
    job_spec: UnslothJobOutput,
    base_env: list[EnvironmentVariable],
    validation_dataset_path: str | None = None,
    profile: str | None = None,
) -> HelixJobStep:
    """Build the GPU training :class:`HelixJobStep`.

    The container reads the step config (a serialized
    :class:`TrainingStepConfig`) and runs ``train_sft`` against the
    paths populated by the file_io download step.

    Args:
        job_spec: Canonical job spec to serialize into the step config.
        base_env: Environment variables carried into every step
            (e.g. ``PERSISTENT_JOB_STORAGE_PATH``).
        validation_dataset_path: Local path the file_io download step
            populated for validation. ``None`` when the job has no
            validation split.
        profile: GPU execution profile (e.g. ``gpu`` or
            ``gpu_distributed``). When ``None`` the executor's default
            is used.
    """
    step_config = TrainingStepConfig(
        spec=job_spec,
        model_path=DEFAULT_MODEL_PATH,
        dataset_path=DEFAULT_DATASET_PATH,
        validation_path=validation_dataset_path,
        output_path=DEFAULT_OUTPUT_MODEL_PATH,
    )

    executor: GPUExecutionProviderSpec = {
        "provider": "gpu",
        "container": ContainerSpec(
            image=get_training_image(),
            entrypoint=UNSLOTH_PYTHON_ENTRYPOINT,
            command=["-m", "nhx.unsloth.tasks.training"],
        ),
        # Resources are provider-decided; the platform's GPU executor
        # selects a node based on the profile.
        "resources": ResourcesSpec(),
    }
    if profile is not None:
        executor["profile"] = profile

    warn_incomplete_integrations(job_spec.integrations)
    return HelixJobStep(
        name="training",
        executor=executor,
        environment=[*base_env, *collect_integration_secret_envs(job_spec.integrations)],
        config=step_config.model_dump(mode="json"),
    )

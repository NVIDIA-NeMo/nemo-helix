# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge Automodel ``TrainingStepConfig`` to shared integration runtime helpers."""

from typing import Any

from nhx.automodel.app.jobs.training.schemas import TrainingStepConfig
from nhx.customization_common.integrations import (
    IntegrationRuntimeContext,
)
from nhx.customization_common.integrations import (
    build_mlflow_config as _build_mlflow_config,
)
from nhx.customization_common.integrations import (
    build_wandb_config as _build_wandb_config,
)
from nhx.customization_common.service.context import NHXJobContext


def integration_context_from_training_step(
    customizer_config: TrainingStepConfig,
    job_ctx: NHXJobContext,
    framework: str,
) -> IntegrationRuntimeContext:
    """Map compiled training-step config to shared integration runtime context."""
    return IntegrationRuntimeContext.from_integrations_spec(
        integrations=customizer_config.integrations,
        output_name=customizer_config.output_model,
        workspace_path=customizer_config.workspace_path,
        model_name=customizer_config.model.name,
        job_ctx=job_ctx,
        framework=framework,
    )


def build_mlflow_config(
    customizer_config: TrainingStepConfig,
    job_ctx: NHXJobContext,
    framework: str,
) -> dict[str, Any] | None:
    """Build MLflow config for Automodel training."""
    return _build_mlflow_config(
        integration_context_from_training_step(customizer_config, job_ctx, framework),
    )


def build_wandb_config(
    customizer_config: TrainingStepConfig,
    job_ctx: NHXJobContext,
    framework: str,
) -> dict[str, Any] | None:
    """Build W&B config for Automodel training."""
    return _build_wandb_config(
        integration_context_from_training_step(customizer_config, job_ctx, framework),
    )

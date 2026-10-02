# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime context for building experiment-tracking integration configs."""

from dataclasses import dataclass
from typing import Self

from nemo_helix_plugin.integrations import IntegrationsSpec, MlflowIntegration, WandbIntegration
from nhx.customization_common.service.context import NHXJobContext


@dataclass(frozen=True)
class IntegrationRuntimeContext:
    """Inputs shared by W&B and MLflow runtime config builders."""

    wandb: WandbIntegration | None
    mlflow: MlflowIntegration | None
    output_name: str
    workspace_path: str
    model_name: str | None
    job_ctx: NHXJobContext
    framework: str

    @classmethod
    def from_integrations_spec(
        cls,
        *,
        integrations: IntegrationsSpec | None,
        output_name: str,
        workspace_path: str,
        model_name: str | None,
        job_ctx: NHXJobContext,
        framework: str,
    ) -> Self:
        """Build context from the canonical job integrations object."""
        return cls(
            wandb=integrations.wandb if integrations else None,
            mlflow=integrations.mlflow if integrations else None,
            output_name=output_name,
            workspace_path=workspace_path,
            model_name=model_name,
            job_ctx=job_ctx,
            framework=framework,
        )

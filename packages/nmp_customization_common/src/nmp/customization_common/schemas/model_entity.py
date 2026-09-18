# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas for the model_entity container task configuration.

Shared by both the unsloth and automodel backends.
"""

from __future__ import annotations

from typing import Optional

from nemo_platform_plugin.deployment import DeploymentParams
from nmp.customization_common.schemas.file_io import FileSetRef
from nmp.customization_common.schemas.values import FinetuningType
from pydantic import BaseModel, Field


class PEFTConfig(BaseModel):
    """PEFT configuration for LoRA / LoRA-merged fine-tuning."""

    type: FinetuningType
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)


class ModelEntityTaskConfig(BaseModel):
    """Configuration for the model_entity task.

    Used when running ``python -m nmp.<backend>.tasks.model_entity``.
    """

    name: str = Field(description="Name of the model entity to create.")
    workspace: str = Field(description="Workspace of the model entity to create.")
    description: Optional[str] = Field(default=None, description="Optional description of the model.")
    fileset: FileSetRef = Field(description="FileSet reference containing the customized model artifacts.")
    model_entity: str = Field(description="The model entity (workspace/name) this model was based on.")
    base_model: Optional[str] = Field(default=None, description="Link to the base model used for customization.")
    peft: Optional[PEFTConfig] = Field(
        default=None,
        description="PEFT configuration. Set for LoRA / LoRA-merged, None for full SFT.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code for the checkpoint.",
    )
    deployment_config: Optional[str | DeploymentParams] = Field(
        default=None,
        description=(
            "Deployment configuration. A string references an existing ModelDeploymentConfig "
            "by name. An object provides inline NIM deployment parameters. Omit to skip deployment."
        ),
    )


class ModelEntityCreationError(Exception):
    """Error creating the output model entity."""

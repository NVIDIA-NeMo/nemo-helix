# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas for the model_entity container task configuration.

Shared by both the unsloth and automodel backends.
"""

from __future__ import annotations

from typing import Optional

from nhx.customization_common.schemas.file_io import FileSetRef
from nhx.customization_common.schemas.values import FinetuningType
from pydantic import BaseModel, Field


class ToolCallConfig(BaseModel):
    """Tool calling configuration for NIM deployments."""

    tool_call_parser: Optional[str] = Field(default=None, description="Name of the tool call parser to use.")
    tool_call_plugin: Optional[str] = Field(
        default=None,
        pattern=r"^[\w\-.]+/[\w\-.]+$",
        description=(
            "Reference to a fileset containing the custom tool call plugin Python file. "
            "Expected format: '{workspace}/{fileset_name}'."
        ),
    )
    auto_tool_choice: Optional[bool] = Field(default=None, description="Whether to enable automatic tool choice.")


class DeploymentParameters(BaseModel):
    """Inline deployment parameters for creating a new ModelDeploymentConfig."""

    gpu: int = Field(default=1, gt=0, description="Number of GPUs required for deployment")
    additional_envs: Optional[dict[str, str]] = Field(
        default=None,
        description="Additional environment variables for deployment",
    )
    disk_size: Optional[str] = Field(default=None, description="Disk size for deployment")
    image_name: Optional[str] = Field(
        default=None,
        description="Container image name from NGC. Defaults to multi-llm when unset",
    )
    image_tag: Optional[str] = Field(default=None, description="Container image tag from NGC")
    lora_enabled: bool = Field(
        default=True,
        description=(
            "When auto-deploying full SFT training, setting this true allows "
            "subsequent LoRA adapters to be deployed against the model."
        ),
    )
    tool_call_config: Optional[ToolCallConfig] = Field(
        default=None,
        description="Tool calling configuration override for the NIM deployment.",
    )


class PEFTConfig(BaseModel):
    """PEFT configuration for LoRA / LoRA-merged fine-tuning."""

    type: FinetuningType
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)


class ModelEntityTaskConfig(BaseModel):
    """Configuration for the model_entity task.

    Used when running ``python -m nhx.<backend>.tasks.model_entity``.
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
    deployment_config: Optional[str | DeploymentParameters] = Field(
        default=None,
        description=(
            "Deployment configuration. A string references an existing ModelDeploymentConfig "
            "by name. An object provides inline NIM deployment parameters. Omit to skip deployment."
        ),
    )


class ModelEntityCreationError(Exception):
    """Error creating the output model entity."""

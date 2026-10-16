# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared auto-deployment schemas for customization job specs.

Every customization backend can ask the platform to deploy a model once training
finishes. The parameters describe a deployment, not a training regime, so the
shape is identical across backends and lives here rather than being restated by
each one.

These are deliberately plain ``BaseModel`` subclasses, not ``NamespacedModel``:
one shared model emits one ``DeploymentParams`` schema in the merged
``/apis/customization`` spec, referenced by every backend's job input.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolCallParams(BaseModel):
    """Tool calling configuration for NIM deployments."""

    model_config = ConfigDict(extra="forbid")

    tool_call_parser: str | None = Field(
        default=None,
        description=(
            "Name of the tool call parser to use (e.g., 'openai', 'hermes', 'pythonic', 'llama3_json', 'mistral')."
        ),
    )
    tool_call_plugin: str | None = Field(
        default=None,
        pattern=r"^[\w\-.]+/[\w\-.]+$",
        description=(
            "Reference to a fileset containing the custom tool call plugin Python file. "
            "Expected format: '{workspace}/{fileset_name}'."
        ),
    )
    auto_tool_choice: bool | None = Field(
        default=None,
        description="Whether to enable automatic tool choice.",
    )


class DeploymentParams(BaseModel):
    """Inline deployment parameters for auto-deploying a trained model.

    Used in each backend's ``job_spec.deployment_config`` and passed through to the
    model_entity task at compile time. When unset, no deployment is launched.
    """

    model_config = ConfigDict(extra="forbid")

    engine: Literal["nim", "vllm"] = Field(
        default="vllm",
        description="Inference engine for the deployment: 'vllm' (default) or 'nim'.",
    )
    gpu: int = Field(default=1, gt=0, description="Number of GPUs required for the deployment.")
    additional_envs: dict[str, str] | None = Field(
        default=None,
        description="Additional environment variables for the deployment.",
    )
    disk_size: str | None = Field(default=None, description="Disk size for the deployment.")
    image_name: str | None = Field(
        default=None,
        description="Container image name. If not specified, the platform default image for the engine is used.",
    )
    image_tag: str | None = Field(
        default=None,
        description="Container image tag. If not specified, the platform default tag for the engine is used.",
    )
    lora_enabled: bool = Field(
        default=True,
        description=(
            "When auto-deploying a full-weight training, setting this true allows subsequent "
            "LoRA adapters to be deployed against it."
        ),
    )
    tool_call_config: ToolCallParams | None = Field(
        default=None,
        description="Tool calling configuration override. Requires engine 'nim'.",
    )

    @model_validator(mode="after")
    def _tool_call_config_requires_nim(self) -> Self:
        # The vLLM compiler does not read tool_call_config, so accepting it there would
        # silently drop the setting.
        if self.tool_call_config is not None and self.engine != "nim":
            raise ValueError("tool_call_config is only supported with engine='nim'")
        return self


DEPLOYMENT_CONFIG_DESCRIPTION = (
    "Deployment configuration for auto-deploying the model after training. "
    "Pass a string to reference an existing ModelDeploymentConfig by name "
    "('my-config' or 'workspace/my-config'). An object provides inline "
    "deployment parameters, deployed with vLLM unless engine is 'nim'. Omit to skip deployment."
)

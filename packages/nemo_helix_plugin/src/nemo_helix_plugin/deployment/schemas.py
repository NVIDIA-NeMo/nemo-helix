# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared auto-deployment schemas for customization job specs.

Every customization backend can ask the platform to deploy a model once training
finishes. The parameters describe a NIM deployment, not a training regime, so the
shape is identical across backends and lives here rather than being restated by
each one.

These are deliberately plain ``BaseModel`` subclasses, not ``NamespacedModel``:
one shared model emits one ``DeploymentParams`` schema in the merged
``/apis/customization`` spec, referenced by every backend's job input.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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

    gpu: int = Field(default=1, gt=0, description="Number of GPUs required for the deployment.")
    additional_envs: dict[str, str] | None = Field(
        default=None,
        description="Additional environment variables for the deployment.",
    )
    disk_size: str | None = Field(default=None, description="Disk size for the deployment.")
    image_name: str | None = Field(
        default=None,
        description="Container image name from NGC. If not specified, defaults to multi-llm.",
    )
    image_tag: str | None = Field(default=None, description="Container image tag from NGC.")
    lora_enabled: bool = Field(
        default=True,
        description=(
            "When auto-deploying a full-weight training, setting this true allows subsequent "
            "LoRA adapters to be deployed against it."
        ),
    )
    tool_call_config: ToolCallParams | None = Field(
        default=None,
        description="Tool calling configuration override for the NIM deployment.",
    )


DEPLOYMENT_CONFIG_DESCRIPTION = (
    "Deployment configuration for auto-deploying the model after training. "
    "Pass a string to reference an existing ModelDeploymentConfig by name "
    "('my-config' or 'workspace/my-config'). An object provides inline NIM "
    "deployment parameters. Omit to skip deployment."
)

LORA_ENABLED_REQUIRED_MESSAGE = (
    "deployment_config.lora_enabled must be true (or omitted) when training a LoRA adapter. "
    "Setting lora_enabled=false would deploy the base model without LoRA support, "
    "making the trained adapter unservable."
)


def reject_lora_without_lora_enabled(
    deployment_config: str | DeploymentParams | None,
    *,
    trains_lora_adapter: bool,
) -> None:
    """Raise when a LoRA job asks for a deployment that cannot load adapters.

    A LoRA adapter is served by its base model's deployment, so a base deployed with
    ``lora_enabled=false`` would refuse it. Backends call this from a submit-time
    validator, passing their own answer for whether the job trains a standalone
    adapter — that predicate differs per backend, the check does not.

    String references are not checked here: resolving them needs a platform client,
    so the compiler validates those.
    """
    if trains_lora_adapter and isinstance(deployment_config, DeploymentParams) and not deployment_config.lora_enabled:
        raise ValueError(LORA_ENABLED_REQUIRED_MESSAGE)

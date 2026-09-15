# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submitter-facing NeMo-RL schemas.

The **canonical** types (``RlJobOutput``, ``DPOTraining``, ``GRPOTraining``, ``OutputResponse``)
live in :mod:`nmp.rl.schemas` and are re-exported here for concise imports. Only
the thin input shape (``RlJobInput`` + ``OutputRequest``) is defined here; the
plugin's :func:`~nemo_rl_plugin.transform.transform_input_to_output` resolves it
into the canonical output.
"""

from __future__ import annotations

from nemo_platform_plugin.deployment import (
    DEPLOYMENT_CONFIG_DESCRIPTION,
    DeploymentParams,
    ToolCallParams,
    reject_lora_without_lora_enabled,
)
from nemo_platform_plugin.integrations import IntegrationsSpec
from nmp.rl.schemas import (
    DPOTraining,
    GRPOTraining,
    LoRAParams,
    OutputResponse,
    ParallelismParams,
    RlJobOutput,
    RlSchema,
    TrainingMethod,
    trains_lora_adapter,
)
from pydantic import ConfigDict, Field, model_validator

__all__ = [
    "DPOTraining",
    "DeploymentParams",
    "GRPOTraining",
    "LoRAParams",
    "OutputRequest",
    "OutputResponse",
    "ParallelismParams",
    "RlJobInput",
    "RlJobOutput",
    "ToolCallParams",
    "TrainingMethod",
]


class OutputRequest(RlSchema):
    """Submitter-facing output preferences. ``name`` is auto-derived if omitted."""

    name: str | None = Field(default=None, max_length=255)


class RlJobInput(RlSchema):
    """POST body / CLI JSON for ``nemo customization rl submit``."""

    # extra="forbid" inherited from RlSchema; protected_namespaces=() kept for the
    # ``model`` field.
    model_config = ConfigDict(protected_namespaces=())

    name: str | None = None
    model: str = Field(description="Model entity reference ('name' or 'workspace/name').")
    dataset: str = Field(
        description=(
            "Dataset fileset reference. DPO: preference JSONL (training.jsonl + validation.jsonl). "
            "GRPO: Gym JSONL (training.jsonl required)."
        ),
    )
    environment: str | None = Field(
        default=None,
        description="Environment fileset reference (required when training.type is grpo).",
    )
    training: TrainingMethod = Field(description="Training method and hyperparameters (DPO or GRPO).")
    integrations: IntegrationsSpec | None = None
    output: OutputRequest | None = None
    deployment_config: str | DeploymentParams | None = Field(
        default=None,
        description=DEPLOYMENT_CONFIG_DESCRIPTION,
    )

    @property
    def trains_lora_adapter(self) -> bool:
        """True when the job produces a LoRA adapter rather than a full-weight model."""
        return trains_lora_adapter(self.training)

    @model_validator(mode="after")
    def _reject_lora_without_lora_enabled(self) -> RlJobInput:
        reject_lora_without_lora_enabled(
            self.deployment_config,
            trains_lora_adapter=self.trains_lora_adapter,
        )
        return self

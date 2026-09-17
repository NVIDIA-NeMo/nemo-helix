# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submitter-facing Unsloth schemas.

The **canonical** types (``UnslothJobOutput``, ``OutputResponse``, and
all shared sub-shapes) live in :mod:`nmp.unsloth.schemas`. They are
re-exported from this module for backward compatibility and to keep
caller imports concise (``from nemo_unsloth_plugin.schema import
UnslothJobInput, ModelLoadSpec`` still works).

Only two types are defined here:

- :class:`OutputRequest` — submitter-facing output preferences. The
  plugin's :func:`~nemo_unsloth_plugin.transform.transform_input_to_output`
  resolves it into the canonical :class:`~nmp.unsloth.schemas.OutputResponse`.
- :class:`UnslothJobInput` — the POST body / CLI JSON shape and the
  validators that mediate between input and canonical (mutexes,
  defaulting, etc.).
"""

from __future__ import annotations

from typing import Literal, Self

from nemo_platform_plugin.deployment import (
    DeploymentParams,
    ToolCallParams,
    reject_lora_without_lora_enabled,
)
from nemo_platform_plugin.integrations import IntegrationsSpec
from nmp.unsloth.schemas import (
    BatchSpec,
    DatasetSpec,
    HardwareSpec,
    LoRAParams,
    ModelLoadSpec,
    OptimizerSpec,
    OutputResponse,
    ScheduleSpec,
    TrainingSpec,
    UnslothJobOutput,
    UnslothSchema,
)
from pydantic import Field, model_validator

__all__ = [
    "BatchSpec",
    "DatasetSpec",
    "DeploymentParams",
    "HardwareSpec",
    "LoRAParams",
    "ModelLoadSpec",
    "OptimizerSpec",
    "OutputRequest",
    "OutputResponse",
    "ScheduleSpec",
    "ToolCallParams",
    "TrainingSpec",
    "UnslothJobInput",
    "UnslothJobOutput",
]


class OutputRequest(UnslothSchema):
    """Submitter-facing output preferences. ``name`` is auto-derived if omitted."""

    name: str | None = None
    description: str | None = None
    save_method: Literal["lora", "merged_16bit", "merged_4bit"] = "lora"


class UnslothJobInput(UnslothSchema):
    """POST body / CLI JSON for ``nemo customization unsloth submit``."""

    name: str | None = None
    model: ModelLoadSpec
    dataset: DatasetSpec
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    schedule: ScheduleSpec = Field(default_factory=ScheduleSpec)
    batch: BatchSpec = Field(default_factory=BatchSpec)
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    integrations: IntegrationsSpec | None = None
    output: OutputRequest | None = None
    deployment_config: str | DeploymentParams | None = Field(
        default=None,
        description=(
            "Deployment configuration for auto-deploying the model after training. "
            "Pass a string to reference an existing ModelDeploymentConfig by name "
            "('my-config' or 'workspace/my-config'). An object provides inline NIM "
            "deployment parameters. Omit to skip deployment."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        # Schedule: epochs defaults to 1; max_steps (when set) caps training. Both may be
        # present — the trainer honours max_steps as an override (consistent with Automodel).
        # 4bit / 8bit mutex (bitsandbytes — they really are exclusive)
        if self.model.load_in_4bit and self.model.load_in_8bit:
            raise ValueError("model.load_in_4bit and model.load_in_8bit are mutually exclusive")
        # all-weights (full) FT cannot quantize. The lora/finetuning_type invariant
        # (auto-fill for lora, reject lora for all_weights) is enforced in TrainingSpec.
        if self.training.finetuning_type == "all_weights":
            if self.model.load_in_4bit or self.model.load_in_8bit:
                raise ValueError(
                    "training.finetuning_type='all_weights' is incompatible with 4-bit/8-bit loading; "
                    "set model.load_in_4bit=false and model.load_in_8bit=false"
                )
        # warmup_steps and warmup_ratio mutex (transformers also enforces this
        # at runtime; we surface it earlier with a clearer message)
        if self.schedule.warmup_steps and self.schedule.warmup_ratio is not None:
            raise ValueError("schedule.warmup_steps and schedule.warmup_ratio are mutually exclusive")
        # merged_* save methods only make sense with LoRA training
        if self.output is not None and self.output.save_method != "lora":
            if self.training.finetuning_type != "lora":
                raise ValueError(
                    f"output.save_method={self.output.save_method!r} is only valid for training.finetuning_type='lora'"
                )
        reject_lora_without_lora_enabled(
            self.deployment_config,
            trains_lora_adapter=self.trains_standalone_lora_adapter,
        )
        return self

    @property
    def trains_standalone_lora_adapter(self) -> bool:
        """True when the job produces a LoRA adapter rather than a full-weight model.

        A merged save method folds the adapter into the base weights, which yields a
        standalone model; only the unmerged case needs a LoRA-enabled deployment.
        """
        return self.training.finetuning_type == "lora" and (self.output is None or self.output.save_method == "lora")

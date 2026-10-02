# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas for the unsloth training task configuration.

The training step config is a thin wrapper around the canonical
:class:`~nhx.unsloth.schemas.UnslothJobOutput` plus the resolved
filesystem paths the file_io download step writes to. The container
runner reads this from the platform Jobs envelope, validates it, and
hands the job spec + paths to :func:`~nhx.unsloth.tasks.training.backends.unsloth_sft.train_sft`.
"""

from __future__ import annotations

from nhx.unsloth.schemas import UnslothJobOutput
from pydantic import BaseModel, ConfigDict, Field


class TrainingStepConfig(BaseModel):
    """Configuration handed to ``python -m nhx.unsloth.tasks.training``."""

    model_config = ConfigDict(extra="forbid")

    spec: UnslothJobOutput = Field(description="Canonical job spec for the training run.")
    model_path: str = Field(description="Local filesystem path where the model weights were downloaded.")
    dataset_path: str = Field(description="Local filesystem path where the training dataset was downloaded.")
    validation_path: str | None = Field(
        default=None,
        description=(
            "Local filesystem path where the validation dataset was downloaded. "
            "When set, overrides ``spec.dataset.validation_path`` so the trainer "
            "reads on-disk JSONL instead of treating the platform ref as an HF id."
        ),
    )
    output_path: str = Field(description="Local filesystem path the training driver should save the checkpoint to.")

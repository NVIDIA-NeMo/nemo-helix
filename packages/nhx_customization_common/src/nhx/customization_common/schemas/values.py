# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value enums shared by the customization backends.

Only the enums with identical members across unsloth and automodel live here.
``TrainingType`` stays per-backend (different supported algorithms), and
``CheckpointFormat`` / ``Precision`` are automodel-only.
"""

from enum import Enum, StrEnum


class FinetuningType(str, Enum):
    """Finetuning strategy (full weights vs PEFT)."""

    ALL_WEIGHTS = "all_weights"
    LORA = "lora"
    LORA_MERGED = "lora_merged"


class OutputNameType(StrEnum):
    """Output artifact type — adapter (LoRA only) or model (merged / full)."""

    ADAPTER = "adapter"
    MODEL = "model"


class TrainingPhase(StrEnum):
    """What the training task reports as ``phase`` in its task ``status_details``.

    In the order a run normally passes through them. ``validation``,
    ``saving_checkpoint``, ``checkpoint_saved`` and ``epoch_end`` repeat during
    ``training``. After the last step, ``merging_adapter`` (LoRA merged) or
    ``finalizing_checkpoint`` (everything else) covers preparing the output
    model, before ``completed``.
    """

    CONVERSIONS = "conversions"
    COMPILING_CONFIG = "compiling_config"
    TRAINING = "training"
    VALIDATION = "validation"
    SAVING_CHECKPOINT = "saving_checkpoint"
    CHECKPOINT_SAVED = "checkpoint_saved"
    EPOCH_END = "epoch_end"
    MERGING_ADAPTER = "merging_adapter"
    FINALIZING_CHECKPOINT = "finalizing_checkpoint"
    COMPLETED = "completed"

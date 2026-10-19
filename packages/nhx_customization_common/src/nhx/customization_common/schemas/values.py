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

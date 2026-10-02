# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value enums for the nhx-rl backend.

``FinetuningType`` is reused from the common package. ``TrainingType`` is
per-backend (RL supports DPO today; GRPO reserved for headroom).
``Precision`` / ``CheckpointFormat`` are RL-local.
"""

from enum import Enum

from nhx.customization_common.schemas.values import FinetuningType  # noqa: F401  (re-export)


class TrainingType(str, Enum):
    """RL training algorithm. DPO is wired; GRPO is headroom.

    PPO is intentionally absent: the backend only compiles DPO/GRPO, so
    accepting a PPO value here would defer the failure to a runtime crash inside
    the training container instead of failing fast at request validation.
    """

    DPO = "dpo"
    GRPO = "grpo"


class Precision(str, Enum):
    """Model weight / compute precision."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class CheckpointFormat(str, Enum):
    """Output checkpoint artifact format."""

    SAFETENSORS = "safetensors"
    HF = "hf"
    HF_PEFT = "hf-peft"  # HuggingFace PEFT adapter (GRPO LoRA)

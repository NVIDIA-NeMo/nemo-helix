# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value types for the Customizer service.

``FinetuningType`` and ``OutputNameType`` are shared with unsloth via
:mod:`nmp.customization_common.schemas.values`. ``TrainingType`` (4 algorithms),
``CheckpointFormat`` and ``Precision`` are automodel-specific.
"""

from enum import Enum

from nmp.customization_common.schemas.values import FinetuningType, OutputNameType

__all__ = [
    "CheckpointFormat",
    "CheckpointSelection",
    "FinetuningType",
    "OutputNameType",
    "Precision",
    "TrainingType",
]


class CheckpointFormat(str, Enum):
    """Model checkpoint format (input or output)."""

    HF = "hf"  # Standard HuggingFace format
    HF_PEFT = "hf-peft"  # HuggingFace PEFT adapter (LoRA, etc.)
    NEMO = "nemo"  # NeMo checkpoint format


class CheckpointSelection(str, Enum):
    """Checkpoint(s) to publish after training."""

    BEST = "best"
    LAST = "last"
    BOTH = "both"


class Precision(str, Enum):
    """Model precision for training."""

    FP8 = "fp8"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"

    def to_torch_dtype(self) -> str:
        """
        Convert to a torch dtype string compatible with HuggingFace/Automodel.

        Returns:
            String like "bfloat16", "float16", "float32" that can be passed to
            from_pretrained(torch_dtype=...) or Automodel's dtype_from_str().

        Raises:
            ValueError: If this precision cannot be represented as a torch dtype.
                FP8 requires separate quantization config, BF16_MIXED is a training mode.
        """
        mapping = {
            Precision.BF16: "bfloat16",
            Precision.FP16: "float16",
            Precision.FP32: "float32",
        }
        if self not in mapping:
            raise ValueError(
                f"Precision '{self.value}' cannot be converted to a torch dtype. "
                f"Supported: {[p.value for p in mapping.keys()]}. "
                f"Note: FP8 requires separate quantization config, BF16_MIXED is a training mode."
            )
        return mapping[self]

    @classmethod
    def from_hf_dtype(cls, hf_dtype: str) -> "Precision":
        """
        Create Precision from a HuggingFace torch_dtype string.

        Args:
            hf_dtype: String like "bfloat16", "float16", "float32", "float".

        Returns:
            Corresponding Precision enum value.

        Raises:
            ValueError: If the dtype string is not recognized.
        """
        mapping = {
            "bfloat16": cls.BF16,
            "float16": cls.FP16,
            "float32": cls.FP32,
            "float": cls.FP32,
        }
        if hf_dtype not in mapping:
            raise ValueError(f"Unknown HuggingFace dtype '{hf_dtype}'. Supported: {list(mapping.keys())}")
        return mapping[hf_dtype]


class TrainingType(str, Enum):
    """Training algorithm type."""

    SFT = "sft"
    DISTILLATION = "distillation"
    DPO = "dpo"
    GRPO = "grpo"

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value types for the unsloth service.

``FinetuningType`` and ``OutputNameType`` are shared with automodel via
:mod:`nhx.customization_common.schemas.values`. ``TrainingType`` is unsloth-specific:
the unsloth backend supports SFT only.
"""

from enum import Enum

from nhx.customization_common.schemas.values import FinetuningType, OutputNameType

__all__ = ["FinetuningType", "OutputNameType", "TrainingType"]


class TrainingType(str, Enum):
    """Training algorithm type. Unsloth backend supports SFT only."""

    SFT = "sft"

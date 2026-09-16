# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Customizer entity definitions.

This module exports:
- Entity classes (database/persistence models)
- Shared value types (enums and read-only metadata)

Configuration types (LoRAConfig, ModelConfig, etc.) are NOT exported here.
They belong in their respective layers:
- API types → api/v2/jobs/schemas.py
- Internal types → app/jobs/training/schemas.py
"""

from .values import (
    CheckpointFormat,
    FinetuningType,
    Precision,
    TrainingType,
)

__all__ = [
    # Enums
    "CheckpointFormat",
    "FinetuningType",
    "Precision",
    "TrainingType",
]

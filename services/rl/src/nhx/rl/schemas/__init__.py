# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical NeMo-RL schemas.

- :mod:`nhx.rl.schemas.job` — job / training method types (compiler + drivers)
- :mod:`nhx.rl.schemas.environment` — environment FileSet manifests + Gym JSONL rows
"""

from nhx.rl.schemas.job import (
    DPOTraining,
    GRPOTraining,
    LoRAParams,
    OutputRequest,
    OutputResponse,
    ParallelismParams,
    RlJobOutput,
    RlSchema,
    TrainingMethod,
    trains_lora_adapter,
)

__all__ = [
    "DPOTraining",
    "GRPOTraining",
    "LoRAParams",
    "OutputRequest",
    "OutputResponse",
    "ParallelismParams",
    "RlJobOutput",
    "RlSchema",
    "TrainingMethod",
    "trains_lora_adapter",
]

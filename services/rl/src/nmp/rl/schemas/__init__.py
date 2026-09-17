# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical NeMo-RL schemas.

- :mod:`nmp.rl.schemas.job` — job / training method types (compiler + drivers)
- :mod:`nmp.rl.schemas.environment` — environment FileSet manifests + Gym JSONL rows
"""

from nmp.rl.schemas.job import (
    DEPLOYMENT_CONFIG_DESCRIPTION,
    DeploymentParams,
    DPOTraining,
    GRPOTraining,
    LoRAParams,
    OutputRequest,
    OutputResponse,
    ParallelismParams,
    RlJobOutput,
    RlSchema,
    ToolCallParams,
    TrainingMethod,
    trains_lora_adapter,
)

__all__ = [
    "DEPLOYMENT_CONFIG_DESCRIPTION",
    "DPOTraining",
    "DeploymentParams",
    "GRPOTraining",
    "LoRAParams",
    "OutputRequest",
    "OutputResponse",
    "ParallelismParams",
    "RlJobOutput",
    "RlSchema",
    "ToolCallParams",
    "TrainingMethod",
    "trains_lora_adapter",
]

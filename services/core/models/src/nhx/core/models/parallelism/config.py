#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parallelism configuration — re-exported from the core models config.

All parallelism config classes and get_config() live in nhx.core.models.config
and use the Pydantic-based models service configuration.
"""

from nhx.core.models.config import (
    BalanceConfig,
    ContextParallelismConfig,
    DataParallelismConfig,
    ExpertParallelismConfig,
    ModelSizeThresholds,
    ParallelismConfig,
    ParallelismMemoryConfig,
    PipelineParallelismConfig,
    TensorParallelismConfig,
)
from nhx.core.models.config import (
    config as models_config,
)

# Backward compatibility: old dataclass was named MemoryConfig
MemoryConfig = ParallelismMemoryConfig

__all__ = [
    "BalanceConfig",
    "ContextParallelismConfig",
    "DataParallelismConfig",
    "ExpertParallelismConfig",
    "MemoryConfig",
    "ModelSizeThresholds",
    "ParallelismConfig",
    "ParallelismMemoryConfig",
    "PipelineParallelismConfig",
    "TensorParallelismConfig",
    "get_config",
]


def get_config() -> ParallelismConfig:
    """Return the parallelism config from the models service configuration."""
    return models_config.parallelism

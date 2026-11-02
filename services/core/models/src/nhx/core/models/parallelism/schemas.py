# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Estimation inputs for parallelism and memory estimation.

Kept apart from ``models.py`` so the memory arithmetic in ``memory.py`` can be
imported without torch or transformers, which the models service does not ship.
"""

from typing import Optional

from pydantic import BaseModel, Field


class EstimationParams(BaseModel):
    """Parameters for parallelism estimation (pure Python API)."""

    gpus: int = Field(..., description="Total number of GPUs available", gt=0)
    gpu_mem_gb: float = Field(..., description="Memory per GPU in GB", gt=0)
    seq_len: int = Field(..., description="Sequence length", gt=0)
    global_batch_size: Optional[int] = Field(default=None, description="Global batch size (optional)", ge=1)
    microbatch_size: Optional[int] = Field(default=None, description="Fixed microbatch size (optional)", ge=1)
    max_microbatch: int = Field(default=64, description="Maximum microbatch size to search", ge=1)
    act_ckpt_ratio: float = Field(default=0.25, description="Activation checkpointing ratio (0.0-1.0)", ge=0.0, le=1.0)
    attn_scratch_factor: float = Field(default=0.0, description="Attention scratch memory factor", ge=0.0)

    # Maximum constraints (upper bounds for search space)
    max_tp: int = Field(default=64, description="Maximum tensor parallelism degree", ge=1)
    max_pp: int = Field(default=64, description="Maximum pipeline parallelism degree", ge=1)
    max_cp: int = Field(default=8, description="Maximum context parallelism degree", ge=1)
    max_dp: int = Field(default=1024, description="Maximum data parallelism degree", ge=1)
    max_ep: int = Field(default=8, description="Maximum expert parallelism degree", ge=1)

    # Disable flags
    no_cp: bool = Field(default=False, description="Disable context parallelism")
    no_ep: bool = Field(default=False, description="Disable expert parallelism")

    # Exact parameter constraints (when set, search space is restricted to this exact value)
    exact_tp: Optional[int] = Field(default=None, description="Force exact tensor parallelism degree", ge=1)
    exact_pp: Optional[int] = Field(default=None, description="Force exact pipeline parallelism degree", ge=1)
    exact_cp: Optional[int] = Field(default=None, description="Force exact context parallelism degree", ge=1)
    exact_dp: Optional[int] = Field(default=None, description="Force exact data parallelism degree", ge=1)
    exact_ep: Optional[int] = Field(default=None, description="Force exact expert parallelism degree", ge=1)

    # LoRA configuration
    lora: bool = Field(default=False, description="Enable LoRA fine-tuning mode")
    lora_r: int = Field(default=16, description="LoRA rank", ge=1)
    lora_include_regex: Optional[list[str]] = Field(default=None, description="LoRA include patterns")
    lora_exclude_regex: Optional[list[str]] = Field(default=None, description="LoRA exclude patterns")
    pretrained: str = Field(default="", description="Model ID for LoRA introspection")
    is_trusted: bool = Field(default=False, description="Trust remote code")

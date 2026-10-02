#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Constants for parallelism estimation.

This module contains true constants (not tunable parameters).
For tunable heuristic parameters, see config.py.
"""

# ============================================================================
# Memory Constants
# ============================================================================
BF16_BYTES = 2
FP32_BYTES = 4


# ============================================================================
# LoRA Module Detection Patterns
# ============================================================================
# Default patterns for LoRA module detection
DEFAULT_INCLUDE_PATTERNS = [
    r"\bq_proj\b",
    r"\bk_proj\b",
    r"\bv_proj\b",
    r"\bo_proj\b",
    r"\bqkv\b",
    r"\bqkv_proj\b",
    r"\bWqkv\b",
    r"\bW_pack\b",
    r"\bc_attn\b",  # merged QKV names
    r"\bgate_proj\b",
    r"\bup_proj\b",
    r"\bdown_proj\b",
    r"\bc_proj\b",  # GPT-2 out proj
]
DEFAULT_EXCLUDE_PATTERNS = [r"embed", r"lm_head", r"final_layer_norm", r"norm", r"layernorm", r"ln_"]
DEFAULT_MAX_SEQ_LENGTH = 4096

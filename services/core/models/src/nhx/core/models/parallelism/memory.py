#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Memory calculations and parameter counting for parallelism estimation.

This module contains:
- Parameter counting functions (standard, MoE, Mamba, GQA)
- Activation memory calculations
- Static memory calculations (params + grads + optimizer)
- Training framework overhead (FSDP2, NCCL, grad accumulation)
- Microbatch size fitting
"""

import math
from typing import Optional

from nhx.core.models.parallelism.constants import (
    BF16_BYTES,
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_INCLUDE_PATTERNS,
    FP32_BYTES,
)
from nhx.core.models.parallelism.models import EstimationParams, ModelSpec
from nhx.core.models.parallelism.utils import compile_patterns, name_matches

# ============================================================================
# Parameter Counting Functions
# ============================================================================


def mamba_layer_params(d_model: int, state_size: int = 16, d_inner_factor: int = 2) -> int:
    """
    Calculate parameters for a Mamba/SSM layer.

    Mamba layer components:
    - Input projection: d_model -> d_inner (typically 2*d_model)
    - Conv1d: d_inner * conv_kernel (small, ~4)
    - SSM parameters: A, B, C, D matrices
    - Output projection: d_inner -> d_model

    Approximation: Similar to attention but slightly different distribution
    """
    d_inner = d_model * d_inner_factor
    # Input proj + output proj + SSM params + conv
    # Roughly: 2*d_model*d_inner + state params ~ 4*d_model^2
    return 2 * d_model * d_inner + d_model * state_size


def param_size_bytes(
    n_layers: int,
    d_model: int,
    d_ff: int,
    vocab_size: int,
    tied_embeddings: bool = True,
    moe_config: Optional[dict] = None,
    mamba_config: Optional[dict] = None,
    n_kv_heads: Optional[int] = None,
    n_heads: Optional[int] = None,
    gated_mlp: bool = True,
) -> int:
    """
    Calculate parameter size in bytes.
    For MoE models, this accounts for multiple experts per layer.
    For Mamba models, this accounts for SSM layers vs attention layers.
    """
    # Just call param_count and multiply by FP32_BYTES
    params = param_count(
        n_layers, d_model, d_ff, vocab_size, tied_embeddings, moe_config, mamba_config, n_kv_heads, n_heads, gated_mlp
    )
    return params * FP32_BYTES


def param_count(
    n_layers: int,
    d_model: int,
    d_ff: int,
    vocab_size: int,
    tied_embeddings: bool = True,
    moe_config: Optional[dict] = None,
    mamba_config: Optional[dict] = None,
    n_kv_heads: Optional[int] = None,
    n_heads: Optional[int] = None,
    gated_mlp: bool = True,
) -> int:
    """
    Calculate total parameter count.

    Args:
        n_kv_heads: Number of KV heads for GQA/MQA (if None, assumes full attention)
        n_heads: Number of query heads (required if n_kv_heads is specified)
        gated_mlp: Whether FFN uses gated activation (3x params vs 2x)
    """
    # Calculate attention parameters (accounting for GQA/MQA)
    if n_kv_heads is not None and n_heads is not None and n_kv_heads < n_heads:
        # GQA/MQA: Q, K, V, O where K and V are smaller
        head_dim = d_model // n_heads
        q_params = d_model * d_model
        kv_params = 2 * d_model * (n_kv_heads * head_dim)
        o_params = d_model * d_model
        attn_per_layer = q_params + kv_params + o_params
    else:
        # Full attention: Q, K, V, O all same size
        attn_per_layer = 4 * d_model * d_model

    # FFN multiplier: 3x for gated (SwiGLU, GELU-gated), 2x for standard
    ffn_multiplier = 3 if gated_mlp else 2

    if mamba_config:
        # Hybrid or pure Mamba model
        num_attn_layers = mamba_config.get("num_attention_layers", 0)
        num_mamba_layers = mamba_config.get("num_mamba_layers", n_layers)
        num_mlp_layers = mamba_config.get("num_mlp_layers", 0)
        state_size = mamba_config.get("state_size", 16)

        if num_mlp_layers > 0:
            # Interleaved architecture (e.g., Nemotron-9B): layers are separate
            # - Pure Mamba layers (SSM only, no FFN)
            # - Pure MLP layers (FFN only)
            # - Pure Attention layers (attention only, no FFN)
            # Each block type has its own layer norm
            mamba_layer = mamba_layer_params(d_model, state_size)
            mamba_params = num_mamba_layers * (mamba_layer + d_model)  # +norm
            attn_params = num_attn_layers * (attn_per_layer + d_model)  # +norm
            mlp_params = num_mlp_layers * (ffn_multiplier * d_model * d_ff + d_model)  # +norm
            total_layers = mamba_params + attn_params + mlp_params
        else:
            # Combined architecture: Mamba/Attention + FFN in each layer
            # Attention layers (with full FFN + 2 layer norms)
            attn_params = num_attn_layers * (attn_per_layer + ffn_multiplier * d_model * d_ff + 2 * d_model)

            # Mamba layers (with full FFN + 2 layer norms)
            mamba_layer = mamba_layer_params(d_model, state_size)
            mamba_params = num_mamba_layers * (mamba_layer + ffn_multiplier * d_model * d_ff + 2 * d_model)

            total_layers = attn_params + mamba_params
    elif moe_config and moe_config.get("num_experts", 0) > 0:
        num_experts = moe_config["num_experts"]  # Routed experts (sharded by EP)
        num_expert_layers = moe_config.get("num_expert_layers", n_layers)
        num_shared_experts = moe_config.get("num_shared_experts", 0)  # Shared experts (replicated)

        # Use expert-specific FFN size if available (e.g., DeepSeek V3), otherwise use main FFN size
        expert_d_ff = moe_config.get("expert_ffn_size") or d_ff

        non_expert_layers = n_layers - num_expert_layers
        standard_ffn = non_expert_layers * (attn_per_layer + ffn_multiplier * d_model * d_ff + 2 * d_model)

        router_params_per_layer = d_model * num_experts
        # Routed experts + shared experts
        routed_expert_ffn = num_experts * ffn_multiplier * d_model * expert_d_ff
        shared_expert_ffn = num_shared_experts * ffn_multiplier * d_model * expert_d_ff
        total_expert_ffn_per_layer = routed_expert_ffn + shared_expert_ffn

        expert_layer_params = num_expert_layers * (
            attn_per_layer + router_params_per_layer + total_expert_ffn_per_layer + 2 * d_model
        )

        total_layers = standard_ffn + expert_layer_params
    else:
        # Standard dense transformer
        per_layer = attn_per_layer + ffn_multiplier * d_model * d_ff + 2 * d_model
        total_layers = n_layers * per_layer

    emb = vocab_size * d_model
    lm_head = 0 if tied_embeddings else vocab_size * d_model
    final_ln = d_model
    return total_layers + emb + lm_head + final_ln


def expert_params_count(n_layers: int, d_model: int, d_ff: int, moe_config: dict, gated_mlp: bool = True) -> int:
    """
    Calculate only the ROUTED expert FFN parameters (for EP sharding).

    Note: Shared experts are NOT included here as they are replicated, not sharded by EP.
    """
    if not moe_config or moe_config.get("num_experts", 0) == 0:
        return 0

    num_experts = moe_config["num_experts"]  # Routed experts only
    num_expert_layers = moe_config.get("num_expert_layers", n_layers)

    # Use expert-specific FFN size if available (e.g., DeepSeek V3), otherwise use main FFN size
    expert_d_ff = moe_config.get("expert_ffn_size") or d_ff

    # FFN multiplier: 3x for gated, 2x for standard
    ffn_multiplier = 3 if gated_mlp else 2

    # Routed expert FFN params: num_expert_layers * num_routed_experts * FFN
    routed_expert_ffn_per_layer = num_experts * ffn_multiplier * d_model * expert_d_ff
    return num_expert_layers * routed_expert_ffn_per_layer


def non_expert_params_count(
    n_layers: int,
    d_model: int,
    d_ff: int,
    vocab_size: int,
    tied_embeddings: bool,
    moe_config: Optional[dict],
    mamba_config: Optional[dict] = None,
    n_kv_heads: Optional[int] = None,
    n_heads: Optional[int] = None,
    gated_mlp: bool = True,
) -> int:
    """Calculate non-expert parameters (not sharded by EP)."""
    total = param_count(
        n_layers, d_model, d_ff, vocab_size, tied_embeddings, moe_config, mamba_config, n_kv_heads, n_heads, gated_mlp
    )
    expert = expert_params_count(n_layers, d_model, d_ff, moe_config, gated_mlp)
    return total - expert


# ============================================================================
# Activation Memory Functions
# ============================================================================


def activations_bytes_per_token(
    d_model: int,
    n_layers: int,
    ckpt_ratio: float = 1.0,
    moe_config: Optional[dict] = None,
    mamba_config: Optional[dict] = None,
) -> float:
    """
    Activation memory per token.
    For MoE, activations scale with experts_per_token, not total experts.
    For Mamba, activations are different (SSM state instead of attention activations).
    """
    if mamba_config:
        # Mamba models have different activation patterns
        num_attn_layers = mamba_config.get("num_attention_layers", 0)
        num_mamba_layers = mamba_config.get("num_mamba_layers", n_layers)
        state_size = mamba_config.get("state_size", 16)

        # Attention layers: standard 6*d per layer
        attn_act = num_attn_layers * 6 * d_model * BF16_BYTES

        # Mamba layers: SSM state + intermediate activations
        # SSM state: d_model * state_size (persistent state)
        # Intermediate: ~4*d_model (input/output projections)
        # Total per Mamba layer ~ 4-5*d_model (less than attention's 6*d)
        mamba_act_per_layer = (4 * d_model + d_model * state_size // 16) * BF16_BYTES
        mamba_act = num_mamba_layers * mamba_act_per_layer

        return (attn_act + mamba_act) * ckpt_ratio

    elif moe_config and moe_config.get("num_experts", 0) > 0:
        num_expert_layers = moe_config.get("num_expert_layers", n_layers)
        experts_per_tok = moe_config.get("num_experts_per_tok", 2)

        # Standard layers: 6 * d_model per layer
        non_expert_layers = n_layers - num_expert_layers
        standard_act = non_expert_layers * 6 * d_model * BF16_BYTES

        # Expert layers: attention activation + expert FFN activation (scaled by active experts)
        # Rough estimate: 4*d (attn) + 2*d*experts_per_tok (active expert FFNs)
        expert_act_per_layer = (4 * d_model + 2 * d_model * experts_per_tok) * BF16_BYTES
        expert_act = num_expert_layers * expert_act_per_layer

        return (standard_act + expert_act) * ckpt_ratio
    else:
        per_layer = 6 * d_model * BF16_BYTES
        return n_layers * per_layer * ckpt_ratio


def attention_kv_bytes_per_token(
    d_model: int, n_heads: int, n_layers: Optional[int] = None, mamba_config: Optional[dict] = None
) -> float:
    """
    Calculate KV cache memory per token.
    For Mamba models, only attention layers have KV cache.
    """
    head_dim = math.ceil(d_model / n_heads)  # padded head dim
    base_kv = 2 * n_heads * head_dim * BF16_BYTES

    if mamba_config and n_layers:
        # Only attention layers have KV cache
        num_attn_layers = mamba_config.get("num_attention_layers", n_layers)
        # Scale by ratio of attention layers
        return base_kv * (num_attn_layers / n_layers)

    return base_kv


def attention_scratch_bytes_per_token(n_heads: int, attn_scratch_factor: float = 0.0) -> float:
    """Calculate attention scratch buffer memory per token."""
    return attn_scratch_factor * n_heads * BF16_BYTES


def optimizer_bytes(param_bytes_fp32):
    """Calculate optimizer state memory (Adam: momentum + variance)."""
    # Adam states ~ 2 * FP32 (m + v)
    return param_bytes_fp32 * 2


def grad_bytes(param_bytes_fp32):
    """Calculate gradient memory."""
    # Grad tensors ~ BF16; use FP32->BF16 ratio
    return param_bytes_fp32 * (BF16_BYTES / FP32_BYTES)


# ============================================================================
# Memory Calculation Functions
# ============================================================================


def lora_params_from_linear_layers(
    linear_layers: list,
    include_patterns=None,
    exclude_patterns=None,
    lora_r: int = 8,
) -> int:
    """
    Compute LoRA parameter count from pre-computed linear layer specifications.

    This is an optimized version that uses pre-computed linear layer information
    from ModelSpec, avoiding the need to instantiate the model.

    Args:
        linear_layers: List of LinearLayerSpec objects from ModelSpec
        include_patterns: Regex patterns to include module names
        exclude_patterns: Regex patterns to exclude module names
        lora_r: LoRA rank

    Returns:
        Total LoRA parameter count
    """

    # In any case if we fail to extract linear layers, return 0
    # so that we don't fail the memory estimation entirely
    if not linear_layers:
        return 0

    total = 0

    # If user provided custom patterns, use those
    if include_patterns or exclude_patterns:
        # Pattern-based filtering (user override)
        include_res = compile_patterns(include_patterns or DEFAULT_INCLUDE_PATTERNS)
        exclude_res = compile_patterns(exclude_patterns or DEFAULT_EXCLUDE_PATTERNS)
        use_patterns = True
    else:
        # Introspection-based filtering (default, more robust)
        # Use DEFAULT_EXCLUDE_PATTERNS for consistency
        exclude_res = compile_patterns(DEFAULT_EXCLUDE_PATTERNS)
        use_patterns = False

    for layer in linear_layers:
        name = layer.name

        # Check eligibility using exclude patterns
        if use_patterns:
            # Pattern-based (user provided both include and exclude)
            if not name_matches(name, include_res, exclude_res):
                continue
        else:
            # Introspection-based (default): only exclude, no include filter
            if any(pattern.search(name) for pattern in exclude_res):
                continue

        total += lora_r * (layer.in_features + layer.out_features)

    return total


def calculate_param_counts(args: EstimationParams, model_cfg: ModelSpec) -> dict[str, int]:
    """Calculate base, expert, non-expert, and LoRA parameter counts."""

    n_layers = model_cfg.num_layers
    d_model = model_cfg.hidden_size
    d_ff = model_cfg.ffn_hidden_size
    vocab = model_cfg.vocab_size
    tied_emb = model_cfg.tied_embeddings
    moe_dict = model_cfg.moe_config.model_dump() if model_cfg.moe_config else None
    mamba_dict = model_cfg.mamba_config.model_dump() if model_cfg.mamba_config else None

    # Base param count
    base_param_cnt = param_count(
        n_layers,
        d_model,
        d_ff,
        vocab,
        tied_emb,
        moe_dict,
        mamba_dict,
        model_cfg.num_kv_heads,
        model_cfg.num_attention_heads,
        model_cfg.gated_mlp,
    )

    # For MoE, separate expert and non-expert params
    if model_cfg.moe_config:
        expert_param_cnt = expert_params_count(n_layers, d_model, d_ff, moe_dict, model_cfg.gated_mlp)
        non_expert_param_cnt = non_expert_params_count(
            n_layers,
            d_model,
            d_ff,
            vocab,
            tied_emb,
            moe_dict,
            mamba_dict,
            model_cfg.num_kv_heads,
            model_cfg.num_attention_heads,
            model_cfg.gated_mlp,
        )
    else:
        expert_param_cnt = 0
        non_expert_param_cnt = base_param_cnt

    # LoRA params from actual modules
    if args.lora:
        lora_params_cnt = lora_params_from_linear_layers(
            model_cfg.linear_layers,
            include_patterns=args.lora_include_regex,
            exclude_patterns=args.lora_exclude_regex,
            lora_r=args.lora_r,
        )

    else:
        lora_params_cnt = 0

    return {
        "base": base_param_cnt,
        "expert": expert_param_cnt,
        "non_expert": non_expert_param_cnt,
        "lora": lora_params_cnt,
    }


def calculate_static_memory_per_rank(
    args: EstimationParams,
    tp: int,
    pp: int,
    dp: int,
    ep: int,
    param_counts: dict[str, int],
) -> float:
    """
    Calculate static memory per rank (parameters, gradients, optimizer states).

    Returns memory in bytes.
    """
    non_expert_param_cnt = param_counts["non_expert"]
    expert_param_cnt = param_counts["expert"]
    lora_params_cnt = param_counts["lora"]

    if args.lora:
        # ----- LoRA (base frozen with Distributed Optimizer) -----
        # Base model (frozen, BF16): replicated across DP
        non_expert_bf16_b = non_expert_param_cnt * BF16_BYTES
        non_expert_per_rank_b = (non_expert_bf16_b / tp) / pp

        expert_bf16_b = expert_param_cnt * BF16_BYTES
        expert_per_rank_b = (expert_bf16_b / tp) / pp / ep

        base_bf16_per_rank_b = non_expert_per_rank_b + expert_per_rank_b

        # LoRA adapters (trainable):
        # - BF16 params: replicated across DP
        # - FP32 gradients: replicated across DP
        # - FP32 optimizer: sharded by DP (distributed optimizer)
        lora_bf16_per_rank_b = (lora_params_cnt * BF16_BYTES) / tp / pp
        lora_grad_per_rank_b = (lora_params_cnt * FP32_BYTES) / tp / pp
        lora_optim_per_rank_b = (lora_params_cnt * 8) / tp / pp / dp  # Distributed!

        static_b = base_bf16_per_rank_b + lora_bf16_per_rank_b + lora_grad_per_rank_b + lora_optim_per_rank_b
    else:
        # ----- Full SFT (Mixed Precision BF16 Training with Distributed Optimizer) -----
        # GPU Memory components (FSDP2/Distributed Optimizer):
        # - BF16 model parameters: 2 bytes/param (replicated, sharded by TP/PP/EP only)
        # - FP32 gradients: 4 bytes/param (replicated, sharded by TP/PP/EP only)
        # - FP32 optimizer states: 8 bytes/param (SHARDED by DP - distributed optimizer!)
        # - FP32 master copy: on disk, NOT in GPU memory

        # BF16 working copy of model (sharded by TP/PP/EP, replicated across DP)
        non_expert_bf16_b = non_expert_param_cnt * BF16_BYTES
        non_expert_bf16_per_rank = (non_expert_bf16_b / tp) / pp

        expert_bf16_b = expert_param_cnt * BF16_BYTES
        expert_bf16_per_rank = (expert_bf16_b / tp) / pp / ep

        model_bf16_b = non_expert_bf16_per_rank + expert_bf16_per_rank

        # FP32 gradients (sharded by TP/PP/EP, replicated across DP)
        non_expert_grad_b = non_expert_param_cnt * FP32_BYTES
        non_expert_grad_per_rank = (non_expert_grad_b / tp) / pp

        expert_grad_b = expert_param_cnt * FP32_BYTES
        expert_grad_per_rank = (expert_grad_b / tp) / pp / ep

        grad_b = non_expert_grad_per_rank + expert_grad_per_rank

        # FP32 optimizer states: 8 bytes/param (SHARDED by TP/PP/EP/DP - distributed!)
        non_expert_optim_b = non_expert_param_cnt * 8
        non_expert_optim_per_rank = (non_expert_optim_b / tp) / pp / dp

        expert_optim_b = expert_param_cnt * 8
        expert_optim_per_rank = (expert_optim_b / tp) / pp / ep / dp

        optim_b = non_expert_optim_per_rank + expert_optim_per_rank

        # Total GPU memory: BF16 model + FP32 grads + FP32 optimizer (distributed)
        static_b = model_bf16_b + grad_b + optim_b

    return static_b


def calculate_activation_bytes_per_token(
    args: EstimationParams, model_cfg: ModelSpec, tp: int, cp: int
) -> tuple[float, float, float]:
    """Calculate activation memory per token (activations, KV cache, scratch)."""
    d_model = model_cfg.hidden_size
    n_heads = model_cfg.num_attention_heads
    n_layers = model_cfg.num_layers
    moe_dict = model_cfg.moe_config.model_dump() if model_cfg.moe_config else None
    mamba_dict = model_cfg.mamba_config.model_dump() if model_cfg.mamba_config else None

    # Activations per token per rank
    # TP shards model dimension, CP shards sequence dimension
    act_tok_b = activations_bytes_per_token(d_model, n_layers, args.act_ckpt_ratio, moe_dict, mamba_dict) / tp / cp
    kv_tok_b = attention_kv_bytes_per_token(d_model, n_heads, n_layers, mamba_dict) / tp / cp
    scr_tok_b = attention_scratch_bytes_per_token(n_heads, args.attn_scratch_factor) / tp / cp

    return act_tok_b, kv_tok_b, scr_tok_b


def calculate_training_overhead_gb(
    per_rank_static_gb: float,
    num_layers: int,
    tp: int,
    pp: int,
    dp: int,
) -> float:
    """
    Calculate training framework overhead that is not captured in static memory or activations.

    This accounts for:
    1. FSDP2 prefetch buffers: Pre-fetches next layer's parameters during forward/backward
    2. NCCL communication buffers: Temporary buffers for all-reduce, all-gather operations
    3. Gradient accumulation buffers: Temporary storage during gradient accumulation

    Args:
        per_rank_static_gb: Static memory per rank (params + grads + optimizer)
        num_layers: Number of transformer layers
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        dp: Data parallelism degree

    Returns:
        Overhead in GB
    """
    # 1. FSDP2 Prefetch Buffers
    # FSDP2 prefetches the next layer's parameters while computing current layer
    # Typically prefetches 1-2 layers ahead
    # Overhead = (params_per_layer + grads_per_layer) * prefetch_layers
    # Static memory includes params (BF16) + grads (FP32) + optimizer (FP32, sharded by DP)
    # For prefetch, we only need params + grads (not optimizer)
    # Approximate: params+grads ≈ 60% of static memory (since optimizer is 8 bytes/param sharded by DP)
    params_grads_ratio = 0.6 if dp > 1 else 0.75  # Higher if no DP sharding of optimizer
    params_grads_per_rank_gb = per_rank_static_gb * params_grads_ratio
    params_grads_per_layer_gb = params_grads_per_rank_gb / num_layers
    prefetch_layers = 2  # FSDP2 typically prefetches 2 layers ahead
    fsdp_prefetch_gb = params_grads_per_layer_gb * prefetch_layers

    # 2. NCCL Communication Buffers
    # NCCL allocates temporary buffers for collective operations (all-reduce, all-gather, etc.)
    # Buffer size depends on the amount of data being communicated
    # For TP: all-reduce of activations and gradients
    # For DP: all-reduce of gradients
    # For PP: send/recv of activations
    # Empirically, NCCL buffers are ~10% of static memory for multi-dimensional parallelism
    nccl_buffer_factor = 0.05  # Base 5%
    if tp > 1:
        nccl_buffer_factor += 0.03  # +3% for TP communication
    if dp > 1:
        nccl_buffer_factor += 0.02  # +2% for DP gradient all-reduce
    if pp > 1:
        nccl_buffer_factor += 0.03  # +3% for PP pipeline communication
    nccl_buffer_gb = per_rank_static_gb * nccl_buffer_factor

    # 3. Gradient Accumulation Temporary Buffers
    # During gradient accumulation, temporary buffers are needed for:
    # - Accumulating gradients across microbatches
    # - FP32 gradient buffers for mixed precision training
    # - Temporary storage during all-reduce operations
    # Overhead is proportional to gradient size, which is ~1/3 of static memory
    # (since static = params + grads + optimizer, and grads ≈ 1/3 of that)
    grad_accum_factor = 0.05  # 5% of static memory for gradient accumulation buffers
    grad_accum_gb = per_rank_static_gb * grad_accum_factor

    # Total overhead
    total_overhead_gb = fsdp_prefetch_gb + nccl_buffer_gb + grad_accum_gb

    return total_overhead_gb


def find_max_microbatch(
    args: EstimationParams,
    model_cfg: ModelSpec,
    static_b: float,
    act_tok_b: float,
    kv_tok_b: float,
    scr_tok_b: float,
    cp: int,
) -> int:
    """
    Find maximum microbatch size that fits in memory.

    If microbatch_size is specified, validate it fits; otherwise binary search.
    """
    mem_budget_b = args.gpu_mem_gb * (1024**3)

    # Effective sequence length per rank with CP
    effective_seq_len = args.seq_len // cp

    # KV cache effective length (reduced by sliding window if present)
    if model_cfg.sliding_window_config:
        # With sliding window, KV cache only stores window_size tokens
        kv_effective_seq_len = min(args.seq_len, model_cfg.sliding_window_config.window_size) // cp
    else:
        kv_effective_seq_len = effective_seq_len

    def fits(mb):
        # Activations use full sequence, KV cache uses sliding window length
        act_b = act_tok_b * (mb * effective_seq_len)
        kv_b = kv_tok_b * (mb * kv_effective_seq_len)
        scr_b = scr_tok_b * (mb * effective_seq_len)
        total = static_b + 1.1 * (act_b + kv_b + scr_b)
        return total <= mem_budget_b

    # If microbatch_size is specified, use it directly; otherwise search
    if hasattr(args, "microbatch_size") and args.microbatch_size is not None:
        best_mb = args.microbatch_size
        if not fits(best_mb):
            # Skip this config if the fixed MBS doesn't fit
            return 0
    else:
        # Binary search for maximum fitting microbatch size
        lo = 0
        hi = args.max_microbatch
        best_mb = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(mid):
                best_mb = mid
                lo = mid + 1
            else:
                hi = mid - 1

    return best_mb

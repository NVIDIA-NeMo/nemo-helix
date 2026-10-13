#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parallelism Helper Library - Production-ready LLM training parallelization estimator.

This library provides accurate memory estimates and parallelization recommendations for:
- Tensor Parallelism (TP): Shards layers across GPUs
- Pipeline Parallelism (PP): Splits layers across stages
- Data Parallelism (DP): Replicates model, shards data (FSDP/ZeRO)
- Context Parallelism (CP): Shards sequence for long contexts (>8K tokens)
- Expert Parallelism (EP): Shards experts in MoE models

Supported Architectures:
- Standard Transformers (GPT, LLaMA, Mistral, Qwen, Phi)
- Mixture of Experts (Mixtral, DeepSeek-MoE, GPT-OSS)
- State-Space Models (Mamba, Hybrid Mamba-Transformer)
- Grouped Query Attention (GQA/MQA)
- Sliding Window Attention (Mistral, Gemma)

Memory Model:
- BF16 mixed precision: 14 bytes/param (2 BF16 + 4 FP32 grad + 8 FP32 optimizer)
- Distributed optimizer (FSDP2): Optimizer states sharded across DP
- Activation checkpointing: Configurable via act_ckpt_ratio
- KV cache: Reduced for Mamba layers and sliding window attention

Validation:
- Tested against NVIDIA NeMo H100 configurations
- Parameter counting: 0.0-0.8% error across 14+ model families
- Communication heuristic tuned to match NeMo preferences

Usage:
    >>> from parallelism_helper import estimate_parallelization
    >>> result = estimate_parallelization("meta-llama/Meta-Llama-3-8B", 8, 80, 8192)
    >>> print(f"Best: TP={result.configs[0].tp} DP={result.configs[0].dp}")

For detailed documentation, see PARALLELISM_README.md
"""

import logging
from typing import Optional

from nhx.core.models.parallelism.config import get_config
from nhx.core.models.parallelism.constants import FP32_BYTES  # Keep for backward compatibility
from nhx.core.models.parallelism.memory import param_count
from nhx.core.models.parallelism.models import EstimationParams, ModelSpec
from nhx.core.models.parallelism.utils import divisors

# Module-level logger
logger = logging.getLogger(__name__)


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


# ---------------- Heuristic communication score ----------------
def _compute_memory_pressure_penalty(static_memory_gb: float, gpu_memory_gb: float) -> float:
    """
    Penalize configurations that leave little room for activations.

    Reasoning:
        Static memory (parameters + gradients + optimizer states) should not consume
        too much GPU memory, as we need headroom for:
        - Activation memory (scales with batch size and sequence length)
        - KV cache (for inference/generation)
        - Temporary buffers (gradient accumulation, communication)

        When static memory exceeds the threshold (default 60%) of GPU memory, the
        configuration becomes memory-constrained and:
        - Limits achievable batch size (hurts throughput)
        - Risks OOM with longer sequences or larger batches
        - Leaves little room for framework overhead

        We use a quadratic penalty to strongly discourage tight memory configs
        and naturally push the heuristic toward higher TP/PP (which reduces
        per-rank static memory through sharding).

    Args:
        static_memory_gb: Static memory per rank (params + grads + optimizer)
        gpu_memory_gb: Total GPU memory available
        config: ParallelismConfig instance (uses global default if None)

    Returns:
        Penalty value (0 if memory ratio <= threshold, quadratic increase above)
    """
    config = get_config()

    if static_memory_gb <= 0 or gpu_memory_gb <= 0:
        return 0

    memory_ratio = static_memory_gb / gpu_memory_gb
    threshold = config.memory.pressure_threshold
    if memory_ratio <= threshold:
        return 0

    # Quadratic penalty: excess=0.1 -> 1e9, excess=0.2 -> 4e9, etc.
    excess = memory_ratio - threshold
    base_penalty = config.memory.base_penalty
    scale_divisor = config.memory.scale_divisor
    return base_penalty * (excess / scale_divisor) ** 2


def _compute_tp_cost(tp: int, n_layers: int, d_model: int, param_count_b: float) -> float:
    """
    Compute cost for Tensor Parallelism (TP).

    Reasoning:
        TP shards individual layers across GPUs, requiring all-reduce communication
        after every forward/backward pass through each layer. This has both benefits
        and costs:

        Benefits:
        - Reduces per-GPU memory by sharding parameters across TP ranks
        - Essential for very large models that don't fit on single GPU
        - Synchronous communication (no pipeline bubbles like PP)

        Costs:
        - Frequent all-reduce operations (2 per layer per step)
        - Communication scales with: n_layers * d_model * TP degree
        - Bandwidth-intensive for large hidden dimensions

        Model size scaling:
        - Small models (<70B): TP is expensive overhead, prefer DP
        - Medium models (70-300B): TP=4 is reasonable, TP=8 has moderate cost
        - Very large models (>300B): TP=8 is necessary, lower per-unit cost

        Empirical observations from NeMo configs:
        - 8B models: TP=1 preferred
        - 70B models: TP=4 standard
        - 340B+ models: TP=8 standard

    Args:
        tp: Tensor parallelism degree
        n_layers: Number of transformer layers
        d_model: Hidden dimension size
        param_count_b: Total parameter count in billions

    Returns:
        TP cost (higher = worse)
    """
    config = get_config()
    tp_cfg = config.tensor_parallelism
    size_thresholds = config.model_size_thresholds

    # Base cost: communication volume per step
    base_multiplier = (
        tp_cfg.base_cost_very_large_model
        if param_count_b > size_thresholds.very_large
        else tp_cfg.base_cost_standard_model
    )
    base_cost = n_layers * tp * d_model * base_multiplier

    # Penalty for excessive TP beyond what's reasonable for model size
    if param_count_b > size_thresholds.very_large:
        excessive_tp = max(0, tp - tp_cfg.excessive_very_large)  # TP=8 is standard for 340B+
    elif param_count_b > size_thresholds.small_tp:
        excessive_tp = max(0, tp - tp_cfg.excessive_standard)  # TP=4 is standard for 70B
    else:
        excessive_tp = max(0, tp - tp_cfg.excessive_standard)  # TP>4 wastes bandwidth for <70B

    penalty_multiplier = (
        tp_cfg.penalty_large_model if param_count_b > size_thresholds.small_tp else tp_cfg.penalty_small_model
    )
    return base_cost + penalty_multiplier * excessive_tp


def _compute_dp_cost(dp: int, tp: int, pp: int, cp: int, ep: int, num_experts: int, param_count_b: float) -> float:
    """
    Compute bonus for Data Parallelism (DP). Note: negative cost = bonus!

    Reasoning:
        DP is the most efficient parallelism strategy for throughput because:
        - Gradient all-reduce happens once per optimizer step (infrequent)
        - Each rank processes different data (embarrassingly parallel)
        - No pipeline bubbles or synchronization overhead
        - Linear scaling in throughput with DP degree (ideal)

        However, DP bonus should be tempered by:

        1. Total parallelism: When TP*PP*CP*EP is already high, adding more DP
           has diminishing returns because:
           - The model is already heavily sharded (memory-constrained)
           - Communication overhead from other dimensions dominates
           - Focus should be on fitting the model, not maximizing DP

        2. MoE models: For Mixture of Experts, Expert Parallelism (EP) is more
           important than DP because:
           - EP provides memory efficiency by sharding experts
           - Large MoE models (>40B) need EP to fit in memory
           - EP enables scaling expert count without memory blowup

        3. Model size: Very large models require high TP/PP/EP for memory,
           leaving less room for DP scaling.

        Empirical tuning based on NeMo configs:
        - Small models (TP=1, PP=1): Strong DP preference
        - Medium models (TP*PP=8-32): Moderate DP preference
        - Large models (TP*PP=64+): Minimal DP preference
        - Large MoE (>40B): No DP preference (EP dominates)

    Args:
        dp: Data parallelism degree
        tp, pp, cp, ep: Other parallelism dimensions
        num_experts: Number of experts (0 for dense models)
        param_count_b: Total parameter count in billions

    Returns:
        DP cost (negative = bonus for throughput)
    """
    config = get_config()
    dp_cfg = config.data_parallelism
    size_thresholds = config.model_size_thresholds

    total_parallelism = tp * pp * cp * ep

    # Large MoE: EP dominates, no DP bonus
    if num_experts > 0 and param_count_b > size_thresholds.small_moe:
        return 0

    # Smaller MoE: minimal DP bonus
    if num_experts > 0:
        return dp_cfg.bonus_small_moe * dp

    # Dense models: DP bonus scales inversely with total parallelism
    bonus = 0.0
    if total_parallelism >= dp_cfg.total_parallelism_very_high:
        bonus = dp_cfg.bonus_minimal * dp  # Minimal: memory-constrained regime, prefer PP
    elif total_parallelism >= dp_cfg.total_parallelism_high:
        bonus = dp_cfg.bonus_small * dp  # Small: high parallelism already, prefer PP
    elif total_parallelism <= dp_cfg.total_parallelism_very_low:
        bonus = dp_cfg.bonus_very_strong * dp  # Very strong: simple setups benefit most
    elif total_parallelism <= dp_cfg.total_parallelism_low:
        bonus = dp_cfg.bonus_strong * dp  # Strong: small clusters, prefer pure DP
    elif total_parallelism <= dp_cfg.total_parallelism_medium:
        bonus = dp_cfg.bonus_moderate * dp  # Moderate: balanced regime
    else:
        bonus = dp_cfg.bonus_medium * dp  # Medium: transition zone

    # When using CP, strongly prefer pure DP over TP/PP splitting
    # CP already handles sequence parallelism, so DP for data is most efficient
    if cp > 1 and tp == 1 and pp == 1:
        bonus *= dp_cfg.cp_bonus_multiplier  # Double the DP bonus for pure DP+CP configurations

    return bonus


def _compute_pp_cost(
    pp: int, tp: int, num_experts: int, param_count_b: float, static_memory_gb: float, gpu_memory_gb: float
) -> float:
    """
    Compute cost for Pipeline Parallelism (PP).

    Reasoning:
        PP divides the model into stages across GPUs, creating a pipeline where
        different micro-batches are processed at different stages simultaneously.

        Benefits:
        - Enables training models that don't fit on single GPU
        - Reduces per-GPU memory by distributing layers across stages
        - Can achieve good throughput with many micro-batches

        Costs:
        - Pipeline bubbles: idle time at start/end of each batch
          * Bubble overhead = (PP - 1) / num_microbatches
          * Requires large global batch size to amortize bubbles
        - Forward/backward dependencies limit parallelism
        - Point-to-point communication between stages

        Model size considerations:
        - Small models (<100B): PP has high bubble overhead, prefer pure DP
        - Medium models (100-300B): PP becomes acceptable if TP alone insufficient
        - Very large models (>300B): PP is often necessary for memory

        Interaction with TP:
        - When TP>1: Model already sharded, PP becomes relatively cheaper
        - When TP=1: Prefer scaling DP for throughput before adding PP

        MoE models:
        - PP is useful for distributing expert-heavy layers across stages
        - Lower bubble impact because expert layers have more compute
        - Enables scaling expert count across pipeline stages

        Memory pressure considerations:
        - When static memory is high (>50% GPU), PP becomes more attractive
        - PP distributes parameters across stages, reducing per-rank memory
        - This aligns with NeMo's strategy of using higher PP under memory pressure

    Args:
        pp: Pipeline parallelism degree
        tp: Tensor parallelism degree (affects relative cost)
        num_experts: Number of experts (>0 for MoE)
        param_count_b: Total parameter count in billions
        static_memory_gb: Static memory per rank (for memory pressure adjustment)
        gpu_memory_gb: Total GPU memory available

    Returns:
        PP cost (0 if PP=1, increases with PP degree)
    """
    if pp == 1:
        return 0

    config = get_config()
    pp_cfg = config.pipeline_parallelism
    size_thresholds = config.model_size_thresholds
    memory_cfg = config.memory

    # MoE: PP is more acceptable due to higher compute per stage
    if num_experts > 0:
        return pp_cfg.cost_moe * (pp - 1)  # Lower to compete with DP bonus, especially for small MoE

    # Dense models: PP cost scales inversely with model size
    if param_count_b > size_thresholds.very_large:
        multiplier = pp_cfg.cost_very_large_model  # Very large: PP necessary, low cost
    elif param_count_b > size_thresholds.large:
        multiplier = pp_cfg.cost_large_model  # Large: PP useful, moderate cost
    elif param_count_b > size_thresholds.medium:
        multiplier = pp_cfg.cost_medium_model  # Medium: PP has cost but acceptable
    elif tp > 1:
        multiplier = pp_cfg.cost_small_with_tp  # Small with TP: PP relatively expensive
    else:
        multiplier = pp_cfg.cost_small_without_tp  # Small without TP: strongly prefer DP scaling

    base_cost = multiplier * (pp - 1)

    # Memory pressure discount: Make PP cheaper when memory is tight
    # This encourages PP for memory distribution when needed
    if static_memory_gb > 0 and gpu_memory_gb > 0:
        memory_ratio = static_memory_gb / gpu_memory_gb
        if memory_ratio > memory_cfg.pressure_low:
            # Above 45% memory usage: apply discount to PP
            # Discount increases with memory pressure (up to 70% discount at 80% memory)
            excess = memory_ratio - memory_cfg.pressure_low
            discount_factor = min(
                memory_cfg.pp_discount_max, excess * memory_cfg.pp_discount_scale
            )  # 0 to 0.7 discount
            base_cost *= 1 - discount_factor

    return base_cost


def _compute_cp_cost(cp: int, seq_len: int, n_layers: int, d_model: int) -> float:
    """
    Compute cost for Context Parallelism (CP).

    Reasoning:
        CP shards the sequence dimension across GPUs, reducing per-GPU memory for
        activations and KV cache. This is beneficial when sequence-related memory
        dominates parameter memory.

        Memory breakdown per token (mixed precision):
        - Activations: ~34 * d_model * n_layers bytes/token
        - KV cache: ~4 * d_model * n_layers bytes/token
        - Total: ~38 * d_model * n_layers bytes/token

        Parameter memory (with distributed optimizer):
        - Model weights: 2 bytes/param (BF16)
        - Gradients: 4 bytes/param (FP32)
        - Optimizer states: 8 bytes/param (FP32 Adam)
        - Total: ~14 bytes/param
        - Approximate params: n_layers * 12 * d_model^2

        When to use CP:
        - seq_to_param_ratio > 1.0: Sequence memory dominates, CP highly beneficial
          * Common with long contexts (32K+) on smaller models (7-13B)
          * Optimal CP scales with ratio (more memory pressure = higher CP)

        - seq_to_param_ratio 0.3-1.0: Moderate sequence memory, CP=2 is good
          * Typical with 8-16K contexts on medium models (70B)

        - seq_to_param_ratio < 0.3: Short sequences, CP=1 is best
          * Standard training sequences (2-4K) on any model size
          * CP overhead not justified

        Communication cost:
        - CP requires all-gather for attention computation
        - Scales with sequence length and number of heads
        - Worth it when memory savings enable larger batches or longer sequences

    Args:
        cp: Context parallelism degree
        seq_len: Sequence length
        n_layers: Number of transformer layers
        d_model: Hidden dimension size

    Returns:
        CP cost (negative = bonus when beneficial, positive = penalty otherwise)
    """
    config = get_config()
    cp_cfg = config.context_parallelism

    # Estimate sequence-to-parameter memory ratio
    param_memory = n_layers * cp_cfg.param_layers_multiplier * (d_model**2) * cp_cfg.param_memory_multiplier
    seq_memory = cp_cfg.seq_memory_multiplier * d_model * n_layers * seq_len
    seq_to_param_ratio = seq_memory / param_memory

    if seq_to_param_ratio > cp_cfg.seq_to_param_ratio_high:
        # Very long sequences: CP is highly beneficial
        optimal_cp = min(cp_cfg.max_value, max(cp_cfg.optimal_value, int(seq_to_param_ratio)))
        if cp == optimal_cp:
            return cp_cfg.bonus_optimal  # Strong bonus for optimal CP
        elif cp < optimal_cp:
            return cp_cfg.penalty_suboptimal * (optimal_cp - cp)  # Penalty: not using enough CP
        else:
            return cp_cfg.penalty_too_much * (cp - optimal_cp)  # Smaller penalty: too much CP
    elif seq_to_param_ratio > cp_cfg.seq_to_param_ratio_medium:
        # Medium sequences: CP=2 is optimal
        if cp == cp_cfg.optimal_value:
            return cp_cfg.bonus_good  # Bonus for CP=2
        elif cp == 1:
            return cp_cfg.penalty_should_use  # Penalty: should use CP
        else:
            return cp_cfg.penalty_too_much * abs(cp - cp_cfg.optimal_value)  # Penalty: wrong CP value
    else:
        # Short sequences: CP adds overhead without benefit
        return cp_cfg.penalty_suboptimal * (cp - 1)  # Penalize any CP > 1


def _compute_ep_cost(ep: int, num_experts: int) -> float:
    """
    Compute cost for Expert Parallelism (EP) in Mixture of Experts (MoE) models.

    Reasoning:
        MoE models have expert layers where each token is routed to a subset of experts.
        This creates unique memory challenges:

        Without EP (EP=1):
        - All ROUTED experts replicated on every GPU
        - Memory scales with: (num_routed_experts * expert_size)
        - Becomes infeasible for models with many large experts
        - Example: 8x7B experts = 56B params per GPU!

        With EP:
        - Routed experts SHARDED across EP ranks
        - Each GPU holds: num_routed_experts / EP experts
        - Memory per GPU reduced by factor of EP
        - Example: 8 experts / EP=8 = 1 expert per GPU

        Note on shared vs routed experts:
        - Routed experts: Sharded by EP (counted in num_experts)
        - Shared experts: Replicated on all ranks (NOT sharded by EP)
        - EP only affects routed expert memory

        Optimal EP strategies:
        - EP = num_routed_experts: Perfect (1 expert/GPU), max memory efficiency
        - EP divides num_routed_experts evenly: Good load balancing
        - EP doesn't divide evenly: Bad (uneven expert distribution)
        - EP = 1 on MoE: Terrible (no memory benefit, model won't fit)

        Communication cost:
        - All-to-all for expert routing (tokens sent to expert GPUs)
        - Scales with: (batch_size * seq_len * num_experts_per_token)
        - Worth it for memory savings (enables training large MoE)

        Empirical observations from NeMo configs:
        - Mixtral 8x7B: EP=8 (1 expert per GPU)
        - DeepSeek V3 671B: EP=8 (256 routed experts / 8 = 32 per GPU)
        - GPT-OSS 120B: EP=4 (16 experts / 4 = 4 per GPU)

    Args:
        ep: Expert parallelism degree
        num_experts: Number of ROUTED experts (excludes shared experts)

    Returns:
        EP cost (negative = bonus for good EP, positive = penalty for bad EP)
    """
    config = get_config()
    ep_cfg = config.expert_parallelism

    if num_experts == 0:
        # Non-MoE: EP must be 1 (no experts to shard)
        return ep_cfg.penalty_non_moe * (ep - 1)

    # MoE: reward high EP for memory efficiency
    # With NeMo Automodel constraint (DP × CP) % EP == 0, we want to maximize EP
    # within the valid range. Add a small bonus per EP degree to prefer higher EP.

    # Check EP=1 first (huge penalty for no sharding on MoE)
    if ep == 1:
        return ep_cfg.penalty_no_sharding  # Huge penalty: no EP on MoE!
    if ep == num_experts:
        # Perfect: 1 routed expert per GPU
        # Also add scaling bonus to ensure it's better than any other EP value
        return ep_cfg.bonus_perfect + (ep_cfg.bonus_high_count * ep)
    elif num_experts % ep == 0:
        experts_per_gpu = num_experts // ep
        if experts_per_gpu <= ep_cfg.experts_per_gpu_very_efficient:
            base_bonus = ep_cfg.bonus_very_efficient  # Very efficient
        elif experts_per_gpu <= ep_cfg.experts_per_gpu_good:
            base_bonus = ep_cfg.bonus_good  # Good
        elif experts_per_gpu <= ep_cfg.experts_per_gpu_acceptable:
            base_bonus = ep_cfg.bonus_acceptable  # Acceptable
        else:
            base_bonus = ep_cfg.bonus_high_count  # High expert count per GPU

        # Add bonus for higher EP to prefer EP=4 over EP=2 over EP=1
        # Each additional EP degree gets a small bonus (negative = better)
        ep_scale_bonus = ep_cfg.bonus_high_count * ep
        return base_bonus + ep_scale_bonus
    else:
        return ep_cfg.penalty_non_divisor * abs(ep - num_experts)  # Non-divisor: uneven distribution


def _compute_balance_bonus(
    tp: int,
    pp: int,
    ep: int,
    num_experts: int,
    param_count_b: float,
    static_memory_gb: float,
    gpu_memory_gb: float,
) -> float:
    """
    Compute bonus for balanced TP/PP combinations.

    Reasoning:
        When using both TP and PP, the balance between them affects efficiency:

        Why balance matters:
        - TP and PP both shard the model, reducing per-GPU memory
        - Balanced TP/PP (e.g., TP=4 PP=4) often better than unbalanced (TP=8 PP=2)
        - Reasons for balance preference:
          1. More uniform memory distribution across GPUs
          2. Better overlap of TP communication with PP computation
          3. Reduced pipeline bubble ratio with more stages
          4. Empirically validated by NeMo configs

        Model size scaling:
        - Small models (<100B): Moderate balance preference
          * Example: TP=2 PP=2 > TP=4 PP=1

        - Large models (100-300B): Strong balance preference
          * Example: TP=4 PP=4 > TP=8 PP=2

        - Very large dense models (>300B): Very strong balance preference
          * Example: TP=8 PP=8 > TP=16 PP=4
          * Perfect balance (TP=PP) is highly optimal

        MoE considerations (EP saturation based on GPU topology):
        - Few experts (<= GPUs per node): Must max EP (EP == num_experts)
          * Example: Mixtral (8 experts) needs EP=8, not EP=4
          * Standard node has 8 GPUs, so 8 experts needs EP=8

        - Many experts (> GPUs per node): EP at node-level is saturated
          * EP >= gpus_per_node (typically 8) is sufficient
          * Example: DeepSeek V3 (256 experts) with EP=8 is saturated

        When EP is saturated:
        - Low/no PP (PP <= 2): No balance bonus (prefer max TP)
        - Significant PP (PP >= 4): Balance bonus applies
          * Quadratic scaling: TP=8 PP=8 >> TP=4 PP=4

        Balance ratio:
        - ratio = 1.0 (perfect balance): Strongest bonus
        - ratio <= 2.0 (good balance): Moderate bonus
        - ratio > 2.0 (unbalanced): No bonus

    Args:
        tp: Tensor parallelism degree
        pp: Pipeline parallelism degree
        ep: Expert parallelism degree
        num_experts: Number of experts (0 for dense models)
        param_count_b: Total parameter count in billions
        static_memory_gb: Static memory per rank (for memory-aware balance decisions)
        gpu_memory_gb: Total GPU memory available

    Returns:
        Balance bonus (negative = bonus, 0 = no bonus)
    """
    if tp <= 1 or pp <= 1:
        return 0  # No balance consideration if not using both TP and PP

    config = get_config()
    balance_cfg = config.balance
    size_thresholds = config.model_size_thresholds
    memory_cfg = config.memory

    ratio = max(tp, pp) / min(tp, pp)

    # Calculate memory utilization ratio
    memory_ratio = static_memory_gb / gpu_memory_gb if (static_memory_gb > 0 and gpu_memory_gb > 0) else 0

    # MoE: balance preference depends on EP usage, PP degree, and memory pressure
    if num_experts > 0:
        # With NeMo Automodel's constraint (DP × CP) % EP == 0, the maximum achievable EP
        # is often limited by DP/CP values, not just by num_experts or topology.
        # For very large MoE models (>100B), if we're using EP at all (EP > 1), we should
        # consider it "effectively saturated" for balance bonus purposes.
        #
        # Rationale:
        # - DeepSeek V3 (256 experts, 512 GPUs, DP=4): Max valid EP is 4, not 8
        # - The constraint prevents higher EP values even though they'd be topologically feasible
        # - If EP > 1 and model is very large, balance bonus should apply
        ep_saturated = ep > 1 or (num_experts <= config.gpus_per_node_default and ep == num_experts)

        if ep_saturated:
            # EP is saturated, now balance matters if PP is significant AND memory is tight
            # For MoE, use memory utilization to decide if balance bonus applies
            if pp >= balance_cfg.pp_significant_threshold:
                # Memory-constrained (>50% usage): Balance helps with memory distribution
                if memory_ratio > memory_cfg.pressure_moderate:
                    # Very large MoE (>100B) with tight memory: Strong balance bonus with TP penalty
                    if param_count_b > size_thresholds.large and ratio == balance_cfg.ratio_perfect:
                        # Perfect balance gets bonus, but penalize high TP (more communication overhead)
                        return balance_cfg.bonus_strong_moe + (tp * tp * balance_cfg.tp_squared_multiplier)
                    # Moderate MoE with tight memory: Standard balance bonus
                    elif ratio == balance_cfg.ratio_perfect:
                        return balance_cfg.bonus_perfect_large  # Perfect balance
                    elif ratio <= balance_cfg.ratio_good:
                        return balance_cfg.bonus_good_large  # Good balance
                # Memory-comfortable (<=50% usage): Prioritize DP throughput over balance
                # Example: Mixtral at 47GB/80GB (59%) should prefer TP=1 PP=1 DP=8 over TP=2 PP=4 DP=1
                else:
                    return 0  # No balance bonus - let DP win for throughput
            # Low/no PP (<=2): Prefer maximizing TP over balance
            return 0
        # EP not saturated: prioritize maxing EP, no balance bonus
        else:
            return 0

    # Very large dense models: strong balance preference
    if param_count_b > size_thresholds.very_large and num_experts == 0:
        if ratio == balance_cfg.ratio_perfect:
            return balance_cfg.bonus_perfect_very_large  # Perfect balance
        elif ratio <= balance_cfg.ratio_good:
            return balance_cfg.bonus_good_very_large  # Good balance

    # Large models: moderate balance preference. By construction MoE models
    # (num_experts > 0) already returned above, so only the dense check needs
    # to fire here.
    if param_count_b > size_thresholds.large:
        if ratio == balance_cfg.ratio_perfect:
            return balance_cfg.bonus_perfect_large  # Perfect balance
        elif ratio <= balance_cfg.ratio_good:
            return balance_cfg.bonus_good_large  # Good balance

    # Smaller dense models: standard balance preference
    if num_experts == 0:
        if ratio == balance_cfg.ratio_perfect:
            return balance_cfg.bonus_perfect_small  # Perfect balance
        elif ratio <= balance_cfg.ratio_good:
            return balance_cfg.bonus_good_small  # Good balance

    return 0


def comm_cost_proxy(
    n_layers: int,
    d_model: int,
    tp: int,
    pp: int,
    dp: int,
    cp: int = 1,
    ep: int = 1,
    seq_len: int = 8192,
    num_experts: int = 0,
    param_count_b: float = 0.0,
    static_memory_gb: float = 0.0,
    gpu_memory_gb: float = 80.0,
) -> int:
    """
    Heuristic communication cost, optimized to match NeMo's preferences.

    Key insights from NeMo configs:
    - For small models that fit easily: Prefer pure DP for maximum throughput
    - TP: all-reduce per layer (COSTLY - prefer TP=1 when possible)
    - PP: pipeline bubbles (VERY costly for small models, necessary for very large)
    - CP: all-gather for attention (costly, scales with sequence length)
    - DP: gradient all-reduce (less frequent, PREFERRED for throughput)
    - EP: all-to-all for expert routing (costly, but necessary for MoE)

    Strategy:
    1. Make DP very attractive (negative cost = throughput benefit)
    2. Make TP very expensive (penalize TP > 1 heavily)
    3. Make PP penalty scale with model size (cheap for huge models, expensive for small)
    4. Penalize configurations with tight memory (static > 75% GPU memory)
    """
    memory_pressure_penalty = _compute_memory_pressure_penalty(static_memory_gb, gpu_memory_gb)
    tp_cost = _compute_tp_cost(tp, n_layers, d_model, param_count_b)
    dp_cost = _compute_dp_cost(dp, tp, pp, cp, ep, num_experts, param_count_b)
    pp_cost = _compute_pp_cost(pp, tp, num_experts, param_count_b, static_memory_gb, gpu_memory_gb)
    cp_cost = _compute_cp_cost(cp, seq_len, n_layers, d_model)
    ep_cost = _compute_ep_cost(ep, num_experts)
    balance_bonus = _compute_balance_bonus(tp, pp, ep, num_experts, param_count_b, static_memory_gb, gpu_memory_gb)

    return tp_cost + dp_cost + pp_cost + cp_cost + ep_cost + balance_bonus + memory_pressure_penalty


def generate_tp_candidates(args: EstimationParams, model_cfg: ModelSpec) -> list[int]:
    """
    Generate Tensor Parallelism candidates.

    CONSTRAINTS (NeMo Automodel requirements):
    1. TP must divide hidden_size
    2. TP must divide num_attention_heads
    3. TP must divide num_kv_heads (critical for GQA/MQA models)

    If exact_tp is set, only that value is returned (after validation).
    """
    # If exact TP is specified, validate and return it
    if args.exact_tp is not None:
        tp = args.exact_tp
        d_model = model_cfg.hidden_size
        n_heads = model_cfg.num_attention_heads
        n_kv_heads = model_cfg.num_kv_heads

        if d_model % tp == 0 and n_heads % tp == 0 and n_kv_heads % tp == 0:
            return [tp]
        else:
            logger.warning(
                f"exact_tp={tp} does not satisfy NeMo Automodel constraints: "
                f"TP must divide hidden_size={d_model}, num_attention_heads={n_heads}, "
                f"and num_kv_heads={n_kv_heads}. No valid configurations found."
            )
            return []  # No valid TP, will result in no valid configs

    n_gpus = args.gpus
    d_model = model_cfg.hidden_size
    n_heads = model_cfg.num_attention_heads
    n_kv_heads = model_cfg.num_kv_heads

    # Filter TP candidates that satisfy ALL constraints
    tp_list = []
    for tp in divisors(n_gpus):
        if tp > args.max_tp:
            continue

        # Check all three divisibility constraints
        if d_model % tp == 0 and n_heads % tp == 0 and n_kv_heads % tp == 0:
            tp_list.append(tp)
        else:
            logger.debug(
                f"Skipping TP={tp}: does not divide "
                f"hidden_size={d_model}, "
                f"num_attention_heads={n_heads}, or "
                f"num_kv_heads={n_kv_heads}"
            )

    if not tp_list:
        logger.warning(
            f"No valid TP configurations found! "
            f"hidden_size={d_model}, "
            f"num_attention_heads={n_heads}, "
            f"num_kv_heads={n_kv_heads}. "
            f"Falling back to TP=1."
        )
        return [1]  # Fallback to TP=1

    return tp_list


def generate_ep_candidates(args: EstimationParams, model_cfg: ModelSpec) -> list[int]:
    """
    Generate Expert Parallelism candidates.

    CONSTRAINT: EP must divide number of experts.

    If exact_ep is set, only that value is returned (after validation).
    """
    # If exact EP is specified, validate and return it
    if args.exact_ep is not None:
        ep = args.exact_ep
        if model_cfg.moe_config:
            num_experts = model_cfg.moe_config.num_experts
            if num_experts % ep != 0:
                logger.warning(
                    f"exact_ep={ep} does not divide num_experts={num_experts}. No valid configurations found."
                )
                return []  # No valid EP
        elif ep != 1:
            logger.warning(
                f"exact_ep={ep} specified but model is not MoE (EP must be 1 for non-MoE models). No valid configurations found."
            )
            return []  # No valid EP
        return [ep]

    if model_cfg.moe_config and not args.no_ep:
        num_experts = model_cfg.moe_config.num_experts
        ep_list = [d for d in divisors(num_experts) if d <= args.max_ep]
    else:
        ep_list = [1]

    return ep_list


def generate_cp_candidates(args: EstimationParams) -> list[int]:
    """
    Generate Context Parallelism candidates.

    CONSTRAINT: CP must divide sequence length.
    CP is beneficial for long sequences (>8K) and very beneficial for very long (>32K).

    If exact_cp is set, only that value is returned (after validation).
    """
    # If exact CP is specified, validate and return it
    if args.exact_cp is not None:
        cp = args.exact_cp
        if args.seq_len % cp != 0:
            logger.warning(f"exact_cp={cp} does not divide seq_len={args.seq_len}. No valid configurations found.")
            return []  # No valid CP
        return [cp]

    config = get_config()
    cp_cfg = config.context_parallelism

    cp_list = [1]  # default: no CP

    if args.seq_len >= cp_cfg.seq_threshold_enable and not args.no_cp:
        # For 8K-16K: CP=2
        # For 16K-32K: CP=2,4
        # For 32K-128K: CP=2,4,8
        # For 128K+: CP=2,4,8,16
        if args.seq_len >= cp_cfg.seq_threshold_cp16:
            cp_candidates = [1, 2, 4, 8, 16]
        elif args.seq_len >= cp_cfg.seq_threshold_cp8:
            cp_candidates = [1, 2, 4, 8]
        elif args.seq_len >= cp_cfg.seq_threshold_cp4:
            cp_candidates = [1, 2, 4]
        else:
            cp_candidates = [1, 2]

        # Filter: must divide sequence length, within max_cp
        cp_list = [c for c in cp_candidates if c <= args.max_cp and args.seq_len % c == 0]
        if not cp_list:
            cp_list = [1]  # Fallback

    return cp_list

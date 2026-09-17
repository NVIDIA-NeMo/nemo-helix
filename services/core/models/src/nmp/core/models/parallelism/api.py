# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from typing import Callable

import torch
from accelerate import init_empty_weights
from nmp.core.models.parallelism.constants import DEFAULT_EXCLUDE_PATTERNS, DEFAULT_INCLUDE_PATTERNS
from nmp.core.models.parallelism.hueristics import (
    comm_cost_proxy,
    divisors,
    generate_cp_candidates,
    generate_ep_candidates,
    generate_tp_candidates,
)
from nmp.core.models.parallelism.memory import (
    calculate_activation_bytes_per_token,
    calculate_param_counts,
    calculate_static_memory_per_rank,
    calculate_training_overhead_gb,
    find_max_microbatch,
    param_count,
)
from nmp.core.models.parallelism.models import (
    EstimationParams,
    ModelSpec,
    ParallelizationConfig,
    ParallelizationRecommendation,
    detect_gated_mlp_from_cfg,
    detect_mamba_config_from_cfg,
    detect_moe_config_from_cfg,
    detect_precision_from_cfg,
    detect_sliding_window_from_cfg,
    extract_basic_config,
    try_load_nemo_yaml_config,
)
from nmp.core.models.parallelism.utils import detect_reasoning_control, is_huggingface_model_directory
from nmp.core.models.schemas import LinearLayerSpec
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def extract_linear_layers(pretrained_or_path: str, is_trusted: bool = False) -> list[LinearLayerSpec]:
    """
    Extract all linear/Conv1D layers from a model without loading weights.

    This function instantiates the model with empty weights and scans all
    nn.Linear and Conv1D modules to extract their dimensions. This information
    is stored in CheckpointMetadata to avoid repeated model instantiation
    during LoRA parameter estimation.

    Args:
        pretrained_or_path: Model ID or path
        is_trusted: Whether to trust remote code

    Returns:
        List of LinearLayerSpec objects containing module names and dimensions
    """
    try:
        logger.info(f"Extracting linear layers from {pretrained_or_path} with trust_remote_code={is_trusted}")
        cfg = AutoConfig.from_pretrained(pretrained_or_path, trust_remote_code=is_trusted)

        ctx = init_empty_weights() if init_empty_weights is not nullcontext else nullcontext()
        with ctx:
            # Try AutoModelForCausalLM first, then fall back to AutoModel for non-causal architectures
            # like bidirectional embedding models with custom config classes
            try:
                model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, trust_remote_code=is_trusted)
            except ValueError:
                logger.debug(f"AutoModelForCausalLM does not support {type(cfg).__name__}, falling back to AutoModel")
                model = AutoModel.from_config(cfg, torch_dtype=torch.bfloat16, trust_remote_code=is_trusted)

        linear_layers = []

        for name, mod in model.named_modules():
            # Check for Linear layers and Conv1D (used by GPT-2 and similar models)
            module_type = type(mod).__name__
            is_linear_like = isinstance(mod, torch.nn.Linear) or module_type == "Conv1D"

            if not is_linear_like:
                continue

            in_f = getattr(mod, "in_features", None)
            out_f = getattr(mod, "out_features", None)

            if in_f is None or out_f is None:
                continue

            linear_layers.append(
                LinearLayerSpec(
                    name=name,
                    in_features=in_f,
                    out_features=out_f,
                )
            )

        logger.info(f"Extracted {len(linear_layers)} linear layers from {pretrained_or_path}")
        return linear_layers

    except Exception as e:
        logger.warning(f"Failed to extract linear layers from {pretrained_or_path}: {e}")
        return []


_MODEL_SPEC_CACHE_MAX = 64
_model_spec_cache: OrderedDict[tuple[str, bool], ModelSpec] = OrderedDict()


def _cache_put(key: tuple[str, bool], value: ModelSpec) -> None:
    _model_spec_cache[key] = value
    if len(_model_spec_cache) > _MODEL_SPEC_CACHE_MAX:
        _model_spec_cache.popitem(last=False)


def infer_model_cfg_from_hf(
    pretrained_or_path: str,
    is_trusted: bool = False,
    file_listing: list[str] | None = None,
    need_linear_layers: bool = True,
) -> ModelSpec:
    """
    Infer model configuration from HuggingFace model.

    This is the main entry point for detecting model architecture from
    a HuggingFace model ID or local path. It delegates to specialized
    helper functions for each architecture component.

    Handles standard HuggingFace configs (config.json) only.

    skip_weights_check - Can be provided if the weights are validated to exist prior
      to calling this function in order to prevent excessive weights downloading

    Args:
        pretrained_or_path: HuggingFace model ID or local directory path.
        is_trusted: Whether to trust remote code when loading the config.
        file_listing: Optional list of file paths from a remote source
            (e.g. a fileset API).  When provided and *pretrained_or_path*
            is a local directory, weight-file validation uses this listing
            instead of checking the local filesystem.
        need_linear_layers: Whether to extract linear layer specs via
            AutoModelForCausalLM.from_config(). Only needed for LoRA parameter
            estimation. Set to False to skip expensive model skeleton instantiation.
    """
    # Cache only remote HF model IDs — skip for local paths and file_listing.
    _use_cache = file_listing is None and not os.path.exists(pretrained_or_path)
    if _use_cache:
        cache_key = (pretrained_or_path, is_trusted)
        cached = _model_spec_cache.get(cache_key)
        if cached is not None:
            if need_linear_layers and cached.linear_layers is None:
                # Upgrade: only extract the missing linear layers, reuse everything else.
                layers = extract_linear_layers(pretrained_or_path, is_trusted) or None
                cached = cached.model_copy(update={"linear_layers": layers})
                _cache_put(cache_key, cached)
            return cached

    if os.path.exists(pretrained_or_path):
        if ("alternates" in os.listdir(pretrained_or_path)) and (
            "hf" in os.listdir(f"{pretrained_or_path}/alternates")
        ):
            pretrained_or_path = f"{pretrained_or_path}/alternates/hf"

        if not is_huggingface_model_directory(
            pretrained_or_path,
            file_listing=file_listing,
        ):
            raise ValueError("Expected checkpoint to be a huggingface directory")

    # Try standard HF config loading first
    cfg = AutoConfig.from_pretrained(pretrained_or_path, trust_remote_code=is_trusted)

    # Check if we got a default config (e.g., Nemotron with 32 layers when it should be 340B)
    # If YAML exists and provides different values, use those instead
    yaml_cfg = try_load_nemo_yaml_config(pretrained_or_path, is_trusted)
    if yaml_cfg is not None:
        cfg = yaml_cfg

    is_chat = False
    # None means undetermined — the tokenizer was unreachable. A tokenizer that
    # loads and carries no chat template is a definite "no control", so the call
    # stays outside the is_chat branch.
    chat_template_controls_reasoning = None
    if os.path.exists(cfg.name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path, trust_remote_code=is_trusted)
        chat_template = getattr(tokenizer, "chat_template", None)
        is_chat = chat_template is not None
        chat_template_controls_reasoning = detect_reasoning_control(chat_template)

    if getattr(cfg, "text_config", None):
        cfg = getattr(cfg, "text_config")

    # Extract basic configuration
    basic_config = extract_basic_config(cfg)
    n_layers = basic_config["num_layers"]

    # Detect architecture-specific features
    moe_config = detect_moe_config_from_cfg(cfg)
    mamba_config = detect_mamba_config_from_cfg(cfg, n_layers, is_trusted)
    gated_mlp = detect_gated_mlp_from_cfg(cfg)
    sliding_window_config = detect_sliding_window_from_cfg(cfg)
    precision = detect_precision_from_cfg(cfg)

    moe_dict = moe_config.model_dump() if moe_config else None
    mamba_dict = mamba_config.model_dump() if mamba_config else None

    total_params = param_count(
        n_layers=basic_config["num_layers"],
        d_model=basic_config["hidden_size"],
        d_ff=basic_config["ffn_hidden_size"],
        vocab_size=basic_config["vocab_size"],
        tied_embeddings=basic_config["tied_embeddings"],
        moe_config=moe_dict,
        mamba_config=mamba_dict,
        n_kv_heads=basic_config["num_kv_heads"],
        n_heads=basic_config["num_attention_heads"],
        gated_mlp=bool(gated_mlp),
    )

    # Extract linear layers for LoRA parameter estimation (expensive — skipped when not needed)
    linear_layers = extract_linear_layers(pretrained_or_path, is_trusted) if need_linear_layers else []

    # Construct and return full model configuration
    result = ModelSpec(
        **basic_config,
        is_chat=is_chat,
        chat_template_controls_reasoning=chat_template_controls_reasoning,
        gated_mlp=bool(gated_mlp),
        base_num_parameters=total_params,
        precision=precision,
        moe_config=moe_config,
        mamba_config=mamba_config,
        sliding_window_config=sliding_window_config,
        linear_layers=linear_layers if linear_layers else None,
    )

    if _use_cache:
        _cache_put(cache_key, result)

    return result


def _get_pp_candidates(args: EstimationParams, n_gpus: int, tp: int, ep: int, cp: int) -> list[int]:
    """
    Get PP candidates based on constraints.

    If exact_pp is set, returns only that value.
    Otherwise, returns all valid PP divisors up to max_pp.
    """
    if args.exact_pp is not None:
        # Check if exact_pp is valid for this TP/EP/CP combination
        if n_gpus % (tp * ep * cp * args.exact_pp) == 0:
            return [args.exact_pp]
        else:
            # Skip this TP/EP/CP combination as exact_pp doesn't work with it
            return []
    return [pp for pp in divisors(n_gpus // (tp * ep * cp)) if pp <= args.max_pp]


def estimate(args: EstimationParams, model_cfg: ModelSpec) -> list[ParallelizationConfig]:
    """
    Estimate parallelization configurations for given model and hardware.

    This function generates and evaluates all valid parallelization configurations,
    returning them sorted by communication cost (best first).
    """
    n_gpus = args.gpus
    n_layers = model_cfg.num_layers

    # Generate parallelism candidates
    tp_list = generate_tp_candidates(args, model_cfg)
    ep_list = generate_ep_candidates(args, model_cfg)
    cp_list = generate_cp_candidates(args)

    # Calculate parameter counts once (expensive operation)
    param_counts = calculate_param_counts(args, model_cfg)

    results = []

    # Evaluate all parallelization combinations
    for tp in tp_list:
        for ep in ep_list:
            for cp in cp_list:
                # CONSTRAINT 2: PP must divide GPU count after TP/EP/CP
                for pp in _get_pp_candidates(args, n_gpus, tp, ep, cp):
                    # Calculate Data Parallelism degree
                    dp = n_gpus // (tp * pp * cp * ep)
                    if dp < 1:
                        continue

                    # If exact_dp is specified, skip configs that don't match
                    if args.exact_dp is not None and dp != args.exact_dp:
                        continue

                    # Validate max_dp constraint
                    if dp > args.max_dp:
                        continue

                    # CONSTRAINT 3: NeMo Automodel EP constraint
                    # For MoE models, EP must divide (DP × CP)
                    # Source: FSDP2Manager asserts dp_cp_size % ep_size == 0
                    # This is required by NeMo Automodel's FSDP2Manager
                    if ep > 1:
                        dp_cp_size = dp * cp
                        if dp_cp_size % ep != 0:
                            logger.debug(
                                f"Skipping config TP={tp} PP={pp} DP={dp} CP={cp} EP={ep}: "
                                f"(DP * CP) % EP != 0 ({dp_cp_size} % {ep} != 0)"
                            )
                            continue

                    # Note: PP doesn't strictly need to divide layers evenly
                    # NeMo uses virtual pipeline parallelism with uneven stages

                    # Calculate static memory (parameters, gradients, optimizer)
                    static_b = calculate_static_memory_per_rank(args, tp, pp, dp, ep, param_counts)

                    # Calculate activation memory per token
                    act_tok_b, kv_tok_b, scr_tok_b = calculate_activation_bytes_per_token(args, model_cfg, tp, cp)

                    # Find maximum microbatch size that fits in memory
                    best_mb = find_max_microbatch(args, model_cfg, static_b, act_tok_b, kv_tok_b, scr_tok_b, cp)
                    if best_mb == 0:
                        continue  # Config doesn't fit

                    # Calculate communication cost score
                    num_experts = model_cfg.moe_config.num_experts if model_cfg.moe_config else 0
                    param_count_b = param_counts["base"] / 1e9  # Convert to billions
                    # Calculate memory breakdown for reporting
                    per_rank_static_gb = round(static_b / 1024**3, 2)

                    score = comm_cost_proxy(
                        n_layers,
                        model_cfg.hidden_size,
                        tp,
                        pp,
                        dp,
                        cp,
                        ep,
                        args.seq_len,
                        num_experts,
                        param_count_b,
                        static_memory_gb=per_rank_static_gb,
                        gpu_memory_gb=args.gpu_mem_gb,
                    )
                    effective_seq_len = args.seq_len // cp
                    est_act_gb_per_mb1 = round(((act_tok_b + kv_tok_b + scr_tok_b) * effective_seq_len) / 1024**3, 2)

                    # Calculate training framework overhead (FSDP2 prefetch, NCCL buffers, grad accumulation)
                    training_overhead_gb = calculate_training_overhead_gb(
                        per_rank_static_gb=per_rank_static_gb,
                        num_layers=n_layers,
                        tp=tp,
                        pp=pp,
                        dp=dp,
                    )

                    # Total memory = static + activations + training overhead
                    total_memory_per_rank_gb = round(
                        per_rank_static_gb + (est_act_gb_per_mb1 * best_mb) + training_overhead_gb, 2
                    )

                    # Create configuration
                    config = ParallelizationConfig(
                        tp=tp,
                        pp=pp,
                        dp=dp,
                        cp=cp,
                        ep=ep,
                        microbatch_per_dp=best_mb,
                        per_rank_static_gb=per_rank_static_gb,
                        est_act_gb_per_mb1=est_act_gb_per_mb1,
                        total_memory_per_rank_gb=total_memory_per_rank_gb,
                        score=int(score),
                    )
                    results.append(config)

    # Sort by communication cost (lower is better), then by microbatch size (higher is better)
    results.sort(key=lambda x: (x.score, -x.microbatch_per_dp))
    return results


def estimate_parallelization(
    model_id: str,
    gpus: int,
    gpu_mem_gb: float,
    seq_len: int,
    act_ckpt_ratio: float = 0.05,
    max_tp: int = 64,
    max_cp: int = 8,
    no_cp: bool = False,
    max_ep: int = 8,
    no_ep: bool = False,
    max_microbatch: int = 64,
    microbatch_size: int | None = None,
    attn_scratch_factor: float = 0.0,
    lora: bool = False,
    lora_r: int = 16,
    is_trusted: bool = False,
) -> ParallelizationRecommendation:
    """
    Simple API to estimate parallelization strategies for any HuggingFace model.

    This is the main entry point for programmatic use.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Meta-Llama-3-8B")
        gpus: Number of GPUs available
        gpu_mem_gb: Memory per GPU in GB
        seq_len: Sequence length in tokens
        act_ckpt_ratio: Fraction of activations kept (0.0-1.0, default 0.05)
        max_tp: Maximum tensor parallelism degree (default 64)
        max_cp: Maximum context parallelism degree (default 8)
        no_cp: Disable context parallelism (default False)
        max_ep: Maximum expert parallelism degree for MoE (default 64)
        no_ep: Disable expert parallelism (default False)
        max_microbatch: Maximum microbatch size to search for (default 64)
        microbatch_size: Fixed microbatch size (if set, skips search; default None)
        attn_scratch_factor: Attention scratch memory factor (default 0.0)
        lora: Enable LoRA fine-tuning mode (default False)
        lora_r: LoRA rank if enabled (default 16)

    Returns:
        ParallelizationRecommendation with sorted list of viable configurations

    Example:
        >>> from parallelism_helper import estimate_parallelization
        >>>
        >>> # Simple usage
        >>> result = estimate_parallelization(
        ...     "openai/gpt-oss-120b",
        ...     gpus=64,
        ...     gpu_mem_gb=80,
        ...     seq_len=4096
        ... )
        >>>
        >>> # Print top 3 configurations
        >>> for i, config in enumerate(result.configs[:3], 1):
        ...     print(f"{i}. TP={config.tp} PP={config.pp} DP={config.dp} EP={config.ep}")
        ...     print(f"   Memory: {config.total_memory_per_rank_gb:.1f}GB")
        >>>
        >>> # Find minimum GPUs needed
        >>> for n in [1, 2, 4, 8, 16, 32, 64]:
        ...     result = estimate_parallelization("openai/gpt-oss-120b", n, 80, 4096)
        ...     if result.configs:
        ...         print(f"Minimum GPUs: {n}")
        ...         break
    """
    # Load model config from HuggingFace (skip linear layer extraction when LoRA is not requested)
    model_cfg = infer_model_cfg_from_hf(model_id, is_trusted, need_linear_layers=lora)

    # Create EstimationParams object
    params = EstimationParams(
        pretrained=model_id,  # Needed for LoRA introspection
        gpus=gpus,
        gpu_mem_gb=gpu_mem_gb,
        seq_len=seq_len,
        act_ckpt_ratio=act_ckpt_ratio,
        max_tp=max_tp,
        max_cp=max_cp,
        no_cp=no_cp,
        max_ep=max_ep,
        no_ep=no_ep,
        max_microbatch=max_microbatch,
        microbatch_size=microbatch_size,
        attn_scratch_factor=attn_scratch_factor,
        lora=lora,
        lora_r=lora_r,
        lora_include_regex=DEFAULT_INCLUDE_PATTERNS if lora else None,
        lora_exclude_regex=DEFAULT_EXCLUDE_PATTERNS if lora else None,
    )

    configs = estimate(params, model_cfg)

    return ParallelizationRecommendation(
        model_info=model_cfg, configs=configs, gpu_count=gpus, gpu_mem_gb=gpu_mem_gb, seq_len=seq_len
    )


def _binary_search(
    gpu_counts: list[int],
    check_gpu_count: Callable[[int], tuple[bool, ParallelizationConfig | None]],
) -> tuple[int, ParallelizationConfig] | tuple[None, None]:
    # Binary search on gpu_counts
    left, right = 0, len(gpu_counts) - 1
    min_gpus_found = None
    best_config_found = None

    while left <= right:
        mid = (left + right) // 2
        n_gpus = gpu_counts[mid]

        fits, config = check_gpu_count(n_gpus)

        if fits:
            # Found a valid config, try to find smaller
            min_gpus_found = config.total_gpus
            best_config_found = config
            right = mid - 1
        else:
            # Need more GPUs
            left = mid + 1
    return min_gpus_found, best_config_found


def find_minimum_gpus(
    model_id: str,
    gpu_mem_gb: float,
    seq_len: int,
    max_gpus: int = 128,
    max_memory_utilization: float = 0.85,
    lora: bool = False,
    lora_r: int = 16,
    microbatch_size: int | None = None,
    act_ckpt_ratio: float = 0.05,
    no_cp: bool = False,
    no_ep: bool = False,
    is_trusted: bool = False,
) -> tuple[int, ParallelizationConfig] | tuple[None, None]:
    """
    Find the minimum number of GPUs needed for a model using binary search.

    This function uses binary search over powers-of-2 GPU counts to efficiently
    find the minimum number of GPUs required to fit the model within memory constraints.

    Args:
        model_id: HuggingFace model ID
        gpu_mem_gb: Memory per GPU in GB
        seq_len: Sequence length
        max_gpus: Maximum GPUs to try (default 128)
        max_memory_utilization: Maximum memory utilization (default 0.85 = 85%)
            Configurations exceeding this threshold are rejected as likely to OOM.
            The function will search through all available configs to find one
            that meets this threshold.
        lora: Enable LoRA fine-tuning mode (default False)
        lora_r: LoRA rank (default 16)
        microbatch_size: Fixed microbatch size, or None to search (default None)
        act_ckpt_ratio: Activation checkpointing ratio (default 0.05)
        no_cp: Disable context parallelism (default False)
        no_ep: Disable expert parallelism for MoE models (default False)

    Returns:
        (min_gpus, best_config) or (None, None) if not feasible

    Examples:
        >>> from parallelism_helper import find_minimum_gpus
        >>>
        >>> # Full training
        >>> min_gpus, config = find_minimum_gpus(
        ...     "meta-llama/Meta-Llama-3-70B",
        ...     gpu_mem_gb=80,
        ...     seq_len=4096
        ... )
        >>>
        >>> # LoRA fine-tuning
        >>> min_gpus, config = find_minimum_gpus(
        ...     "meta-llama/Meta-Llama-3-70B",
        ...     gpu_mem_gb=80,
        ...     seq_len=4096,
        ...     lora=True,
        ...     lora_r=8
        ... )
        >>>
        >>> # Disable CP for standard sequences
        >>> min_gpus, config = find_minimum_gpus(
        ...     "meta-llama/Meta-Llama-3-8B",
        ...     gpu_mem_gb=80,
        ...     seq_len=2048,
        ...     no_cp=True
        ... )
        >>>
        >>> if min_gpus:
        ...     print(f"Minimum: {min_gpus} GPUs")
        ...     print(f"Config: TP={config.tp} PP={config.pp} DP={config.dp} EP={config.ep}")
    """
    # Powers of 2 and common GPU counts as search space
    gpu_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    gpu_counts = [n for n in gpu_counts if n <= max_gpus]

    if not gpu_counts:
        return None, None

    def check_gpu_count(n_gpus: int) -> tuple[bool, ParallelizationConfig | None]:
        """
        Check if n_gpus is sufficient. Returns (fits, config).

        Searches through all configs for this GPU count to find one that fits.
        """
        result = estimate_parallelization(
            model_id,
            n_gpus,
            gpu_mem_gb,
            seq_len,
            act_ckpt_ratio=act_ckpt_ratio,
            no_cp=no_cp,
            no_ep=no_ep,
            microbatch_size=microbatch_size,
            lora=lora,
            lora_r=lora_r,
            is_trusted=is_trusted,
        )

        if not result.configs:
            return False, None

        # Search through all configs to find one that fits
        for config in result.configs:
            memory_utilization = config.total_memory_per_rank_gb / gpu_mem_gb
            if memory_utilization <= max_memory_utilization:
                return True, config

        # All configs exceeded memory threshold
        logger.debug(f"All configs for {n_gpus} GPUs exceed memory threshold {max_memory_utilization:.1%}")
        return False, None

    return _binary_search(gpu_counts, check_gpu_count)


def find_minimum_gpus_from_metadata(
    checkpoint_metadata: ModelSpec,
    gpu_mem_gb: float,
    seq_len: int,
    max_gpus: int = 128,
    max_memory_utilization: float = 0.85,
    lora: bool = False,
    lora_r: int = 16,
    microbatch_size: int | None = None,
    act_ckpt_ratio: float = 0.05,
    no_cp: bool = False,
    no_ep: bool = False,
    is_trusted: bool = False,
) -> tuple[int, ParallelizationConfig] | tuple[None, None]:
    # Powers of 2 and common GPU counts as search space
    gpu_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    gpu_counts = [n for n in gpu_counts if n <= max_gpus]

    if not gpu_counts:
        return None, None

    def check_gpu_count(n_gpus: int) -> tuple[bool, ParallelizationConfig | None]:
        """
        Check if n_gpus is sufficient. Returns (fits, config).

        Searches through all configs for this GPU count to find one that fits.
        """
        args = EstimationParams(
            gpus=n_gpus,
            gpu_mem_gb=gpu_mem_gb,
            seq_len=seq_len,
            act_ckpt_ratio=act_ckpt_ratio,
            no_cp=no_cp,
            no_ep=no_ep,
            microbatch_size=microbatch_size,
            lora=lora,
            lora_r=lora_r,
            is_trusted=is_trusted,
        )

        result = estimate(args, checkpoint_metadata)
        if not result:
            return False, None

        # Search through all configs to find one that fits
        for config in result:
            memory_utilization = config.total_memory_per_rank_gb / gpu_mem_gb
            if memory_utilization <= max_memory_utilization:
                return True, config

        # All configs exceeded memory threshold
        logger.debug(f"All configs for {n_gpus} GPUs exceed memory threshold {max_memory_utilization:.1%}")
        return False, None

    return _binary_search(gpu_counts, check_gpu_count)


def normalize_config_scores_to_ranks(configs: list[ParallelizationConfig]) -> list[ParallelizationConfig]:
    """
    Normalize configuration scores to ranks (1 = best, 2 = second best, etc.).

    This is useful for presenting results to users where rank is more intuitive
    than raw communication cost scores.

    Args:
        configs: List of ParallelizationConfig objects sorted by score (best first)

    Returns:
        New list of configs with scores replaced by ranks (1-indexed)

    Example:
        Input configs with scores: [100, 200, 200, 500]
        Output configs with ranks: [1, 2, 2, 4]  (ties get same rank)
    """
    if not configs:
        return []

    # Create a mapping of original scores to ranks
    # Configs with the same score get the same rank
    sorted_configs = sorted(configs, key=lambda c: c.score)
    score_to_rank = {}
    current_rank = 1

    for i, config in enumerate(sorted_configs):
        if config.score not in score_to_rank:
            score_to_rank[config.score] = current_rank
            current_rank += 1

    # Create new configs with ranks instead of scores
    ranked_configs = []
    for config in configs:
        ranked_config = config.model_copy(update={"score": score_to_rank[config.score]})
        ranked_configs.append(ranked_config)

    return ranked_configs

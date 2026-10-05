#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI tool for estimating LLM parallelization strategies.

This is a command-line wrapper around the parallelism_helper library.
For programmatic use, import parallelism_helper directly.
"""

import argparse
import logging

from nhx.core.models.parallelism.api import estimate, infer_model_cfg_from_hf
from nhx.core.models.parallelism.constants import (
    DEFAULT_EXCLUDE_PATTERNS,
    DEFAULT_INCLUDE_PATTERNS,
)
from nhx.core.models.parallelism.models import (
    EstimationParams,
    ParallelizationRecommendation,
)

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Recommend TP/PP/DP/CP/EP and microbatch for LLM training")
    ap.add_argument("--pretrained", type=str, required=True, help="HuggingFace model ID")
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--gpu-mem-gb", type=float, required=True)
    ap.add_argument("--seq-len", type=int, required=True)

    # Sizing knobs
    ap.add_argument(
        "--act-ckpt-ratio", type=float, default=0.25, help="Fraction of activations kept due to checkpointing"
    )
    ap.add_argument("--max-tp", type=int, default=64)
    ap.add_argument("--max-cp", type=int, default=8, help="Maximum Context Parallelism degree (for long sequences)")
    ap.add_argument("--no-cp", action="store_true", help="Disable Context Parallelism even for long sequences")
    ap.add_argument("--max-ep", type=int, default=8, help="Maximum Expert Parallelism degree (for MoE models)")
    ap.add_argument("--no-ep", action="store_true", help="Disable Expert Parallelism even for MoE models")
    ap.add_argument("--max-microbatch", type=int, default=64)
    ap.add_argument(
        "--microbatch-size",
        type=int,
        default=None,
        help="Fixed microbatch size (if set, skips binary search)",
    )
    ap.add_argument(
        "--attn-scratch-factor",
        type=float,
        default=0.0,
        help="Optional per-token per-layer scratch (bf16 bytes per head)",
    )

    # LoRA mode
    ap.add_argument("--lora", action="store_true", help="Estimate memory for PEFT LoRA finetuning (base frozen)")

    ap.add_argument("--lora-r", type=int, default=8, help="LoRA rank")

    ap.add_argument(
        "--lora-include-regex",
        nargs="*",
        default=None,
        help="Regex patterns to INCLUDE module names (default covers q/k/v/o, qkv, c_attn, up/down/gate/c_proj)",
    )

    ap.add_argument(
        "--lora-exclude-regex",
        nargs="*",
        default=None,
        help="Regex patterns to EXCLUDE module names (default excludes norms, embeddings, lm_head)",
    )

    args = ap.parse_args()

    # Fill default include/exclude if not provided
    if args.lora and args.lora_include_regex is None:
        args.lora_include_regex = DEFAULT_INCLUDE_PATTERNS
    if args.lora and args.lora_exclude_regex is None:
        args.lora_exclude_regex = DEFAULT_EXCLUDE_PATTERNS

    model_cfg = infer_model_cfg_from_hf(args.pretrained)

    # Display model information
    print(f"\n{'=' * 80}")
    print(f"Model: {model_cfg.checkpoint_model_name}")
    print(f"Architecture: {model_cfg.family}")
    print(f"Layers: {model_cfg.num_layers}, Hidden: {model_cfg.hidden_size}, FFN: {model_cfg.ffn_hidden_size}")
    print(f"Attention Heads: {model_cfg.num_attention_heads}, KV Heads: {model_cfg.num_kv_heads}")
    print(f"Vocab: {model_cfg.vocab_size}, Tied Embeddings: {model_cfg.tied_embeddings}")

    if model_cfg.mamba_config:
        print("\n[*] Mamba/SSM Model Detected!")
        if model_cfg.mamba_config.is_hybrid:
            print(
                f"Hybrid Architecture: {model_cfg.mamba_config.num_attention_layers} Attention layers + "
                f"{model_cfg.mamba_config.num_mamba_layers} Mamba layers"
            )
        else:
            print(f"Pure Mamba/SSM: {model_cfg.mamba_config.num_mamba_layers} layers")
        print(f"State size: {model_cfg.mamba_config.state_size}, Conv kernel: {model_cfg.mamba_config.conv_kernel}")
        print("Note: Mamba layers have NO KV cache (major memory savings!)")

    if model_cfg.moe_config:
        print("\n[*] MoE Model Detected!")
        print(
            f"Experts: {model_cfg.moe_config.num_experts}, Active per token: {model_cfg.moe_config.num_experts_per_tok}"
        )
        print(f"Expert layers: {model_cfg.moe_config.num_expert_layers}/{model_cfg.num_layers}")
        if not args.no_ep:
            print(f"Expert Parallelism: ENABLED (max EP={args.max_ep})")
        else:
            print("Expert Parallelism: DISABLED")

    if not model_cfg.moe_config and not model_cfg.mamba_config:
        print("\nStandard Transformer (dense model)")

    # CP recommendation
    if args.seq_len >= 8192:
        print(f"\n[*] Sequence Length: {args.seq_len:,} tokens")
        if args.seq_len >= 32768:
            print("   [!] Very long context detected! Context Parallelism (CP) highly recommended.")
        elif args.seq_len >= 16384:
            print("   [i] Long context detected. Context Parallelism (CP) recommended for memory savings.")
        elif args.seq_len >= 8192:
            print("   [i] Moderate context length. Context Parallelism (CP) may be beneficial.")
        if not args.no_cp:
            print(f"   Context Parallelism: ENABLED (max CP={args.max_cp})")
        else:
            print("   Context Parallelism: DISABLED")

    print(f"{'=' * 80}\n")

    # Create EstimationParams from argparse args
    params = EstimationParams(
        pretrained=args.pretrained,
        gpus=args.gpus,
        gpu_mem_gb=args.gpu_mem_gb,
        seq_len=args.seq_len,
        act_ckpt_ratio=args.act_ckpt_ratio,
        max_tp=args.max_tp,
        max_cp=args.max_cp,
        no_cp=args.no_cp,
        max_ep=args.max_ep,
        no_ep=args.no_ep,
        max_microbatch=args.max_microbatch,
        microbatch_size=args.microbatch_size,
        attn_scratch_factor=args.attn_scratch_factor,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_include_regex=args.lora_include_regex,
        lora_exclude_regex=args.lora_exclude_regex,
    )

    configs = estimate(params, model_cfg)

    if not configs:
        print("No feasible parallelization strategy found under memory budget.")
        print("Try: lower seq len, heavier activation checkpointing, higher TP/PP/EP, or more GPUs.")
        return

    # Create recommendation object
    recommendation = ParallelizationRecommendation(
        model_info=model_cfg,
        gpu_count=args.gpus,
        gpu_mem_gb=args.gpu_mem_gb,
        seq_len=args.seq_len,
        configs=configs,
        lora_enabled=args.lora,
        lora_r=args.lora_r if args.lora else None,
    )

    # Print results
    print("Top parallelization candidates (lower score = better; higher microbatch = better):")
    print()
    for i, config in enumerate(recommendation.configs[:10], 1):
        # Build parallelism string
        parallel_dims = []
        if config.cp > 1:
            parallel_dims.append(f"CP={config.cp}")
        if config.ep > 1:
            parallel_dims.append(f"EP={config.ep}")
        extra_str = (" " + " ".join(parallel_dims)) if parallel_dims else ""

        print(
            f"{i:2}. TP={config.tp:2} PP={config.pp:2} DP={config.dp:2}{extra_str:11} | "
            f"MB/DP={config.microbatch_per_dp:2} | "
            f"static~{config.per_rank_static_gb:5.1f}GB | "
            f"act~{config.est_act_gb_per_mb1:5.2f}GB/MB | "
            f"total~{config.total_memory_per_rank_gb:5.1f}GB | "
            f"score={config.score}"
        )

    # Return the recommendation object (useful for API/programmatic use)
    return recommendation


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data models and model introspection for parallelism estimation.

This module contains:
- Pydantic models for configuration and results
- HuggingFace model introspection (MoE, Mamba, GQA detection)
- LoRA module introspection
- NeMo YAML config loading
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from huggingface_hub import hf_hub_download
from nhx.core.models.schemas import (
    MambaConfig,
    ModelSpec,
    MoEConfig,
    SlidingWindowConfig,
)
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================


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


class ParallelizationConfig(BaseModel):
    """Parallelization strategy configuration."""

    tp: int = Field(..., description="Tensor Parallelism degree")
    pp: int = Field(..., description="Pipeline Parallelism degree")
    dp: int = Field(..., description="Data Parallelism degree")
    cp: int = Field(1, description="Context Parallelism degree (1 = disabled)")
    ep: int = Field(1, description="Expert Parallelism degree (1 = disabled)")
    microbatch_per_dp: int = Field(..., description="Microbatch size per DP rank")
    per_rank_static_gb: float = Field(..., description="Static memory per rank (params + optimizer + grads) in GB")
    est_act_gb_per_mb1: float = Field(..., description="Estimated activation memory per rank for microbatch=1 in GB")
    total_memory_per_rank_gb: float = Field(..., description="Total estimated memory per rank in GB")
    score: int = Field(..., description="Communication cost heuristic (lower is better)")

    @property
    def total_gpus(self) -> int:
        """Calculate total number of GPUs."""
        return self.tp * self.pp * self.dp * self.cp * self.ep

    @property
    def global_batch_size(self) -> int:
        """Calculate global batch size."""
        return self.microbatch_per_dp * self.dp

    @property
    def effective_seq_len_per_rank(self) -> int:
        """Calculate effective sequence length per rank with CP."""
        # This would need seq_len passed in, so it's more of a conceptual property
        # Actual calculation: seq_len / cp
        return -1  # Placeholder, actual value calculated in estimator


class ParallelizationRecommendation(BaseModel):
    """Complete parallelization recommendation including model info and configs."""

    model_info: ModelSpec = Field(..., description="Model architecture configuration")
    gpu_count: int = Field(..., description="Total number of GPUs available")
    gpu_mem_gb: float = Field(..., description="Memory per GPU in GB")
    seq_len: int = Field(..., description="Sequence length")
    configs: list[ParallelizationConfig] = Field(..., description="Sorted list of parallelization configurations")
    lora_enabled: bool = Field(False, description="Whether LoRA fine-tuning is enabled")
    lora_r: Optional[int] = Field(None, description="LoRA rank if enabled")

    model_config = {
        "protected_namespaces": (),
    }


# ============================================================================
# Model Introspection Functions
# ============================================================================


def detect_precision_from_cfg(cfg: PretrainedConfig) -> str:
    """
    Detect model precision from HuggingFace config.

    Common precision attributes in HF configs:
    - torch_dtype: torch.float16, torch.bfloat16, torch.float32
    - quantization_config: for quantized models (int8, int4, etc.)

    Returns:
        Precision string (e.g., 'float16', 'bfloat16', 'float32', 'int8', 'int4')
        Defaults to 'float32' if no precision is specified.
    """
    # Check torch_dtype first (most common)
    torch_dtype = getattr(cfg, "torch_dtype", None)
    if torch_dtype is not None:
        # Convert torch dtype to string
        dtype_str = str(torch_dtype)
        if "bfloat16" in dtype_str:
            return "bfloat16"
        elif "float16" in dtype_str:
            return "float16"
        elif "float32" in dtype_str:
            return "float32"

    # Check quantization_config for quantized models
    quant_config = getattr(cfg, "quantization_config", None)
    if quant_config is not None:
        # Try to extract bits from quantization config
        if isinstance(quant_config, dict):
            bits = quant_config.get("bits")
            if bits:
                return f"int{bits}"

    # Default to float32 if no precision is specified
    return "float32"


def extract_basic_config(cfg: PretrainedConfig) -> dict[str, Any]:
    """Extract basic model dimensions from HuggingFace config."""
    # Extract layer count
    if getattr(cfg, "text_config", None):
        cfg = getattr(cfg, "text_config")

    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    if n_layers is None:
        raise ValueError(f"Could not infer num_layers from HF config for {cfg.model_type}")

    # Extract dimensions
    d_model = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None)
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)

    # Extract KV heads (for GQA/MQA)
    n_kv_heads = getattr(cfg, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = n_heads  # Default: same as query heads (standard attention)

    # Extract FFN dimension
    d_ff = getattr(cfg, "intermediate_size", None) or getattr(cfg, "n_inner", None)
    if d_ff is None:
        d_ff = 4 * d_model  # Default FFN is 4x hidden size

    # Extract vocabulary size
    vocab_size = getattr(cfg, "vocab_size", 50257)

    # Check if embeddings are tied
    tied = getattr(cfg, "tie_word_embeddings", False)

    family = cfg.model_type.lower()

    return {
        "checkpoint_model_name": cfg.architectures[0] if cfg.architectures else cfg._name_or_path,
        "family": family,
        "num_layers": int(n_layers),
        "hidden_size": int(d_model),
        "num_attention_heads": int(n_heads),
        "num_kv_heads": int(n_kv_heads),
        "ffn_hidden_size": int(d_ff),
        "vocab_size": int(vocab_size),
        "tied_embeddings": tied,
    }


def detect_moe_config_from_cfg(cfg: PretrainedConfig) -> Optional[MoEConfig]:
    """
    Detect and return MoE configuration if present.

    Uses config attributes rather than family name checks.
    """
    # Check various expert count attributes (different models use different names)
    num_experts = (
        getattr(cfg, "num_local_experts", None)
        or getattr(cfg, "num_experts", None)
        or getattr(cfg, "n_routed_experts", None)  # DeepSeek V3
    )

    # Detect MoE-specific attributes that indicate MoE architecture
    has_moe_routing = (
        hasattr(cfg, "num_experts_per_tok")
        or hasattr(cfg, "num_selected_experts")
        or hasattr(cfg, "router_aux_loss_coef")  # MoE routers have auxiliary loss
        or hasattr(cfg, "expert_capacity")  # Expert capacity is MoE-specific
        or hasattr(cfg, "router_jitter_noise")  # MoE routing noise
        or hasattr(cfg, "n_routed_experts")  # DeepSeek V3 style
    )

    # Only treat as MoE if num_experts > 1 AND we have MoE routing configuration
    if num_experts and num_experts > 1 and has_moe_routing:
        experts_per_tok = (
            getattr(cfg, "num_experts_per_tok", None)
            or getattr(cfg, "num_selected_experts", None)
            or 2  # default to 2 if not specified
        )

        # Some models have MoE only in certain layers
        # DeepSeek V3 uses moe_layer_freq to determine which layers have MoE
        moe_layer_freq = getattr(cfg, "moe_layer_freq", None)
        if moe_layer_freq is not None and moe_layer_freq > 0:
            # If moe_layer_freq=1, all layers have MoE
            # If moe_layer_freq=2, every other layer has MoE
            num_expert_layers = cfg.num_hidden_layers // moe_layer_freq
        else:
            num_expert_layers = getattr(cfg, "num_expert_layers", None) or cfg.num_hidden_layers

        # Check for separate MoE FFN size (DeepSeek V3 uses smaller FFN for experts)
        expert_ffn_size = getattr(cfg, "moe_intermediate_size", None)

        # Check for shared experts (DeepSeek V3 has shared + routed experts)
        # Shared experts are replicated (not sharded by EP)
        num_shared_experts = getattr(cfg, "n_shared_experts", 0)

        return MoEConfig(
            num_experts=int(num_experts),
            num_experts_per_tok=int(experts_per_tok),
            num_expert_layers=int(num_expert_layers),
            expert_ffn_size=int(expert_ffn_size) if expert_ffn_size is not None else None,
            num_shared_experts=int(num_shared_experts),
        )

    return None


def detect_gated_mlp_from_cfg(cfg: PretrainedConfig) -> bool:
    """
    Detect if model uses gated MLP (e.g., SwiGLU, GeGLU).

    Gated MLPs use 3x parameter multiplier vs 2x for standard MLPs.
    """
    mlp_act = getattr(cfg, "hidden_act", None) or getattr(cfg, "activation_function", None)
    mlp_hidden_act = getattr(cfg, "mlp_hidden_act", None)

    # Explicit gated activation names
    if mlp_act:
        mlp_act_lower = mlp_act.lower()
        if any(x in mlp_act_lower for x in ["swiglu", "geglu", "siglu"]):
            return True
        if "relu" in mlp_act_lower:
            return False

    if mlp_hidden_act:
        mlp_hidden_act_lower = mlp_hidden_act.lower()
        if any(x in mlp_hidden_act_lower for x in ["swiglu", "geglu", "siglu"]):
            return True
        if "relu" in mlp_hidden_act_lower:
            return False

    # For silu/gelu activations, assume gated by default (safer for memory estimation)
    if mlp_act and mlp_act.lower() in ["silu", "gelu"]:
        return True

    return False


def detect_sliding_window_from_cfg(cfg: PretrainedConfig) -> Optional[SlidingWindowConfig]:
    """Detect sliding window attention configuration."""
    if hasattr(cfg, "sliding_window") and cfg.sliding_window is not None:
        return SlidingWindowConfig(window_size=int(cfg.sliding_window))
    return None


def detect_interleaved_mamba_from_cfg(cfg: PretrainedConfig) -> MambaConfig:
    """Detect interleaved Mamba architecture (e.g., Nemotron-Nano-9B)."""

    layer_counts = Counter(cfg.layers_block_type)
    num_mamba_layers = layer_counts.get("mamba", 0)
    num_attn_layers_from_type = layer_counts.get("attention", 0)
    num_mlp_layers = layer_counts.get("mlp", 0)

    state_size = (
        getattr(cfg, "state_size", None) or getattr(cfg, "ssm_state_size", None) or getattr(cfg, "mamba_state_dim", 16)
    )
    conv_kernel = getattr(cfg, "conv_kernel", None) or getattr(cfg, "d_conv", 4)

    return MambaConfig(
        is_hybrid=True,
        num_mamba_layers=int(num_mamba_layers),
        num_attention_layers=int(num_attn_layers_from_type),
        num_mlp_layers=int(num_mlp_layers),
        state_size=int(state_size),
        conv_kernel=int(conv_kernel),
    )


def detect_hybrid_mamba_from_cfg(cfg: PretrainedConfig, n_layers: int) -> MambaConfig:
    """Detect hybrid Mamba with explicit num_attention_layers."""
    num_attn_layers = cfg.num_attention_layers
    num_mamba_layers = n_layers - num_attn_layers

    state_size = (
        getattr(cfg, "state_size", None) or getattr(cfg, "ssm_state_size", None) or getattr(cfg, "mamba_state_dim", 16)
    )
    conv_kernel = getattr(cfg, "conv_kernel", None) or getattr(cfg, "d_conv", 4)

    return MambaConfig(
        is_hybrid=True,
        num_mamba_layers=int(num_mamba_layers),
        num_attention_layers=int(num_attn_layers),
        num_mlp_layers=0,
        state_size=int(state_size),
        conv_kernel=int(conv_kernel),
    )


def detect_hybrid_mamba_via_introspection(
    cfg: PretrainedConfig, n_layers: int, is_trusted: bool = False
) -> MambaConfig:
    """Detect hybrid Mamba via model introspection (fallback)."""
    try:
        # Load model structure on meta device to inspect layer types
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=is_trusted)

        # Find the layers container
        layers = None
        for attr_name in ["layers", "h", "blocks", "decoder_layers"]:
            if hasattr(model, attr_name):
                layers = getattr(model, attr_name)
                break
            if hasattr(model, "transformer") and hasattr(model.transformer, attr_name):
                layers = getattr(model.transformer, attr_name)
                break
            if hasattr(model, "model") and hasattr(model.model, attr_name):
                layers = getattr(model.model, attr_name)
                break

        num_mamba_layers_found = 0
        num_attn_layers_found = 0

        if layers is not None and len(layers) > 0:
            # Inspect each layer to determine its type
            for layer in layers:
                # Check if this layer contains Mamba/SSM components
                has_mamba = any(hasattr(layer, attr) for attr in ["mixer", "ssm", "mamba", "in_proj"])
                # Check if this layer contains attention components
                has_attn = any(hasattr(layer, attr) for attr in ["self_attn", "attention", "attn"])

                if has_mamba:
                    num_mamba_layers_found += 1
                elif has_attn:
                    num_attn_layers_found += 1

            # Use introspected counts if we found layers
            if num_mamba_layers_found > 0 or num_attn_layers_found > 0:
                num_mamba_layers = num_mamba_layers_found
                num_attn_layers_introspected = num_attn_layers_found
            else:
                # Fallback: estimate
                num_attn_layers_introspected = max(1, n_layers // 10)
                num_mamba_layers = n_layers - num_attn_layers_introspected
        else:
            # Fallback
            num_attn_layers_introspected = max(1, n_layers // 10)
            num_mamba_layers = n_layers - num_attn_layers_introspected
    except Exception:
        # If introspection fails, fall back to estimate
        num_attn_layers_introspected = max(1, n_layers // 10)
        num_mamba_layers = n_layers - num_attn_layers_introspected

    state_size = (
        getattr(cfg, "state_size", None) or getattr(cfg, "ssm_state_size", None) or getattr(cfg, "mamba_state_dim", 16)
    )
    conv_kernel = getattr(cfg, "conv_kernel", None) or getattr(cfg, "d_conv", 4)

    return MambaConfig(
        is_hybrid=True,
        num_mamba_layers=int(num_mamba_layers),
        num_attention_layers=int(num_attn_layers_introspected),
        num_mlp_layers=0,
        state_size=int(state_size),
        conv_kernel=int(conv_kernel),
    )


def detect_pure_mamba_from_cfg(cfg: PretrainedConfig, n_layers: int) -> MambaConfig:
    """Detect pure Mamba model (no attention layers)."""
    state_size = (
        getattr(cfg, "state_size", None) or getattr(cfg, "ssm_state_size", None) or getattr(cfg, "mamba_state_dim", 16)
    )
    conv_kernel = getattr(cfg, "conv_kernel", None) or getattr(cfg, "d_conv", 4)

    return MambaConfig(
        is_hybrid=False,
        num_mamba_layers=int(n_layers),
        num_attention_layers=0,
        num_mlp_layers=0,
        state_size=int(state_size),
        conv_kernel=int(conv_kernel),
    )


def detect_mamba_config_from_cfg(
    cfg: PretrainedConfig, family: str, n_layers: int, is_trusted: bool = False
) -> Optional[MambaConfig]:
    """
    Detect and return Mamba/SSM configuration if present.

    Uses config attributes rather than family name checks.
    """
    # Check for SSM/Mamba by looking for actual config attributes
    has_ssm_config = (
        hasattr(cfg, "ssm_state_size")
        or hasattr(cfg, "state_size")
        or hasattr(cfg, "mamba_state_dim")
        or hasattr(cfg, "layers_block_type")  # Interleaved architectures
        or hasattr(cfg, "ssm_cfg")  # Some models have ssm_cfg dict
    )

    if not has_ssm_config:
        return None

    # Priority 1: Interleaved architecture with explicit layer types
    if hasattr(cfg, "layers_block_type"):
        return detect_interleaved_mamba_from_cfg(cfg)

    # Priority 2: Hybrid with explicit num_attention_layers
    num_attn_layers = getattr(cfg, "num_attention_layers", None)
    if num_attn_layers is not None and num_attn_layers < n_layers:
        return detect_hybrid_mamba_from_cfg(cfg, n_layers)

    # Priority 3: Hybrid inferred from '_h' suffix (with introspection)
    if has_ssm_config and "_h" in family:
        return detect_hybrid_mamba_via_introspection(cfg, n_layers, is_trusted)

    # Priority 4: Pure Mamba model
    return detect_pure_mamba_from_cfg(cfg, n_layers)


def _has_config_json(pretrained_or_path: str) -> bool:
    """Check if config.json exists (locally or remotely)."""
    local_path = Path(pretrained_or_path)
    if local_path.is_dir():
        return (local_path / "config.json").exists()

    # Remote model - try to download config.json
    try:
        hf_hub_download(pretrained_or_path, "config.json")
        return True
    except Exception:
        return False


def _get_yaml_path(pretrained_or_path: str) -> Optional[Path]:
    """Get path to model_config.yaml (locally or from HF Hub)."""
    local_path = Path(pretrained_or_path)
    if local_path.is_dir():
        yaml_path = local_path / "model_config.yaml"
        return yaml_path if yaml_path.exists() else None

    # Remote model - try to download from HF Hub
    try:
        return Path(hf_hub_download(pretrained_or_path, "model_config.yaml"))
    except Exception:
        return None


def try_load_nemo_yaml_config(pretrained_or_path: str, is_trusted: bool = False) -> Optional[PretrainedConfig]:
    """
    Attempt to load configuration from NeMo's model_config.yaml if config.json is missing.

    Some NVIDIA models (like Nemotron-4-340B) only have NeMo YAML configs, not HF configs.
    This function handles that case by finding and parsing the YAML (either locally or from HF Hub),
    then constructing a proper HuggingFace config from it.

    Returns:
        PretrainedConfig if YAML was found and parsed, None otherwise
    """
    # Skip YAML fallback if config.json exists
    if _has_config_json(pretrained_or_path):
        return None

    # Try to find model_config.yaml
    yaml_path = _get_yaml_path(pretrained_or_path)
    if yaml_path is None:
        return None

    try:
        # Load and parse YAML
        with open(yaml_path) as f:
            nemo_config = yaml.safe_load(f)

        # Extract relevant fields from NeMo config
        num_layers = nemo_config.get("num_layers")
        hidden_size = nemo_config.get("hidden_size")
        ffn_hidden_size = nemo_config.get("ffn_hidden_size")
        num_attention_heads = nemo_config.get("num_attention_heads")
        num_query_groups = nemo_config.get("num_query_groups")
        vocab_size = nemo_config.get("make_vocab_size_divisible_by", 128) * 256  # Approximation

        if all(v is not None for v in [num_layers, hidden_size, ffn_hidden_size, num_attention_heads]):
            # Get the model type to construct the right config class
            base_cfg = AutoConfig.from_pretrained(pretrained_or_path, trust_remote_code=is_trusted)

            # Override with YAML values
            base_cfg.num_hidden_layers = num_layers
            base_cfg.hidden_size = hidden_size
            base_cfg.intermediate_size = ffn_hidden_size
            base_cfg.num_attention_heads = num_attention_heads
            if num_query_groups:
                base_cfg.num_key_value_heads = num_query_groups
            if vocab_size:
                base_cfg.vocab_size = vocab_size

            return base_cfg

    except Exception:
        # YAML parsing failed - not an error, just use standard config
        pass

    return None

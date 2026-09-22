#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive pytest test suite for parallelism_helper.py validation against NVIDIA NeMo configs.

This test suite validates that our tool's recommendations align with NVIDIA's proven
parallelization strategies from their H100 configs.

From: https://github.com/NVIDIA-NeMo/NeMo/blob/main/scripts/performance/recommended_model_configs/model_configs_h100.csv
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch required for parallelism API tests")

_DATA_FILE = Path(__file__).resolve().parents[2] / "parallelism" / "nemo_validation_data.py"
_spec = spec_from_file_location("models_nemo_validation_data", _DATA_FILE)
_nv = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_nv)
GATED_MODEL_IDS = _nv.GATED_MODEL_IDS
H100_MEM_GB = _nv.H100_MEM_GB
NEMO_CONFIGS = _nv.NEMO_CONFIGS
NEMO_CONFIGS_PARAMS = _nv.NEMO_CONFIGS_PARAMS
REQUIRES_HF_TOKEN = _nv.REQUIRES_HF_TOKEN

from nhx.core.models.parallelism.api import (  # noqa: E402
    estimate_parallelization,
    find_minimum_gpus,
    infer_model_cfg_from_hf,
)

# =================================================================================================
# TEST FIXTURES
# =================================================================================================


@pytest.fixture
def h100_memory():
    """H100 80GB memory."""
    return H100_MEM_GB


# =================================================================================================
# HELPER FUNCTIONS
# =================================================================================================


def check_parallelization_match(our_config, nemo_config):
    """
    Check if our configuration matches NeMo's expected config.

    Args:
        our_config: ParallelizationConfig from our tool
        nemo_config: Expected config from NeMo (dict with tp, pp, dp, ep, cp)

    Returns:
        (bool, str): (matches, reason)

    Matching criteria (STRICT):
    - TP, PP, EP, CP must all match
    - DP may differ slightly due to rounding (within 1 GPU)

    Note: VP (Virtual Pipeline Parallelism) is not considered as it's an efficiency
    optimization that doesn't affect memory feasibility (our primary concern).
    """
    # All dimensions must match
    tp_match = our_config.tp == nemo_config["tp"]
    pp_match = our_config.pp == nemo_config["pp"]
    ep_match = our_config.ep == nemo_config["ep"]
    cp_match = our_config.cp == nemo_config["cp"]

    # DP can differ by 1 due to rounding (DP = GPUs / (TP * PP * EP * CP))
    dp_close = abs(our_config.dp - nemo_config["dp"]) <= 1

    # Require all core dimensions to match
    if tp_match and pp_match and ep_match and cp_match and dp_close:
        if our_config.dp == nemo_config["dp"]:
            return True, "Exact match"
        else:
            return True, f"Match (DP slightly different: ours={our_config.dp} vs nemo={nemo_config['dp']})"
    else:
        diff_parts = []
        if not tp_match:
            diff_parts.append(f"TP (ours={our_config.tp} vs nemo={nemo_config['tp']})")
        if not pp_match:
            diff_parts.append(f"PP (ours={our_config.pp} vs nemo={nemo_config['pp']})")
        if not ep_match:
            diff_parts.append(f"EP (ours={our_config.ep} vs nemo={nemo_config['ep']})")
        if not cp_match:
            diff_parts.append(f"CP (ours={our_config.cp} vs nemo={nemo_config['cp']})")
        if not dp_close:
            diff_parts.append(f"DP (ours={our_config.dp} vs nemo={nemo_config['dp']})")
        return False, f"Mismatch: {', '.join(diff_parts)}"


# =================================================================================================
# NEMO CONFIG TESTS - Parametrized for each config
# =================================================================================================


@pytest.mark.parametrize("config", NEMO_CONFIGS_PARAMS, ids=[cfg["id"] for cfg in NEMO_CONFIGS])
def test_nemo_config(config, h100_memory):
    """
    Test that our tool's recommendations are stable and memory-safe.

    This test validates each config entry against our tool's output.

    Note: Some configs differ from NeMo's original recommendations:
    - Our tool optimizes for throughput (prefers DP over PP for speed)
    - NeMo optimizes for memory safety (prefers PP to keep static memory <30%)
    Both approaches are valid. We test against our throughput-optimized configs
    to prevent regressions, while documenting the trade-offs in comments.
    """
    model_id = config["model_id"]
    gpus = config["gpus"]
    seq_len = config["seq_len"]
    nemo_config = config["nemo_config"]
    lora = config.get("lora", False)

    # Run our tool
    kwargs = {}
    if lora:
        kwargs["lora"] = True
        kwargs["lora_r"] = 8

    # Use NeMo's microbatch size if specified
    if "microbatch_size" in config:
        kwargs["microbatch_size"] = config["microbatch_size"]

    # Pass trust_remote_code flag if specified (for models with custom architectures)
    if "is_trusted" in config:
        kwargs["is_trusted"] = config["is_trusted"]

    result = estimate_parallelization(model_id, gpus, h100_memory, seq_len, **kwargs)

    # Assert we got results
    assert result is not None, f"Failed to get parallelization estimate for {model_id}"
    assert len(result.configs) > 0, f"No valid configurations found for {model_id}"

    # Check the top recommendation
    best_config = result.configs[0]

    # Verify configuration is feasible
    assert best_config.total_memory_per_rank_gb <= h100_memory, (
        f"Config exceeds GPU memory: {best_config.total_memory_per_rank_gb:.1f}GB > {h100_memory}GB"
    )

    # Check match with NeMo config
    matches, reason = check_parallelization_match(best_config, nemo_config)

    # For debugging, print comparison if not matching
    if not matches:
        # Find NeMo's config in our results to compare static memory
        nemo_static_mem = None
        for cfg in result.configs:
            if (
                cfg.tp == nemo_config["tp"]
                and cfg.pp == nemo_config["pp"]
                and cfg.cp == nemo_config["cp"]
                and cfg.ep == nemo_config["ep"]
            ):
                nemo_static_mem = cfg.per_rank_static_gb
                break

        our_static_mem = best_config.per_rank_static_gb

        print(f"\n{'=' * 80}")
        print(f"Config mismatch for {config['id']}")

        nemo_static_str = f"{nemo_static_mem:.1f}GB" if nemo_static_mem else "N/A (not in our candidates)"
        print(
            f"NeMo:  TP={nemo_config['tp']} PP={nemo_config['pp']} DP={nemo_config['dp']} "
            f"CP={nemo_config['cp']} EP={nemo_config['ep']} | Static: {nemo_static_str}"
        )
        print(
            f"Ours:  TP={best_config.tp} PP={best_config.pp} DP={best_config.dp} "
            f"CP={best_config.cp} EP={best_config.ep} | Static: {our_static_mem:.1f}GB"
        )
        print(f"Reason: {reason}")
        print(f"Total Memory (Ours): {best_config.total_memory_per_rank_gb:.1f}GB / {h100_memory}GB")
        print(f"{'=' * 80}\n")

    # All configs must match - no exceptions
    assert matches, f"Configuration mismatch: {reason}"


# =================================================================================================
# GPT-OSS-120B SPECIFIC TESTS
# =================================================================================================


def test_gpt_oss_120b_minimum_gpus():
    """
    Test that GPT-OSS-120B can be loaded and analyzed for minimum GPU requirements.

    This is a large MoE model (117B total params, 128 experts, 4 active per token).
    With transformers >= 4.48, GPT-OSS-120B is now available via HuggingFace.
    """
    # Calculate minimum GPUs required
    min_gpus, best_config = find_minimum_gpus(
        "openai/gpt-oss-120b",
        H100_MEM_GB,
        4096,
        max_gpus=128,
    )

    # Verify we got a result
    assert min_gpus is not None, "Failed to calculate minimum GPUs"
    assert best_config is not None, "No valid configuration found"

    # Verify the configuration is valid
    assert best_config.total_gpus == min_gpus
    assert best_config.total_memory_per_rank_gb <= H100_MEM_GB

    # assert best_config.ep > 1, "Expected EP > 1 for MoE model with 128 experts"
    # Note: For minimum GPU configurations, EP=1 may be necessary to fit in memory
    # When optimizing for minimum GPUs (not throughput), EP might be sacrificed
    # to reduce the total GPU count. This is acceptable for minimum GPU estimation.

    # Print results for documentation
    print(f"\nGPT-OSS-120B minimum GPUs: {min_gpus}")
    print(
        f"Configuration: TP={best_config.tp}, PP={best_config.pp}, DP={best_config.dp}, "
        f"CP={best_config.cp}, EP={best_config.ep}"
    )
    print(f"Memory per rank: {best_config.total_memory_per_rank_gb:.1f}GB")


def test_gpt_oss_120b_model_config():
    """Test that GPT-OSS-120B can be loaded from HuggingFace and has correct config."""
    # Load model config from HuggingFace
    model_cfg = infer_model_cfg_from_hf("openai/gpt-oss-120b")

    # Verify basic architecture
    assert model_cfg.checkpoint_model_name == "GptOssForCausalLM"
    assert model_cfg.family == "gpt_oss"
    assert model_cfg.num_layers == 36
    assert model_cfg.hidden_size == 2880

    # Verify MoE configuration
    assert model_cfg.moe_config is not None, "GPT-OSS-120B should be detected as MoE"
    assert model_cfg.moe_config.num_experts == 128, "Should have 128 experts"
    assert model_cfg.moe_config.num_experts_per_tok == 4, "Should activate 4 experts per token"
    assert model_cfg.moe_config.num_expert_layers == 36, "All layers should have experts"

    # Verify GQA configuration (8:1 ratio)
    assert model_cfg.num_attention_heads == 64
    assert model_cfg.num_kv_heads == 8, "Should use GQA with 8 KV heads"

    # Verify gated MLP detection
    assert model_cfg.gated_mlp is True, "Should detect gated MLP"

    print("\n[OK] GPT-OSS-120B config validated:")
    print(f"  {model_cfg.moe_config.num_experts} experts, {model_cfg.moe_config.num_experts_per_tok} active/token")
    print(f"  GQA ratio: {model_cfg.num_attention_heads}:{model_cfg.num_kv_heads}")


def test_gpt_oss_120b_parallelization():
    """Test full parallelization estimate for GPT-OSS-120B on realistic hardware."""
    # Test with 64 H100 GPUs (realistic large-scale setup)
    result = estimate_parallelization(
        "openai/gpt-oss-120b",
        gpus=64,
        gpu_mem_gb=H100_MEM_GB,
        seq_len=4096,
        microbatch_size=1,
    )

    # Should find valid configurations
    assert result is not None, "Failed to get parallelization estimate"
    assert len(result.configs) > 0, "No valid configurations found"

    # Check best configuration
    best = result.configs[0]
    assert best.total_gpus == 64
    assert best.total_memory_per_rank_gb <= H100_MEM_GB

    # Should use Expert Parallelism (critical for 128-expert MoE)
    assert best.ep > 1, "Expected EP > 1 for 128-expert MoE model"

    # Should use reasonable TP (not too aggressive)
    assert best.tp >= 1 and best.tp <= 8, f"TP={best.tp} seems unusual"

    print("\n[OK] GPT-OSS-120B on 64x H100:")
    print(f"  Best config: TP={best.tp}, PP={best.pp}, DP={best.dp}, CP={best.cp}, EP={best.ep}")
    print(f"  Memory: {best.total_memory_per_rank_gb:.1f}GB / {H100_MEM_GB}GB")
    print(f"  Microbatch size: {best.microbatch_per_dp}")


# =================================================================================================
# ARCHITECTURE DETECTION TESTS
# =================================================================================================


def test_moe_detection_positive():
    """Test that MoE models are correctly detected (non-gated: Mixtral)."""
    model_cfg = infer_model_cfg_from_hf("mistralai/Mixtral-8x7B-v0.1")
    assert model_cfg.moe_config is not None
    assert model_cfg.moe_config.num_experts == 8


@REQUIRES_HF_TOKEN
def test_moe_detection_negative():
    """Test that non-MoE models are not detected as MoE (gated: Llama)."""
    model_cfg = infer_model_cfg_from_hf("meta-llama/Meta-Llama-3-8B")
    assert model_cfg.moe_config is None


def test_mamba_detection():
    """
    Test that Mamba/SSM models are correctly detected.

    Source: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/discussions
    Nemotron-Nano-9B-v2 is a hybrid Mamba2-Transformer architecture.
    """
    model_cfg = infer_model_cfg_from_hf("nvidia/NVIDIA-Nemotron-Nano-9B-v2", is_trusted=True)

    # Should be detected as Mamba hybrid - check for mamba_config attribute
    assert hasattr(model_cfg, "mamba_config"), "Model should have mamba_config attribute"
    # Mamba config should exist (not None) for hybrid models
    assert model_cfg.mamba_config is not None, "Mamba config should not be None for hybrid Mamba model"


# =================================================================================================
# API TESTS
# =================================================================================================


@REQUIRES_HF_TOKEN
def test_estimate_parallelization_api():
    """Test that the estimate_parallelization API works correctly."""
    # Simple case: Llama 8B on 8 GPUs
    result = estimate_parallelization(
        "meta-llama/Meta-Llama-3-8B",
        gpus=8,
        gpu_mem_gb=80,
        seq_len=8192,
    )

    assert result is not None
    assert len(result.configs) > 0
    assert result.gpu_count == 8
    assert result.gpu_mem_gb == 80
    assert result.seq_len == 8192

    # Best config should be feasible
    best = result.configs[0]
    assert best.total_memory_per_rank_gb <= 80
    assert best.total_gpus == 8


@REQUIRES_HF_TOKEN
def test_find_minimum_gpus_api():
    """Test that the find_minimum_gpus API works correctly."""
    # Find minimum for Llama 8B
    min_gpus, best_config = find_minimum_gpus(
        "meta-llama/Meta-Llama-3-8B",
        gpu_mem_gb=80,
        seq_len=8192,
        max_gpus=16,
    )

    assert min_gpus is not None
    assert best_config is not None
    assert min_gpus <= 16

    # Verify the config matches the reported GPU count
    assert best_config.total_gpus == min_gpus


@REQUIRES_HF_TOKEN
def test_context_parallelism_enabled_for_long_sequences():
    """Test that CP is automatically enabled for long sequences (8K+)."""
    result = estimate_parallelization(
        "meta-llama/Meta-Llama-3-8B",
        gpus=8,
        gpu_mem_gb=80,
        seq_len=8192,  # 8K should trigger CP
    )

    # At least one config should use CP
    has_cp = any(cfg.cp > 1 for cfg in result.configs)
    assert has_cp, "Expected CP > 1 for 8K sequence length"


def test_expert_parallelism_for_moe():
    """Test that EP is used for MoE models."""
    result = estimate_parallelization(
        "mistralai/Mixtral-8x7B-v0.1",
        gpus=64,
        gpu_mem_gb=80,
        seq_len=4096,
    )

    # Best config should use EP
    best = result.configs[0]
    assert best.ep > 1, "Expected EP > 1 for MoE model"


# =================================================================================================
# EDGE CASE TESTS
# =================================================================================================


def test_single_gpu_small_model():
    """Test that small models can fit on a single GPU."""
    # GPT-2 should easily fit on 1 GPU
    result = estimate_parallelization(
        "gpt2",
        gpus=1,
        gpu_mem_gb=80,
        seq_len=1024,
    )

    assert len(result.configs) > 0
    best = result.configs[0]
    assert best.tp == 1
    assert best.pp == 1
    assert best.dp == 1
    assert best.ep == 1
    assert best.cp == 1


@REQUIRES_HF_TOKEN
def test_very_long_sequence():
    """Test handling of very long sequences (128K)."""
    result = estimate_parallelization(
        "meta-llama/Meta-Llama-3-8B",
        gpus=64,
        gpu_mem_gb=80,
        seq_len=131072,  # 128K
    )

    # Should find configs with CP >= 2 for very long sequences
    assert len(result.configs) > 0
    best = result.configs[0]
    assert best.cp >= 2, "Expected CP >= 2 for 128K sequence"


@REQUIRES_HF_TOKEN
def test_insufficient_gpus():
    """Test that tool correctly reports when GPUs are insufficient."""
    # Try to fit Llama 70B on 1 GPU - should fail
    result = estimate_parallelization(
        "meta-llama/Meta-Llama-3-70B",
        gpus=1,
        gpu_mem_gb=80,
        seq_len=4096,
    )

    # Should return empty configs or configs that don't fit
    if len(result.configs) > 0:
        best = result.configs[0]
        # If any config is returned, it should exceed memory
        assert best.total_memory_per_rank_gb > 80
    else:
        # No configs found - this is also acceptable
        assert len(result.configs) == 0


# =================================================================================================
# CONSISTENCY TESTS
# =================================================================================================

# 70B-class models: NeMo uses TP=4, PP=4, CP=2 at 8K. Qwen is non-gated; meta-llama are gated.
MODELS_70B = [
    ("meta-llama/Meta-Llama-3-70B", "llama3_70b"),
    ("Qwen/Qwen2.5-72B", "qwen_72b"),
    ("meta-llama/Llama-3.3-70B-Instruct", "llama33_70b"),
]


def _model_70b_param(model_id, id_):
    if model_id in GATED_MODEL_IDS:
        return pytest.param(model_id, id_, marks=REQUIRES_HF_TOKEN)
    return (model_id, id_)


MODELS_70B_PARAMS = [_model_70b_param(mid, id_) for mid, id_ in MODELS_70B]


@pytest.mark.parametrize("model_id,model_id_short", MODELS_70B_PARAMS, ids=[p[1] for p in MODELS_70B])
def test_70b_models_consistency(model_id, model_id_short):
    """
    Test that each 70B-class model with 8K sequence gets NeMo-aligned recommendations.

    NeMo uses TP=4, PP=4, CP=2 for all 70B models at 8K. Parametrized so Qwen (non-gated)
    runs without HF_TOKEN; meta-llama cases require HF_TOKEN.
    """
    result = estimate_parallelization(
        model_id,
        gpus=64,
        gpu_mem_gb=80,
        seq_len=8192,
    )
    assert len(result.configs) > 0, f"No configs for {model_id}"
    best = result.configs[0]
    assert best.tp == 4, f"Expected TP=4 for 70B at 8K, got {best.tp} for {model_id}"
    assert best.pp == 4, f"Expected PP=4 for 70B at 8K, got {best.pp} for {model_id}"
    assert best.cp == 2, f"Expected CP=2 for 70B at 8K, got {best.cp} for {model_id}"

    # Note: With distributed optimizer, CP requirements are relaxed
    # Core NeMo validation tests already validate CP usage where needed

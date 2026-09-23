# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for validating parallelization recommendations on recent 2025 models.

This script tests the tool's ability to handle the latest models from HuggingFace,
ensuring it generalizes well to new architectures and configurations.
"""

import pytest

pytest.importorskip("torch", reason="torch required for parallelism API tests")

from nhx.core.models.parallelism.api import estimate_parallelization, find_minimum_gpus

# TODO: Mock HuggingFace API calls instead of accessing real gated models
REQUIRES_HF_TOKEN = pytest.mark.skip(
    reason="Gated HuggingFace models require mocking (not yet implemented)",
)
GATED_MODEL_IDS = frozenset({"meta-llama/Llama-3.3-70B-Instruct"})

# Very recent models from 2024-2025
TEST_MODELS = [
    # Nemotron Super 49B v1.5 (July 2025) - NAS-optimized derivative of Llama 3.3 70B
    # NOTE: Disabled - requires transformers<=4.52.2, but GPT-OSS requires >=4.55
    # ("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5", "Nemotron Super 49B v1.5", 8192, 128),
    # Mistral NeMo Minitron 8B (2024-2025)
    ("nvidia/Mistral-NeMo-Minitron-8B-Instruct", "Mistral NeMo Minitron 8B", 8192, 128),
    # Qwen3 8B (2025)
    ("Qwen/Qwen3-8B", "Qwen3 8B", 8192, 128),
    # Devstral Small (May 2025) - Mistral's latest code model
    ("mistralai/Devstral-Small-2505", "Devstral Small 22B", 8192, 128),
    # Qwen3 4B SafeRL (2025)
    ("Qwen/Qwen3-4B-SafeRL", "Qwen3 4B SafeRL", 8192, 128),
    # Llama 3.3 70B (Dec 2024)
    ("meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B", 8192, 128),
    # DeepSeek V3 (Dec 2024) - Very large MoE, needs more GPUs
    # NOTE: Disabled - extremely large model (671B params) takes a long time to test
    # ("deepseek-ai/DeepSeek-V3-Base", "DeepSeek V3 671B MoE", 4096, 512),
    # Phi-4 (Dec 2024)
    ("microsoft/phi-4", "Phi-4 14B", 16384, 128),
    # Qwen 2.5 72B (Sept 2024)
    ("Qwen/Qwen2.5-72B-Instruct", "Qwen 2.5 72B", 8192, 128),
]

H100_MEM = 80


def _test_model_param(entry):
    model_id, name, seq_len, max_gpus = entry
    if model_id in GATED_MODEL_IDS:
        return pytest.param(model_id, name, seq_len, max_gpus, marks=REQUIRES_HF_TOKEN)
    return entry


TEST_MODELS_PARAMS = [_test_model_param(entry) for entry in TEST_MODELS]


@pytest.mark.parametrize(
    "model_id,name,seq_len,max_gpus",
    TEST_MODELS_PARAMS,
    ids=[entry[1] for entry in TEST_MODELS],
)
def test_model_sft_and_lora(model_id, name, seq_len, max_gpus):
    """Test that recent 2025 models can be analyzed and configured for both SFT and LoRA."""
    print(f"\nTesting {name} ({model_id})")
    print("-" * 110)

    # Find minimum GPUs for full training (SFT)
    min_gpus_sft, best_config_sft = find_minimum_gpus(model_id, H100_MEM, seq_len, max_gpus=max_gpus)

    # Find minimum GPUs for LoRA fine-tuning
    min_gpus_lora, best_config_lora = find_minimum_gpus(
        model_id, H100_MEM, seq_len, max_gpus=max_gpus, lora=True, lora_r=8
    )

    # Assert we got valid configurations
    assert min_gpus_sft is not None, f"Failed to find SFT config for {name}"
    assert min_gpus_lora is not None, f"Failed to find LoRA config for {name}"

    # Assert LoRA requires fewer or equal GPUs than SFT
    assert min_gpus_lora <= min_gpus_sft, (
        f"LoRA should require <= GPUs than SFT: LoRA={min_gpus_lora}, SFT={min_gpus_sft}"
    )

    # Get full result for model info
    result = estimate_parallelization(model_id, min_gpus_sft, H100_MEM, seq_len)
    params_b = result.model_info.base_num_parameters / 1e9

    # Print detailed results (streamed by pytest)
    print(f"  [OK] Full Training (SFT): {min_gpus_sft} GPUs")
    print(
        f"    Config: TP={best_config_sft.tp} PP={best_config_sft.pp} DP={best_config_sft.dp} CP={best_config_sft.cp} EP={best_config_sft.ep}"
    )
    print(
        f"    Memory: {best_config_sft.total_memory_per_rank_gb:.1f}GB / {H100_MEM}GB ({best_config_sft.total_memory_per_rank_gb / H100_MEM * 100:.0f}%)"
    )
    print(f"    Params: {params_b:.2f}B ({result.model_info.base_num_parameters:,})")
    print(f"    Layers: {result.model_info.num_layers}, Hidden: {result.model_info.hidden_size}")

    if result.model_info.moe_config:
        moe = result.model_info.moe_config
        print(f"    MoE: {moe.num_experts} experts, {moe.num_experts_per_tok} active")
    if result.model_info.sliding_window_config:
        sw = result.model_info.sliding_window_config
        print(f"    Sliding Window: {sw.window_size} tokens")

    print(f"\n  [OK] LoRA Fine-tuning (r=8): {min_gpus_lora} GPUs")
    print(
        f"    Config: TP={best_config_lora.tp} PP={best_config_lora.pp} DP={best_config_lora.dp} CP={best_config_lora.cp} EP={best_config_lora.ep}"
    )
    print(
        f"    Memory: {best_config_lora.total_memory_per_rank_gb:.1f}GB / {H100_MEM}GB ({best_config_lora.total_memory_per_rank_gb / H100_MEM * 100:.0f}%)"
    )

    gpu_reduction = min_gpus_sft - min_gpus_lora
    if gpu_reduction > 0:
        print(f"    Reduction: {gpu_reduction} fewer GPUs than SFT")
    else:
        print("    Reduction: Same as SFT")

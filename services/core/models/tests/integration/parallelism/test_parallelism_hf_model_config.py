# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: Hugging Face model config inference for parallelism (network I/O)."""

import pytest

pytest.importorskip("torch", reason="torch required for parallelism API tests")

from nhx.core.models.parallelism.api import infer_model_cfg_from_hf

# TODO: Mock HuggingFace API calls instead of accessing real gated models
REQUIRES_HF_TOKEN = pytest.mark.skip(
    reason="Gated HuggingFace models require mocking (not yet implemented)",
)
GATED_MODEL_IDS = frozenset({"meta-llama/Llama-3.1-8B"})

MODEL_TEST_CASES = [
    {
        "id": "gpt2",
        "model_id": "gpt2",
        "checkpoint_model_name": "GPT2LMHeadModel",
        "expected_family": "gpt2",
        "expected_layers": 12,
        "expected_hidden": 768,
        "expected_moe": False,
        "description": "GPT-2 (small dense model)",
    },
    {
        "id": "mixtral_8x7b",
        "model_id": "mistralai/Mixtral-8x7B-v0.1",
        "checkpoint_model_name": "MixtralForCausalLM",
        "expected_family": "mixtral",
        "expected_layers": 32,
        "expected_hidden": 4096,
        "expected_moe": True,
        "expected_num_experts": 8,
        "description": "Mixtral-8x7B (MoE model)",
    },
    {
        "id": "phi2",
        "model_id": "microsoft/phi-2",
        "checkpoint_model_name": "PhiForCausalLM",
        "expected_family": "phi",
        "expected_layers": 32,
        "expected_hidden": 2560,
        "expected_moe": False,
        "description": "Phi-2 (efficient dense model)",
    },
    {
        "id": "phi4",
        "model_id": "microsoft/phi-4",
        "checkpoint_model_name": "Phi3ForCausalLM",
        "expected_family": "phi",
        "expected_moe": False,
        "description": "Phi-4 (latest Microsoft efficient model)",
    },
    {
        "id": "nemotron_nano_9b",
        "model_id": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        "checkpoint_model_name": "NemotronHForCausalLM",
        "expected_moe": False,
        "description": "Nemotron-Nano-9B-v2 (Hybrid Mamba2-Transformer, 9B params)",
        "note": "Source: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/discussions",
    },
    {
        "id": "gpt_oss_20b",
        "model_id": "openai/gpt-oss-20b",
        "checkpoint_model_name": "GptOssForCausalLM",
        "expected_moe": True,
        "expected_num_experts": 32,
        "description": "GPT-OSS-20B (OpenAI open source MoE model, 32 experts)",
    },
    {
        "id": "gptj_6b",
        "model_id": "EleutherAI/gpt-j-6b",
        "checkpoint_model_name": "GPTJForCausalLM",
        "expected_family": "gptj",
        "expected_layers": 28,
        "expected_hidden": 4096,
        "expected_moe": False,
        "description": "GPT-J-6B (EleutherAI open source GPT variant)",
    },
    {
        "id": "gpt_neox_20b",
        "model_id": "EleutherAI/gpt-neox-20b",
        "checkpoint_model_name": "GPTNeoXForCausalLM",
        "expected_family": "gpt_neox",
        "expected_layers": 44,
        "expected_hidden": 6144,
        "expected_moe": False,
        "description": "GPT-NeoX-20B (EleutherAI open source GPT variant)",
    },
]


@pytest.mark.parametrize("test_case", MODEL_TEST_CASES, ids=[tc["id"] for tc in MODEL_TEST_CASES])
def test_model_config_detection(test_case):
    """
    Test model configuration detection from HuggingFace models.

    Each test case validates:
    - Model can be loaded and config inferred
    - Basic fields are present and valid
    - Family, layers, hidden size match expectations (if specified)
    - MoE is correctly detected (or not)
    """
    model_id = test_case["model_id"]

    is_trusted = "NVIDIA-Nemotron" in model_id
    config = infer_model_cfg_from_hf(model_id, is_trusted=is_trusted)

    ckpt_model_name = test_case["checkpoint_model_name"]
    assert config.checkpoint_model_name == ckpt_model_name, f"Model name should be {ckpt_model_name}"
    assert config.num_layers > 0, "Should have positive number of layers"
    assert config.hidden_size > 0, "Should have positive hidden size"
    assert config.num_attention_heads > 0, "Should have positive number of attention heads"

    if "expected_family" in test_case:
        assert test_case["expected_family"] in config.family.lower(), (
            f"Family should contain {test_case['expected_family']}, got {config.family}"
        )

    if "expected_layers" in test_case:
        assert config.num_layers == test_case["expected_layers"], (
            f"Expected {test_case['expected_layers']} layers, got {config.num_layers}"
        )

    if "expected_hidden" in test_case:
        assert config.hidden_size == test_case["expected_hidden"], (
            f"Expected hidden size {test_case['expected_hidden']}, got {config.hidden_size}"
        )

    if test_case["expected_moe"]:
        assert config.moe_config is not None, f"Should detect MoE for {model_id}"
        if "expected_num_experts" in test_case:
            assert config.moe_config.num_experts == test_case["expected_num_experts"], (
                f"Expected {test_case['expected_num_experts']} experts, got {config.moe_config.num_experts}"
            )
    else:
        assert config.moe_config is None, f"Should not detect MoE for {model_id}"


PRECISION_TEST_CASES = [
    {
        "id": "gpt2_fp32_default",
        "model_id": "gpt2",
        "expected_precision": "float32",
        "description": "GPT-2 (no torch_dtype specified, defaults to float32)",
    },
    {
        "id": "llama3_8b_fp16",
        "model_id": "meta-llama/Llama-3.1-8B",
        "expected_precision": "bfloat16",
        "description": "Llama 3.1 8B (float16 in config)",
    },
    {
        "id": "mistral_7b_fp16",
        "model_id": "mistralai/Mistral-7B-v0.1",
        "expected_precision": "bfloat16",
        "description": "Mistral 7B (float16 in config)",
    },
    {
        "id": "phi2_fp16",
        "model_id": "microsoft/phi-2",
        "expected_precision": "float16",
        "description": "Phi-2 (float16)",
    },
]


def _precision_param(tc):
    if tc["model_id"] in GATED_MODEL_IDS:
        return pytest.param(tc, marks=REQUIRES_HF_TOKEN)
    return tc


PRECISION_TEST_CASES_PARAMS = [_precision_param(tc) for tc in PRECISION_TEST_CASES]


@pytest.mark.parametrize("test_case", PRECISION_TEST_CASES_PARAMS, ids=lambda tc: tc["id"])
def test_precision_detection(test_case):
    """Test that precision is correctly detected from HuggingFace configs."""
    model_id = test_case["model_id"]
    expected_precision = test_case["expected_precision"]
    description = test_case["description"]

    print(f"\n{'=' * 80}")
    print(f"Testing: {description}")
    print(f"Model: {model_id}")
    print(f"Expected Precision: {expected_precision}")
    print(f"{'=' * 80}")

    config = infer_model_cfg_from_hf(model_id)

    assert config.precision == expected_precision, (
        f"Precision mismatch for {model_id}: expected {expected_precision}, got {config.precision}"
    )

    print(f"✓ Precision correctly detected: {config.precision}")

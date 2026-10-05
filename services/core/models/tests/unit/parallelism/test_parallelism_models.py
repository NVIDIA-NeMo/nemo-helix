#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest test suite to validate Pydantic model serialization for parallelism types.

Hugging Face model-config detection tests live under tests/integration/parallelism/.
"""

import pytest

pytest.importorskip("torch", reason="torch required for parallelism API tests")

from nhx.core.models.parallelism.models import (
    ParallelizationConfig,
    ParallelizationRecommendation,
)
from nhx.core.models.schemas import (
    ModelSpec,
    MoEConfig,
)
from pydantic import ValidationError

# =================================================================================================
# PYDANTIC MODEL TESTS
# =================================================================================================


def test_pydantic_json_serialization():
    """Test that ParallelizationRecommendation can be serialized to JSON."""
    # Create a sample MoE model configuration
    moe_config = MoEConfig(num_experts=8, num_experts_per_tok=2, num_expert_layers=32)

    model_info = ModelSpec(
        checkpoint_model_name="mistralai/Mixtral-8x7B-v0.1",
        family="mixtral",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        ffn_hidden_size=14336,
        vocab_size=32000,
        tied_embeddings=False,
        gated_mlp=True,
        base_num_parameters=46702792704,  # ~47B
        precision="bfloat16",
        moe_config=moe_config,
    )

    configs = [
        ParallelizationConfig(
            tp=1,
            pp=2,
            dp=1,
            ep=4,
            microbatch_per_dp=1,
            per_rank_static_gb=71.2,
            est_act_gb_per_mb1=4.12,
            total_memory_per_rank_gb=75.3,
            score=790226355,
        ),
    ]

    recommendation = ParallelizationRecommendation(
        model_info=model_info,
        gpu_count=8,
        gpu_mem_gb=80.0,
        seq_len=8192,
        configs=configs,
        lora_enabled=False,
        lora_r=None,
    )

    # Test JSON serialization
    json_str = recommendation.model_dump_json(indent=2)
    assert isinstance(json_str, str)
    assert len(json_str) > 0


def test_pydantic_dict_serialization():
    """Test that ParallelizationRecommendation can be serialized to dict."""
    moe_config = MoEConfig(num_experts=8, num_experts_per_tok=2, num_expert_layers=32)
    model_info = ModelSpec(
        checkpoint_model_name="mistralai/Mixtral-8x7B-v0.1",
        family="mixtral",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        ffn_hidden_size=14336,
        vocab_size=32000,
        tied_embeddings=False,
        gated_mlp=True,
        base_num_parameters=46702792704,  # ~47B
        precision="bfloat16",
        moe_config=moe_config,
    )

    configs = [
        ParallelizationConfig(
            tp=1,
            pp=2,
            dp=1,
            ep=4,
            microbatch_per_dp=1,
            per_rank_static_gb=71.2,
            est_act_gb_per_mb1=4.12,
            total_memory_per_rank_gb=75.3,
            score=790226355,
        ),
    ]

    recommendation = ParallelizationRecommendation(
        model_info=model_info,
        gpu_count=8,
        gpu_mem_gb=80.0,
        seq_len=8192,
        configs=configs,
        lora_enabled=False,
        lora_r=None,
    )

    # Test dict serialization
    dict_repr = recommendation.model_dump()
    assert isinstance(dict_repr, dict)
    assert "model_info" in dict_repr
    assert "configs" in dict_repr
    assert len(dict_repr["configs"]) == 1


def test_pydantic_json_deserialization():
    """Test that ParallelizationRecommendation can be deserialized from JSON."""
    moe_config = MoEConfig(num_experts=8, num_experts_per_tok=2, num_expert_layers=32)
    model_info = ModelSpec(
        checkpoint_model_name="mistralai/Mixtral-8x7B-v0.1",
        family="mixtral",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        ffn_hidden_size=14336,
        vocab_size=32000,
        tied_embeddings=False,
        gated_mlp=True,
        base_num_parameters=46702792704,  # ~47B
        precision="bfloat16",
        moe_config=moe_config,
    )

    configs = [
        ParallelizationConfig(
            tp=1,
            pp=2,
            dp=1,
            ep=4,
            microbatch_per_dp=1,
            per_rank_static_gb=71.2,
            est_act_gb_per_mb1=4.12,
            total_memory_per_rank_gb=75.3,
            score=790226355,
        ),
    ]

    recommendation = ParallelizationRecommendation(
        model_info=model_info,
        gpu_count=8,
        gpu_mem_gb=80.0,
        seq_len=8192,
        configs=configs,
        lora_enabled=False,
        lora_r=None,
    )

    # Serialize and deserialize
    json_str = recommendation.model_dump_json(indent=2)
    reconstructed = ParallelizationRecommendation.model_validate_json(json_str)

    # Verify reconstruction
    assert isinstance(reconstructed, ParallelizationRecommendation)
    assert reconstructed.model_info.checkpoint_model_name == "mistralai/Mixtral-8x7B-v0.1"
    assert reconstructed.model_info.family == "mixtral"
    assert reconstructed.gpu_count == 8
    assert reconstructed.seq_len == 8192
    assert len(reconstructed.configs) == 1


def test_pydantic_moe_config_preservation():
    """Test that MoE configuration is preserved through serialization."""
    moe_config = MoEConfig(num_experts=8, num_experts_per_tok=2, num_expert_layers=32)
    model_info = ModelSpec(
        checkpoint_model_name="mistralai/Mixtral-8x7B-v0.1",
        family="mixtral",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        ffn_hidden_size=14336,
        vocab_size=32000,
        tied_embeddings=False,
        gated_mlp=True,
        base_num_parameters=46702792704,  # ~47B
        precision="bfloat16",
        moe_config=moe_config,
    )

    configs = [
        ParallelizationConfig(
            tp=1,
            pp=2,
            dp=1,
            ep=4,
            microbatch_per_dp=1,
            per_rank_static_gb=71.2,
            est_act_gb_per_mb1=4.12,
            total_memory_per_rank_gb=75.3,
            score=790226355,
        ),
    ]

    recommendation = ParallelizationRecommendation(
        model_info=model_info,
        gpu_count=8,
        gpu_mem_gb=80.0,
        seq_len=8192,
        configs=configs,
        lora_enabled=False,
        lora_r=None,
    )

    # Serialize and deserialize
    json_str = recommendation.model_dump_json()
    reconstructed = ParallelizationRecommendation.model_validate_json(json_str)

    # Verify MoE config
    assert reconstructed.model_info.moe_config is not None
    assert reconstructed.model_info.moe_config.num_experts == 8
    assert reconstructed.model_info.moe_config.num_experts_per_tok == 2


def test_pydantic_computed_properties():
    """Test that computed properties (total_gpus, global_batch_size) work correctly."""
    config = ParallelizationConfig(
        tp=2,
        pp=2,
        dp=4,
        cp=2,
        ep=1,
        microbatch_per_dp=3,
        per_rank_static_gb=71.2,
        est_act_gb_per_mb1=4.12,
        total_memory_per_rank_gb=75.3,
        score=790226355,
    )

    assert config.total_gpus == 32  # TP * PP * DP * CP * EP = 2 * 2 * 4 * 2 * 1
    assert config.global_batch_size == 12  # microbatch_per_dp * DP = 3 * 4


def test_pydantic_dense_model():
    """Test serialization of dense models (no MoE)."""
    dense_model = ModelSpec(
        checkpoint_model_name="gpt2",
        family="gpt2",
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        num_kv_heads=12,
        ffn_hidden_size=3072,
        vocab_size=50257,
        tied_embeddings=True,
        gated_mlp=False,
        base_num_parameters=123551232,  # ~124M
        precision="float32",
        moe_config=None,
    )

    assert dense_model.moe_config is None

    # Test round-trip
    dense_json = dense_model.model_dump_json()
    dense_reconstructed = ModelSpec.model_validate_json(dense_json)
    assert dense_reconstructed.moe_config is None


# =================================================================================================
# EDGE CASE TESTS
# =================================================================================================


def test_context_parallelism_field():
    """Test that CP field is included in ParallelizationConfig."""
    config = ParallelizationConfig(
        tp=1,
        pp=1,
        dp=8,
        cp=2,
        ep=1,
        microbatch_per_dp=1,
        per_rank_static_gb=10.0,
        est_act_gb_per_mb1=1.0,
        total_memory_per_rank_gb=11.0,
        score=100,
    )

    assert config.cp == 2
    assert config.total_gpus == 16  # TP * PP * DP * CP * EP = 1 * 1 * 8 * 2 * 1


def test_expert_parallelism_field():
    """Test that EP field is included in ParallelizationConfig."""
    config = ParallelizationConfig(
        tp=1,
        pp=1,
        dp=1,
        cp=1,
        ep=8,
        microbatch_per_dp=1,
        per_rank_static_gb=10.0,
        est_act_gb_per_mb1=1.0,
        total_memory_per_rank_gb=11.0,
        score=100,
    )

    assert config.ep == 8
    assert config.total_gpus == 8  # TP * PP * DP * CP * EP = 1 * 1 * 1 * 1 * 8


def test_multiple_parallelism_dimensions():
    """Test configuration with multiple parallelism dimensions active."""
    config = ParallelizationConfig(
        tp=2,
        pp=4,
        dp=8,
        cp=2,
        ep=4,
        microbatch_per_dp=2,
        per_rank_static_gb=20.0,
        est_act_gb_per_mb1=2.0,
        total_memory_per_rank_gb=24.0,
        score=1000,
    )

    # Verify all dimensions
    assert config.tp == 2
    assert config.pp == 4
    assert config.dp == 8
    assert config.cp == 2
    assert config.ep == 4

    # Verify computed properties
    assert config.total_gpus == 512  # 2 * 4 * 8 * 2 * 4
    assert config.global_batch_size == 16  # 2 * 8


def test_precision_in_model_config_serialization():
    """Test that precision field is properly serialized in ModelConfig."""
    config = ModelSpec(
        checkpoint_model_name="test-model",
        family="test",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=32,
        ffn_hidden_size=11008,
        vocab_size=32000,
        tied_embeddings=False,
        gated_mlp=True,
        base_num_parameters=7_000_000_000,
        precision="bfloat16",
    )

    # Test serialization
    config_dict = config.model_dump()
    assert "precision" in config_dict
    assert config_dict["precision"] == "bfloat16"

    # Test JSON serialization
    config_json = config.model_dump_json()
    assert "precision" in config_json
    assert "bfloat16" in config_json


def test_precision_required():
    """Test that precision field is required and must be provided."""
    # Test that precision is required
    with pytest.raises(ValidationError):
        ModelSpec(
            checkpoint_model_name="test-model",
            family="test",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
            ffn_hidden_size=11008,
            vocab_size=32000,
            tied_embeddings=False,
            gated_mlp=True,
            base_num_parameters=7_000_000_000,
            # precision not provided - should fail
        )

    # Test that precision can be provided
    config = ModelSpec(
        checkpoint_model_name="test-model",
        family="test",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=32,
        ffn_hidden_size=11008,
        vocab_size=32000,
        tied_embeddings=False,
        gated_mlp=True,
        base_num_parameters=7_000_000_000,
        precision="float32",
    )

    assert config.precision == "float32"


# =================================================================================================
# MODEL SPEC CACHE TESTS
# =================================================================================================

_MINIMAL_SPEC_KWARGS = dict(
    checkpoint_model_name="test-model",
    family="test",
    num_layers=12,
    hidden_size=768,
    num_attention_heads=12,
    num_kv_heads=12,
    ffn_hidden_size=3072,
    vocab_size=32000,
    tied_embeddings=False,
    gated_mlp=False,
    base_num_parameters=125_000_000,
    precision="float16",
)


@pytest.fixture(autouse=False)
def _clear_model_spec_cache():
    """Reset the module-level cache before and after each cache test."""
    from nhx.core.models.parallelism.api import _model_spec_cache

    _model_spec_cache.clear()
    yield
    _model_spec_cache.clear()


class TestModelSpecCache:
    """Tests for _model_spec_cache eviction and ordering."""

    def test_cache_respects_max_size(self, _clear_model_spec_cache):
        from nhx.core.models.parallelism.api import (
            _MODEL_SPEC_CACHE_MAX,
            _cache_put,
            _model_spec_cache,
        )

        for i in range(_MODEL_SPEC_CACHE_MAX + 10):
            spec = ModelSpec(**{**_MINIMAL_SPEC_KWARGS, "checkpoint_model_name": f"model-{i}"})
            _cache_put((f"model-{i}", False), spec)

        assert len(_model_spec_cache) == _MODEL_SPEC_CACHE_MAX

    def test_cache_evicts_oldest_first(self, _clear_model_spec_cache):
        from nhx.core.models.parallelism.api import (
            _MODEL_SPEC_CACHE_MAX,
            _cache_put,
            _model_spec_cache,
        )

        for i in range(_MODEL_SPEC_CACHE_MAX + 5):
            spec = ModelSpec(**{**_MINIMAL_SPEC_KWARGS, "checkpoint_model_name": f"model-{i}"})
            _cache_put((f"model-{i}", False), spec)

        # The first 5 entries (model-0 through model-4) should have been evicted.
        for i in range(5):
            assert (f"model-{i}", False) not in _model_spec_cache

        # The remaining entries (model-5 onward) should still be present.
        for i in range(5, _MODEL_SPEC_CACHE_MAX + 5):
            assert (f"model-{i}", False) in _model_spec_cache

    def test_cache_put_overwrites_existing_key(self, _clear_model_spec_cache):
        from nhx.core.models.parallelism.api import _cache_put, _model_spec_cache

        spec_v1 = ModelSpec(**{**_MINIMAL_SPEC_KWARGS, "num_layers": 12})
        spec_v2 = ModelSpec(**{**_MINIMAL_SPEC_KWARGS, "num_layers": 24})

        _cache_put(("same-model", False), spec_v1)
        _cache_put(("same-model", False), spec_v2)

        assert len(_model_spec_cache) == 1
        assert _model_spec_cache[("same-model", False)].num_layers == 24

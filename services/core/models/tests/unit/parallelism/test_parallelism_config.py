# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for parallelism configuration system.

Tests cover:
- Default configuration values
- Partial configuration overrides
- Configuration merging behavior (Pydantic model instantiation)
"""

from nhx.core.models.parallelism.config import (
    BalanceConfig,
    ContextParallelismConfig,
    DataParallelismConfig,
    ExpertParallelismConfig,
    MemoryConfig,
    ModelSizeThresholds,
    ParallelismConfig,
    PipelineParallelismConfig,
    TensorParallelismConfig,
    get_config,
)


def test_default_configuration():
    """Test that default configuration has expected values."""
    config = ParallelismConfig()

    # Model size thresholds
    assert config.model_size_thresholds.very_large == 300.0
    assert config.model_size_thresholds.large == 100.0
    assert config.model_size_thresholds.medium == 50.0
    assert config.model_size_thresholds.small_tp == 70.0
    assert config.model_size_thresholds.small_moe == 40.0

    # Memory config
    assert config.memory.pressure_threshold == 0.60
    assert config.memory.pressure_moderate == 0.50
    assert config.memory.pressure_low == 0.45
    assert config.memory.base_penalty == 1e9
    assert config.memory.scale_divisor == 0.1

    # TP config
    assert config.tensor_parallelism.base_cost_very_large_model == 50.0
    assert config.tensor_parallelism.base_cost_standard_model == 100.0
    assert config.tensor_parallelism.excessive_very_large == 8
    assert config.tensor_parallelism.excessive_standard == 4

    # CP config
    assert config.context_parallelism.seq_threshold_enable == 8192
    assert config.context_parallelism.seq_threshold_cp4 == 16384
    assert config.context_parallelism.seq_threshold_cp8 == 32768
    assert config.context_parallelism.seq_threshold_cp16 == 131072


def test_partial_override_memory_only():
    """Test that partial override only changes specified values."""
    config = ParallelismConfig(
        memory=MemoryConfig(
            pressure_threshold=0.55,
            base_penalty=2e9,
        )
    )

    # Overridden values
    assert config.memory.pressure_threshold == 0.55
    assert config.memory.base_penalty == 2e9

    # Non-overridden values should remain default
    assert config.memory.pressure_moderate == 0.50
    assert config.memory.pressure_low == 0.45
    assert config.memory.scale_divisor == 0.1

    # Other sections should be default
    assert config.model_size_thresholds.very_large == 300.0
    assert config.tensor_parallelism.base_cost_very_large_model == 50.0


def test_partial_override_multiple_sections():
    """Test overriding multiple sections with partial values."""
    config = ParallelismConfig(
        memory=MemoryConfig(pressure_threshold=0.50),
        tensor_parallelism=TensorParallelismConfig(excessive_very_large=16),
        context_parallelism=ContextParallelismConfig(
            seq_threshold_enable=4096,
            seq_threshold_cp16=65536,
        ),
    )

    # Memory overrides
    assert config.memory.pressure_threshold == 0.50
    assert config.memory.base_penalty == 1e9  # default

    # TP overrides
    assert config.tensor_parallelism.excessive_very_large == 16
    assert config.tensor_parallelism.excessive_standard == 4  # default

    # CP overrides
    assert config.context_parallelism.seq_threshold_enable == 4096
    assert config.context_parallelism.seq_threshold_cp16 == 65536
    assert config.context_parallelism.seq_threshold_cp4 == 16384  # default
    assert config.context_parallelism.seq_threshold_cp8 == 32768  # default


def test_complete_override():
    """Test overriding all values in a section."""
    config = ParallelismConfig(
        model_size_thresholds=ModelSizeThresholds(
            very_large=500.0,
            large=200.0,
            medium=100.0,
            small_tp=50.0,
            small_moe=30.0,
        )
    )

    assert config.model_size_thresholds.very_large == 500.0
    assert config.model_size_thresholds.large == 200.0
    assert config.model_size_thresholds.medium == 100.0
    assert config.model_size_thresholds.small_tp == 50.0
    assert config.model_size_thresholds.small_moe == 30.0


def test_empty_config():
    """Test that empty config uses all defaults."""
    config = ParallelismConfig()

    # Should be identical to default config
    default = ParallelismConfig()

    assert config.memory.pressure_threshold == default.memory.pressure_threshold
    assert config.tensor_parallelism.base_cost_very_large_model == default.tensor_parallelism.base_cost_very_large_model
    assert config.context_parallelism.seq_threshold_enable == default.context_parallelism.seq_threshold_enable


def test_nested_partial_override():
    """Test that nested structures merge correctly."""
    config = ParallelismConfig(
        data_parallelism=DataParallelismConfig(
            bonus_very_strong=-1e8,
            total_parallelism_very_high=128,
        )
    )

    # Overridden values
    assert config.data_parallelism.bonus_very_strong == -1e8
    assert config.data_parallelism.total_parallelism_very_high == 128

    # Default values
    assert config.data_parallelism.bonus_strong == -3e7
    assert config.data_parallelism.bonus_moderate == -1.5e7
    assert config.data_parallelism.total_parallelism_high == 32


def test_type_conversion():
    """Test that Pydantic correctly coerces types when building from dict."""
    config = ParallelismConfig.model_validate(
        {
            "memory": {
                "pressure_threshold": 0.55,
                "base_penalty": 2000000000,
            },
            "tensor_parallelism": {
                "excessive_very_large": 16,
            },
        }
    )

    assert isinstance(config.memory.pressure_threshold, float)
    assert config.memory.pressure_threshold == 0.55

    assert isinstance(config.memory.base_penalty, float)
    assert config.memory.base_penalty == 2e9

    assert isinstance(config.tensor_parallelism.excessive_very_large, int)
    assert config.tensor_parallelism.excessive_very_large == 16


def test_expert_parallelism_config():
    """Test expert parallelism configuration overrides."""
    config = ParallelismConfig(
        expert_parallelism=ExpertParallelismConfig(
            bonus_perfect=-6e8,
            experts_per_gpu_very_efficient=4,
        )
    )

    # Overridden
    assert config.expert_parallelism.bonus_perfect == -6e8
    assert config.expert_parallelism.experts_per_gpu_very_efficient == 4

    # Default
    assert config.expert_parallelism.bonus_very_efficient == -4e8
    assert config.expert_parallelism.experts_per_gpu_good == 32


def test_balance_config():
    """Test balance configuration overrides."""
    config = ParallelismConfig(
        gpus_per_node_default=4,
        balance=BalanceConfig(
            ratio_perfect=1.0,
            tp_squared_multiplier=1e7,
        ),
    )

    # Overridden
    assert config.balance.ratio_perfect == 1.0
    assert config.gpus_per_node_default == 4
    assert config.balance.tp_squared_multiplier == 1e7

    # Default
    assert config.balance.ratio_good == 2.0
    assert config.balance.pp_significant_threshold == 4


def test_pipeline_parallelism_config():
    """Test pipeline parallelism configuration overrides."""
    config = ParallelismConfig(
        pipeline_parallelism=PipelineParallelismConfig(
            cost_moe=1e6,
            cost_small_with_tp=2e8,
        )
    )

    # Overridden
    assert config.pipeline_parallelism.cost_moe == 1e6
    assert config.pipeline_parallelism.cost_small_with_tp == 2e8

    # Default
    assert config.pipeline_parallelism.cost_very_large_model == 5e6
    assert config.pipeline_parallelism.cost_large_model == 1e7


def test_to_dict():
    """Test converting config to dictionary."""
    config = ParallelismConfig()
    config_dict = config.model_dump()

    assert "model_size_thresholds" in config_dict
    assert "memory" in config_dict
    assert "tensor_parallelism" in config_dict
    assert "data_parallelism" in config_dict
    assert "pipeline_parallelism" in config_dict
    assert "context_parallelism" in config_dict
    assert "expert_parallelism" in config_dict
    assert "balance" in config_dict

    # Check nested structure
    assert config_dict["memory"]["pressure_threshold"] == 0.60
    assert config_dict["context_parallelism"]["seq_threshold_enable"] == 8192


def test_conservative_memory_scenario():
    """Test a realistic scenario: more conservative memory settings."""
    config = ParallelismConfig(
        memory=MemoryConfig(
            pressure_threshold=0.50,
            base_penalty=1.5e9,
        ),
        expert_parallelism=ExpertParallelismConfig(bonus_perfect=-6e8),
    )

    assert config.memory.pressure_threshold == 0.50
    assert config.memory.base_penalty == 1.5e9
    assert config.expert_parallelism.bonus_perfect == -6e8

    # Verify other settings remain default
    assert config.memory.pressure_moderate == 0.50  # Default
    assert config.tensor_parallelism.base_cost_very_large_model == 50.0


def test_custom_cp_thresholds_scenario():
    """Test a realistic scenario: custom CP thresholds for different workload."""
    config = ParallelismConfig(
        context_parallelism=ContextParallelismConfig(
            seq_threshold_enable=4096,
            seq_threshold_cp4=8192,
            seq_threshold_cp8=16384,
            seq_threshold_cp16=65536,
        )
    )

    assert config.context_parallelism.seq_threshold_enable == 4096
    assert config.context_parallelism.seq_threshold_cp4 == 8192
    assert config.context_parallelism.seq_threshold_cp8 == 16384
    assert config.context_parallelism.seq_threshold_cp16 == 65536

    # Other CP settings should be default
    assert config.context_parallelism.optimal_value == 2
    assert config.context_parallelism.max_value == 8


def test_get_config_singleton():
    """Test that get_config() returns a singleton that loads from cfg()."""
    # Note: This test assumes cfg() is available in the test environment
    # In a real deployment, cfg() would have the parallelism section
    config1 = get_config()
    config2 = get_config()

    # Should return the same instance
    assert config1 is config2

    # Should have default values if cfg().parallelism doesn't exist
    assert config1.memory.pressure_threshold == 0.60


def test_all_sections_present():
    """Test that all configuration sections are present in default config."""
    config = ParallelismConfig()

    assert isinstance(config.model_size_thresholds, ModelSizeThresholds)
    assert isinstance(config.memory, MemoryConfig)
    assert isinstance(config.tensor_parallelism, TensorParallelismConfig)
    assert isinstance(config.data_parallelism, DataParallelismConfig)
    assert isinstance(config.pipeline_parallelism, PipelineParallelismConfig)
    assert isinstance(config.context_parallelism, ContextParallelismConfig)
    assert isinstance(config.expert_parallelism, ExpertParallelismConfig)
    assert isinstance(config.balance, BalanceConfig)


def test_model_validate_partial_dict():
    """Test that building from a partial dict uses defaults for missing keys."""
    config = ParallelismConfig.model_validate(
        {
            "memory": {
                "pressure_threshold": 0.55,
            },
        }
    )

    assert config.memory.pressure_threshold == 0.55
    assert config.memory.base_penalty == 1e9
    assert config.tensor_parallelism.base_cost_very_large_model == 50.0


def test_4gpu_topology_scenario():
    """Test a realistic scenario: 4-GPU topology instead of 8-GPU."""
    config = ParallelismConfig(gpus_per_node_default=4)

    assert config.gpus_per_node_default == 4

    # Other balance settings should be default
    assert config.balance.ratio_perfect == 1.0
    assert config.balance.ratio_good == 2.0

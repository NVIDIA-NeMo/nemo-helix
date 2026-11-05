# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from nhx.core.models.parallelism.checkpoint import inspect_checkpoint


def test_vision_tower_is_seen_on_the_root_and_on_a_nested_config() -> None:
    assert inspect_checkpoint({"vision_config": {"hidden_size": 4}}).has_vision_tower
    assert inspect_checkpoint({"text_config": {"vision_config": {"hidden_size": 4}}}).has_vision_tower
    assert not inspect_checkpoint({"hidden_size": 8}).has_vision_tower
    assert not inspect_checkpoint({"vision_config": {}}).has_vision_tower


def test_audio_tower_follows_the_same_nesting() -> None:
    traits = inspect_checkpoint({"thinker_config": {"audio_config": {"d_model": 2}}})

    assert traits.has_audio_tower
    assert not traits.has_vision_tower


def test_omni_experts_and_mtp_live_on_the_language_config() -> None:
    """A top-level read of ``n_routed_experts`` misses Nemotron Omni."""
    traits = inspect_checkpoint(
        {
            "model_type": "nemotron_h_omni",
            "vision_config": {"hidden_size": 8},
            "text_config": {"n_routed_experts": 128, "num_nextn_predict_layers": 1},
        }
    )

    assert traits.has_vision_tower
    assert traits.expert_count == 128
    assert traits.num_nextn_predict_layers == 1


def test_a_single_expert_is_not_mixture_of_experts() -> None:
    assert inspect_checkpoint({"num_local_experts": 1}).expert_count is None


def test_top_level_expert_count_still_wins_when_it_is_set() -> None:
    traits = inspect_checkpoint({"num_experts": 8, "text_config": {"n_routed_experts": 32}})

    assert traits.expert_count == 8


def test_moe_detection_reads_a_nested_language_config() -> None:
    pytest.importorskip("torch")
    from nhx.core.models.parallelism.models import detect_moe_config_from_cfg

    language = SimpleNamespace(
        num_local_experts=None,
        num_experts=None,
        n_routed_experts=16,
        num_experts_per_tok=2,
        num_hidden_layers=8,
        moe_layer_freq=None,
        num_expert_layers=None,
        moe_intermediate_size=None,
        n_shared_experts=0,
    )
    root = SimpleNamespace(
        num_local_experts=None,
        num_experts=None,
        n_routed_experts=None,
        text_config=language,
        thinker_config=None,
        language_config=None,
        llm_config=None,
    )

    moe = detect_moe_config_from_cfg(root)

    assert moe is not None
    assert moe.num_experts == 16

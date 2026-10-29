# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from nhx.customization_common.schemas.checkpoint import (
    CheckpointTraits,
    lora_uses_default_targets,
    reject_incompatible_checkpoint,
    traits_from_model_spec,
    traits_from_step_model,
)


def test_the_api_proj_backfill_is_not_a_caller_filter() -> None:
    assert lora_uses_default_targets(None, None)
    assert lora_uses_default_targets(["*proj"], None)
    assert not lora_uses_default_targets(["*.q_proj"], None)
    assert not lora_uses_default_targets(None, ["*lm_head*"])


def test_a_spec_without_the_vision_field_stays_text_and_keeps_moe() -> None:
    spec = SimpleNamespace(
        model_fields_set={"moe_config", "checkpoint_model_name"},
        moe_config=SimpleNamespace(num_experts=8),
        checkpoint_model_name="NemotronHForCausalLM",
        has_vision_tower=False,
    )

    traits = traits_from_model_spec(spec)

    assert not traits.has_vision_tower
    assert traits.expert_count == 8
    assert traits.checkpoint_model_name == "NemotronHForCausalLM"


def test_a_recorded_spec_is_trusted() -> None:
    spec = SimpleNamespace(
        model_fields_set={"has_vision_tower", "has_audio_tower", "num_experts", "num_nextn_predict_layers"},
        has_vision_tower=True,
        has_audio_tower=True,
        num_experts=16,
        num_nextn_predict_layers=1,
        checkpoint_model_name="NemotronH_Omni",
        moe_config=None,
    )

    traits = traits_from_model_spec(spec)

    assert traits.has_vision_tower
    assert traits.has_audio_tower
    assert traits.expert_count == 16
    assert traits.declares_nextn_predict_layers
    assert traits.num_nextn_predict_layers == 1


def test_step_model_fields_round_trip() -> None:
    traits = traits_from_step_model(
        SimpleNamespace(
            has_vision_tower=True,
            has_audio_tower=False,
            num_experts=4,
            num_nextn_predict_layers=None,
            checkpoint_model_name="Qwen2_5_VLForConditionalGeneration",
        )
    )

    assert traits == CheckpointTraits(
        has_vision_tower=True,
        expert_count=4,
        checkpoint_model_name="Qwen2_5_VLForConditionalGeneration",
    )


def test_distillation_and_packing_are_rejected_for_a_vision_checkpoint() -> None:
    traits = CheckpointTraits(has_vision_tower=True)

    with pytest.raises(ValueError, match="distillation"):
        reject_incompatible_checkpoint(
            traits,
            training_type="distillation",
            recipe="sft",
            sequence_packing=False,
            expert_parallel_size=1,
            num_gpus_per_node=1,
            num_nodes=1,
        )
    with pytest.raises(ValueError, match="Sequence packing"):
        reject_incompatible_checkpoint(
            traits,
            training_type="sft",
            recipe="sft",
            sequence_packing=True,
            expert_parallel_size=1,
            num_gpus_per_node=1,
            num_nodes=1,
        )


def test_text_moe_without_expert_parallel_is_rejected_on_multiple_gpus() -> None:
    traits = CheckpointTraits(expert_count=8)

    with pytest.raises(ValueError, match="expert_parallel_size"):
        reject_incompatible_checkpoint(
            traits,
            training_type="sft",
            recipe="sft",
            sequence_packing=False,
            expert_parallel_size=None,
            num_gpus_per_node=4,
            num_nodes=1,
        )


def test_a_text_spec_does_not_reject_sequence_packing() -> None:
    reject_incompatible_checkpoint(
        CheckpointTraits(),
        training_type="sft",
        recipe="sft",
        sequence_packing=True,
        expert_parallel_size=None,
        num_gpus_per_node=4,
        num_nodes=1,
    )

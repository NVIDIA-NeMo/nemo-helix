# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-configuration training memory estimate (no torch required)."""

import pytest
from nhx.core.models.parallelism.memory import calculate_memory_per_rank, calculate_param_counts
from nhx.core.models.parallelism.schemas import EstimationParams
from nhx.core.models.parallelism.training_memory import (
    ACT_CKPT_RATIO_DISABLED,
    InvalidTrainingLayoutError,
    estimate_training_memory,
)
from nhx.core.models.schemas import (
    LinearLayerSpec,
    ModelSpec,
    TrainingMemoryComponent,
    TrainingMemoryEstimateRequest,
    TrainingMemoryStatus,
)


@pytest.fixture
def llama_8b_spec() -> ModelSpec:
    """Llama-3.1-8B architecture, with one layer's projections for LoRA counting."""
    return ModelSpec(
        checkpoint_model_name="LlamaForCausalLM",
        family="llama",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        ffn_hidden_size=14336,
        vocab_size=128256,
        tied_embeddings=False,
        gated_mlp=True,
        base_num_parameters=8_030_261_248,
        precision="bfloat16",
        linear_layers=[
            LinearLayerSpec(name="model.layers.0.self_attn.q_proj", in_features=4096, out_features=4096),
            LinearLayerSpec(name="model.layers.0.mlp.up_proj", in_features=4096, out_features=14336),
            LinearLayerSpec(name="lm_head", in_features=4096, out_features=128256),
        ],
    )


def _estimate(spec: ModelSpec, gpu_memory_gb: float = 80, **kwargs):
    request = TrainingMemoryEstimateRequest(**{"max_seq_length": 4096, "micro_batch_size": 1, **kwargs})
    return estimate_training_memory(
        spec, request, gpu_memory_gb=gpu_memory_gb, gpu_memory_assumed=True, warning_threshold=0.85
    )


def test_full_finetune_on_one_gpu_exceeds_capacity(llama_8b_spec):
    estimate = _estimate(llama_8b_spec)

    assert estimate.gpu_count == 1
    assert estimate.data_parallel_size == 1
    assert estimate.status == TrainingMemoryStatus.EXCEEDS_CAPACITY
    assert estimate.limiting_factor == TrainingMemoryComponent.MODEL_STATES
    assert estimate.utilization > 1.0


def test_lora_fits_on_one_gpu(llama_8b_spec):
    estimate = _estimate(llama_8b_spec, lora_rank=16)

    assert estimate.status == TrainingMemoryStatus.OK
    assert estimate.total_gb < _estimate(llama_8b_spec).total_gb


def test_components_sum_to_total(llama_8b_spec):
    estimate = _estimate(llama_8b_spec, num_gpus_per_node=8, micro_batch_size=2)

    assert estimate.total_gb == pytest.approx(
        estimate.model_states_gb + estimate.activations_gb + estimate.overhead_gb, abs=0.02
    )
    assert estimate.utilization == pytest.approx(estimate.total_gb / 80, abs=1e-4)


def test_near_capacity_between_threshold_and_full(llama_8b_spec):
    # 8 GPUs at mb=1 lands at ~83% of 80 GB; shrinking the card to 75 GB pushes it past 85%.
    assert _estimate(llama_8b_spec, num_gpus_per_node=8).status == TrainingMemoryStatus.OK
    assert _estimate(llama_8b_spec, gpu_memory_gb=75, num_gpus_per_node=8).status == (
        TrainingMemoryStatus.NEAR_CAPACITY
    )


def test_activations_scale_with_micro_batch_size(llama_8b_spec):
    one = _estimate(llama_8b_spec, num_gpus_per_node=8, micro_batch_size=1)
    eight = _estimate(llama_8b_spec, num_gpus_per_node=8, micro_batch_size=8)

    assert eight.activations_gb == pytest.approx(8 * one.activations_gb, abs=0.01)
    assert eight.model_states_gb == one.model_states_gb


def test_activation_checkpointing_lowers_activations(llama_8b_spec):
    off = _estimate(llama_8b_spec, num_gpus_per_node=8, micro_batch_size=8)
    on = _estimate(llama_8b_spec, num_gpus_per_node=8, micro_batch_size=8, activation_checkpointing=True)

    assert on.activations_gb < off.activations_gb
    assert on.model_states_gb == off.model_states_gb


def test_tensor_parallel_splits_model_states(llama_8b_spec):
    single = _estimate(llama_8b_spec)
    tp2 = _estimate(llama_8b_spec, num_gpus_per_node=2, tensor_parallel_size=2)

    assert tp2.data_parallel_size == 1
    assert tp2.model_states_gb == pytest.approx(single.model_states_gb / 2, abs=0.01)


def test_packed_sequence_size_replaces_max_seq_length(llama_8b_spec):
    unpacked = _estimate(llama_8b_spec, max_seq_length=2048)
    packed = _estimate(llama_8b_spec, max_seq_length=2048, sequence_packing=True, packed_sequence_size=8192)
    ignored = _estimate(llama_8b_spec, max_seq_length=2048, packed_sequence_size=8192)

    assert packed.activations_gb > unpacked.activations_gb
    assert ignored.activations_gb == unpacked.activations_gb


def test_gpu_count_spans_nodes(llama_8b_spec):
    estimate = _estimate(llama_8b_spec, num_nodes=2, num_gpus_per_node=8, tensor_parallel_size=4)

    assert estimate.gpu_count == 16
    assert estimate.data_parallel_size == 4


def test_parallel_sizes_must_divide_gpu_count(llama_8b_spec):
    with pytest.raises(InvalidTrainingLayoutError, match=r"\(3\) must divide the GPU count \(8\)"):
        _estimate(llama_8b_spec, num_gpus_per_node=8, tensor_parallel_size=3)


def test_matches_shared_rank_memory_calculation(llama_8b_spec):
    """The endpoint and the parallelism search must report the same number for the same layout."""
    args = EstimationParams(
        gpus=8, gpu_mem_gb=80, seq_len=4096, microbatch_size=2, act_ckpt_ratio=ACT_CKPT_RATIO_DISABLED
    )
    expected = calculate_memory_per_rank(
        args, llama_8b_spec, calculate_param_counts(args, llama_8b_spec), tp=2, pp=1, dp=4, cp=1, ep=1, microbatch=2
    )

    estimate = _estimate(llama_8b_spec, num_gpus_per_node=8, tensor_parallel_size=2, micro_batch_size=2)

    assert estimate.total_gb == expected.total_gb

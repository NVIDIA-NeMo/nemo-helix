# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-GPU memory estimate for one user-chosen training configuration.

Unlike ``api.estimate``, this does not search the parallelism space: the caller
has already picked TP/PP/CP/EP and the microbatch size. It reuses the same
arithmetic, so its numbers agree with the stored ``minimum_gpus_*`` values.
Imports no torch, so it can run inside the models service.
"""

from nhx.core.models.parallelism.memory import calculate_memory_per_rank, calculate_param_counts
from nhx.core.models.parallelism.schemas import EstimationParams
from nhx.core.models.schemas import (
    ModelSpec,
    TrainingMemoryComponent,
    TrainingMemoryEstimate,
    TrainingMemoryEstimateRequest,
    TrainingMemoryStatus,
)

# Fraction of activations kept in memory. The checkpointed value matches the
# ratio used when computing minimum GPU counts for model entities.
ACT_CKPT_RATIO_ENABLED = 0.05
ACT_CKPT_RATIO_DISABLED = 1.0


class InvalidTrainingLayoutError(ValueError):
    """The requested parallel sizes do not fit the requested GPU count."""


def estimate_training_memory(
    spec: ModelSpec,
    request: TrainingMemoryEstimateRequest,
    gpu_memory_gb: float,
    gpu_memory_assumed: bool,
    warning_threshold: float,
) -> TrainingMemoryEstimate:
    """Estimate per-GPU memory for the requested configuration and compare it with GPU memory."""
    gpus = request.num_nodes * request.num_gpus_per_node
    tp = request.tensor_parallel_size
    pp = request.pipeline_parallel_size
    cp = request.context_parallel_size
    ep = request.expert_parallel_size or 1

    model_parallel = tp * pp * cp * ep
    if gpus % model_parallel != 0:
        raise InvalidTrainingLayoutError(
            f"tensor x pipeline x context x expert parallel size ({model_parallel}) must divide the GPU count ({gpus})"
        )
    dp = gpus // model_parallel

    seq_len = request.max_seq_length
    if request.sequence_packing and request.packed_sequence_size is not None:
        seq_len = request.packed_sequence_size

    args = EstimationParams(
        gpus=gpus,
        gpu_mem_gb=gpu_memory_gb,
        seq_len=seq_len,
        microbatch_size=request.micro_batch_size,
        act_ckpt_ratio=ACT_CKPT_RATIO_ENABLED if request.activation_checkpointing else ACT_CKPT_RATIO_DISABLED,
    )
    if request.lora_rank is not None:
        args = args.model_copy(update={"lora": True, "lora_r": request.lora_rank})
    param_counts = calculate_param_counts(args, spec)
    memory = calculate_memory_per_rank(args, spec, param_counts, tp, pp, dp, cp, ep, request.micro_batch_size)

    utilization = memory.total_gb / gpu_memory_gb
    if utilization > 1.0:
        status = TrainingMemoryStatus.EXCEEDS_CAPACITY
    elif utilization > warning_threshold:
        status = TrainingMemoryStatus.NEAR_CAPACITY
    else:
        status = TrainingMemoryStatus.OK

    components = {
        TrainingMemoryComponent.MODEL_STATES: memory.static_gb,
        TrainingMemoryComponent.ACTIVATIONS: memory.activations_gb,
        TrainingMemoryComponent.OVERHEAD: memory.overhead_gb,
    }

    return TrainingMemoryEstimate(
        gpu_count=gpus,
        gpu_memory_gb=gpu_memory_gb,
        gpu_memory_assumed=gpu_memory_assumed,
        data_parallel_size=dp,
        model_states_gb=round(memory.static_gb, 2),
        activations_gb=round(memory.activations_gb, 2),
        overhead_gb=round(memory.overhead_gb, 2),
        total_gb=memory.total_gb,
        utilization=round(utilization, 4),
        warning_threshold=warning_threshold,
        status=status,
        limiting_factor=max(components, key=lambda c: components[c]),
    )

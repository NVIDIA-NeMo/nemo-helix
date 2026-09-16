# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nhx-automodel training error handling.

Maps Automodel runtime exceptions to user-facing error types via error_rules.yaml.
See services/automodel/docs/automodel_errors.md for the full error catalog.
"""

import subprocess

from nhx.automodel.tasks.training.errors.converter import create_error_details, get_error_converter


class TestGetErrorConverter:
    """Tests for error converter initialization."""

    def test_converter_loads_rules(self):
        converter = get_error_converter()
        assert converter.rule_count > 0


class TestAutomodelDatasetErrors:
    """Tests for Automodel dataset error conversion."""

    def test_unsupported_role_error(self):
        original = ValueError("Unsupported role in messages: invalid_role")
        details = create_error_details(original)

        assert details["type"] == "DatasetFormatError"
        assert "invalid role" in details["message"].lower()

    def test_unrelated_value_error_uses_fallback(self):
        original = ValueError("Something completely different")
        details = create_error_details(original)

        assert details["type"] == "InternalError"


class TestAutomodelModelLoadErrors:
    """Tests for Automodel model load error conversion."""

    def test_weight_swap_failure(self):
        original = RuntimeError("_apply(): Couldn't swap Linear.weight")
        details = create_error_details(original)

        assert details["type"] == "ModelLoadError"
        assert "weights could not be applied" in details["message"].lower()

    def test_patch_failure(self):
        original = RuntimeError("Failed to patch model")
        details = create_error_details(original)

        assert details["type"] == "ModelLoadError"
        assert "optimizations" in details["message"].lower()

    def test_signature_mismatch(self):
        original = AssertionError("Signature mismatch:\n  original: foo\n  patched : bar")
        details = create_error_details(original)

        assert details["type"] == "ModelLoadError"
        assert "signature" in details["message"].lower()

    def test_missing_lm_head(self):
        original = ValueError("lm_head.weight not found in model")
        details = create_error_details(original)

        assert details["type"] == "ModelLoadError"
        assert "language model head" in details["message"].lower()


class TestAutomodelTrainingConfigErrors:
    """Tests for Automodel training config error conversion."""

    def test_tied_embeddings_error(self):
        original = ValueError(
            "Model 'test-model' is not compatible with pipeline parallelism:\n\n"
            "1. tie_word_embeddings=True is not supported for pipelining."
        )
        details = create_error_details(original)

        assert details["type"] == "TrainingConfigError"
        assert "tied embeddings" in details["message"].lower()

    def test_encoder_decoder_error(self):
        original = ValueError(
            "Model 'test-model' is not compatible with pipeline parallelism:\n\n"
            "1. Encoder-Decoder models with cross-attention are not supported yet."
        )
        details = create_error_details(original)

        assert details["type"] == "TrainingConfigError"
        assert "encoder-decoder" in details["message"].lower()

    def test_pp_batch_size_error(self):
        original = AssertionError("pp_batch_size // pp_microbatch_size must be >= pp_size")
        details = create_error_details(original)

        assert details["type"] == "TrainingConfigError"
        assert "pipeline parallelism" in details["message"].lower()

    def test_sdpa_error(self):
        original = ValueError("Model does not support SDPA required for context parallelism")
        details = create_error_details(original)

        assert details["type"] == "TrainingConfigError"
        assert "SDPA" in details["message"] or "context parallelism" in details["message"].lower()

    def test_triton_not_installed(self):
        original = ImportError("triton is not installed. Please install it.")
        details = create_error_details(original)

        assert details["type"] == "TrainingConfigError"
        assert "triton" in details["message"].lower()

    def test_lora_dimensions_mismatch(self):
        original = AssertionError("Incompatible X and LoRA A dimensions")
        details = create_error_details(original)

        assert details["type"] == "TrainingConfigError"
        assert "LoRA" in details["message"]


class TestAutomodelCheckpointErrors:
    """Tests for Automodel checkpoint error conversion."""

    def test_checkpoint_directory_exists(self):
        original = AssertionError("Checkpoint directory /path/to/ckpt already exists")
        details = create_error_details(original)

        assert details["type"] == "CheckpointError"
        assert "already exists" in details["message"].lower()

    def test_global_plan_validation(self):
        original = ValueError("Failed to validate global plan")
        details = create_error_details(original)

        assert details["type"] == "CheckpointError"
        assert "validation failed" in details["message"].lower()

    def test_missing_checkpoint_key(self):
        original = RuntimeError("Missing key in checkpoint state_dict: model.layer.weight")
        details = create_error_details(original)

        assert details["type"] == "CheckpointError"
        assert "missing" in details["message"].lower()

    def test_moe_expert_weights_missing(self):
        original = RuntimeError("Expert weights missing from checkpoint for layer 0")
        details = create_error_details(original)

        assert details["type"] == "CheckpointError"
        assert "MoE" in details["message"] or "expert" in details["message"].lower()


class TestAutomodelCudaErrors:
    """Tests for Automodel CUDA error conversion."""

    def test_cuda_oom_message(self):
        original = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        details = create_error_details(original)

        assert details["type"] == "CudaError"
        assert "memory" in details["message"].lower()

    def test_out_of_memory_generic(self):
        original = RuntimeError("out of memory")
        details = create_error_details(original)

        assert details["type"] == "CudaError"

    def test_cuda_error_generic(self):
        original = RuntimeError("CUDA error: device-side assert triggered")
        details = create_error_details(original)

        assert details["type"] == "CudaError"


class TestAutomodelDistributedErrors:
    """Tests for Automodel distributed error conversion."""

    def test_distributed_not_available(self):
        original = RuntimeError("torch.distributed not available")
        details = create_error_details(original)

        assert details["type"] == "DistributedError"
        assert "not available" in details["message"].lower()

    def test_distributed_not_initialized(self):
        original = RuntimeError("expected torch.distributed to be initialized")
        details = create_error_details(original)

        assert details["type"] == "DistributedError"
        assert "not properly initialized" in details["message"].lower()

    def test_nccl_error(self):
        original = RuntimeError("NCCL error in: ncclAllReduce")
        details = create_error_details(original)

        assert details["type"] == "DistributedError"
        assert "NCCL" in details["message"]

    def test_timeout_in_cause_chain(self):
        timeout_exc = TimeoutError("Timed out waiting for worker")
        original = RuntimeError("Distributed operation failed")
        original.__cause__ = timeout_exc

        details = create_error_details(original)

        assert details["type"] == "DistributedError"
        assert "timed out" in details["message"].lower()

    def test_timeout_in_nested_cause_chain(self):
        timeout_exc = TimeoutError("Connection timed out")
        middle_exc = ValueError("Worker communication failed")
        middle_exc.__cause__ = timeout_exc
        original = RuntimeError("Training failed")
        original.__cause__ = middle_exc

        details = create_error_details(original)

        assert details["type"] == "DistributedError"
        assert "timed out" in details["message"].lower()


class TestAutomodelTimeoutError:
    """Tests for training timeout error conversion."""

    def test_subprocess_timeout(self):
        original = subprocess.TimeoutExpired(cmd="torchrun", timeout=3600)
        details = create_error_details(original)

        assert details["type"] == "TrainingTimeoutError"
        assert "time limit" in details["message"].lower()


class TestAutomodelInternalErrors:
    """Tests for Automodel internal error conversion."""

    def test_pipeline_missing_inputs(self):
        original = ValueError("You must provide either input_ids or inputs_embeds")
        details = create_error_details(original)

        assert details["type"] == "InternalError"
        assert "pipeline" in details["message"].lower()

    def test_pipeline_missing_embeddings(self):
        original = ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")
        details = create_error_details(original)

        assert details["type"] == "InternalError"
        assert "pipeline" in details["message"].lower()

    def test_moe_mesh_error(self):
        original = AssertionError("We only support 1D mesh for MoE")
        details = create_error_details(original)

        assert details["type"] == "ParallelismConfigError"
        assert "moe" in details["message"].lower()

    def test_dtensor_placement_error(self):
        original = ValueError("tensor has unsupported DTensor placement: Partial")
        details = create_error_details(original)

        assert details["type"] == "ParallelismConfigError"
        assert "moe" in details["message"].lower() or "expert" in details["message"].lower()

    def test_fused_loss_error(self):
        original = ValueError("FusedLinearCrossEntropy requires the model to output hidden states")
        details = create_error_details(original)

        assert details["type"] == "InternalError"
        assert "hidden states" in details["message"].lower()

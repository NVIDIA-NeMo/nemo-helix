# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom exceptions for Customizer training errors.

These exceptions provide user-friendly error messages for errors that may occur
during training with various backends:
- Automodel
- NeMo-RL
- Megatron Bridge
"""

from dataclasses import dataclass
from typing import TypedDict


def format_exception_string(exc: BaseException) -> str:
    """Format an exception as ``TypeName: message`` matching Python's traceback style.

    This is the canonical format used throughout the error-handling pipeline:
    - ``ray_bootstrap`` writes it into the driver output buffer so the parser
      can extract exceptions that occurred outside the subprocess.
    - ``default_exception_handler`` uses it for the ``detail`` field reported
      to the Jobs service.
    - The parser's ``_EXCEPTION_RE`` regex is designed to match this format
      when reading subprocess output.
    """
    return f"{type(exc).__name__}: {exc}"


class ErrorDetails(TypedDict):
    """Error details dict for Jobs service reporting."""

    message: str
    type: str
    detail: str | None


@dataclass
class CustomizerTrainingError(Exception):
    """
    Base exception for Customizer training errors.

    Attributes:
        message: User-friendly error message shown to the user.
        detail: Technical details about the original error (for debugging).
        user_message: Class-level default message used as fallback when the YAML rule
            does not specify an `error_details` field. Subclasses override this.
    """

    message: str
    detail: str | None = None

    # Default user-facing message - subclasses override this.
    # Used as fallback when YAML rule omits `error_details` field.
    # See default_exception_handler() for usage.
    user_message: str = "An error occurred during training."

    def __post_init__(self):
        # Call Exception.__init__ with the message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def to_error_details(self) -> ErrorDetails:
        """Convert to error_details dict for Jobs service reporting."""
        return ErrorDetails(
            message=self.message,
            type=type(self).__name__,
            detail=self.detail,
        )


# =============================================================================
# CLIENT ERRORS (400)
# =============================================================================


@dataclass
class DatasetFormatError(CustomizerTrainingError):
    """
    Dataset has invalid format or schema.

    Raised when:
    - Dataset sample has unsupported role (not system/user/assistant/tool)
    - Dataset is empty or has zero valid samples
    - Text input is not a string or list of strings
    - Required field missing from dataset sample
    - Prompt file does not exist
    """

    user_message: str = "Dataset format error. Please check your dataset matches the expected schema."


@dataclass
class TrainingConfigError(CustomizerTrainingError):
    """
    Invalid training configuration.

    Raised when:
    - Model incompatible with pipeline parallelism (tied embeddings, encoder-decoder)
    - PP batch/microbatch configuration invalid
    - Model doesn't support SDPA for context parallelism
    - Triton not installed for optimized LoRA kernels
    - LoRA adapter dimensions mismatch
    - DPO with dynamic batching or sequence packing
    - GRPO missing generation config or validation dataset
    - Async GRPO configuration errors
    - Batch size not divisible by data parallel size
    - World size insufficient for parallelism configuration
    """

    user_message: str = (
        "Training configuration error. Please check your parallelism settings "
        "(tensor_parallel_size, pipeline_parallel_size, expert_model_parallel_size), "
        "batch settings (batch_size, micro_batch_size), or training type configuration."
    )


@dataclass
class TrainingEnvironmentError(CustomizerTrainingError):
    """
    Invalid environment configuration for GRPO.

    Raised when:
    - GRPO environment name is not recognized
    - GRPO environment not configured
    - No environment found for task type
    """

    user_message: str = "Environment configuration error. Please check your GRPO environment settings."


@dataclass
class ParallelismConfigError(CustomizerTrainingError):
    """
    Invalid parallelism configuration for MoE models.

    Raised when:
    - MoE model uses tensor parallelism with expert parallelism (only 1D mesh supported)
    - DTensor placement incompatible with expert parallelism settings
    - Checkpoint parallelism settings don't match training configuration
    """

    user_message: str = (
        "Parallelism configuration error for Mixture-of-Experts (MoE) model. "
        "MoE models do not support combining tensor_parallel_size > 1 with expert_model_parallel_size > 1. "
        "To fix: either set tensor_parallel_size=1 when using expert parallelism, "
        "or set expert_model_parallel_size=1 when using tensor parallelism."
    )


# =============================================================================
# NOT FOUND ERRORS (404)
# =============================================================================


@dataclass
class ModelNotFoundError(CustomizerTrainingError):
    """
    Model or checkpoint path doesn't exist.

    Raised when:
    - The specified checkpoint path does not exist
    - The checkpoint directory is empty when resuming
    - Nemotron model missing required HF source code
    """

    user_message: str = (
        "Model or checkpoint not found. The specified model path does not exist or is inaccessible. "
        "Please verify the model identifier is correct and the model was successfully downloaded."
    )


# =============================================================================
# SERVER ERRORS (500)
# =============================================================================


@dataclass
class ModelLoadError(CustomizerTrainingError):
    """
    Failed to load or initialize model.

    Raised when:
    - Model weights could not be applied to a layer (corruption)
    - Model optimizations/patches failed
    - Method signature mismatch during patching
    - Missing lm_head.weight in model
    - vLLM library not installed
    - Shape mismatch for model parameters or buffers
    - Generation output missing required fields
    """

    user_message: str = (
        "Failed to load the model. This can happen when: "
        "1) The model checkpoint is corrupted or incomplete, "
        "2) The model architecture is incompatible with the training configuration, "
        "3) There is a version mismatch between the model and the training framework. "
        "Please verify the model checkpoint is valid and complete."
    )


@dataclass
class CheckpointError(CustomizerTrainingError):
    """
    Checkpoint save or load failure.

    Raised when:
    - Checkpoint directory already exists
    - Failed to validate global plan (distributed checkpoint corruption)
    - Missing key in checkpoint state_dict
    - Expert weights missing from MoE checkpoint
    - Training interrupted during checkpoint save
    - Parallelism settings don't match checkpoint
    - Model export or upload failed
    """

    user_message: str = (
        "Checkpoint save or load failed. This can happen when: "
        "1) The checkpoint is corrupted or was saved incompletely (e.g., training was interrupted), "
        "2) Disk space is insufficient for saving checkpoints, "
        "3) The base model checkpoint is incompatible with the current training configuration."
    )


@dataclass
class CudaError(CustomizerTrainingError):
    """
    GPU/CUDA runtime error.

    Raised when:
    - GPU out of memory (OOM)
    - General CUDA runtime errors
    """

    user_message: str = (
        "GPU memory exhausted. To reduce memory usage: "
        "1) Reduce batch_size or micro_batch_size, "
        "2) Reduce max_seq_length, "
        "3) Use LoRA fine-tuning instead of full fine-tuning, "
        "4) Increase tensor_parallel_size to distribute the model across more GPUs."
    )


@dataclass
class DistributedError(CustomizerTrainingError):
    """
    Distributed training or Ray cluster failure.

    Raised when:
    - torch.distributed not available
    - torch.distributed not initialized
    - Distributed operation timeout
    - NCCL communication errors
    - Ray cluster resource insufficiency
    - Placement group allocation failure
    """

    user_message: str = "Distributed training error. Please check cluster resources and try again."


@dataclass
class GenerationError(CustomizerTrainingError):
    """
    vLLM generation/inference failure.

    Raised when:
    - Failed to update vLLM weights from training policy
    - Sync method called on async engine
    - Error during rollout for a sample
    - Async generation called without async engine
    - Penguin requires async vLLM
    """

    user_message: str = (
        "Generation error during reinforcement learning training. "
        "DPO and GRPO training generate model responses during the training loop to compute rewards. "
        "This error indicates the generation step failed, which may be caused by vLLM backend issues "
        "or incompatible generation settings."
    )


@dataclass
class TrainingTimeoutError(CustomizerTrainingError):
    """
    Training exceeded time limit.

    Raised when:
    - Training subprocess exceeded configured timeout
    """

    user_message: str = (
        "Training exceeded the maximum allowed time limit. "
        "To reduce training time: reduce epochs or max_steps, use a smaller dataset, "
        "use a smaller model, or use LoRA fine-tuning instead of full fine-tuning. "
        "Contact your administrator if you need longer training time limits."
    )


@dataclass
class InternalError(CustomizerTrainingError):
    """
    Unexpected internal error.

    Raised when:
    - Pipeline stage missing input_ids or inputs_embeds
    - MoE device mesh configuration error
    - DTensor placement error for expert parallelism
    - FusedLinearCrossEntropy configuration error
    - Tensor dimension/dtype/device mismatch
    - Logger misconfiguration
    - Any unmatched error (fallback)
    """

    user_message: str = (
        "An unexpected internal error occurred during training. "
        "This is typically caused by framework-level issues such as tensor misconfigurations, "
        "device mesh errors, or internal pipeline failures. "
        "Please try running your job again. If the issue persists, contact your administrator "
        "with the job ID and error details for further investigation."
    )


@dataclass
class GenericTrainingError(CustomizerTrainingError):
    """
    Fallback when error classification is ambiguous.

    Used when multiple error rules match the same exception,
    making classification unreliable.
    """

    user_message: str = (
        "Training failed due to an error that could not be precisely categorized. "
        "Please review the error details for more information. "
        "If the issue persists, try adjusting your training configuration."
    )


# =============================================================================
# EXCEPTION REGISTRY
# =============================================================================

# Maps exception class names (strings in YAML) to actual Python classes
EXCEPTION_REGISTRY: dict[str, type[Exception]] = {
    # Base
    "CustomizerTrainingError": CustomizerTrainingError,
    # Client errors (400)
    "DatasetFormatError": DatasetFormatError,
    "TrainingConfigError": TrainingConfigError,
    "TrainingEnvironmentError": TrainingEnvironmentError,
    "ParallelismConfigError": ParallelismConfigError,
    # Not found (404)
    "ModelNotFoundError": ModelNotFoundError,
    # Server errors (500)
    "ModelLoadError": ModelLoadError,
    "CheckpointError": CheckpointError,
    "CudaError": CudaError,
    "DistributedError": DistributedError,
    "GenerationError": GenerationError,
    "TrainingTimeoutError": TrainingTimeoutError,
    "InternalError": InternalError,
    "GenericTrainingError": GenericTrainingError,
}


# =============================================================================
# DEFAULT EXCEPTION HANDLER
# =============================================================================


def default_exception_handler(
    exception_class: type[Exception],
    original_exception: Exception,
    error_details: str | None,
) -> Exception:
    """
    Default handler for creating Customizer training exceptions.

    This handler is used by RulesLoader when:
    1. A rule matches but doesn't have a custom handler
    2. No rule matches and fallback_exception is set

    Args:
        exception_class: The exception class to create (from EXCEPTION_REGISTRY)
        original_exception: The original exception that was caught
        error_details: User-friendly message from the rule's error_details field,
                       or None if not specified

    Returns:
        A new instance of exception_class with appropriate message and detail
    """
    # Get the default user message from the class if no error_details provided
    if issubclass(exception_class, CustomizerTrainingError):
        user_message = error_details or exception_class.user_message
        # For InternalError fallback (no matching rule), include the original error
        # in the message so users get actionable information instead of a vague message
        if exception_class is InternalError and error_details is None:
            user_message = f"{user_message} ({format_exception_string(original_exception)})"
        return exception_class(
            message=user_message,
            detail=format_exception_string(original_exception),
        )
    else:
        # For non-CustomizerTrainingError classes (shouldn't happen, but be safe)
        return exception_class(error_details or str(original_exception))


__all__ = [
    "CheckpointError",
    "CudaError",
    "CustomizerTrainingError",
    "DatasetFormatError",
    "DistributedError",
    "ErrorDetails",
    "EXCEPTION_REGISTRY",
    "format_exception_string",
    "GenerationError",
    "GenericTrainingError",
    "InternalError",
    "ModelLoadError",
    "ModelNotFoundError",
    "ParallelismConfigError",
    "TrainingConfigError",
    "TrainingEnvironmentError",
    "TrainingTimeoutError",
    "default_exception_handler",
]

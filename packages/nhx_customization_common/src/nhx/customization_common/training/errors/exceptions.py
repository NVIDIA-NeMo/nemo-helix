# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing errors for customization training failures (Automodel, NeMo-RL, Unsloth).

Each error carries the id of the rule that classified it (``cause``) and whether a rerun of the same
job could succeed (``rerunnable``).
"""

from dataclasses import dataclass
from typing import TypedDict


def format_exception_string(exc: BaseException) -> str:
    """Format an exception as ``TypeName: message``, the traceback format the log parser reads."""
    return f"{type(exc).__name__}: {exc}"


class ErrorDetails(TypedDict):
    """Error details reported to the Jobs service."""

    message: str
    type: str
    detail: str | None
    cause: str | None


@dataclass
class CustomizerTrainingError(Exception):
    """Base class for classified training errors.

    Attributes:
        message: User-facing message.
        detail: The original error, for debugging.
        user_message: Default message when the rule gives none.
        cause: Id of the rule that matched, or ``None`` when no rule did.
        rerunnable: Whether a rerun of the same job could succeed. False for deterministic failures,
            such as a bad dataset, a bad configuration, or running out of GPU memory.
    """

    message: str
    detail: str | None = None
    user_message: str = "An error occurred during training."
    cause: str | None = None
    rerunnable: bool = True

    def __post_init__(self):
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def to_error_details(self) -> ErrorDetails:
        return ErrorDetails(message=self.message, type=type(self).__name__, detail=self.detail, cause=self.cause)


# =============================================================================
# CLIENT ERRORS (400)
# =============================================================================


@dataclass
class DatasetFormatError(CustomizerTrainingError):
    """The dataset is missing, empty, or does not match the expected schema."""

    user_message: str = "Dataset format error. Please check your dataset matches the expected schema."


@dataclass
class TrainingConfigError(CustomizerTrainingError):
    """The training configuration is invalid for this model, backend, or cluster."""

    user_message: str = (
        "Training configuration error. Please check your parallelism settings "
        "(tensor_parallel_size, pipeline_parallel_size, expert_model_parallel_size), "
        "batch settings (batch_size, micro_batch_size), or training type configuration."
    )


@dataclass
class TrainingEnvironmentError(CustomizerTrainingError):
    """The GRPO environment is missing or misconfigured."""

    user_message: str = "Environment configuration error. Please check your GRPO environment settings."


@dataclass
class ParallelismConfigError(CustomizerTrainingError):
    """The parallelism settings are not supported for this Mixture-of-Experts model."""

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
    """The model or checkpoint path does not exist."""

    user_message: str = (
        "Model or checkpoint not found. The specified model path does not exist or is inaccessible. "
        "Please verify the model identifier is correct and the model was successfully downloaded."
    )


# =============================================================================
# SERVER ERRORS (500)
# =============================================================================


@dataclass
class ModelLoadError(CustomizerTrainingError):
    """The model could not be loaded or is incompatible with the training setup."""

    user_message: str = (
        "Failed to load the model. This can happen when: "
        "1) The model checkpoint is corrupted or incomplete, "
        "2) The model architecture is incompatible with the training configuration, "
        "3) There is a version mismatch between the model and the training framework. "
        "Please verify the model checkpoint is valid and complete."
    )


@dataclass
class CheckpointError(CustomizerTrainingError):
    """A checkpoint could not be saved, loaded, or published."""

    user_message: str = (
        "Checkpoint save or load failed. This can happen when: "
        "1) The checkpoint is corrupted or was saved incompletely (e.g., training was interrupted), "
        "2) Disk space is insufficient for saving checkpoints, "
        "3) The base model checkpoint is incompatible with the current training configuration."
    )


@dataclass
class GpuOutOfMemoryError(CustomizerTrainingError):
    """A GPU ran out of memory."""

    user_message: str = (
        "GPU out of memory. To reduce memory usage: "
        "1) Reduce batch_size or micro_batch_size, "
        "2) Reduce max_seq_length, "
        "3) Use LoRA fine-tuning instead of full fine-tuning, "
        "4) Increase tensor_parallel_size to distribute the model across more GPUs."
    )


@dataclass
class CudaError(CustomizerTrainingError):
    """A CUDA runtime or device error other than running out of memory."""

    user_message: str = (
        "A GPU/CUDA error occurred. This can be a transient hardware or driver fault; if it repeats, "
        "contact your administrator with the job ID."
    )


@dataclass
class NodeResourceError(CustomizerTrainingError):
    """The node ran out of disk space, shared memory (/dev/shm), or host memory."""

    user_message: str = (
        "The training node ran out of disk space, shared memory, or host memory. "
        "Contact your administrator if the node's storage or memory limits need to change."
    )


@dataclass
class DistributedError(CustomizerTrainingError):
    """Distributed communication or cluster scheduling failed."""

    user_message: str = "Distributed training error. Please check cluster resources and try again."


@dataclass
class GenerationError(CustomizerTrainingError):
    """vLLM generation failed during DPO or GRPO training."""

    user_message: str = (
        "Generation error during reinforcement learning training. "
        "DPO and GRPO training generate model responses during the training loop to compute rewards. "
        "This error indicates the generation step failed, which may be caused by vLLM backend issues "
        "or incompatible generation settings."
    )


@dataclass
class TrainingTimeoutError(CustomizerTrainingError):
    """Training exceeded its time limit."""

    user_message: str = (
        "Training exceeded the maximum allowed time limit. "
        "To reduce training time: reduce epochs or max_steps, use a smaller dataset, "
        "use a smaller model, or use LoRA fine-tuning instead of full fine-tuning. "
        "Contact your administrator if you need longer training time limits."
    )


@dataclass
class InternalError(CustomizerTrainingError):
    """An unexpected internal error, and the fallback when no rule matches."""

    user_message: str = (
        "An unexpected internal error occurred during training. "
        "This is typically caused by framework-level issues such as tensor misconfigurations, "
        "device mesh errors, or internal pipeline failures. "
        "Please try running your job again. If the issue persists, contact your administrator "
        "with the job ID and error details for further investigation."
    )


EXCEPTION_REGISTRY: dict[str, type[CustomizerTrainingError]] = {
    cls.__name__: cls
    for cls in (
        CustomizerTrainingError,
        DatasetFormatError,
        TrainingConfigError,
        TrainingEnvironmentError,
        ParallelismConfigError,
        ModelNotFoundError,
        ModelLoadError,
        CheckpointError,
        GpuOutOfMemoryError,
        CudaError,
        NodeResourceError,
        DistributedError,
        GenerationError,
        TrainingTimeoutError,
        InternalError,
    )
}

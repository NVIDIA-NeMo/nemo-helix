# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Literal, Optional

from nemo_platform_plugin.integrations import IntegrationsSpec
from nmp.automodel.app.constants import (
    DEFAULT_OUTPUT_MODEL_PATH,
    DEFAULT_SEED,
    DEFAULT_TRAINING_OUTPUT_PATH,
)
from nmp.automodel.entities.values import (
    CheckpointFormat,
    CheckpointSelection,
    FinetuningType,
    Precision,
    TrainingType,
)
from nmp.customization_common.training.reporting import ProgressReportingConfig
from pydantic import BaseModel, Field


class OptimizerType(str, Enum):
    """Optimizer and scheduler combination types."""

    ADAMW_WITH_COSINE_ANNEALING = "adamw_with_cosine_annealing"
    ADAM_WITH_COSINE_ANNEALING = "adam_with_cosine_annealing"
    ADAMW_WITH_FLAT_LR = "adamw_with_flat_lr"
    ADAM_WITH_FLAT_LR = "adam_with_flat_lr"


class TrainingRecipe(str, Enum):
    AUTO = "auto"
    SFT = "sft"
    BI_ENCODER = "bi_encoder"
    CROSS_ENCODER = "cross_encoder"


class LoRAConfig(BaseModel):
    """Internal LoRA configuration with implementation details.

    This differs from the API LoRAParams:
    - Includes use_triton, match_all_linear (implementation details)
    - exclude_modules for advanced control
    - Can add new fields freely without breaking API
    """

    # Core LoRA parameters (from API)
    rank: int = Field(default=8, description="LoRA rank (low-rank dimension)")
    alpha: int = Field(default=32, description="LoRA alpha scaling factor")
    dropout: float = Field(default=0.0, description="LoRA dropout probability")

    # Module targeting
    target_modules: Optional[list[str]] = Field(
        default=None, description="Module name patterns to apply LoRA to (e.g., ['*.proj'])"
    )
    exclude_modules: Optional[list[str]] = Field(default=None, description="Module name patterns to exclude from LoRA")

    # Implementation details (not in API)
    use_triton: bool = Field(default=True, description="Use optimized Triton LoRA kernel")


class ModelConfig(BaseModel):
    """Internal model configuration."""

    path: str = Field(description="Path to a model directory (contains config, weights, tokenizer etc.)")
    name: Optional[str] = Field(
        default=None,
        description="Model identifier (e.g., 'meta/llama-3.1-8b-instruct')",
    )
    max_seq_length: int = Field(
        default=2048,
        description="Maximum token sequence length for training; longer sequences are truncated",
    )

    # Model loading options
    precision: Optional[Precision] = Field(
        default=None,
        description="Model weight dtype (e.g., 'bf16', 'fp16'). None implies auto-detects from model config",
    )
    attn_implementation: Optional[str] = Field(
        default="sdpa",
        description="Attention backend: 'sdpa' (PyTorch native), 'flash_attention_2' (requires flash-attn), 'eager' (no optimization)",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Allow executing custom model code from the checkpoint. Required for some community models",
    )
    is_embedding_model: bool = Field(
        default=False,
        description="Deprecated compatibility alias for checkpoint_head_type == 'embedding'.",
    )
    checkpoint_head_type: str = Field(
        default="unknown",
        description="Task-specific head persisted in the source checkpoint.",
    )
    chat_template: Optional[str] = Field(
        default=None,
        description="Jinja2 chat template from the model entity spec or fileset metadata. "
        "Takes highest priority in resolve_chat_template when set.",
    )

    override_custom_impl: bool = Field(
        default=False,
        description="Some of the custom implementations in nemo automodel cause loading failures when used with other models in the same family, this forces the use_hf=True flag to use non custom implementations.",
    )

    v4_compatible: bool = Field(
        default=False,
        description="Enable transformers-v4-compatible checkpoint output that preserves the original transformers-v4-style config.json output.",
    )


class DistillationConfig(BaseModel):
    """Internal Knowledge Distillation configuration.

    teacher is a ModelConfig with resolved path, not a URN.
    """

    # Teacher model (resolved path)
    teacher_model: ModelConfig = Field(description="Teacher model configuration with resolved path")

    # KD hyperparameters
    ratio: float = Field(default=0.5, description="Balance between CE loss and KD loss")
    temperature: float = Field(default=1.0, description="Softmax temperature for KD")

    # Implementation detail (not in API)
    offload_teacher: bool = Field(default=False, description="Offload teacher model to CPU for memory efficiency")


class ExportConfig(BaseModel):
    """ONNX export and fileset layout. ``primary`` is the artifact at the root; the other is under ``alternates/``."""

    primary: Literal["onnx", "hf"] = Field(
        default="onnx",
        description="Artifact at the fileset root. Use 'hf' when the NIM loads the PyTorch checkpoint.",
    )
    opset: int = Field(default=17, gt=0, description="ONNX opset version.")
    precision: Literal["fp32", "fp16"] = Field(
        default="fp16",
        description="ONNX graph dtype. Defaults to fp16 to match typical Hugging Face checkpoints.",
    )
    attn_implementation: str = Field(
        default="eager",
        description="Attention backend for the traced model. The exporter cannot trace SDPA/GQA.",
    )
    pooling: Literal["avg", "cls", "last"] = Field(
        default="avg",
        description="Embedding pooling over hidden states. Ignored for cross_encoder.",
    )
    normalize: bool = Field(default=True, description="L2-normalize pooled embeddings. Ignored for cross_encoder.")
    dimensions: bool = Field(
        default=False,
        description="Add a Matryoshka 'dimensions' input that truncates and renormalizes embeddings.",
    )


class RetrievalConfig(BaseModel):
    """Collator, dataset, and export knobs for ``bi_encoder`` / ``cross_encoder``."""

    # Training configuration
    train_n_passages: int = Field(
        default=5,
        description=(
            "Total number of passages per query during training: 1 positive + (n-1) negatives. "
            "For example, train_n_passages=5 means 1 positive and 4 negative passages per query."
        ),
    )
    eval_negative_size: Optional[int] = Field(
        default=None,
        description=(
            "Number of negative passages per query during validation. "
            "Recommended to keep as train_n_passages - 1 for consistent train/eval behavior. "
            "If not set, defaults to train_n_passages - 1."
        ),
    )

    # Memory optimization
    do_gradient_checkpointing: bool = Field(
        default=False,
        description=(
            "Enable gradient checkpointing to reduce memory usage at the cost of slower training. "
            "Useful for larger embedding models or memory-constrained environments."
        ),
    )

    # Contrastive objective
    do_distributed_inbatch_negative: bool = Field(
        default=False,
        description=("Use other queries' passages in the global batch as extra negatives. Ignored for cross_encoder."),
    )

    # Tokenization configuration
    query_max_length: int = Field(default=512, description="Maximum token length for query tokenization")
    passage_max_length: int = Field(default=512, description="Maximum token length for passage tokenization")
    query_prefix: str = Field(default="query:", description="Prefix to prepend to queries before tokenization")
    passage_prefix: str = Field(default="passage:", description="Prefix to prepend to passages before tokenization")

    # Post-training export
    export: Optional[ExportConfig] = Field(
        default=None,
        description="Output artifact layout and ONNX export knobs. Defaults are applied when omitted.",
    )


class TrainingStepConfig(BaseModel):
    """Normalized training configuration compiled into nemo-automodel recipe YAML."""

    class DatasetConfig(BaseModel):
        path: str
        prompt_template: Optional[str] = None
        add_bos: Optional[bool] = None
        add_eos: Optional[bool] = None

    class TrainingConfig(BaseModel):
        training_type: TrainingType
        recipe: TrainingRecipe = TrainingRecipe.AUTO
        finetuning_type: Optional[FinetuningType] = None
        lora: Optional[LoRAConfig] = None
        kd: Optional[DistillationConfig] = None

    class ScheduleConfig(BaseModel):
        epochs: int = 1
        max_steps: Optional[int] = None
        val_check_interval: Optional[float] = None
        validation_split: Optional[float] = Field(default=0.1, gt=0, lt=1)
        checkpoint_selection: CheckpointSelection = Field(
            default=CheckpointSelection.BEST,
            description="Checkpoint(s) published after training: lowest validation loss, last, or both.",
        )
        progress_reporting: ProgressReportingConfig = Field(default_factory=ProgressReportingConfig)

    class BatchConfig(BaseModel):
        global_batch_size: int = Field(default=32, gt=0)
        micro_batch_size: int = Field(default=1, gt=0)
        sequence_packing: bool = False
        sequence_packing_max_samples: int = 1000

    class OptimizerConfig(BaseModel):
        optimizer_type: Optional[OptimizerType] = Field(default=None)
        optimizer_name: str = "auto"
        lr_decay_style: str = "cosine"
        learning_rate: float = 1e-4
        min_learning_rate: Optional[float] = None
        eps: float = 1e-8
        weight_decay: float = 0.01
        beta1: float = 0.9
        beta2: float = 0.999
        warmup_steps: int = 0

    class ParallelismConfig(BaseModel):
        num_nodes: int = 1
        num_gpus_per_node: int = 1
        tensor_parallel_size: int = 1
        pipeline_parallel_size: int = 1
        context_parallel_size: int = 1
        expert_parallel_size: Optional[int] = None
        sequence_parallel: bool = False

    # === Main Config Fields ===
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    schedule: ScheduleConfig
    batch: BatchConfig
    optimizer: OptimizerConfig
    parallelism: ParallelismConfig
    integrations: IntegrationsSpec | None = None

    # === Output Paths ===
    output_model: str  # Set at compile-time from CustomizationJobOutput
    workspace_path: str = Field(default=DEFAULT_TRAINING_OUTPUT_PATH)
    output_path: str = Field(default=DEFAULT_OUTPUT_MODEL_PATH)

    # === Miscellaneous ===
    seed: int = Field(
        default=DEFAULT_SEED, description="Random seed for ensuring reproducibility in all random processes."
    )
    training_timeout: Optional[int] = None
    retrieval: Optional[RetrievalConfig] = Field(
        default=None,
        description="Retrieval dataset, collator, and export knobs for bi_encoder / cross_encoder recipes.",
    )

    model_config = {"populate_by_name": True}


class GPUInfo(BaseModel):
    """GPU architecture information captured during training."""

    architecture: str
    device_name: str
    memory_gb: float
    cuda_version: str


class CheckpointInfo(BaseModel):
    """Output checkpoint information."""

    path: str
    format: CheckpointFormat
    precision: Optional[Precision] = Field(
        default=None, description="Checkpoint precision. None when auto-detected from model config."
    )


class TrainingMetrics(BaseModel):
    """Final training metrics."""

    final_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    total_steps: int = 0
    total_epochs: int = 0


class TrainingResult(BaseModel):
    """
    Result written by training task.

    Written to: {workspace_path}/training_result.json
    """

    success: bool
    error_message: Optional[str] = None
    checkpoint: Optional[CheckpointInfo] = None
    gpu_info: Optional[GPUInfo] = None
    metrics: TrainingMetrics = Field(default_factory=TrainingMetrics)
    training_duration_seconds: Optional[float] = None

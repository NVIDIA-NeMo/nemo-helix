# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automodel job input/output schemas (simplified JSON v1)."""

from __future__ import annotations

from typing import Literal, Self

from nemo_platform_plugin.integrations import IntegrationsSpec
from nmp.customization_common.schema import NamespacedModel
from nmp.customization_common.training.reporting import ProgressReportingConfig
from pydantic import Field, model_validator

__all__ = [
    "AutomodelJobInput",
    "AutomodelJobOutput",
    "BatchSpec",
    "DatasetSpec",
    "ExportSpec",
    "LoRAParams",
    "OptimizerSpec",
    "OutputRequest",
    "OutputResponse",
    "ParallelismSpec",
    "RetrievalSpec",
    "ScheduleSpec",
    "TrainingSpec",
    "ValidationError",
]


class ValidationError(ValueError):
    """Raised when automodel job input validation fails."""


class AutomodelSchema(NamespacedModel):
    """Backend base: every Automodel-owned model emits an ``Automodel``-prefixed
    OpenAPI schema name (``TrainingSpec`` -> ``AutomodelTrainingSpec``), so it
    can't collide with another backend's same-named model in the merged
    ``/apis/customization`` spec. ``extra='forbid'`` is inherited from the base."""

    __schema_namespace__ = "Automodel"


class LoRAParams(AutomodelSchema):
    rank: int = Field(
        default=16,
        gt=0,
        description="Rank of the LoRA update matrices. Higher ranks add capacity and trainable parameters.",
    )
    alpha: int = Field(
        default=32, gt=0, description="LoRA scaling factor. The update is scaled by alpha divided by rank."
    )
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="LoRA dropout probability for regularization.")
    merge: bool = Field(
        default=False,
        description=(
            "Merge the adapter into the base model after training, producing a full-weight checkpoint "
            "rather than an adapter. Has the same effect as finetuning_type 'lora_merged'."
        ),
    )
    target_modules: list[str] | None = Field(
        default=None,
        description=(
            "Module name patterns to apply LoRA to (e.g. ['*.q_proj', '*.v_proj']). Applies to every "
            "'*proj' linear layer when unset."
        ),
    )
    exclude_modules: list[str] | None = Field(
        default=None, description="Module name patterns to exclude from LoRA (e.g. ['*.out_proj'])."
    )
    use_triton: bool = Field(default=True, description="Use the optimized Triton LoRA kernel.")


class DatasetSpec(AutomodelSchema):
    training: str = Field(description="Training fileset as 'name' or 'workspace/name'.")
    validation: str | None = Field(
        default=None,
        description=(
            "Validation fileset as 'name' or 'workspace/name'. When unset, schedule.validation_split "
            "holds part of the training data out instead."
        ),
    )
    prompt_template: str | None = Field(
        default=None,
        description=(
            "Row template for datasets whose columns are neither 'messages' nor 'prompt'/'completion', "
            "e.g. '{input} {output}'. Takes exactly two placeholders, each naming a column."
        ),
    )


class ExportSpec(AutomodelSchema):
    """ONNX export and fileset layout. ``primary`` is the artifact at the root; the other is under ``alternates/``."""

    primary: Literal["onnx", "hf"] = Field(
        default="onnx", description="Artifact at the fileset root. Use 'hf' when the NIM loads PyTorch weights."
    )
    opset: int = Field(default=17, gt=0, description="ONNX opset version the graph is exported against.")
    precision: Literal["fp32", "fp16"] = Field(
        default="fp16",
        description="ONNX graph dtype. Defaults to fp16 to match typical Hugging Face checkpoints.",
    )
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = Field(
        default="eager", description="Attention backend for tracing. The exporter cannot trace SDPA/GQA."
    )
    pooling: Literal["avg", "cls", "last"] = Field(
        default="avg", description="Embedding pooling. Ignored for cross_encoder."
    )
    normalize: bool = Field(default=True, description="L2-normalize embeddings. Ignored for cross_encoder.")
    dimensions: bool = Field(
        default=False, description="Add a Matryoshka 'dimensions' input that truncates and renormalizes embeddings."
    )


class RetrievalSpec(AutomodelSchema):
    """Dataset, collator, and export knobs for bi_encoder / cross_encoder recipes."""

    train_n_passages: int = Field(
        default=5, ge=2, description="Passages per training example: one positive and the rest as negatives."
    )
    eval_negative_size: int | None = Field(
        default=None, ge=1, description="Negatives per query at evaluation. Follows the training count when unset."
    )
    do_gradient_checkpointing: bool = Field(
        default=False, description="Recompute activations during the backward pass to save memory."
    )
    query_max_length: int = Field(default=512, ge=1, description="Maximum token length of the query side.")
    passage_max_length: int = Field(default=512, ge=1, description="Maximum token length of the passage side.")
    query_prefix: str = Field(default="query:", description="Collator-side prefix; BiEncoderCollator adds a space.")
    passage_prefix: str = Field(default="passage:", description="Collator-side prefix; BiEncoderCollator adds a space.")
    export: ExportSpec | None = Field(
        default=None, description="Artifact layout and ONNX export settings. Defaults are applied when omitted."
    )


class TrainingSpec(AutomodelSchema):
    model_config = AutomodelSchema.model_config | {"populate_by_name": True}

    training_type: Literal["sft", "distillation"] = Field(
        default="sft",
        description="'distillation' trains against a teacher model and requires teacher_model; 'sft' does not.",
    )
    recipe: Literal["auto", "sft", "bi_encoder", "cross_encoder"] = Field(
        default="auto",
        description=(
            "Training recipe. 'auto' uses the model checkpoint head; explicit encoder recipes can wrap a causal-LM "
            "backbone for retrieval training."
        ),
    )
    finetuning_type: Literal["lora", "all_weights", "lora_merged"] = Field(
        default="lora",
        description=(
            "'lora' trains an adapter and ships it as one, 'all_weights' trains every weight, and "
            "'lora_merged' trains an adapter then folds it into the base weights. Setting lora.merge "
            "on 'lora' produces the same full-weight result as 'lora_merged'."
        ),
    )
    lora: LoRAParams | None = Field(
        default=None, description="LoRA settings. Defaulted automatically when finetuning_type is a LoRA variant."
    )
    max_seq_length: int = Field(
        default=2048, gt=0, description="Maximum token sequence length for training; longer sequences are truncated."
    )
    precision: Literal["bf16", "fp16", "fp32", "fp8"] | None = Field(
        default=None,
        description="Model precision for training. Auto-detected from the checkpoint when unset.",
    )
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = Field(
        default="sdpa",
        description="Attention backend: 'sdpa' (PyTorch native), 'flash_attention_2', or 'eager'.",
    )
    execution_profile: str | None = Field(
        default=None, min_length=1, description="Named tuning profile to apply. The backend chooses one when unset."
    )
    teacher_model: str | None = Field(
        default=None, description="Teacher model to distill from. Required when training_type is 'distillation'."
    )
    distillation_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight of the teacher distillation loss against the ground-truth loss. 0 ignores the teacher.",
    )
    distillation_temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Softens the teacher distribution before the student matches it. Higher values soften more.",
    )
    teacher_precision: Literal["bf16", "fp16", "fp32"] = Field(
        default="bf16", description="Precision the teacher runs at. It is never trained, so this only affects memory."
    )
    offload_teacher: bool = Field(
        default=False, description="Keep the teacher on CPU between forward passes, trading speed for GPU memory."
    )
    retrieval: RetrievalSpec | None = Field(
        default=None,
        description="Retrieval dataset, collator, and export knobs. Used when recipe is bi_encoder or cross_encoder.",
    )

    @model_validator(mode="after")
    def _training_type_fields(self) -> Self:
        if self.training_type == "distillation" and not self.teacher_model:
            raise ValueError("teacher_model is required when training_type is distillation")
        if self.training_type == "distillation" and self.recipe not in ("auto", "sft"):
            raise ValueError("distillation only supports the sft recipe")
        if self.finetuning_type.startswith("lora") and self.lora is None:
            self.lora = LoRAParams()
        return self


class ScheduleSpec(AutomodelSchema):
    epochs: int = Field(default=1, gt=0, description="Number of complete passes through the dataset.")
    max_steps: int | None = Field(
        default=None,
        gt=0,
        description="Hard cap on optimizer steps. Training stops at whichever comes first, this or the epoch count.",
    )
    val_check_interval: float | None = Field(
        default=None,
        description="How often to run validation: a fraction of an epoch below 1, or a step count at 1 and above.",
    )
    validation_split: float | None = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="Validation split to use when a validation dataset is not provided.",
    )
    seed: int | None = Field(
        default=None, description="Seed for shuffling and initialisation. A run is only reproducible when this is set."
    )
    progress_reporting: ProgressReportingConfig = Field(default_factory=ProgressReportingConfig)


# (global_batch_size, micro_batch_size) per retrieval recipe. bi_encoder takes its
# in-batch negatives from the micro batch, which accumulation does not widen, so
# lowering micro costs retrieval quality; cross_encoder scores pairs independently.
RETRIEVAL_BATCH_DEFAULTS: dict[str, tuple[int, int]] = {
    "bi_encoder": (256, 8),
    "cross_encoder": (128, 8),
}


class BatchSpec(AutomodelSchema):
    global_batch_size: int = Field(default=8, gt=0, description="Examples per optimizer step, summed across all GPUs.")
    micro_batch_size: int = Field(
        default=1,
        gt=0,
        description="Examples each GPU processes at once. Lower this first when training runs out of memory.",
    )
    sequence_packing: bool = Field(
        default=False, description="Pack several short examples into one sequence to cut padding waste."
    )
    sequence_packing_max_samples: int = Field(
        default=1000, gt=0, description="Samples analyzed to estimate the optimal pack size when packing is enabled."
    )


class OptimizerSpec(AutomodelSchema):
    learning_rate: float = Field(
        default=5e-6,
        gt=0.0,
        description="Peak learning rate, reached at the end of warmup. Around 5e-5 suits full SFT and 1e-4 LoRA.",
    )
    min_learning_rate: float | None = Field(
        default=None, ge=0.0, description="Minimum learning rate for the cosine decay schedule."
    )
    weight_decay: float = Field(
        default=0.01, ge=0.0, description="Penalty on large weights. Higher regularizes more; 0 disables it."
    )
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0, description="Adam optimizer beta1.")
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0, description="Adam optimizer beta2.")
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Steps spent ramping the learning rate up from zero. Around 10% of total steps is a stable start.",
    )
    adam_eps: float = Field(default=1e-8, gt=0.0, description="Adam/AdamW epsilon for numerical stability.")
    optimizer: Literal["auto", "Adam", "AdamW", "FusedAdam"] = Field(
        default="auto",
        description=(
            "Optimizer algorithm. 'auto' selects Transformer Engine FusedAdam for retrieval recipes "
            "and torch Adam for SFT."
        ),
    )
    lr_decay_style: Literal["cosine", "linear", "constant"] = Field(
        default="cosine", description="Learning-rate decay schedule."
    )


class ParallelismSpec(AutomodelSchema):
    num_nodes: int = Field(default=1, gt=0, description="Number of nodes to train on.")
    num_gpus_per_node: int = Field(
        default=1, gt=0, description="GPUs used on each node. Total GPUs is num_nodes multiplied by this."
    )
    tensor_parallel_size: int = Field(default=1, gt=0, description="Splits each layer across this many GPUs.")
    pipeline_parallel_size: int = Field(
        default=1, gt=0, description="Splits the layer stack into this many sequential stages across GPUs."
    )
    context_parallel_size: int = Field(
        default=1, gt=0, description="Splits the sequence dimension across GPUs, for long-sequence training."
    )
    expert_parallel_size: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Expert parallel size for MoE models. Must divide data_parallel_size x context_parallel_size, and "
            "tensor_parallel_size must be 1 when this is above 1."
        ),
    )
    sequence_parallel: bool = Field(default=False, description="Enable sequence parallelism.")


class OutputRequest(AutomodelSchema):
    name: str = Field(description="Name for the model this job produces. Generated from the inputs when omitted.")
    description: str | None = Field(default=None, description="Free-text description stored on the output model.")


class OutputResponse(AutomodelSchema):
    name: str
    type: Literal["model", "adapter"]
    fileset: str
    description: str | None = None


class AutomodelJobInput(AutomodelSchema):
    """POST body / CLI JSON."""

    name: str | None = Field(default=None, description="Name for the job. Generated when omitted.")
    model: str = Field(description="Model to fine-tune, as 'name' or 'workspace/name'.")
    dataset: DatasetSpec = Field(description="Training and validation data.")
    training: TrainingSpec = Field(description="What kind of fine-tuning to run and how the model is adapted.")
    schedule: ScheduleSpec = Field(
        default_factory=ScheduleSpec,
        description="How long training runs, and how often it validates and reports progress.",
    )
    batch: BatchSpec = Field(default_factory=BatchSpec, description="Batch sizes and sequence packing.")
    optimizer: OptimizerSpec = Field(
        default_factory=OptimizerSpec, description="Optimizer choice and learning-rate schedule."
    )
    parallelism: ParallelismSpec = Field(
        default_factory=ParallelismSpec, description="How the job is spread across nodes and GPUs."
    )
    output: OutputRequest | None = Field(
        default=None, description="Naming for the model this job produces. Derived from the inputs when omitted."
    )
    integrations: IntegrationsSpec | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_fields(cls, data: object) -> object:
        if isinstance(data, dict) and "output_model" in data:
            raise ValueError("spec.output_model was removed. Use spec.output instead.")
        return data

    def with_resolved_recipe(self, checkpoint_head_type: str) -> Self:
        """Return the canonical job input after resolving its recipe and defaults."""
        recipe = self.training.recipe
        if recipe == "auto" and checkpoint_head_type in ("embedding", "cross_encoder"):
            recipe = "bi_encoder" if checkpoint_head_type == "embedding" else "cross_encoder"
        if self.training.training_type == "distillation" and recipe not in ("auto", "sft"):
            raise ValueError("distillation only supports the sft recipe")

        training = self.training.model_copy(update={"recipe": recipe})
        if recipe == "bi_encoder":
            learning_rate, warmup_steps = 1e-5, 5
            global_batch_size, micro_batch_size = RETRIEVAL_BATCH_DEFAULTS["bi_encoder"]
        elif recipe == "cross_encoder":
            learning_rate, warmup_steps = 3e-6, 100
            global_batch_size, micro_batch_size = RETRIEVAL_BATCH_DEFAULTS["cross_encoder"]
        else:
            return self.model_copy(update={"training": training})

        batch_updates: dict[str, int] = {}
        if "global_batch_size" not in self.batch.model_fields_set:
            batch_updates["global_batch_size"] = global_batch_size
        if "micro_batch_size" not in self.batch.model_fields_set:
            batch_updates["micro_batch_size"] = micro_batch_size

        optimizer_updates: dict[str, float | int] = {}
        if "learning_rate" not in self.optimizer.model_fields_set:
            optimizer_updates["learning_rate"] = learning_rate
        if "warmup_steps" not in self.optimizer.model_fields_set:
            optimizer_updates["warmup_steps"] = warmup_steps

        return self.model_copy(
            update={
                "training": training,
                "batch": self.batch.model_copy(update=batch_updates),
                "optimizer": self.optimizer.model_copy(update=optimizer_updates),
            }
        )


class AutomodelJobOutput(AutomodelSchema):
    """Stored canonical spec after ``to_spec()``."""

    name: str | None = None
    model: str
    dataset: DatasetSpec
    training: TrainingSpec
    schedule: ScheduleSpec
    batch: BatchSpec
    optimizer: OptimizerSpec
    parallelism: ParallelismSpec
    output: OutputResponse
    integrations: IntegrationsSpec | None = None

    def validate_for_training(self) -> None:
        """MoE / parallelism constraints (ported from legacy CustomizationJobOutput)."""
        p = self.parallelism
        num_nodes = p.num_nodes
        num_gpus_per_node = p.num_gpus_per_node
        tp = p.tensor_parallel_size
        pp = p.pipeline_parallel_size
        cp = p.context_parallel_size
        ep = p.expert_parallel_size

        total_gpus = num_gpus_per_node * num_nodes
        model_parallel_size = tp * pp * cp
        if total_gpus % model_parallel_size != 0:
            raise ValidationError(
                f"Total GPUs ({total_gpus}) must be divisible by "
                f"tensor_parallel_size ({tp}) * pipeline_parallel_size ({pp}) * "
                f"context_parallel_size ({cp}) = {model_parallel_size}"
            )

        derived_dp = total_gpus // model_parallel_size
        gb = self.batch.global_batch_size
        mb = self.batch.micro_batch_size
        divisor = mb * derived_dp
        if gb % divisor != 0:
            raise ValidationError(
                f"global_batch_size ({gb}) must be divisible by "
                f"micro_batch_size ({mb}) * data_parallel_size ({derived_dp}) = {divisor}"
            )

        if ep is not None:
            dp_cp = derived_dp * cp
            if dp_cp % ep != 0:
                raise ValidationError(
                    f"(data_parallel_size * context_parallel_size) ({dp_cp}) "
                    f"must be divisible by expert_parallel_size ({ep})"
                )
            if ep > 1 and tp > 1 and total_gpus > 1:
                raise ValidationError(
                    f"Tensor parallelism (tensor_parallel_size={tp}) is not supported for MoE models "
                    f"when expert_parallel_size > 1 ({ep}); tensor_parallel_size must be 1."
                )

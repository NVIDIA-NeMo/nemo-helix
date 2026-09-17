# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automodel job input/output schemas (simplified JSON v1)."""

from __future__ import annotations

from typing import Literal, Self

from nemo_platform_plugin.deployment import (
    DEPLOYMENT_CONFIG_DESCRIPTION,
    DeploymentParams,
    ToolCallParams,
    reject_lora_without_lora_enabled,
)
from nemo_platform_plugin.integrations import IntegrationsSpec
from nmp.customization_common.schema import NamespacedModel
from nmp.customization_common.training.reporting import ProgressReportingConfig
from pydantic import Field, model_validator

__all__ = [
    "AutomodelJobInput",
    "AutomodelJobOutput",
    "BatchSpec",
    "DatasetSpec",
    "DeploymentParams",
    "ExportSpec",
    "LoRAParams",
    "OptimizerSpec",
    "OutputRequest",
    "OutputResponse",
    "ParallelismSpec",
    "RetrievalSpec",
    "ScheduleSpec",
    "ToolCallParams",
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
    rank: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="LoRA dropout probability for regularization.")
    merge: bool = False
    target_modules: list[str] | None = None
    exclude_modules: list[str] | None = Field(
        default=None, description="Module name patterns to exclude from LoRA (e.g. ['*.out_proj'])."
    )
    use_triton: bool = Field(default=True, description="Use the optimized Triton LoRA kernel.")


class DatasetSpec(AutomodelSchema):
    training: str = Field(description="Training fileset as 'name' or 'workspace/name'.")
    validation: str | None = None
    prompt_template: str | None = None


class ExportSpec(AutomodelSchema):
    """ONNX export and fileset layout. ``primary`` is the artifact at the root; the other is under ``alternates/``."""

    primary: Literal["onnx", "hf"] = Field(
        default="onnx", description="Artifact at the fileset root. Use 'hf' when the NIM loads PyTorch weights."
    )
    opset: int = Field(default=17, gt=0)
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

    train_n_passages: int = Field(default=5, ge=2)
    eval_negative_size: int | None = Field(default=None, ge=1)
    do_gradient_checkpointing: bool = False
    query_max_length: int = Field(default=512, ge=1)
    passage_max_length: int = Field(default=512, ge=1)
    query_prefix: str = Field(default="query:", description="Collator-side prefix; BiEncoderCollator adds a space.")
    passage_prefix: str = Field(default="passage:", description="Collator-side prefix; BiEncoderCollator adds a space.")
    export: ExportSpec | None = Field(
        default=None, description="Artifact layout and ONNX export settings. Defaults are applied when omitted."
    )


class TrainingSpec(AutomodelSchema):
    model_config = AutomodelSchema.model_config | {"populate_by_name": True}

    training_type: Literal["sft", "distillation"] = "sft"
    recipe: Literal["auto", "sft", "bi_encoder", "cross_encoder"] = Field(
        default="auto",
        description=(
            "Training recipe. 'auto' uses the model checkpoint head; explicit encoder recipes can wrap a causal-LM "
            "backbone for retrieval training."
        ),
    )
    finetuning_type: Literal["lora", "all_weights", "lora_merged"] = "lora"
    lora: LoRAParams | None = None
    max_seq_length: int = Field(default=2048, gt=0)
    precision: Literal["bf16", "fp16", "fp32", "fp8"] | None = Field(
        default=None,
        description="Model precision for training. Auto-detected from the checkpoint when unset.",
    )
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = Field(
        default="sdpa",
        description="Attention backend: 'sdpa' (PyTorch native), 'flash_attention_2', or 'eager'.",
    )
    execution_profile: str | None = Field(default=None, min_length=1)
    teacher_model: str | None = None
    distillation_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    distillation_temperature: float = Field(default=1.0, gt=0.0)
    teacher_precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    offload_teacher: bool = False
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
    epochs: int = Field(default=1, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    val_check_interval: float | None = None
    validation_split: float | None = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="Validation split to use when a validation dataset is not provided.",
    )
    seed: int | None = None
    progress_reporting: ProgressReportingConfig = Field(default_factory=ProgressReportingConfig)


# (global_batch_size, micro_batch_size) per retrieval recipe. bi_encoder takes its
# in-batch negatives from the micro batch, which accumulation does not widen, so
# lowering micro costs retrieval quality; cross_encoder scores pairs independently.
RETRIEVAL_BATCH_DEFAULTS: dict[str, tuple[int, int]] = {
    "bi_encoder": (256, 8),
    "cross_encoder": (128, 8),
}


class BatchSpec(AutomodelSchema):
    global_batch_size: int = Field(default=8, gt=0)
    micro_batch_size: int = Field(default=1, gt=0)
    sequence_packing: bool = False
    sequence_packing_max_samples: int = Field(
        default=1000, gt=0, description="Samples analyzed to estimate the optimal pack size when packing is enabled."
    )


class OptimizerSpec(AutomodelSchema):
    learning_rate: float = Field(default=5e-6, gt=0.0)
    min_learning_rate: float | None = Field(
        default=None, ge=0.0, description="Minimum learning rate for the cosine decay schedule."
    )
    weight_decay: float = Field(default=0.01, ge=0.0)
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0, description="Adam optimizer beta1.")
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0, description="Adam optimizer beta2.")
    warmup_steps: int = Field(default=0, ge=0)
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
    num_nodes: int = Field(default=1, gt=0)
    num_gpus_per_node: int = Field(default=1, gt=0)
    tensor_parallel_size: int = Field(default=1, gt=0)
    pipeline_parallel_size: int = Field(default=1, gt=0)
    context_parallel_size: int = Field(default=1, gt=0)
    expert_parallel_size: int | None = Field(default=None, gt=0)
    sequence_parallel: bool = Field(default=False, description="Enable sequence parallelism.")


class OutputRequest(AutomodelSchema):
    name: str
    description: str | None = None


class OutputResponse(AutomodelSchema):
    name: str
    type: Literal["model", "adapter"]
    fileset: str
    description: str | None = None


class AutomodelJobInput(AutomodelSchema):
    """POST body / CLI JSON."""

    name: str | None = None
    model: str
    dataset: DatasetSpec
    training: TrainingSpec
    schedule: ScheduleSpec = Field(default_factory=ScheduleSpec)
    batch: BatchSpec = Field(default_factory=BatchSpec)
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    parallelism: ParallelismSpec = Field(default_factory=ParallelismSpec)
    output: OutputRequest | None = None
    integrations: IntegrationsSpec | None = None
    deployment_config: str | DeploymentParams | None = Field(
        default=None,
        description=DEPLOYMENT_CONFIG_DESCRIPTION,
    )

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_fields(cls, data: object) -> object:
        if isinstance(data, dict) and "output_model" in data:
            raise ValueError("spec.output_model was removed. Use spec.output instead.")
        return data

    def trains_standalone_lora_adapter(self) -> bool:
        """True when the job produces a LoRA adapter rather than a full-weight model.

        ``merge=True`` folds the adapter back into the base weights, which yields a
        standalone model; only the unmerged case needs a LoRA-enabled deployment.
        """
        lora = self.training.lora
        return self.training.finetuning_type == "lora" and lora is not None and not lora.merge

    @model_validator(mode="after")
    def _reject_lora_without_lora_enabled(self) -> Self:
        reject_lora_without_lora_enabled(
            self.deployment_config,
            trains_lora_adapter=self.trains_standalone_lora_adapter(),
        )
        return self

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
    deployment_config: str | DeploymentParams | None = Field(
        default=None,
        description=DEPLOYMENT_CONFIG_DESCRIPTION,
    )

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

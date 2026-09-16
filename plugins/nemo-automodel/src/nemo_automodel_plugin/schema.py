# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automodel job input/output schemas (simplified JSON v1)."""

from __future__ import annotations

from typing import Literal, Self

from nemo_helix_plugin.deployment import (
    DEPLOYMENT_CONFIG_DESCRIPTION,
    DeploymentParams,
    ToolCallParams,
    reject_lora_without_lora_enabled,
)
from nemo_helix_plugin.integrations import IntegrationsSpec
from nhx.customization_common.schema import NamespacedModel
from nhx.customization_common.training.reporting import ProgressReportingConfig
from pydantic import Field, model_validator

__all__ = [
    "AutomodelJobInput",
    "AutomodelJobOutput",
    "BackendSpec",
    "BatchSpec",
    "DatasetSpec",
    "DeploymentParams",
    "ExportSpec",
    "LoRAParams",
    "MTPSpec",
    "OptimizerSpec",
    "OutputRequest",
    "OutputResponse",
    "ParallelismSpec",
    "PipelineSpec",
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
    use_memory_efficient_lora: bool | None = Field(
        default=None,
        description="Use Automodel's lower-memory LoRA path. Recommended for large MoE checkpoints. Omit to use Automodel's default, which is on.",
    )

    @model_validator(mode="after")
    def _module_filters_are_mutually_exclusive(self) -> Self:
        """Automodel's PeftConfig takes one filter or the other, never both.

        It raises "target_modules and exclude_modules are mutually exclusive" inside the
        training container, which is an expensive place to learn the job spec named both.
        """
        if self.target_modules and self.exclude_modules:
            raise ValueError(
                "lora.target_modules and lora.exclude_modules are mutually exclusive; "
                "name the modules to adapt, or the ones to skip, but not both."
            )
        return self


class DatasetSpec(AutomodelSchema):
    training: str = Field(
        description="Training fileset as 'name', 'workspace/name', or either form with a '#path/' directory."
    )
    validation: str | None = None
    prompt_template: str | None = None
    shuffle: bool = Field(
        default=True,
        description="Reshuffle training examples each epoch. Disable only for a deliberate curriculum order.",
    )


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
    do_distributed_inbatch_negative: bool = Field(
        default=False,
        description=(
            "Score each query against every passage in the global batch rather than only its own "
            "train_n_passages. Ignored for cross_encoder."
        ),
    )
    query_max_length: int = Field(default=512, ge=1)
    passage_max_length: int = Field(default=512, ge=1)
    query_prefix: str = Field(
        default="query: ",
        description="Literal prefix prepended to each query. Empty string disables prefixing.",
    )
    passage_prefix: str = Field(
        default="passage: ",
        description="Literal prefix prepended to each passage. Empty string disables prefixing.",
    )
    export: ExportSpec | None = Field(
        default=None, description="Artifact layout and ONNX export settings. Defaults are applied when omitted."
    )


class MTPSpec(AutomodelSchema):
    """Multi-Token Prediction overrides for checkpoints trained with MTP heads, e.g. Nemotron 3.5 Lightning.

    Automodel builds MTP from the checkpoint's own config; each field here overrides one
    setting, and only the fields that are set reach the model. Support differs per model.
    """

    num_nextn_predict_layers: int | None = Field(
        default=None, gt=0, description="How many tokens ahead to predict. Omit to keep the checkpoint's own value."
    )
    use_repeated_layer: bool | None = Field(
        default=None,
        description=(
            "Share one weight-tied layer across the prediction depths. Only some Automodel models accept it "
            "(Nemotron-H in r0.6.0); omit it for others."
        ),
    )
    loss_scaling_factor: float | None = Field(
        default=None,
        ge=0.0,
        description="Weight of the MTP loss term relative to the main loss. Omit to keep the model's default (0.1).",
    )


class BackendSpec(AutomodelSchema):
    """Which implementation Automodel uses for each model component.

    Every field defaults to ``None``, meaning "leave it to Automodel". Its own defaults
    depend on what the training node has available (Transformer Engine, DeepEP, CUDA), so
    only explicitly set values are forwarded.
    """

    attn: Literal["te", "sdpa", "flex", "eager", "tilelang"] | None = Field(
        default=None,
        description="Attention kernel. 'te' uses Transformer Engine (Hopper or newer); "
        "'sdpa' is the portable PyTorch implementation.",
    )
    linear: Literal["torch", "te", "quack"] | None = Field(default=None, description="Linear-layer kernel.")
    rms_norm: Literal["torch", "torch_fp32", "te", "quack"] | None = Field(
        default=None, description="RMSNorm kernel. 'torch_fp32' normalises in fp32 for numerical stability."
    )
    rope: Literal["torch", "quack"] | None = Field(default=None, description="Rotary position embedding kernel.")
    rope_fusion: bool | None = Field(default=None, description="Fuse the rotary embedding into the attention kernel.")
    experts: Literal["torch", "te", "gmm", "torch_mm", "torch_mm_mxfp8"] | None = Field(
        default=None,
        description="MoE expert compute kernel. 'gmm' is the grouped-matmul path used by the large MoE recipes.",
    )
    dispatcher: Literal["torch", "deepep", "hybridep", "uccl_ep"] | None = Field(
        default=None,
        description="How MoE tokens are routed between expert-parallel ranks. 'deepep' is the "
        "high-throughput path and requires the DeepEP library on the node.",
    )
    fake_balanced_gate: bool | None = Field(
        default=None,
        description="Route tokens evenly across experts instead of using the trained router. Benchmarking aid.",
    )
    enable_hf_state_dict_adapter: bool | None = Field(
        default=None,
        description="Save and load checkpoints in HuggingFace layout. Needed when the trained model "
        "is consumed by HuggingFace tooling.",
    )
    enable_fsdp_optimizations: bool | None = Field(
        default=None, description="Enable Automodel's additional FSDP2 sharding optimizations."
    )


class PipelineSpec(AutomodelSchema):
    """How work is scheduled across pipeline stages. Read only when pipeline_parallel_size > 1."""

    pp_schedule: str | None = Field(
        default=None,
        description="Pipeline schedule name, e.g. '1f1b', 'interleaved1f1b', 'gpipe'. Defaults to 'interleaved1f1b'.",
    )
    pp_microbatch_size: int | None = Field(
        default=None, gt=0, description="Micro-batch size flowing through each pipeline stage."
    )
    round_virtual_stages_to_pp_multiple: Literal["up", "down"] | None = Field(
        default=None,
        description="Round the number of virtual stages to a multiple of the pipeline size, in the given direction.",
    )
    scale_grads_in_schedule: bool | None = Field(
        default=None, description="Scale gradients inside the schedule rather than afterwards."
    )
    patch_inner_model: bool | None = Field(
        default=None, description="Apply Automodel's pipeline patch to the inner transformer module."
    )
    patch_causal_lm_model: bool | None = Field(
        default=None, description="Apply Automodel's pipeline patch to the causal-LM wrapper."
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
    activation_checkpointing: bool = Field(
        default=False,
        description="Recompute intermediate activations in the backward pass instead of storing them. "
        "Substantially lowers memory use for a modest slowdown, and is commonly enabled when training "
        "large MoE models.",
    )
    mtp: MTPSpec | None = Field(
        default=None,
        description="Multi-Token Prediction settings. Omit for checkpoints trained without MTP heads.",
    )
    backend: BackendSpec | None = Field(
        default=None,
        description="Low-level kernel selection for the model's components. Omit to let Automodel "
        "choose based on the hardware it lands on.",
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
    checkpoint_selection: Literal["best", "last", "both"] = Field(
        default="best",
        description=(
            "Checkpoint(s) to publish: 'best' selects the lowest validation loss, "
            "'last' preserves the end of training, and 'both' publishes best at the root "
            "with last under alternates/last."
        ),
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
    packed_sequence_size: int | None = Field(
        default=None,
        gt=0,
        description="Pin the packed sequence length instead of estimating it. Requires sequence_packing.",
    )

    @model_validator(mode="after")
    def _explicit_pack_size_needs_packing(self) -> Self:
        # Automodel reads packed_sequence_size only when packing is on, so a size set
        # against packing=false is silently ignored -- and the run is slower than asked for.
        if self.packed_sequence_size is not None and not self.sequence_packing:
            raise ValueError("batch.packed_sequence_size requires batch.sequence_packing=true.")
        return self


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
    pipeline: PipelineSpec | None = Field(
        default=None,
        description="Pipeline schedule settings. Only read when pipeline_parallel_size is greater than 1.",
    )


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

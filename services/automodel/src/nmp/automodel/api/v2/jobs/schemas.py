# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API schemas for customization job endpoints."""

from typing import Annotated, Any, Dict, Literal, Optional, Self, Union

from nemo_platform_plugin.deployment import DeploymentParams
from nemo_platform_plugin.integrations import IntegrationsSpec
from nmp.automodel.entities.validators import validate_fileset_uri
from nmp.automodel.entities.values import CheckpointSelection, FinetuningType, OutputNameType, Precision
from nmp.common.entities.constants import (
    MAX_LENGTH_255,
    REGEX_WORD_CHARACTER_DOT_DASH,
)
from nmp.customization_common.training.reporting import ProgressReportingConfig
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    model_validator,
)

# Important!!! Do not import Pydantic models from this file into tasks.
# Instead, duplicate models from this file into corresponding task module schemas.py.


class ValidationError(ValueError):
    """Raised when job input validation fails."""

    pass


# ============================================================
# Sub-Configurations
# ============================================================


class QuantizationParams(BaseModel):
    """Base model quantization for memory-efficient PEFT training.

    Supports two scenarios:
    - Full-precision base model: quantized on-the-fly at load time
    - Pre-quantized base model: loaded directly at the specified precision

    In both cases, base model weights are frozen and only the PEFT adapter
    parameters are trained in full precision.
    """

    precision: Literal["4bit", "8bit"] = Field(
        default="4bit",
        description="Quantization precision. '4bit' (NF4) for maximum memory savings, "
        "'8bit' (LLM.int8) for a balance of quality and memory.",
    )


class _PEFTParams(BaseModel):
    """Base configuration shared by all PEFT methods."""

    # Quantization only makes sense with PEFT (quantized base weights are frozen, so you need trainable
    # adapter parameters), which is why it lives here rather than on _TrainingBase.
    quantization: Optional[QuantizationParams] = Field(
        default=None,
        description="Enable quantized training to reduce GPU memory. "
        "If the base model is full-precision, it will be quantized at load time. "
        "If the base model is already pre-quantized, this configures the expected precision. "
        "The trained adapter remains full-precision.",
    )


class LoRAParams(_PEFTParams):
    """LoRA adapter configuration."""

    type: Literal["lora"] = "lora"

    rank: int = Field(
        default=8,
        ge=1,
        le=256,
        description="LoRA rank (low-rank dimension). Higher values increase capacity but use more memory.",
    )
    alpha: int = Field(
        default=32,
        ge=1,
        description="LoRA alpha scaling factor. Common practice: alpha = 2-4x rank.",
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LoRA dropout probability for regularization.",
    )
    target_modules: Optional[list[str]] = Field(
        default=None,
        description="Module name patterns to apply LoRA to (e.g., ['*.q_proj', '*.v_proj']). "
        "If not set, applies to all '*proj' linear layers.",
    )
    exclude_modules: Optional[list[str]] = Field(
        default=None,
        description="Module name patterns to exclude from LoRA (e.g., ['*.out_proj']).",
    )
    use_triton: bool = Field(
        default=True,
        description="Use the optimized Triton LoRA kernel.",
    )
    merge: bool = Field(
        default=False,
        description="Merge LoRA weights into base model after training. "
        "Produces a full-weight checkpoint instead of an adapter.",
    )
    use_dora: bool = Field(
        default=False,
        description="Enable DoRA (Weight-Decomposed Low-Rank Adaptation). "
        "Decomposes weight updates into magnitude and direction components. "
        "Can improve quality especially at low ranks, but adds training overhead.",
    )

    @model_validator(mode="after")
    def _validate_unsupported_features(self) -> Self:
        if self.quantization is not None:
            raise ValueError("Quantized LoRA training is not yet supported.")
        if self.use_dora:
            raise ValueError("DoRA is not yet supported.")
        return self


# When a second PEFT method is added (e.g., IA3Config), change this to:
#   PeftMethod = Annotated[Union[LoRAParams, IA3Config], Discriminator("type")]
PeftMethod = LoRAParams


class ExportParams(BaseModel):
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
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = Field(
        default="eager",
        description="Attention backend for the traced model. The exporter cannot trace SDPA/GQA.",
    )
    pooling: Literal["avg", "cls", "last"] = Field(
        default="avg", description="Embedding pooling over hidden states. Ignored for cross_encoder."
    )
    normalize: bool = Field(default=True, description="L2-normalize pooled embeddings. Ignored for cross_encoder.")
    dimensions: bool = Field(
        default=False,
        description="Add a Matryoshka 'dimensions' input that truncates and renormalizes embeddings.",
    )


class RetrievalParams(BaseModel):
    """Collator, dataset, and export knobs for ``bi_encoder`` / ``cross_encoder``."""

    train_n_passages: int = Field(default=5, ge=2, description="Passages per query: 1 positive + (n-1) negatives.")
    eval_negative_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Negatives per query at eval. Defaults to train_n_passages - 1.",
    )
    do_gradient_checkpointing: bool = Field(default=False)
    do_distributed_inbatch_negative: bool = Field(
        default=False,
        description=(
            "Score each query against every passage in the global batch instead of only its own "
            "train_n_passages. Widens the negative pool to num_gpus * micro_batch_size * train_n_passages "
            "at the cost of an all-gather per step. Ignored for cross_encoder."
        ),
    )
    query_max_length: int = Field(default=512, ge=1)
    passage_max_length: int = Field(default=512, ge=1)
    query_prefix: str = Field(default="query:", description="Collator-side prefix; do not include a trailing space.")
    passage_prefix: str = Field(
        default="passage:", description="Collator-side prefix; do not include a trailing space."
    )
    export: Optional[ExportParams] = Field(
        default=None,
        description="Output artifact layout and ONNX export settings. Defaults are applied when omitted.",
    )


# (batch_size, micro_batch_size) defaults per retrieval recipe.
RETRIEVAL_BATCH_DEFAULTS: dict[str, tuple[int, int]] = {
    "bi_encoder": (256, 8),
    "cross_encoder": (128, 8),
}


class ParallelismParams(BaseModel):
    """Distributed training parallelism configuration.

    Most users only need num_gpus_per_node. Advanced users can configure
    tensor/pipeline/context/expert parallelism for large models.
    """

    num_gpus_per_node: int = Field(default=1, gt=0, description="Number of gpus per node.")
    num_nodes: int = Field(default=1, gt=0, description="Number of nodes.")
    tensor_parallel_size: int = Field(default=1, gt=0, description="Tensor parallel size.")
    pipeline_parallel_size: int = Field(default=1, gt=0, description="Pipeline parallel size.")
    context_parallel_size: int = Field(default=1, gt=0, description="Context parallel size.")
    expert_parallel_size: Optional[int] = Field(default=None, gt=0, description="Expert parallel size (MoE models).")
    sequence_parallel: bool = Field(default=False, description="Enable sequence parallelism.")


# ============================================================
# Training Method Discriminated Union
# ============================================================


class _TrainingBase(BaseModel):
    """Common training configuration shared by all methods.

    Flat hyperparameters match the ML practitioner mental model
    (like HuggingFace TrainingArguments / TRL SFTConfig).
    Only parallelism is grouped — it's enterprise infrastructure.
    """

    recipe: Literal["auto", "sft", "bi_encoder", "cross_encoder"] = Field(
        default="auto",
        description="Training recipe; explicit encoder recipes may wrap a causal-LM checkpoint.",
    )

    # --- PEFT (orthogonal to training method) ---
    peft: Optional[PeftMethod] = Field(
        default=None,
        description="PEFT adapter configuration. If set, trains a parameter-efficient adapter. "
        "If omitted, performs full-weight fine-tuning.",
    )

    # --- Optimizer ---
    learning_rate: float = Field(
        default=1e-4,
        description="Peak learning rate. Optimal value will depend on training type and PEFT. "
        "For SFT without LoRA, start with 5e-5. If using LoRA start with 1e-4.  Lowering the value "
        "can enable for slower, more precise training; Raising the value speeds up learning.",
    )
    min_learning_rate: Optional[float] = Field(
        default=None,
        description="Minimum learning rate for cosine decay. Optional; used with learning rate schedules.",
    )
    weight_decay: float = Field(
        default=0.01,
        description="Weight decay coefficient. Helps prevent overfitting.",
    )
    adam_beta1: float = Field(
        default=0.9,
        description="Adam beta1 parameter. Adjust for optimizer tuning.",
    )
    adam_beta2: float = Field(
        default=0.999,
        description="Adam beta2 parameter. Adjust for optimizer tuning.",
    )
    adam_eps: float = Field(
        default=1e-8,
        gt=0.0,
        description="Adam/AdamW epsilon for numerical stability.",
    )
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Linear warmup steps. Recommended: 10% of total training steps for stable training.",
    )
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

    # --- Schedule ---
    epochs: int = Field(
        default=1,
        gt=0,
        description="Number of complete passes through the dataset. The ideal number of epochs depends "
        "on the training method, the number of training samples, and size of the model. Start with 3 for "
        "a reasonable value. Monitor the validation and training loss curves. If both are still "
        "decreasing, you can increase this number.",
    )
    max_steps: Optional[int] = Field(
        default=None,
        gt=0,
        description="Max training steps. Overrides epochs if set.",
    )
    val_check_interval: Optional[float] = Field(
        default=None,
        description="Validation interval. Float <= 1.0 is fraction of epoch; > 1.0 is step count.",
    )
    validation_split: Optional[float] = Field(
        default=0.1,
        gt=0,
        lt=1,
        description="Validation split to use when a validation dataset is not provided.",
    )
    checkpoint_selection: CheckpointSelection = Field(
        default=CheckpointSelection.BEST,
        description=(
            "Checkpoint(s) to publish: 'best' selects the lowest validation loss, "
            "'last' preserves the end of training, and 'both' publishes best at the root "
            "with last under alternates/last."
        ),
    )
    # `log_every_n_steps` used to sit here, described as "Logging frequency in steps.
    # Controls how often training metrics are logged." It controlled nothing: no
    # code read it, it never reached the recipe config, and it was absent from the
    # submitter-facing plugin schema and from every generated spec, so no request
    # could set it in the first place. `progress_reporting` below is the knob it
    # was reaching for. Removing it is invisible -- this model ignores extras, so a
    # stored spec that still carries the key parses exactly as it did before.
    progress_reporting: ProgressReportingConfig = Field(default_factory=ProgressReportingConfig)

    # --- Batch ---
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Global batch size across all GPUs. Higher = faster but more memory. If OOM, reduce this first.",
    )
    micro_batch_size: int = Field(
        default=1,
        gt=0,
        description="Per-GPU micro batch size. Keep small (1-2) for large models to avoid OOM.",
    )
    sequence_packing: bool = Field(
        default=False,
        description="Enable sequence packing for efficiency. Can improve training speed.",
    )
    sequence_packing_max_samples: int = Field(
        default=1000,
        gt=0,
        description="Samples analyzed to estimate the optimal pack size when sequence packing is enabled.",
    )

    # --- Model ---
    max_seq_length: int = Field(
        default=2048,
        gt=0,
        description="Maximum token sequence length for training. Higher = more memory, longer training.",
    )
    precision: Optional[Precision] = Field(
        default=None,
        description="Model precision for training. Auto-detected if unset.",
    )
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = Field(
        default="sdpa",
        description="Attention backend: 'sdpa' (PyTorch native), 'flash_attention_2', or 'eager'.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. Optional.",
    )
    retrieval: Optional[RetrievalParams] = Field(
        default=None,
        description="Retrieval dataset, collator, and export knobs. Used when recipe is bi_encoder or cross_encoder.",
    )

    # --- Enterprise Infrastucture ---
    parallelism: ParallelismParams = Field(default_factory=ParallelismParams)
    execution_profile: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Execution profile for the GPU training step. Maps to an operator-configured profile "
        "(e.g., 'a100', 'high_priority'). If omitted, uses the service-level default.",
    )

    model_config = {"protected_namespaces": (), "populate_by_name": True}

    def with_resolved_recipe(self, recipe: str) -> Self:
        """Return this training config with its resolved recipe defaults."""
        if recipe == "bi_encoder":
            lr, warmup = 1e-5, 5
            batch, micro_batch = RETRIEVAL_BATCH_DEFAULTS["bi_encoder"]
        elif recipe == "cross_encoder":
            lr, warmup = 3e-6, 100
            batch, micro_batch = RETRIEVAL_BATCH_DEFAULTS["cross_encoder"]
        else:
            return self.model_copy(update={"recipe": recipe})
        if getattr(self, "type", None) == "distillation":
            raise ValueError("Knowledge distillation only supports the sft recipe.")
        updates: dict[str, object] = {"recipe": recipe}
        if "batch_size" not in self.model_fields_set:
            updates["batch_size"] = batch
        if "micro_batch_size" not in self.model_fields_set:
            updates["micro_batch_size"] = micro_batch
        if "learning_rate" not in self.model_fields_set:
            updates["learning_rate"] = lr
        if "warmup_steps" not in self.model_fields_set:
            updates["warmup_steps"] = warmup
        return self.model_copy(update=updates)

    @property
    def finetuning_type(self) -> FinetuningType:
        """Derived from peft config: presence → adapter type, absence → full-weight."""
        if self.peft is None:
            return FinetuningType.ALL_WEIGHTS
        if isinstance(self.peft, LoRAParams):
            return FinetuningType.LORA_MERGED if self.peft.merge else FinetuningType.LORA
        raise ValueError(f"Unknown PEFT type: {type(self.peft).__name__}")


class SFTTraining(_TrainingBase):
    """Supervised Fine-Tuning."""

    type: Literal["sft"] = "sft"


class DistillationTraining(_TrainingBase):
    """Knowledge Distillation with a teacher model.

    Customizer's differentiator — not available in Unsloth.
    Trains the student model to match the teacher's output distribution.
    """

    type: Literal["distillation"] = "distillation"
    teacher_model: str = Field(
        description="Teacher model URN (e.g., 'workspace/model-name'). "
        "Must have the same vocabulary as the student model.",
    )
    teacher_precision: Literal["bf16", "fp16", "fp32"] = Field(
        default="bf16",
        description="Precision for loading the frozen teacher model. "
        "Lower precision reduces memory but may affect logit quality.",
    )
    distillation_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Balance between CE loss and KD loss. 0.0 = CE only, 1.0 = KD only.",
    )
    distillation_temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Softmax temperature for KD. Higher = softer probability distributions.",
    )


class DPOTraining(_TrainingBase):
    """Direct Preference Optimization."""

    type: Literal["dpo"] = "dpo"
    ref_policy_kl_penalty: float = Field(
        default=0.05, ge=0.0, description="KL penalty coefficient (beta in DPO paper)."
    )
    preference_average_log_probs: bool = Field(
        default=False, description="Average log probabilities for preference loss calculation."
    )
    sft_average_log_probs: bool = Field(
        default=False, description="Average log probabilities for SFT regularization loss."
    )
    preference_loss_weight: float = Field(default=1.0, ge=0.0, description="Weight for the preference (DPO) loss term.")
    sft_loss_weight: float = Field(
        default=0.0, ge=0.0, description="Weight for SFT regularization loss (0 = disabled)."
    )
    max_grad_norm: float = Field(default=1.0, ge=0.0, description="Maximum gradient norm for clipping.")

    @model_validator(mode="after")
    def _peft_not_yet_supported(self) -> Self:
        if self.peft is not None:
            raise ValueError(
                "PEFT is not yet supported with DPO training. Use full-weight training by omitting the 'peft' field."
            )
        return self


AnyTraining = Union[SFTTraining, DistillationTraining, DPOTraining]
TrainingMethod = Annotated[AnyTraining, Discriminator("type")]


# ============================================================
# Output
# ============================================================


class _OutputBase(BaseModel):
    """Shared fields for output artifact request and response."""

    name: str = Field(
        pattern=REGEX_WORD_CHARACTER_DOT_DASH,
        max_length=MAX_LENGTH_255,
        description="Name of the output artifact. Used to identify it during deployment and inference.",
        examples=["my-finetuned-llama", "llama-3-8b-lora-v2"],
    )


class OutputRequest(_OutputBase):
    """Output artifact configuration provided by the user."""


class OutputResponse(_OutputBase):
    """Resolved output artifact details returned by the server."""

    type: OutputNameType = Field(
        description="Output artifact type. Either `model` (full fine-tuned weights) or `adapter` (LoRA adapter weights).",
        examples=["model", "adapter"],
    )
    fileset: str = Field(
        pattern=REGEX_WORD_CHARACTER_DOT_DASH,
        max_length=MAX_LENGTH_255,
        description="FileSet name where output artifacts are stored.",
        examples=["my-model-a1b2c3d4e5f6"],
    )


# ============================================================
# Job Schemas
# ============================================================


class _CustomizationJobBase(BaseModel):
    """Base schema with common fields for customization jobs."""

    model: str = Field(description="Model reference (e.g., 'workspace/model-name').")
    dataset: Annotated[str, AfterValidator(validate_fileset_uri)] = Field(
        description="Training dataset fileset as 'workspace/name' or 'name' (resolved in the job path workspace)."
    )
    training: TrainingMethod = Field(description="Training method and hyperparameters.")
    integrations: Optional[IntegrationsSpec] = Field(
        default=None,
        description="Third-party integrations (e.g., Weights & Biases, MLflow).",
    )
    deployment_config: Optional[str | DeploymentParams] = Field(
        default=None,
        description="Deployment configuration for auto-deploying the model after training. "
        "Pass a string to reference an existing ModelDeploymentConfig by name "
        "(e.g., 'my-config' or 'workspace/my-config'). "
        "An object provides inline NIM deployment parameters. "
        "Omit to skip deployment.",
    )
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom user-defined fields.")

    model_config = ConfigDict(protected_namespaces=(), regex_engine="python-re")


class CustomizationJobInput(_CustomizationJobBase):
    """Input schema for creating customization jobs."""

    output: Optional[OutputRequest] = Field(
        default=None,
        description="Output artifact configuration. If omitted, name is auto-generated as "
        "`{model}-{dataset}-<random-hex>`. The output type (model vs adapter) is always "
        "inferred from the training configuration.",
        examples=[{"name": "my-finetuned-llama"}],
    )

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_fields(cls, data: object) -> object:
        if isinstance(data, dict) and "output_model" in data:
            raise ValueError("spec.output_model was removed. Use spec.output instead.")
        return data

    @model_validator(mode="after")
    def _reject_lora_without_lora_enabled(self) -> Self:
        peft = self.training.peft
        dc = self.deployment_config
        if isinstance(peft, LoRAParams) and not peft.merge and isinstance(dc, DeploymentParams) and not dc.lora_enabled:
            raise ValueError(
                "deployment_config.lora_enabled must be true (or omitted) when training a LoRA adapter. "
                "Setting lora_enabled=false would deploy the base model without LoRA support, "
                "making the trained adapter unservable."
            )
        return self


class CustomizationJobOutput(_CustomizationJobBase):
    """Customization job details returned by the server."""

    output: OutputResponse = Field(
        description="Output artifact created by this job.",
        examples=[
            {"name": "my-finetuned-llama", "type": "model", "fileset": "my-finetuned-llama"},
            {"name": "llama-3-8b-lora-v2", "type": "adapter", "fileset": "llama-3-8b-lora-v2-a1b2c3d4e5f6"},
        ],
    )

    def validate_for_training(self) -> None:
        """Validate this job input for training execution.

        Call this after any enrichment has been applied.

        Raises:
            ValidationError: If validation fails.
        """
        training = self.training
        p = training.parallelism
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
                f"tensor_parallel_size ({tp}) * "
                f"pipeline_parallel_size ({pp}) * "
                f"context_parallel_size ({cp}) = {model_parallel_size}"
            )

        derived_dp = total_gpus // model_parallel_size

        # Note: Expert model parallelism (EP) is NOT a dimension that divides world_size like TP/PP.
        # Instead, EP operates orthogonally, therefore we validate it separately.
        # It distributes experts across the dp × cp dimension.
        # FSDP2 requires: (dp_size × cp_size) % ep_size == 0
        if ep is not None:
            dp_cp = derived_dp * cp
            if dp_cp % ep != 0:
                raise ValidationError(
                    f"(data_parallel_size * context_parallel_size) ({derived_dp} * {cp} = {dp_cp}) "
                    f"must be divisible by expert_parallel_size ({ep})"
                )
            # MoE models on multi-GPU don't support tensor parallelism
            # in Automodel's MoE parallelizer. See: nemo_automodel/components/moe/parallelizer.py
            if ep > 1 and tp > 1 and total_gpus > 1:
                raise ValidationError(
                    f"Tensor parallelism (tensor_parallel_size={tp}) is not supported for MoE models. "
                    f"When expert_parallel_size > 1 ({ep}), tensor_parallel_size must be 1."
                )

        gb = training.batch_size
        mb = training.micro_batch_size
        divisor = mb * derived_dp
        if gb % divisor != 0:
            raise ValidationError(
                f"batch_size ({gb}) must be divisible by "
                f"micro_batch_size ({mb}) * data_parallel_size ({derived_dp}) = {divisor}. "
                f"Consider adjusting batch_size to {divisor * max(1, gb // divisor)} or {divisor * (gb // divisor + 1)}."
            )

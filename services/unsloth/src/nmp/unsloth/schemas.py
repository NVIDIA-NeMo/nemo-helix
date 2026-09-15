# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Unsloth schemas — consumed by ``train_sft`` (and ``compile`` later).

Why these live in the service, not the plugin:

- Both compile-time (``compile.platform_job_config_compiler``, when wired
  later) and runtime (``train_sft``) consume the canonical shape. The
  container entrypoint reads it from the platform Jobs envelope.
- The plugin's ``transform.py`` produces ``UnslothJobOutput`` from
  ``UnslothJobInput`` (which lives in the plugin) — but the *output*
  type is what flows downstream, so it belongs with the downstream
  consumers.

Each field maps onto a real argument of one of three call-sites the
training driver hits:

- ``unsloth.FastLanguageModel.from_pretrained(...)``  → :class:`ModelLoadSpec`
- ``unsloth.FastLanguageModel.get_peft_model(...)``   → :class:`LoRAParams`
- ``trl.SFTConfig`` + ``trl.SFTTrainer(...)`` →
  :class:`ScheduleSpec`, :class:`BatchSpec`, :class:`OptimizerSpec`,
  :class:`HardwareSpec`, :class:`IntegrationsSpec`
- Output saving (``model.save_pretrained{,_merged}``) → :class:`OutputResponse`

``extra="forbid"`` everywhere — typos in the JSON shape become
validation errors, not silently-ignored fields.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from nemo_platform_plugin.integrations import IntegrationsSpec
from nmp.customization_common.schema import NamespacedModel
from nmp.customization_common.training.reporting import ProgressReportingConfig
from pydantic import Field, model_validator


class UnslothSchema(NamespacedModel):
    """Backend base: every Unsloth-owned model emits an ``Unsloth``-prefixed
    OpenAPI schema name (``TrainingSpec`` -> ``UnslothTrainingSpec``), so it
    can't collide with another backend's same-named model in the merged
    ``/apis/customization`` spec. ``extra='forbid'`` is inherited from the base.

    Defined here (with the canonical schemas) and re-imported by the plugin's
    submitter-facing module so both prefix identically."""

    __schema_namespace__ = "Unsloth"


class ModelLoadSpec(UnslothSchema):
    """Args to ``FastLanguageModel.from_pretrained``.

    ``name`` is a NeMo Platform model entity reference (``"name"`` or
    ``"workspace/name"``). The plugin's run orchestration resolves the
    entity, downloads its fileset to a local path, and hands that path
    to :func:`train_sft`.
    """

    name: str = Field(
        description=(
            "Model entity reference. Accepts 'name' (uses the job's workspace) "
            "or 'workspace/name'. The plugin's run resolves this to a local "
            "path before training."
        ),
    )
    max_seq_length: int = Field(default=2048, gt=0)
    load_in_4bit: bool = Field(
        default=True,
        description="bitsandbytes 4-bit. Mutex with load_in_8bit. Default for Unsloth's headline path.",
    )
    load_in_8bit: bool = False
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    trust_remote_code: bool = False
    device_map: str | int | dict[str, int] | None = Field(
        default=None,
        description=(
            "Device placement forwarded to FastLanguageModel.from_pretrained. "
            "Omit (null) to pin the whole model to the single visible GPU "
            "({'': 0}) — the right default for this single-GPU backend, and it "
            "avoids accelerate's auto-placement under-sizing GPU memory on "
            "unified-memory parts (e.g. GB10 / DGX Spark), which otherwise "
            "spills layers to CPU and aborts 4-bit loads. Set 'auto', "
            "'balanced', 'sequential', a device index, or a custom map for "
            "multi-device experiments."
        ),
    )
    rope_scaling: dict[str, Any] | None = Field(
        default=None,
        description=(
            "RoPE scaling config for long-context extension, passed to "
            "FastLanguageModel.from_pretrained (e.g. {'type': 'linear', 'factor': 2.0}). "
            "None uses the model's native context length."
        ),
    )


class LoRAParams(UnslothSchema):
    """Args to ``FastLanguageModel.get_peft_model``."""

    rank: int = Field(default=16, gt=0, description="LoRA rank.")
    alpha: int = Field(default=16, gt=0, description="LoRA scaling factor (alpha).")
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        # Unsloth's recommended 7-module set: full attention + MLP.
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = False
    random_state: int = 3407
    use_dora: bool = Field(
        default=False,
        description="DoRA (weight-decomposed LoRA). Improves quality at low ranks; adds training overhead.",
    )
    loftq_config: dict[str, Any] | None = Field(
        default=None,
        description="LoftQ initialization config for quantized bases. None disables LoftQ.",
    )
    modules_to_save: list[str] | None = Field(
        default=None,
        description=(
            "Extra non-LoRA modules to train and save in full (e.g. ['embed_tokens', 'lm_head']). "
            "Needed for vocab changes / continued pretraining."
        ),
    )
    layers_to_transform: int | list[int] | None = Field(
        default=None,
        description="Restrict LoRA to specific layer index(es). None applies to all layers.",
    )
    layer_replication: list[list[int]] | None = Field(
        default=None,
        description="Layer-replication ranges for stacking, e.g. [[0, 16], [8, 24]]. None disables.",
    )
    init_lora_weights: bool | Literal["gaussian", "pissa", "olora", "loftq"] = Field(
        default=True,
        description="LoRA weight init scheme. True = PEFT default; 'pissa'/'olora'/'loftq' for advanced inits.",
    )


class TrainingSpec(UnslothSchema):
    """Algorithm + adapter shape selectors."""

    training_type: Literal["sft"] = "sft"
    finetuning_type: Literal["lora", "all_weights"] = "lora"
    lora: LoRAParams | None = Field(
        default=None,
        description="Required when finetuning_type='lora'. Auto-filled with defaults if omitted.",
    )
    use_gradient_checkpointing: Literal["unsloth", "true", "false"] = "unsloth"

    @model_validator(mode="after")
    def _enforce_lora_invariant(self) -> Self:
        """Keep ``lora`` consistent with ``finetuning_type`` at the schema level.

        ``build_peft_kwargs`` (and the training driver) assume a LoRA run always
        carries a populated ``lora`` block. Enforcing it here means every path
        that builds a ``TrainingSpec`` — the plugin's ``UnslothJobInput``, a
        directly-constructed ``UnslothJobOutput``, SDK callers, tests — gets the
        invariant for free, instead of relying on a downstream ``assert``.
        """
        if self.finetuning_type == "lora" and self.lora is None:
            self.lora = LoRAParams()
        if self.finetuning_type == "all_weights" and self.lora is not None:
            raise ValueError("training.lora must be unset when finetuning_type='all_weights'")
        return self


class DatasetSpec(UnslothSchema):
    """Training data location + shape.

    ``path`` and ``validation_path`` are platform fileset references
    (``"name"`` or ``"workspace/name"``). The plugin's run downloads
    each fileset before training; ``train_sft`` only ever sees a local
    filesystem path.
    """

    path: str = Field(
        description=(
            "Training fileset reference: 'name' (uses the job's workspace) "
            "or 'workspace/name'. Resolved to a local path by the plugin run."
        ),
    )
    text_field: str = Field(default="text", description="Row field consumed by SFTTrainer.")
    apply_chat_template: bool = Field(
        default=False,
        description=(
            "If True, expects rows with a 'messages' field and applies tokenizer.apply_chat_template at training time."
        ),
    )
    validation_path: str | None = Field(
        default=None,
        description=(
            "Optional validation fileset reference (same format as 'path'). Downloaded under the same scheme."
        ),
    )
    packing: bool = Field(default=False, description="trl.SFTTrainer packing flag.")


class ScheduleSpec(UnslothSchema):
    """Training schedule, scheduler, logging cadence."""

    # Consistent with Automodel: train for ``epochs`` (default 1) unless ``max_steps``
    # is set, in which case the trainer caps training at that many steps.
    epochs: int = Field(default=1, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    warmup_steps: int = Field(default=0, ge=0)
    warmup_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "constant",
        "constant_with_warmup",
        "cosine_with_restarts",
    ] = "linear"
    # HuggingFace's own logging cadence, which drives stdout and any configured
    # W&B/MLflow run. Deliberately *not* the Jobs-service reporting cadence: a user
    # who wants verbose HF logs should not get one Jobs report per step as a side
    # effect. 1 is the right default now that a downstream gate decides what to
    # keep -- it gives that gate the finest resolution available to sample from.
    logging_steps: int = Field(default=1, gt=0)
    save_steps: int | None = Field(default=None, gt=0)
    eval_steps: int | None = Field(default=None, gt=0)
    progress_reporting: ProgressReportingConfig = Field(default_factory=ProgressReportingConfig)
    seed: int = 3407
    lr_scheduler_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Extra kwargs for the LR scheduler, e.g. {'num_cycles': 3} for cosine_with_restarts. "
            "None uses scheduler defaults."
        ),
    )


class BatchSpec(UnslothSchema):
    per_device_train_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)


class OptimizerSpec(UnslothSchema):
    learning_rate: float = Field(default=2e-4, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    # Unsloth's notebooks default to adamw_8bit (bitsandbytes-backed) — much
    # smaller optimizer state than adamw_torch, which lets users fit larger
    # adapters on the same GPU. Users on Hopper+ may prefer adamw_torch_fused.
    optim: Literal[
        "adamw_torch",
        "adamw_torch_fused",
        "adamw_8bit",
        "paged_adamw_8bit",
        "sgd",
    ] = "adamw_8bit"
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0, description="Adam/AdamW beta1.")
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0, description="Adam/AdamW beta2.")
    adam_epsilon: float = Field(default=1e-8, gt=0.0, description="Adam/AdamW epsilon for numerical stability.")
    max_grad_norm: float = Field(default=1.0, ge=0.0, description="Gradient-clipping max norm (TRL default 1.0).")
    label_smoothing_factor: float = Field(
        default=0.0, ge=0.0, lt=1.0, description="Label smoothing for the cross-entropy loss. 0.0 disables."
    )
    neftune_noise_alpha: float | None = Field(
        default=None,
        ge=0.0,
        description="NEFTune embedding-noise alpha (quality boost). None disables.",
    )


class HardwareSpec(UnslothSchema):
    gpus: str | None = Field(
        default=None,
        description=(
            "Comma-separated GPU indices ('0' or '0,1') for CUDA_VISIBLE_DEVICES. Selection, not reservation."
        ),
    )
    precision: Literal["bf16", "fp16"] = Field(
        default="bf16",
        description="Mixed-precision dtype for training. bf16 recommended for Ampere+.",
    )


class OutputResponse(UnslothSchema):
    """Stored on the canonical UnslothJobOutput. Output naming is resolved during ``to_spec``.

    ``type`` is the high-level shape (``adapter`` for a saved LoRA, ``model``
    for a merged checkpoint). ``save_method`` keeps the original Unsloth
    save verb so the training driver can dispatch correctly without
    re-deriving it. ``fileset`` is the platform fileset name the trained
    artefacts will be uploaded to (the plugin's ``transform`` defaults
    this to ``name``).
    """

    name: str
    type: Literal["adapter", "model"]
    save_method: Literal["lora", "merged_16bit", "merged_4bit"]
    fileset: str = Field(
        description=("Platform fileset name the trained checkpoint will be uploaded to. Defaults to the entity name."),
    )
    description: str | None = None


class ToolCallParams(UnslothSchema):
    """Tool calling configuration for NIM deployments."""

    tool_call_parser: str | None = Field(
        default=None,
        description=(
            "Name of the tool call parser to use (e.g., 'openai', 'hermes', 'pythonic', 'llama3_json', 'mistral')."
        ),
    )
    tool_call_plugin: str | None = Field(
        default=None,
        pattern=r"^[\w\-.]+/[\w\-.]+$",
        description=(
            "Reference to a fileset containing the custom tool call plugin Python file. "
            "Expected format: '{workspace}/{fileset_name}'."
        ),
    )
    auto_tool_choice: bool | None = Field(
        default=None,
        description="Whether to enable automatic tool choice.",
    )


class DeploymentParams(UnslothSchema):
    """Inline deployment parameters for auto-deploying a trained model.

    Used in :class:`UnslothJobInput.deployment_config` and passed through to
    the model_entity task at compile time. When unset, no deployment is launched.
    """

    gpu: int = Field(default=1, gt=0, description="Number of GPUs required for the deployment.")
    additional_envs: dict[str, str] | None = Field(
        default=None,
        description="Additional environment variables for the deployment.",
    )
    disk_size: str | None = Field(default=None, description="Disk size for the deployment.")
    image_name: str | None = Field(
        default=None,
        description="Container image name from NGC. If not specified, defaults to multi-llm.",
    )
    image_tag: str | None = Field(default=None, description="Container image tag from NGC.")
    lora_enabled: bool = Field(
        default=True,
        description=(
            "When auto-deploying a full SFT training, setting this true allows subsequent "
            "LoRA adapters to be deployed against it."
        ),
    )
    tool_call_config: ToolCallParams | None = Field(
        default=None,
        description="Tool calling configuration override for the NIM deployment.",
    )


class UnslothJobOutput(UnslothSchema):
    """Canonical spec stored after the plugin's ``to_spec()`` resolves output naming.

    Defaults match :class:`~nemo_unsloth_plugin.schema.UnslothJobInput` so SDK
    callers and tests can construct :class:`UnslothJobOutput` directly without
    restating every sub-section. The plugin's ``to_spec`` always passes the
    resolved input values through, so these defaults never override real input.
    """

    name: str | None = None
    model: ModelLoadSpec
    dataset: DatasetSpec
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    schedule: ScheduleSpec = Field(default_factory=ScheduleSpec)
    batch: BatchSpec = Field(default_factory=BatchSpec)
    optimizer: OptimizerSpec = Field(default_factory=OptimizerSpec)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    integrations: IntegrationsSpec | None = None
    output: OutputResponse
    deployment_config: str | DeploymentParams | None = Field(
        default=None,
        description=(
            "Deployment configuration for auto-deploying the model after training. "
            "Pass a string to reference an existing ModelDeploymentConfig by name "
            "('my-config' or 'workspace/my-config'). An object provides inline NIM "
            "deployment parameters. Omit to skip deployment."
        ),
    )

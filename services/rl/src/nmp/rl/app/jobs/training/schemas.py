# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-agnostic training-step config consumed by the container runner.

The compiler serializes a :class:`TrainingStepConfig` into the training
``PlatformJobStep``; the runner deserializes it and ``dpo_config.compile_dpo_config``
turns it into the NeMo-RL YAML.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from nmp.customization_common.training.reporting import ProgressReportingConfig
from nmp.rl.app.constants import DEFAULT_OUTPUT_MODEL_PATH, DEFAULT_SEED, DEFAULT_TRAINING_OUTPUT_PATH
from nmp.rl.entities.values import CheckpointFormat, FinetuningType, Precision, TrainingType
from pydantic import BaseModel, Field


class TrainingBackend(str, Enum):
    """Training backend identifier."""

    NEMO_RL = "nemo_rl"


class PolicyBackend(str, Enum):
    """Which NeMo-RL policy worker trains the model.

    The value picks the worker class, which picks the Ray actor's venv and kernels.
    Asking for a capability the backend lacks is a validation error, not a silent upgrade.
    """

    # _v2: false -> DTensorPolicyWorker. Stock HuggingFace + PyTorch FSDP2, no
    # Transformer Engine (so pre-Hopper works), no LoRA, no expert parallelism.
    DTENSOR = "dtensor"
    # _v2: true -> DTensorPolicyWorkerV2. Automodel rebuilds the model on TE.
    # Superset of DTENSOR; upstream NeMo-RL's reference configs default to it.
    AUTOMODEL = "automodel"
    # megatron_cfg.enabled -> MegatronPolicyWorker. The image builds the `mcore`
    # extra and prefetches the venv, but the compiler still emits an inert
    # megatron_cfg (dpo_config._megatron_cfg_disabled). Enabling it needs HF->Megatron
    # conversion, a shared checkpoint dir, and megatron_cfg.optimizer/scheduler.
    # MEGATRON = "megatron"


class OptimizerType(str, Enum):
    """Optimizer and scheduler combination types."""

    ADAMW_WITH_COSINE_ANNEALING = "adamw_with_cosine_annealing"
    ADAM_WITH_COSINE_ANNEALING = "adam_with_cosine_annealing"
    ADAMW_WITH_FLAT_LR = "adamw_with_flat_lr"
    ADAM_WITH_FLAT_LR = "adam_with_flat_lr"


class BatchingStrategy(str, Enum):
    """How the trainer groups rollouts into micro-batches.

    One enum, not two flags: NeMo-RL asserts the modes are mutually exclusive in
    ``lm_policy`` and ``BatchedDataDict.shard_by_batch_size``.
    """

    STATIC = "static"  # one rollout per slot, padded
    DYNAMIC = "dynamic"  # bucket rollouts up to a token budget
    SEQUENCE_PACKING = "sequence_packing"  # concatenate into packed sequences


class ModelConfig(BaseModel):
    """Internal model configuration with a resolved local path."""

    path: str = Field(description="Local path to the downloaded model directory.")
    name: str | None = Field(default=None, description="Model entity identifier.")
    max_seq_length: int = Field(default=2048)
    precision: Precision | None = Field(default=None, description="Weight dtype; auto-detected when None.")
    chat_template: str | None = Field(default=None, description="Jinja2 chat template override.")
    trust_remote_code: bool = Field(default=False)
    v4_compatible: bool = Field(
        default=True,
        description="Keep the base checkpoint's transformers-v4 config.json on consolidated "
        "exports. Set false to write the in-memory v5 config instead.",
    )


class DPOConfig(BaseModel):
    """DPO hyperparameters controlling the loss and optimization behavior."""

    ref_policy_kl_penalty: float = Field(default=0.05, ge=0.0, description="KL penalty (beta in the DPO paper).")
    preference_average_log_probs: bool = Field(default=False)
    sft_average_log_probs: bool = Field(default=False)
    preference_loss_weight: float = Field(default=1.0, ge=0.0)
    sft_loss_weight: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, ge=0.0)


class GRPOConfig(BaseModel):
    """GRPO hyperparameters for NeMo Gym rollouts."""

    num_generations_per_prompt: int = Field(default=8, gt=0)
    num_prompts_per_step: int | None = Field(default=None, gt=0)
    # Rollout sampling. Both land in NeMo-RL's policy.generation block, which is what
    # _prepare_nemo_gym_rows stamps onto every Gym row, so they reach colocated and
    # sandboxed rollouts alike.
    temperature: float = Field(default=1.0, gt=0.0)
    max_new_tokens: int | None = Field(default=None, gt=0)
    top_k: int | None = Field(default=None, gt=0)
    # Advantage shaping. normalize_rewards and use_leave_one_out_baseline reach NeMo-RL
    # through the grpo.adv_estimator block; the clip bounds sit on grpo itself.
    normalize_rewards: bool = True
    use_leave_one_out_baseline: bool = True
    advantage_clip_low: float | None = None
    advantage_clip_high: float | None = None
    overlong_filtering: bool = False
    max_rollout_turns: int = Field(default=1, gt=0)
    ref_policy_kl_penalty: float = Field(default=0.0, ge=0.0)
    ratio_clip_min: float = Field(default=0.2, ge=0.0)
    ratio_clip_max: float = Field(default=0.28, ge=0.0)
    ratio_clip_c: float | None = Field(default=None, gt=1.0)
    use_on_policy_kl_approximation: bool = True
    use_importance_sampling_correction: bool = True
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    # DAPO components; see GRPOTraining in nmp.rl.schemas.job for what each does. All three
    # sit off by default so an unstated job compiles to the same NeMo-RL config as before.
    truncated_importance_sampling_type: str | None = None
    truncated_importance_sampling_ratio: float | None = Field(default=None, gt=0.0)
    truncated_importance_sampling_ratio_min: float | None = Field(default=None, ge=0.0)
    use_dynamic_sampling: bool = False
    dynamic_sampling_max_gen_batches: int = Field(default=10, gt=0)
    batch_multiplier: float = Field(default=1.0, gt=0.0)
    reward_shaping: dict[str, Any] | None = None
    reward_scaling: dict[str, Any] | None = None
    # Defaults to DYNAMIC: this backend always trains on DTensor, and dynamic batching is the
    # DTensor/Automodel norm in NeMo-RL's recipes (sequence packing is the Megatron norm).
    # It is also the only mode with no architecture restrictions on the GRPO + DTensor path --
    # packing is rejected for VLM, multimodal and context parallel.
    batching_strategy: BatchingStrategy = BatchingStrategy.DYNAMIC
    # None derives max_seq_length * micro_batch_size: the peak STATIC already provisions for.
    train_mb_tokens: int | None = Field(default=None, gt=0)
    sequence_length_round: int = Field(default=64, gt=0)
    # Per-architecture backend knobs; see GRPOTraining in nmp.rl.schemas.job for what each does.
    automodel_kwargs: dict[str, Any] | None = None
    router_aux_loss_coef: float | None = Field(default=None, ge=0.0)
    # Passed to NeMo-RL's policy.hf_config_overrides verbatim, so nested keys reach models
    # that namespace their config (Qwen3.5 reads router_aux_loss_coef under text_config).
    hf_config_overrides: dict[str, Any] | None = None
    vllm_tensor_parallel_size: int | None = Field(default=None, gt=0)
    vllm_gpu_memory_utilization: float = Field(default=0.5, gt=0.0, le=1.0)


class LoRAConfig(BaseModel):
    """LoRA settings carried on the training step (maps to NeMo-RL lora_cfg)."""

    rank: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    target_modules: list[str] | None = None
    exclude_modules: list[str] | None = None
    # None = compiler picks (true at TP 1, false above). Explicit values pass through;
    # GRPOTraining rejects true with TP > 1.
    use_triton: bool | None = None


class WandBConfig(BaseModel):
    project: str | None = None
    name: str | None = None
    entity: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    base_url: str | None = None


class MLflowConfig(BaseModel):
    experiment_name: str | None = None
    run_name: str | None = None
    tags: dict[str, str] | None = None
    description: str | None = None
    tracking_uri: str | None = None


class TrainingStepConfig(BaseModel):
    """Normalized, backend-agnostic training configuration.

    The training container deserializes this and the NeMo-RL backend transforms
    it into library-specific YAML at runtime.
    """

    class DatasetConfig(BaseModel):
        path: str
        prompt_template: str | None = None
        add_bos: bool | None = None
        add_eos: bool | None = None

    class TrainingConfig(BaseModel):
        training_type: TrainingType
        finetuning_type: FinetuningType | None = None
        dpo: DPOConfig | None = None
        grpo: GRPOConfig | None = None
        lora: LoRAConfig | None = None

    class GymConfig(BaseModel):
        """NeMo Gym environment paths and sandbox mode (GRPO only)."""

        environment_path: str | None = None
        sandbox_environment_path: str | None = None
        sandbox_dataset_path: str | None = None
        sandboxed: bool = True
        gym_runtime_image: str | None = None
        allow_internet: bool = False
        public_dns_allow: list[str] = Field(default_factory=list)
        # Scheme the cluster's OpenSandbox server speaks. Operator-scoped like the egress
        # settings; resolved by the compiler because RlConfig is not readable from the
        # training pod. None takes NeMo-RL's default.
        sandbox_server_protocol: str | None = None
        # Operator-scoped, from platformConfig.rl.sandbox_resources / sandbox_ttl_s.
        sandbox_resources: dict[str, str] | None = None
        sandbox_ttl_s: int | None = None
        # Rollouts per POST to the sandbox, and how many of those POSTs may run at once.
        # None takes NeMo-RL's default. In-flight rollouts are the product of the two.
        sandbox_rollout_chunk_size: int | None = Field(default=None, gt=0)
        sandbox_rollout_max_in_flight: int | None = Field(default=None, gt=0)

    class ScheduleConfig(BaseModel):
        epochs: int = 1
        max_steps: int | None = None
        val_check_interval: float | None = None
        val_at_start: bool = False
        val_at_end: bool = True
        keep_top_k: int = 1
        progress_reporting: ProgressReportingConfig = Field(default_factory=ProgressReportingConfig)

    class BatchConfig(BaseModel):
        global_batch_size: int = Field(default=32, gt=0)
        micro_batch_size: int = Field(default=1, gt=0)
        sequence_packing: bool = False
        sequence_packing_max_samples: int = 1000

    class OptimizerConfig(BaseModel):
        optimizer_type: OptimizerType | None = Field(default=None)
        learning_rate: float = 1e-4
        min_learning_rate: float | None = None
        eps: float = 1e-5
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
        expert_parallel_size: int = 1
        sequence_parallel: bool = False
        activation_checkpointing: bool = False
        # GRPO only; default matches GRPOTraining so a hand-built config compiles the
        # same YAML. DPO ignores it -- its dtensor_cfg is a literal with no ``_v2``.
        policy_backend: PolicyBackend = PolicyBackend.AUTOMODEL

    class IntegrationsConfig(BaseModel):
        wandb: WandBConfig | None = None
        mlflow: MLflowConfig | None = None

    # === Main config fields ===
    backend: TrainingBackend = TrainingBackend.NEMO_RL
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    gym: GymConfig | None = None
    schedule: ScheduleConfig
    batch: BatchConfig
    optimizer: OptimizerConfig
    parallelism: ParallelismConfig
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)

    # === Output paths ===
    output_model: str
    workspace_path: str = Field(default=DEFAULT_TRAINING_OUTPUT_PATH)
    output_path: str = Field(default=DEFAULT_OUTPUT_MODEL_PATH)

    # === Misc ===
    seed: int = Field(default=DEFAULT_SEED)
    training_timeout: int | None = None


class GPUInfo(BaseModel):
    architecture: str
    device_name: str
    memory_gb: float
    cuda_version: str


class CheckpointInfo(BaseModel):
    path: str
    format: CheckpointFormat
    precision: Precision | None = None


class TrainingMetrics(BaseModel):
    final_loss: float | None = None
    final_val_loss: float | None = None
    best_val_loss: float | None = None
    total_steps: int = 0
    total_epochs: int = 0


class TrainingResult(BaseModel):
    """Result written by the training task to ``{workspace_path}/training_result.json``."""

    success: bool
    error_message: str | None = None
    checkpoint: CheckpointInfo | None = None
    gpu_info: GPUInfo | None = None
    metrics: TrainingMetrics = Field(default_factory=TrainingMetrics)
    training_duration_seconds: float | None = None

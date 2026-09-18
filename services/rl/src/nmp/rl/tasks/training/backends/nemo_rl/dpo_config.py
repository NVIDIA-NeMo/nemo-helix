# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""TrainingStepConfig -> NeMo-RL DPO YAML generation.

Converts the internal :class:`TrainingStepConfig` into the complete YAML config
NeMo-RL's DPO training expects.

The full config is generated here in one place — there is no external base file
to merge against. Fields driven by the job spec (model, batch sizes, parallelism,
schedule, DPO hyperparameters, optimizer/scheduler, integrations) are computed
from ``TrainingStepConfig``; every other key NeMo-RL's schema requires
(``policy.megatron_cfg``, ``dtensor_cfg.lora_cfg``, ``fp8_cfg``,
``dpo.val_at_end``, ``checkpointing.save_optimizer``, the logger subsections, …)
is set explicitly to a known-good default. The Megatron backend block is inert
(``enabled: False``) since training runs on DTensor, but must still be fully
populated to satisfy the schema.
"""

import logging
from pathlib import Path
from typing import Any

from nmp.customization_common.service.context import NMPJobContext
from nmp.rl.app.jobs.training.schemas import (
    DPOConfig,
    OptimizerType,
    TrainingStepConfig,
)
from nmp.rl.tasks.training.chat_templates import resolve_chat_template
from nmp.rl.tasks.training.datasets.preparation import (
    PreparedDataset,
    compute_val_check_interval,
    prepare_dataset,
)
from nmp.rl.tasks.training.datasets.validation import DatasetValidator, detect_dpo_schema_name
from nmp.rl.tasks.training.integrations import (
    build_mlflow_config,
    build_wandb_config,
)

logger = logging.getLogger(__name__)


def compile_dpo_config(
    customizer_config: TrainingStepConfig,
    job_ctx: NMPJobContext,
) -> dict[str, Any]:
    """
    Compile TrainingStepConfig to NeMo RL DPO configuration dict.

    This transforms the standardized TrainingStepConfig into the format
    expected by NeMo RL's DPO training. The output dict will be serialized
    to YAML by the training runner.

    Args:
        customizer_config: The training step configuration
        job_ctx: Job context

    Returns:
        Configuration dict for NeMo RL DPO training

    Reference: https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/dpo.yaml
    """
    cfg: dict[str, Any] = {}
    workspace_dir = Path(customizer_config.workspace_path)

    # === Dataset Preparation ===
    prepared = prepare_dataset(
        dataset_path=Path(customizer_config.dataset.path),
        output_dir=workspace_dir / "dataset",
    )
    logger.info(
        f"Prepared dataset: train={prepared.train_samples} samples, validation={prepared.validation_samples} samples"
    )
    validator = DatasetValidator(training_type=customizer_config.training.training_type)
    validator.validate_dataset(str(prepared.train_file))
    validator.validate_dataset(str(prepared.validation_file))
    logger.info("Validated datasets successfully")

    # === Training Schedule Calculations ===
    batch_size = customizer_config.batch.global_batch_size
    micro_batch_size = customizer_config.batch.micro_batch_size
    epochs = customizer_config.schedule.epochs

    # Compute steps per epoch (round up to ensure all samples are used)
    steps_per_epoch = max((prepared.train_samples + batch_size - 1) // batch_size, 1)
    total_steps = steps_per_epoch * epochs

    # Determine effective max_steps
    user_max_steps = customizer_config.schedule.max_steps
    if user_max_steps and user_max_steps > 0:
        max_steps = min(user_max_steps, total_steps)
    else:
        max_steps = total_steps

    # Compute validation interval
    val_check_interval = compute_val_check_interval(
        steps_per_epoch=steps_per_epoch,
        max_steps=max_steps,
        val_check_interval=customizer_config.schedule.val_check_interval,
    )

    logger.info(
        f"Training schedule: {prepared.train_samples} samples, batch_size={batch_size}, "
        f"steps_per_epoch={steps_per_epoch}, epochs={epochs}, max_steps={max_steps}, "
        f"val_period={val_check_interval}"
    )

    # === Get DPO Hyperparameters ===
    dpo_hp = customizer_config.training.dpo or DPOConfig()

    # Checkpoint selection ranks by the validation metric (`metric_name` below),
    # so the saved checkpoint must carry validation metrics. NeMo-RL always saves a
    # checkpoint on the LAST step (is_last_step), and the last step is frequently
    # not a validation step — e.g. with the small-run default (max_steps caps the
    # run at 7 steps with 200 rows / batch 32) the last step never aligns with
    # val_period, so that checkpoint has no `val:...` metric. NeMo-RL then warns and
    # falls back to "latest" instead of best.
    #
    # Two measures keep validation aligned with checkpointing:
    #   1. `val_at_end` (the dpo section sets it from schedule.val_at_end, which
    #      defaults to True) forces a validation pass on the FINAL step, so the
    #      is_last_step checkpoint carries the metric regardless of whether
    #      max_steps is a multiple of val_period. This is the primary guarantee.
    #      A user CAN opt out (val_at_end=False, to skip the final eval); in that
    #      case the last-step checkpoint may lack the metric and best-checkpoint
    #      selection degrades to "latest" — the accepted trade-off of opting out.
    #   2. `save_period == val_period` so intermediate saves also land on validation
    #      steps. (val_period=steps_per_epoch when val_check_interval is unset.)
    # NeMo-RL's get_best_checkpoint_path() itself is the safety net: it filters out
    # checkpoints missing the metric and, if none have it, returns the latest — so
    # the val_at_end=False path degrades gracefully to "latest" instead of crashing.
    val_period = val_check_interval

    # === DPO Section ===
    cfg["dpo"] = {
        "max_num_epochs": epochs,
        "max_num_steps": max_steps,
        # Neither of these is a NeMo-RL field. Both ride along as extras -- DPOConfig
        # allows them -- because this dict is the only channel from the compiled job
        # config to the driver, and the driver is where our own logger is built.
        # Read back with getattr, never attribute access: a config compiled
        # elsewhere simply omits them.
        "steps_per_epoch": steps_per_epoch,
        "progress_time_series_metrics": customizer_config.schedule.progress_reporting.time_series_metrics,
        "progress_min_report_interval_seconds": customizer_config.schedule.progress_reporting.min_report_interval_seconds,
        "val_period": val_period,
        "val_batches": 0,  # Run the entire validation dataset
        "val_global_batch_size": batch_size,
        "val_micro_batch_size": micro_batch_size,
        "val_at_start": True,
        "val_at_end": customizer_config.schedule.val_at_end,
        "seed": customizer_config.seed,
        # DPO-specific hyperparameters
        "reference_policy_kl_penalty": dpo_hp.ref_policy_kl_penalty,
        "preference_average_log_probs": dpo_hp.preference_average_log_probs,
        "sft_average_log_probs": dpo_hp.sft_average_log_probs,
        "preference_loss_weight": dpo_hp.preference_loss_weight,
        "sft_loss_weight": dpo_hp.sft_loss_weight,
    }

    # === Checkpointing Section ===
    # save_period == val_period so intermediate saves land on validation steps; the
    # always-saved last-step checkpoint gets its validation metric from `val_at_end`
    # (see the val_period comment above). `metric_name` + `keep_top_k` then select
    # the best checkpoint by validation loss.
    cfg["checkpointing"] = {
        "enabled": True,
        "checkpoint_dir": str(workspace_dir / "checkpoints"),
        "metric_name": "val:validation-default_loss",
        "higher_is_better": False,
        "keep_top_k": customizer_config.schedule.keep_top_k,
        "save_period": val_period,
        "checkpoint_must_save_by": None,
        "save_optimizer": True,
    }

    # === Policy Section ===
    model_path = customizer_config.model.path
    precision = _adapt_precision(customizer_config.model.precision)
    parallelism = customizer_config.parallelism

    # Resolve chat template with priority:
    # 1. Fileset metadata chat_template (from model entity spec)
    # 2. Custom template from DEFAULT_CHAT_TEMPLATES (if model.name matches)
    # 3. Model's built-in tokenizer template (fallback)
    chat_template = resolve_chat_template(
        model_path=model_path,
        model_name=customizer_config.model.name,
        user_template=customizer_config.model.chat_template,
        trust_remote_code=customizer_config.model.trust_remote_code,
    )

    cfg["policy"] = {
        "model_name": model_path,
        "tokenizer": _build_tokenizer_config(model_path, chat_template),
        "train_global_batch_size": batch_size,
        "train_micro_batch_size": micro_batch_size,
        "max_total_sequence_length": customizer_config.model.max_seq_length,
        "precision": precision,
        "offload_optimizer_for_logprob": False,
        # Training runs on the DTensor backend. We propagate tensor / sequence /
        # context parallelism from the parallelism config; the remaining keys are
        # NeMo-RL defaults (LoRA disabled — DPO is full-weight here).
        "dtensor_cfg": {
            "env_vars": {"PYTORCH_CUDA_ALLOC_CONF": ""},
            "enabled": True,
            "cpu_offload": False,
            "sequence_parallel": parallelism.sequence_parallel,
            "activation_checkpointing": parallelism.activation_checkpointing,
            "tensor_parallel_size": parallelism.tensor_parallel_size,
            "context_parallel_size": parallelism.context_parallel_size,
            "custom_parallel_plan": None,
            "clear_cache_every_n_steps": None,
            "automodel_kwargs": {},
            "lora_cfg": {
                "enabled": False,
                "target_modules": [],
                "exclude_modules": [],
                "match_all_linear": True,
                "dim": 8,
                "alpha": 32,
                "dropout": 0.0,
                "dropout_position": "post",
                "lora_A_init": "xavier",
                "use_triton": True,
            },
        },
        "dynamic_batching": {"enabled": False},
        "sequence_packing": _build_sequence_packing_config(customizer_config),
        "make_sequence_length_divisible_by": parallelism.tensor_parallel_size,
        "max_grad_norm": dpo_hp.max_grad_norm,
        # Optimizer and scheduler
        "optimizer": _build_optimizer_config(customizer_config),
        # Schedule LR over the steps that will actually execute (max_steps after
        # user capping), so warmup + cosine decay complete within the run instead of
        # being stretched across the uncapped epoch length and never reaching min_lr.
        "scheduler": _build_scheduler_config(customizer_config, max_steps),
        # Megatron backend is disabled (we train on DTensor). NeMo-RL's config
        # schema still requires this block to be fully populated, so it is
        # reproduced inert here; none of these values take effect while
        # ``enabled`` is False.
        "megatron_cfg": _megatron_cfg_disabled(precision, dpo_hp.max_grad_norm),
    }

    # === Data Section ===
    cfg["data"] = _build_data_config(customizer_config, prepared)

    # === Logger Section ===
    cfg["logger"] = _build_logger_config(customizer_config, job_ctx, workspace_dir)

    # === Cluster Section ===
    cfg["cluster"] = {
        "gpus_per_node": parallelism.num_gpus_per_node,
        "num_nodes": parallelism.num_nodes,
    }

    return cfg


def _build_tokenizer_config(model_path: str, chat_template: str | None) -> dict[str, Any]:
    """Build ``policy.tokenizer``, omitting ``chat_template`` when there is none.

    NeMo-RL's TokenizerConfig declares ``chat_template: NotRequired[str]`` -- absent
    is fine, ``None`` is not. ``resolve_chat_template`` returns ``None`` when the
    model ships no template and the user supplied none, so emitting the key
    unconditionally fails MasterConfig validation in the driver with
    "Input should be a valid string". ``chat_template_kwargs`` is
    ``NotRequired[dict | None]`` and does accept ``None``.
    """
    tokenizer: dict[str, Any] = {"name": model_path, "chat_template_kwargs": None}
    if chat_template is not None:
        tokenizer["chat_template"] = chat_template
    return tokenizer


def _megatron_cfg_disabled(precision: str, max_grad_norm: float) -> dict[str, Any]:
    """Return the (inert) Megatron backend config block.

    Training uses the DTensor backend, so ``enabled`` is False and none of these
    values take effect. NeMo-RL's config schema still requires the block to be
    present and fully populated, so it is reproduced here. ``pipeline_dtype`` and
    ``optimizer.clip_grad`` track ``policy.precision`` / ``policy.max_grad_norm``.
    """
    return {
        "enabled": False,
        "use_linear_ce_fusion_loss": False,
        "linear_ce_fusion_chunk_size": 256,
        "force_reconvert_from_hf": False,
        "empty_unused_memory_level": 1,
        "activation_checkpointing": False,
        "tensor_model_parallel_size": 2,
        "expert_tensor_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "pipeline_dtype": precision,
        "num_layers_in_first_pipeline_stage": None,
        "num_layers_in_last_pipeline_stage": None,
        "sequence_parallel": True,
        "freeze_moe_router": False,
        "moe_router_dtype": "fp64",
        "moe_router_load_balancing_type": "aux_loss",
        "moe_router_bias_update_rate": 1e-3,
        "moe_permute_fusion": False,
        "apply_rope_fusion": True,
        "bias_activation_fusion": True,
        "defer_fp32_logits": False,
        "moe_per_layer_logging": False,
        "moe_enable_deepep": False,
        "moe_token_dispatcher_type": "alltoall",
        "moe_shared_expert_overlap": False,
        "gradient_accumulation_fusion": False,
        "peft": {
            "enabled": False,
            "target_modules": [],
            "exclude_modules": [],
            "dim": 8,
            "alpha": 32,
            "dropout": 0.0,
            "dropout_position": "post",
            "lora_A_init_method": "xavier",
            "lora_B_init_method": "zero",
            "a2a_experimental": False,
            "lora_dtype": None,
        },
        "optimizer": {
            "optimizer": "adam",
            "lr": 5.0e-6,
            "min_lr": 5.0e-6,
            "weight_decay": 0.1,
            "bf16": True,
            "fp16": False,
            "params_dtype": "float32",
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
            "adam_eps": 1e-8,
            "sgd_momentum": 0.9,
            "use_distributed_optimizer": True,
            "use_precision_aware_optimizer": True,
            "clip_grad": max_grad_norm,
            "optimizer_cpu_offload": False,
            "optimizer_offload_fraction": 0.0,
        },
        "scheduler": {
            "start_weight_decay": 0.1,
            "end_weight_decay": 0.1,
            "weight_decay_incr_style": "constant",
            "lr_decay_style": "constant",
            "lr_warmup_iters": 1,
            "lr_warmup_init": 0.00000001,
        },
        "distributed_data_parallel_config": {
            "grad_reduce_in_fp32": False,
            "overlap_grad_reduce": True,
            "overlap_param_gather": True,
            "data_parallel_sharding_strategy": "optim_grads_params",
            "use_custom_fsdp": False,
        },
        "fp8_cfg": {
            "enabled": False,
            "fp8": "e4m3",
            "fp8_recipe": "blockwise",
            "fp8_param": False,
        },
    }


def _build_data_config(customizer_config: TrainingStepConfig, prepared: PreparedDataset) -> dict[str, Any]:
    """Build the NeMo-RL ``data`` config.

    NeMo-RL's ``setup_preference_data`` reads nested ``train`` / ``validation``
    dataset specs (``dataset_name`` + ``data_path``) and builds each split by
    instantiating the class registered under ``dataset_name`` in NeMo-RL's
    ``DATASET_REGISTRY`` as ``cls(**spec)`` — no custom preprocessor.
    ``detect_dpo_schema_name`` returns that registry key, one of
    ``BinaryPreferenceDataset`` / ``PreferenceDataset`` / ``HelpSteer3`` /
    ``Tulu3Preference``.

    All four load from the local ``data_path`` here. ``BinaryPreferenceDataset`` and
    ``PreferenceDataset`` accept a local path natively; ``HelpSteer3`` and
    ``Tulu3Preference`` only do so because the DPO driver re-points those registry
    entries to our local-file-capable subclasses via ``register_preference_datasets()``
    (NeMo-RL's built-ins for those two always download from HuggingFace and ignore
    ``data_path``). If that registration is ever removed, a user-uploaded
    HelpSteer3/Tulu3 dataset would silently train on the public HF dataset instead.

    The schema is detected per split: train and validation may legitimately be in
    different supported formats, so inferring once from the train file and reusing
    it for validation would point validation at the wrong loader.
    """

    def _dataset_spec(path: Path) -> dict[str, Any]:
        dataset_name = detect_dpo_schema_name(path)
        spec: dict[str, Any] = {"dataset_name": dataset_name, "data_path": str(path)}
        # BinaryPreferenceDataset reads explicit prompt/chosen/rejected keys; our
        # datasets use those exact field names.
        if dataset_name == "BinaryPreferenceDataset":
            spec.update({"prompt_key": "prompt", "chosen_key": "chosen", "rejected_key": "rejected"})
        return spec

    return {
        "max_input_seq_length": customizer_config.model.max_seq_length,
        # Disable dataloader shuffling for deterministic, reproducible training
        # order (and consistent ordering across distributed ranks). The
        # train/validation split is already randomized at preparation time.
        "shuffle": False,
        "num_workers": 1,
        "train": _dataset_spec(prepared.train_file),
        "validation": _dataset_spec(prepared.validation_file),
    }


def _adapt_precision(precision: str | None) -> str:
    """

    Returns in the format that is expected by NeMo FW:
    ('transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed',
    'bf16-true', 'bf16-mixed', '32-true', '64-true', 64, 32, 16, '64', '32', '16', 'bf16')
    """
    precision_map = {
        "bf16": "bfloat16",
        "bf16-mixed": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
        None: "bfloat16",  # Default
    }
    result = precision_map.get(precision)
    if result is None:
        logger.warning(f"Unknown precision '{precision}', defaulting to bfloat16")
        return "bfloat16"
    return result


def _build_sequence_packing_config(customizer_config: TrainingStepConfig) -> dict[str, Any]:
    """Build sequence packing configuration."""
    logger.warning("Sequence packing is currently not supported with DPO.")
    return {"enabled": False}

    ## TODO: uncomment below code when sequence packing is supported by nemo-rl
    ## Sequence packing is currently not supported with DPO. See https://github.com/NVIDIA-NeMo/RL/issues/719
    # if not customizer_config.batch.sequence_packing:
    #     return {"enabled": False}

    # return {
    #     "enabled": True,
    #     "train_mb_tokens": 2048,
    #     "logprob_mb_tokens": 2048,
    #     "algorithm": "modified_first_fit_decreasing",
    #     "sequence_length_round": 64,  # Hardware alignment
    # }


def _build_optimizer_config(customizer_config: TrainingStepConfig) -> dict[str, Any]:
    """Build optimizer configuration for NeMo RL.

    Supports:
    - AdamW (with weight decay)
    - Adam (without weight decay correction)

    The optimizer type is determined by the optimizer_type field in OptimizerConfig.
    """
    opt = customizer_config.optimizer
    optimizer_type = opt.optimizer_type or OptimizerType.ADAMW_WITH_COSINE_ANNEALING

    # Determine optimizer name based on type
    if optimizer_type in (OptimizerType.ADAM_WITH_COSINE_ANNEALING, OptimizerType.ADAM_WITH_FLAT_LR):
        optimizer_name = "torch.optim.Adam"
    else:
        # Default: AdamW for ADAMW_WITH_COSINE_ANNEALING and ADAMW_WITH_FLAT_LR
        optimizer_name = "torch.optim.AdamW"

    return {
        "name": optimizer_name,
        "kwargs": {
            "lr": opt.learning_rate,
            "weight_decay": opt.weight_decay,
            "betas": [opt.beta1, opt.beta2],
            "eps": opt.eps,
            "foreach": False,
            "fused": False,
        },
    }


def _build_scheduler_config(
    customizer_config: TrainingStepConfig,
    num_steps: int,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Build learning rate scheduler configuration.

    Supports two scheduler types based on optimizer_type:
    - Cosine Annealing: LinearLR warmup followed by CosineAnnealingLR decay
    - Flat LR: ConstantLR (constant learning rate throughout training)

    ``num_steps`` is the number of steps that will actually run (``max_steps`` after
    capping), so the warmup + decay horizon matches the executed run.
    """
    opt = customizer_config.optimizer
    optimizer_type = opt.optimizer_type or OptimizerType.ADAMW_WITH_COSINE_ANNEALING
    warmup_steps = opt.warmup_steps
    lr = opt.learning_rate
    min_lr = opt.min_learning_rate or 0.0

    # Check if using flat LR scheduler
    if optimizer_type in (OptimizerType.ADAM_WITH_FLAT_LR, OptimizerType.ADAMW_WITH_FLAT_LR):
        # Flat LR. List form is required by NeMo-RL; empty milestones because
        # SequentialLR needs one fewer milestone than schedulers.
        return [
            {
                "name": "torch.optim.lr_scheduler.ConstantLR",
                "kwargs": {
                    "factor": 1.0,
                    "total_iters": num_steps,
                },
            },
            {
                "milestones": [],
            },
        ]

    if optimizer_type in (OptimizerType.ADAM_WITH_COSINE_ANNEALING, OptimizerType.ADAMW_WITH_COSINE_ANNEALING):
        # Default: Cosine Annealing with warmup
        # Compute start_factor for warmup (avoid division by zero)
        start_factor = max(min_lr / lr, 1e-5) if lr > 0 else 1e-5
        # Clamp warmup_steps to >= 1 for cosine schedulers; LinearLR(total_iters=0)
        # and milestones=[0] produce invalid scheduler behavior
        effective_warmup_steps = max(warmup_steps or 0, 1)

        return [
            {
                "name": "torch.optim.lr_scheduler.LinearLR",
                "kwargs": {
                    "start_factor": start_factor,
                    "end_factor": 1.0,
                    "total_iters": effective_warmup_steps,
                },
            },
            {
                "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "kwargs": {
                    "T_max": max(num_steps - effective_warmup_steps, 1),
                    "eta_min": min_lr,
                },
            },
            {
                "milestones": [effective_warmup_steps],
            },
        ]

    return {}


def _build_logger_config(
    customizer_config: TrainingStepConfig,
    job_ctx: NMPJobContext,
    workspace_dir: Path,
) -> dict[str, Any]:
    """Build logger configuration for NeMo RL.

    WandB logging is handled by nemo-rl's Logger class when wandb_enabled is True.
    The wandb config is passed directly to wandb.init().
    """
    wandb_config = build_wandb_config(
        customizer_config=customizer_config,
        job_ctx=job_ctx,
        framework="nemo_rl",
    )
    wandb_enabled = wandb_config is not None
    # NeMo-RL's WandbLogger always passes `dir=` when initializing wandb.
    # Avoid duplicate keyword errors by removing it from shared config here.
    if wandb_config is not None:
        wandb_config.pop("dir", None)
    mlflow_config = build_mlflow_config(
        customizer_config=customizer_config,
        job_ctx=job_ctx,
        framework="nemo_rl",
    )
    mlflow_enabled = mlflow_config is not None

    # All four backend subsections (wandb / swanlab / tensorboard / mlflow) are
    # always present — NeMo-RL's config schema expects them even when the
    # corresponding ``*_enabled`` flag is False. We overlay the user's wandb /
    # mlflow config when those integrations are enabled; the rest carry inert
    # defaults.
    return {
        "log_dir": str(workspace_dir / "logs"),
        "num_val_samples_to_print": 0,
        "monitor_gpus": False,
        "wandb_enabled": wandb_enabled,
        "tensorboard_enabled": False,
        "mlflow_enabled": mlflow_enabled,
        "swanlab_enabled": False,
        "wandb": wandb_config if (wandb_enabled and wandb_config) else {"project": "dpo", "name": "dpo"},
        "swanlab": {"project": "dpo", "name": "dpo"},
        "tensorboard": {"log_dir": str(workspace_dir / "tb_logs")},
        "mlflow": mlflow_config
        if (mlflow_enabled and mlflow_config)
        else {"experiment_name": "dpo", "run_name": "dpo", "tracking_uri": "http://localhost:5000"},
        "gpu_monitoring": {
            "collection_interval": 10,
            "flush_interval": 10,
        },
    }

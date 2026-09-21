# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Automodel configuration compiler.

This module transforms the standardized TrainingStepConfig into the format
expected by nemo_automodel's TrainFinetuneRecipeForNextTokenPrediction
or KnowledgeDistillationRecipeForNextTokenPrediction.
"""

import logging
import math
import os
from pathlib import Path
from typing import Any

from nemo_automodel._transformers.registry import ModelRegistry
from nmp.automodel.tasks.training.chat_templates import resolve_chat_template
from nmp.automodel.tasks.training.datasets.preparation import (
    DatasetSchema,
    PreparedDataset,
    compute_val_check_interval,
    detect_dataset_schema,
    prepare_dataset,
)
from nmp.automodel.tasks.training.datasets.validation import DatasetValidator
from nmp.automodel.tasks.training.integrations import (
    build_mlflow_config,
    build_wandb_config,
)
from nmp.automodel.tasks.training.schemas import (
    FinetuningType,
    LoRAConfig,
    RetrievalConfig,
    TrainingRecipe,
    TrainingStepConfig,
    TrainingType,
)
from nmp.automodel.tasks.training.sequence_packing import (
    calculate_optimal_pack_size,
    calculate_target_packing_factor,
    estimate_dataset_sequence_lengths,
)
from nmp.customization_common.service.context import NMPJobContext

logger = logging.getLogger(__name__)

_OPTIMIZER_TARGETS = {
    "Adam": "torch.optim.Adam",
    "AdamW": "torch.optim.AdamW",
    "FusedAdam": "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam",
}


def resolve_compiled_recipe(customizer_config: TrainingStepConfig) -> TrainingRecipe:
    """Resolve AUTO from the checkpoint head. Cross-encoder must not fall through to SFT."""
    recipe = getattr(customizer_config.training, "recipe", TrainingRecipe.AUTO)
    if not isinstance(recipe, TrainingRecipe):
        recipe = TrainingRecipe.AUTO
    if recipe != TrainingRecipe.AUTO:
        return recipe
    head = getattr(customizer_config.model, "checkpoint_head_type", "unknown")
    if head == "cross_encoder":
        return TrainingRecipe.CROSS_ENCODER
    if head == "embedding":
        return TrainingRecipe.BI_ENCODER
    if head in (None, "unknown") and getattr(customizer_config.model, "is_embedding_model", False):
        return TrainingRecipe.BI_ENCODER
    return TrainingRecipe.SFT


def _collator_prefix(value: str) -> str:
    """BiEncoderCollator inserts a space after the prefix; Nemotron YAML uses ``"query: "``."""
    return value.rstrip(" ")


_RETRIEVAL_RECIPES = (TrainingRecipe.BI_ENCODER, TrainingRecipe.CROSS_ENCODER)


def compile_automodel_config(
    customizer_config: TrainingStepConfig,
    workspace_dir: Path,
    job_ctx: NMPJobContext,
) -> dict[str, Any]:
    """
    Compile Automodel-specific configuration.

    This transforms the standardized TrainingStepConfig into the format expected
    by nemo_automodel's recipes.

    Retrieval and causal-LM recipes do not differ by a few values in otherwise
    common sections -- they want different sections: a bare seed against an rng
    block, an encoder target plus tokenizer against a causal-LM target, their own
    collating dataloaders against the default collater, and no packing, LM loss,
    or MoE backend at all. Each is compiled by its own function over the section
    builders they share, so a recipe reads top to bottom in one place.
    """
    recipe = resolve_compiled_recipe(customizer_config)
    if recipe in _RETRIEVAL_RECIPES:
        return _compile_retrieval_config(customizer_config, workspace_dir, job_ctx, recipe)
    return _compile_llm_config(customizer_config, workspace_dir, job_ctx, recipe)


def _compile_retrieval_config(
    customizer_config: TrainingStepConfig,
    workspace_dir: Path,
    job_ctx: NMPJobContext,
    recipe: TrainingRecipe,
) -> dict[str, Any]:
    """Compile a bi-encoder or cross-encoder recipe config.

    Mirrors Nemotron Stage 2 (``nemotron embed|rerank finetune``): TE FusedAdam
    by default,
    inline retrieval data with its own collator, and no sequence packing or LM loss.
    """
    if customizer_config.training.training_type == TrainingType.DISTILLATION:
        raise ValueError(
            f"Knowledge distillation is not supported by the '{recipe.value}' recipe; "
            "use the 'sft' recipe or drop the teacher model."
        )

    cfg: dict[str, Any] = {}
    is_bi_encoder = recipe == TrainingRecipe.BI_ENCODER
    retrieval_config = customizer_config.retrieval or RetrievalConfig()
    seed = _resolve_seed(customizer_config)

    _build_dist_env(cfg)

    # Retrieval recipes construct StatefulRNG from the seed internally, so they
    # take a bare seed rather than the LLM recipe's full rng config object.
    cfg["seed"] = seed
    cfg["temperature"] = 0.02 if is_bi_encoder else 1.0

    _build_base_model(cfg, customizer_config)
    cfg["model"].update(
        {
            "_target_": (
                "nemo_automodel._transformers.auto_model.NeMoAutoModelBiEncoder.from_pretrained"
                if is_bi_encoder
                else "nemo_automodel._transformers.auto_model.NeMoAutoModelCrossEncoder.from_pretrained"
            ),
            "pooling": "avg",
            "use_liger_kernel": True,
            "use_sdpa_patching": True,
        }
    )
    if is_bi_encoder:
        cfg["model"]["l2_normalize"] = True
        cfg["model"]["do_distributed_inbatch_negative"] = retrieval_config.do_distributed_inbatch_negative
    else:
        cfg["model"]["num_labels"] = 1
        cfg["model"]["temperature"] = 1.0
    if customizer_config.model.attn_implementation:
        cfg["model"]["attn_implementation"] = customizer_config.model.attn_implementation

    cfg["tokenizer"] = {
        "_target_": "nemo_automodel._transformers.auto_tokenizer.NeMoAutoTokenizer.from_pretrained",
        "pretrained_model_name_or_path": customizer_config.model.path,
    }

    _build_distributed(
        cfg,
        customizer_config,
        activation_checkpointing=retrieval_config.do_gradient_checkpointing,
    )

    prepared = _prepare_and_validate_dataset(customizer_config, workspace_dir)

    # A retrieval sample is a query with its passages, so there is nothing to pack.
    max_steps = _build_schedule(cfg, customizer_config, prepared, packing_factor=None)

    _build_optimizer(
        cfg,
        customizer_config,
        max_steps=max_steps,
        optimizer_target=_resolve_optimizer_target(customizer_config, recipe),
    )
    _build_checkpoint(cfg, customizer_config, workspace_dir)

    # Retrieval datasets bring their own dataloaders and collators.
    _configure_datasets(
        cfg,
        customizer_config,
        prepared,
        customizer_config.model.max_seq_length,
        seed,
        recipe,
        retrieval_config,
    )

    _build_peft(cfg, customizer_config)
    _build_integrations(cfg, customizer_config, job_ctx)

    return cfg


def _compile_llm_config(
    customizer_config: TrainingStepConfig,
    workspace_dir: Path,
    job_ctx: NMPJobContext,
    recipe: TrainingRecipe,
) -> dict[str, Any]:
    """Compile a causal-LM recipe config (SFT or knowledge distillation)."""
    cfg: dict[str, Any] = {}
    trust_remote_code = customizer_config.model.trust_remote_code
    seed = _resolve_seed(customizer_config)

    _build_dist_env(cfg)

    # The LLM recipe expects the full rng config object.
    cfg["rng"] = {
        "_target_": "nemo_automodel.components.training.rng.StatefulRNG",
        "seed": seed,
        "ranked": True,  # Different seed per rank for data augmentation
    }

    _build_base_model(cfg, customizer_config)
    cfg["model"].update(
        {
            "_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
            "attn_implementation": customizer_config.model.attn_implementation,
        }
    )

    _build_distributed(cfg, customizer_config)

    prepared = _prepare_and_validate_dataset(customizer_config, workspace_dir)

    # Compute packing before the schedule so its estimated reduction in dataset
    # length is reflected in steps_per_epoch, validation, and warmup.
    effective_seq_length, packing_factor = _configure_sequence_packing(
        cfg,
        customizer_config,
        prepared,
        trust_remote_code=trust_remote_code,
    )

    max_steps = _build_schedule(cfg, customizer_config, prepared, packing_factor=packing_factor)
    _build_optimizer(
        cfg,
        customizer_config,
        max_steps=max_steps,
        optimizer_target=_resolve_optimizer_target(customizer_config, recipe),
    )
    _build_checkpoint(cfg, customizer_config, workspace_dir)

    _configure_datasets(cfg, customizer_config, prepared, effective_seq_length, seed, recipe, None)

    cfg["dataloader"] = {
        "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
        "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
        "shuffle": True,
    }
    cfg["validation_dataloader"] = {
        "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
        "collate_fn": "nemo_automodel.components.datasets.utils.default_collater",
    }

    _build_peft(cfg, customizer_config)

    cfg["loss_fn"] = {
        "_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy",
    }

    # Check for custom Automodel implementations (e.g., MoE models) and
    # configure backend/parallelizer settings.
    _configure_moe_backend(cfg, customizer_config, trust_remote_code=trust_remote_code)

    if customizer_config.training.training_type == TrainingType.DISTILLATION:
        _configure_kd(cfg, customizer_config, trust_remote_code=trust_remote_code)

    _build_integrations(cfg, customizer_config, job_ctx)

    return cfg


def _resolve_seed(customizer_config: TrainingStepConfig) -> int:
    """Seed for reproducibility across restarts and multi-node training."""
    return int(os.environ.get("PL_GLOBAL_SEED", customizer_config.seed))


def _build_dist_env(cfg: dict[str, Any]) -> None:
    """Required for torch.distributed initialization."""
    cfg["dist_env"] = {
        "backend": "nccl",
        "timeout_minutes": 30,  # Higher timeout for large model loading
    }


def _build_base_model(cfg: dict[str, Any], customizer_config: TrainingStepConfig) -> None:
    """Model fields common to the retrieval and causal-LM recipes."""
    cfg["model"] = {
        "pretrained_model_name_or_path": customizer_config.model.path,
        "torch_dtype": customizer_config.model.precision.to_torch_dtype()
        if customizer_config.model.precision
        else "auto",
        # trust_remote_code is required for models like nvidia/llama-nemotron-embed-1b-v2
        # which use custom model_type "llama_bidirec" with custom modeling code.
        "trust_remote_code": customizer_config.model.trust_remote_code,
    }
    if customizer_config.model.override_custom_impl:
        cfg["model"]["force_hf"] = True


def _build_distributed(
    cfg: dict[str, Any],
    customizer_config: TrainingStepConfig,
    *,
    activation_checkpointing: bool = False,
) -> None:
    """Configure FSDP2 sharding and, for retrieval, activation checkpointing."""
    p = customizer_config.parallelism
    total_gpus = p.num_nodes * p.num_gpus_per_node
    # Note dp_size is typically auto-derived by Automodel (world_size / (tp * pp * cp)),
    # but we calculate it explicitly here because:
    # 1. It's validated upstream in validators.py
    # 2. Passing an explicit value ensures consistency rather than relying on Automodel's derivation
    dp = total_gpus // (p.tensor_parallel_size * p.pipeline_parallel_size * p.context_parallel_size)

    cfg["distributed"] = {
        "_target_": "nemo_automodel.components.distributed.fsdp2.FSDP2Manager",
        "dp_size": dp,
        "tp_size": p.tensor_parallel_size,
        "pp_size": p.pipeline_parallel_size,
        "cp_size": p.context_parallel_size,
        "ep_size": p.expert_parallel_size,
        "sequence_parallel": p.sequence_parallel,
    }
    if activation_checkpointing:
        cfg["distributed"]["activation_checkpointing"] = True
    if p.pipeline_parallel_size > 1:
        cfg["distributed"]["pipeline"] = {
            "pp_schedule": "interleaved1f1b",
            "pp_microbatch_size": 1,
            "scale_grads_in_schedule": False,
        }


def _prepare_and_validate_dataset(
    customizer_config: TrainingStepConfig,
    workspace_dir: Path,
) -> PreparedDataset:
    """Discover, merge, and optionally split dataset files."""
    split_kwargs = (
        {}
        if customizer_config.schedule.validation_split is None
        else {"val_split_ratio": customizer_config.schedule.validation_split}
    )
    prepared = prepare_dataset(
        dataset_path=Path(customizer_config.dataset.path),
        output_dir=workspace_dir / "dataset",
        seed=customizer_config.seed,
        **split_kwargs,
    )
    logger.info(
        f"Prepared dataset: train={prepared.train_samples} samples, validation={prepared.validation_samples} samples, files: "
        f"train={prepared.train_file.absolute()}, validation={prepared.validation_file.absolute()}"
    )
    validator = DatasetValidator(training_type=customizer_config.training.training_type)
    validator.validate_dataset(str(prepared.train_file))
    validator.validate_dataset(str(prepared.validation_file))
    logger.info("Validated datasets successfully")
    return prepared


def _configure_sequence_packing(
    cfg: dict[str, Any],
    customizer_config: TrainingStepConfig,
    prepared: PreparedDataset,
    *,
    trust_remote_code: bool,
) -> tuple[int, float | None]:
    """Write ``packed_sequence`` when packing is on.

    Returns the effective sequence length and packing factor, which the caller
    feeds to the schedule and the dataset configuration.
    """
    if not customizer_config.batch.sequence_packing:
        return customizer_config.model.max_seq_length, None

    packing_estimate = estimate_dataset_sequence_lengths(
        customizer_config,
        train_file=prepared.train_file,
        max_samples=customizer_config.batch.sequence_packing_max_samples,
        seed=customizer_config.seed,
        trust_remote_code=trust_remote_code,
    )

    if packing_estimate is not None:
        optimal_pack_size = packing_estimate.pack_size
        packing_factor = packing_estimate.packing_factor
        logger.info(
            f"Sequence packing enabled: pack_size={optimal_pack_size}, "
            f"avg_seq={packing_estimate.avg_seq_length}, max_seq={packing_estimate.max_seq_length}, "
            f"packing_factor={packing_factor}, samples={packing_estimate.samples_analyzed}"
        )
    else:
        optimal_pack_size = calculate_optimal_pack_size(customizer_config)
        packing_factor = float(calculate_target_packing_factor(customizer_config))
        logger.info(
            f"Sequence packing enabled with fallback pack_size={optimal_pack_size}, packing_factor={packing_factor}"
        )

    cfg["packed_sequence"] = {
        "packed_sequence_size": optimal_pack_size,
    }
    return optimal_pack_size, packing_factor


def _build_schedule(
    cfg: dict[str, Any],
    customizer_config: TrainingStepConfig,
    prepared: PreparedDataset,
    *,
    packing_factor: float | None,
) -> int:
    """Write the step scheduler and progress reporting; return effective max_steps."""
    batch_size = customizer_config.batch.global_batch_size
    epochs = customizer_config.schedule.epochs

    steps_per_epoch = estimate_steps_per_epoch(
        train_samples=prepared.train_samples,
        batch_size=batch_size,
        packing_factor=packing_factor,
    )
    total_steps = steps_per_epoch * epochs

    # Determine effective max_steps
    user_max_steps = customizer_config.schedule.max_steps
    if user_max_steps and user_max_steps > 0:
        max_steps = min(user_max_steps, total_steps)
    else:
        max_steps = total_steps

    logger.info(
        f"Training schedule: {prepared.train_samples} samples, batch_size={batch_size}, "
        f"packing_factor={packing_factor or 1.0}, steps_per_epoch={steps_per_epoch}, "
        f"epochs={epochs}, max_steps={max_steps}"
    )

    cfg["step_scheduler"] = {
        "global_batch_size": batch_size,
        "local_batch_size": customizer_config.batch.micro_batch_size,
        "max_steps": max_steps,
        "num_epochs": epochs,
    }

    # Ours, not the recipe's -- underscore-prefixed like `_resolved_chat_template`
    # below, which is the existing convention for a key that rides in the recipe
    # config because the config file is the only channel to the training process.
    # AutomodelRecipeWrapper reads it back off `recipe.cfg`.
    cfg["_progress_reporting"] = {
        "time_series_metrics": customizer_config.schedule.progress_reporting.time_series_metrics,
        "min_report_interval_seconds": customizer_config.schedule.progress_reporting.min_report_interval_seconds,
    }

    val_every_steps = compute_val_check_interval(
        steps_per_epoch=steps_per_epoch,
        max_steps=max_steps,
        val_check_interval=customizer_config.schedule.val_check_interval,
    )
    cfg["step_scheduler"]["val_every_steps"] = val_every_steps
    cfg["step_scheduler"]["ckpt_every_steps"] = val_every_steps
    logger.info(f"Validation interval: {customizer_config.schedule.val_check_interval} -> {val_every_steps} steps")

    return max_steps


def _resolve_optimizer_target(customizer_config: TrainingStepConfig, recipe: TrainingRecipe) -> str:
    """Resolve ``auto`` by recipe, then map the optimizer to its implementation.

    Nemotron retrieval defaults ``auto`` to Transformer Engine FusedAdam.
    Existing causal-LM jobs retain torch Adam as their ``auto`` default. Explicit
    choices override either default.
    """
    optimizer_name = customizer_config.optimizer.optimizer_name
    if optimizer_name == "auto":
        optimizer_name = "FusedAdam" if recipe in _RETRIEVAL_RECIPES else "Adam"
    if optimizer_name not in _OPTIMIZER_TARGETS:
        raise ValueError(
            f"Unsupported optimizer_name {optimizer_name!r}; expected 'auto' or one of {sorted(_OPTIMIZER_TARGETS)}."
        )
    return _OPTIMIZER_TARGETS[optimizer_name]


def _build_optimizer(
    cfg: dict[str, Any],
    customizer_config: TrainingStepConfig,
    *,
    max_steps: int,
    optimizer_target: str,
) -> None:
    """Write the optimizer and its learning-rate schedule."""
    warmup_steps = resolve_warmup_steps(
        warmup_steps=customizer_config.optimizer.warmup_steps,
        max_steps=max_steps,
    )

    cfg["optimizer"] = {
        "_target_": optimizer_target,
        "lr": customizer_config.optimizer.learning_rate,
        "weight_decay": customizer_config.optimizer.weight_decay,
        "betas": [customizer_config.optimizer.beta1, customizer_config.optimizer.beta2],
        "eps": customizer_config.optimizer.eps,  # Adam epsilon for numerical stability
    }

    cfg["lr_scheduler"] = {
        "lr_decay_style": customizer_config.optimizer.lr_decay_style,
        "lr_warmup_steps": warmup_steps,
    }
    if customizer_config.optimizer.min_learning_rate:
        cfg["lr_scheduler"]["min_lr"] = customizer_config.optimizer.min_learning_rate


def _build_checkpoint(
    cfg: dict[str, Any],
    customizer_config: TrainingStepConfig,
    workspace_dir: Path,
) -> None:
    cfg["checkpoint"] = {
        "enabled": True,
        "model_save_format": "safetensors",
        "checkpoint_dir": str(workspace_dir / "checkpoints"),
        "save_consolidated": True,
        # Required for models with quantized base weights (e.g., GPT-OSS)
        # Safe to enable even for non-quantized models
        "dequantize_base_checkpoint": True,
        "v4_compatible": customizer_config.model.v4_compatible,
    }


def _build_peft(cfg: dict[str, Any], customizer_config: TrainingStepConfig) -> None:
    """Write the LoRA adapter config for both unmerged and merged LoRA."""
    if customizer_config.training.training_type not in (
        TrainingType.SFT,
        TrainingType.DISTILLATION,
    ):
        return
    if customizer_config.training.finetuning_type not in (
        FinetuningType.LORA,
        FinetuningType.LORA_MERGED,
    ):
        return

    lora = customizer_config.training.lora
    if lora is None:
        lora = LoRAConfig()
    peft_cfg: dict[str, Any] = {
        "_target_": "nemo_automodel.components._peft.lora.PeftConfig",
        "dim": lora.rank,
        "alpha": lora.alpha,
        "dropout": lora.dropout,
        "use_triton": lora.use_triton,
        "target_modules": lora.target_modules,
    }
    if lora.exclude_modules:
        peft_cfg["exclude_modules"] = lora.exclude_modules
    cfg["peft"] = peft_cfg


def _build_integrations(
    cfg: dict[str, Any],
    customizer_config: TrainingStepConfig,
    job_ctx: NMPJobContext,
) -> None:
    """Attach WandB and MLflow when the runtime environment supplies credentials."""
    wandb_config = build_wandb_config(
        customizer_config=customizer_config,
        job_ctx=job_ctx,
        framework="automodel",
    )
    if wandb_config:
        cfg["wandb"] = wandb_config
        logger.info(f"WandB enabled: project={wandb_config.get('project')}")

    mlflow_config = build_mlflow_config(
        customizer_config=customizer_config,
        job_ctx=job_ctx,
        framework="automodel",
    )
    if mlflow_config:
        cfg["mlflow"] = mlflow_config
        logger.info(f"MLflow enabled: {mlflow_config.get('tracking_uri')}")


def estimate_steps_per_epoch(
    *,
    train_samples: int,
    batch_size: int,
    packing_factor: float | None = None,
) -> int:
    """Estimate optimizer steps per epoch, accounting for sequence packing."""
    effective_packing_factor = packing_factor if packing_factor and packing_factor > 1 else 1.0
    return math.ceil(train_samples / (batch_size * effective_packing_factor))


def resolve_warmup_steps(*, warmup_steps: int, max_steps: int) -> int:
    """Return warmup steps that satisfy Automodel's ``lr_warmup_steps < lr_decay_steps``.

    A large global batch size can leave a run with fewer optimizer steps than the
    requested warmup. That is a schedule too short to warm up over rather than a
    misconfiguration, so warm up over the whole run instead of failing.
    """
    if warmup_steps <= 0:
        return warmup_steps

    if warmup_steps < max_steps:
        return warmup_steps

    clamped_warmup_steps = max(0, max_steps - 1)
    logger.warning(
        f"warmup_steps ({warmup_steps}) must be less than lr_decay_steps ({max_steps}); "
        f"clamping warmup_steps to {clamped_warmup_steps}"
    )
    return clamped_warmup_steps


def _configure_moe_backend(
    cfg: dict[str, Any], customizer_config: TrainingStepConfig, trust_remote_code: bool = False
) -> None:
    """
    Configure custom Automodel model implementations for MoE models.

    Automodel has optimized implementations for certain model architectures.
    Only MoE models (those with num_local_experts, num_experts, or n_routed_experts in config)
    require additional backend and parallelizer configuration.

    Dense models like LlamaForCausalLM may have custom Automodel implementations
    (for combined QKV projections, etc.) but don't need MoE-specific config.

    This function:
    1. Detects if the model is an MoE model via config attributes
    2. Only for MoE: Configures the backend (with deepep disabled for stability)
    3. Only for MoE: Configures the parallelizer for expert distribution
    """
    # Import here to avoid ModuleNotFoundError in environments where
    # transformers is not installed (e.g., during test collection)
    from transformers import AutoConfig

    model_path = customizer_config.model.path

    try:
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        architectures = getattr(hf_config, "architectures", None)

        # Check if model has a custom Automodel implementation
        has_custom_impl = (
            architectures and len(architectures) > 0 and architectures[0] in ModelRegistry.model_arch_name_to_cls
        )

        if has_custom_impl:
            # Check if model is MoE by looking for expert-related config attributes
            # MoE models use num_local_experts (Mixtral-style), num_experts (older), or n_routed_experts (NemotronH)
            num_experts = (
                getattr(hf_config, "num_local_experts", None)
                or getattr(hf_config, "num_experts", None)
                or getattr(hf_config, "n_routed_experts", None)
            )
            is_moe_model = num_experts is not None and num_experts > 1
            if is_moe_model:
                logger.info(
                    f"Detected MoE model with custom Automodel implementation for architecture: {architectures[0]}. "
                    f"Adding MoE-specific configurations (num_experts={num_experts})."
                )

                # Validate MoE parallelism constraints.
                # Automodel's MoE parallelizer does not support tensor parallelism:
                #   assert tp_axis_name is None or world_mesh[tp_axis_name].size() == 1
                # See: nemo_automodel/components/moe/parallelizer.py
                p = customizer_config.parallelism
                total_gpus = p.num_nodes * p.num_gpus_per_node
                if total_gpus > 1:
                    if p.tensor_parallel_size > 1:
                        raise ValueError(
                            f"Tensor parallelism (tensor_parallel_size={p.tensor_parallel_size}) is not supported for MoE models."
                        )
                    ep = p.expert_parallel_size
                    if ep is None or ep <= 1:
                        raise ValueError(
                            f"MoE model detected (num_experts={num_experts}) but expert_parallel_size "
                            f"is {ep or 'not set'}. Multi-GPU MoE training requires expert_parallel_size > 1."
                        )

                # Backend configuration for MoE models
                # DeepEP is disabled for stability - it's a newer feature that can cause issues
                cfg.setdefault("model", {})["backend"] = {
                    "_target_": "nemo_automodel.components.models.common.utils.BackendConfig",
                    "enable_deepep": False,
                }

            else:
                logger.info(
                    f"Detected custom Automodel implementation for architecture: {architectures[0]}. "
                    "Not an MoE model, skipping MoE-specific configurations."
                )
        else:
            logger.debug(
                f"No custom Automodel implementation found for {model_path}. "
                "Using standard HuggingFace model implementation."
            )
    except ValueError:
        raise  # Re-raise validation errors
    except Exception as e:
        # Don't fail training if we can't check for custom implementations
        logger.warning(
            f"Failed to check for custom model implementation: {e}. Using standard HuggingFace model implementation."
        )


def _configure_datasets(
    cfg: dict[str, Any],
    customizer_config: TrainingStepConfig,
    prepared: PreparedDataset,
    seq_length: int,
    seed: int,
    recipe: TrainingRecipe = TrainingRecipe.SFT,
    retrieval_config: RetrievalConfig | None = None,
) -> None:
    """
    Configure dataset sections based on detected schema.

    Supports:
    - Chat format (OpenAI messages): Uses ChatDataset
    - SFT format (prompt/completion): Uses ColumnMappedTextInstructionDataset
    - Custom format (via prompt_template): Uses ColumnMappedTextInstructionDataset with custom columns
    - Embedding format (query/pos_doc/neg_doc): Uses inline retrieval dataset

    Args:
        cfg: Configuration dictionary to populate.
        customizer_config: Training step configuration.
        prepared: Prepared dataset with merged train/val files.
        seq_length: Effective sequence length for dataset configuration.
            When sequence packing is enabled, this is the pack size.
            Otherwise, this is the model's max_seq_length.
        seed: Random seed for reproducibility.
        recipe: Selected training recipe.
        retrieval_config: Retrieval model configuration (required for retrieval datasets).
    """
    train_file = prepared.train_file
    validation_file = prepared.validation_file

    # Detect schema from training data
    schema, column_keys = detect_dataset_schema(
        train_file,
        prompt_template=customizer_config.dataset.prompt_template,
    )

    is_retrieval = recipe in (TrainingRecipe.BI_ENCODER, TrainingRecipe.CROSS_ENCODER)

    # Validate the dataset against the selected recipe, not the checkpoint's original head.
    if is_retrieval and schema != DatasetSchema.EMBEDDING:
        raise ValueError(
            f"Recipe '{recipe.value}' requires a retrieval dataset, but the dataset "
            f"is in '{schema.value}' format. Retrieval recipes require 'query', 'pos_doc', "
            "and 'neg_doc' fields. Please provide a dataset in embedding format."
        )
    if schema == DatasetSchema.EMBEDDING and not is_retrieval:
        raise ValueError(
            "Dataset is in retrieval format (query/pos_doc/neg_doc), but the selected "
            f"recipe is '{recipe.value}'. Select 'bi_encoder' or 'cross_encoder'."
        )

    if schema == DatasetSchema.EMBEDDING:
        # Embedding/retrieval dataset - uses inline format directly
        if retrieval_config is None:
            raise ValueError("retrieval_config is required for retrieval dataset configuration")
        _configure_retrieval_dataset(
            cfg,
            customizer_config,
            train_file,
            validation_file,
            seed,
            retrieval_config,
            recipe,
        )
    elif schema == DatasetSchema.CHAT:
        # Chat dataset (OpenAI messages format)
        _configure_chat_dataset(cfg, customizer_config, train_file, validation_file, seq_length)
    else:
        # SFT/Custom dataset (prompt/completion or custom columns)
        assert column_keys is not None, "column_keys must be set for SFT/CUSTOM schema"
        question_col, answer_col = column_keys
        _configure_sft_dataset(
            cfg,
            customizer_config,
            train_file,
            validation_file,
            question_col,
            answer_col,
            seq_length,
        )


def _configure_chat_dataset(
    cfg: dict[str, Any],
    customizer_config: TrainingStepConfig,
    train_file: Path,
    val_file: Path,
    seq_length: int,
) -> None:
    """Configure ChatDataset for OpenAI messages format."""
    logger.info(f"Configuring ChatDataset for chat format data with seq_length={seq_length}")

    # Resolve chat template using priority-based selection:
    # 1. Fileset metadata chat_template (from model entity spec, highest priority)
    # 2. Custom template from DEFAULT_CHAT_TEMPLATES (if model.name matches)
    # 3. Model's built-in tokenizer template (fallback)
    chat_template = resolve_chat_template(
        model_path=customizer_config.model.path,
        model_name=customizer_config.model.name,
        user_template=customizer_config.model.chat_template,
    )
    pp_enabled = customizer_config.parallelism.pipeline_parallel_size > 1
    # Note: "split" is required by Automodel's pack_dataset() when sequence packing is enabled.
    # Without it, build_dataloader() raises AttributeError accessing cfg_ds.split.
    cfg["dataset"] = {
        "_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset",
        "path_or_dataset_id": str(train_file),
        "split": "train",
        "seq_length": seq_length,
        "padding": "do_not_pad" if not pp_enabled else "max_length",
    }
    cfg["validation_dataset"] = {
        "_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset",
        "path_or_dataset_id": str(val_file),
        "split": "validation",
        "seq_length": seq_length,
        "padding": "do_not_pad" if not pp_enabled else "max_length",
    }

    # Add chat template if available
    if chat_template:
        cfg["dataset"]["chat_template"] = chat_template
        cfg["validation_dataset"]["chat_template"] = chat_template
        logger.info("Added chat template to dataset config")
    else:
        logger.warning("No chat template found - ChatDataset may fail")

    # Store resolved template in config for checkpoint processing
    # This ensures the same template is used during training and applied to output
    cfg["_resolved_chat_template"] = chat_template


def _configure_sft_dataset(
    cfg: dict[str, Any],
    customizer_config: TrainingStepConfig,
    train_file: Path,
    val_file: Path,
    question_col: str,
    answer_col: str,
    seq_length: int,
) -> None:
    """Configure ColumnMappedTextInstructionDataset for SFT/custom format."""
    logger.info(
        f"Configuring SFT dataset with columns: question={question_col}, answer={answer_col}, seq_length={seq_length}"
    )
    pp_enabled = customizer_config.parallelism.pipeline_parallel_size > 1
    # Note: "split" is required by Automodel's pack_dataset() when sequence packing is enabled.
    # Without it, build_dataloader() raises AttributeError accessing cfg_ds.split.
    cfg["dataset"] = {
        "_target_": "nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset",
        "path_or_dataset_id": str(train_file),
        "split": "train",
        "column_mapping": {
            "question": question_col,
            "answer": answer_col,
        },
        "seq_length": seq_length,
        "answer_only_loss_mask": True,
        "padding": "do_not_pad" if not pp_enabled else "max_length",
        "truncation": "longest_first",
    }
    cfg["validation_dataset"] = {
        "_target_": "nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset",
        "path_or_dataset_id": str(val_file),
        "split": "validation",
        "column_mapping": {
            "question": question_col,
            "answer": answer_col,
        },
        "seq_length": seq_length,
        "answer_only_loss_mask": True,
        "padding": "do_not_pad" if not pp_enabled else "max_length",
        "truncation": "longest_first",
    }


def _configure_retrieval_dataset(
    cfg: dict[str, Any],
    customizer_config: TrainingStepConfig,
    train_file: Path,
    val_file: Path,
    seed: int,
    retrieval_config: RetrievalConfig,
    recipe: TrainingRecipe,
) -> None:
    """Configure inline retrieval data for bi-encoder or cross-encoder training.

    Uses Automodel's inline retrieval dataset format which directly accepts
    Customizer's embedding format without conversion:
        {"query": "...", "pos_doc": "...", "neg_doc": ["...", "..."]}

    This uses retrieval_dataset_inline.make_retrieval_dataset which handles:
        - Loading inline text directly from JSONL
        - BiEncoderCollator for tokenization and batching

    Args:
        cfg: Configuration dictionary to populate.
        customizer_config: Training step configuration.
        train_file: Path to training JSONL file.
        val_file: Path to validation JSONL file.
        seed: Random seed for reproducibility.
        retrieval_config: Retrieval dataset, collator, and export configuration.
    """

    model_type = "bi_encoder" if recipe == TrainingRecipe.BI_ENCODER else "cross_encoder"
    collator_target = (
        "nemo_automodel.components.datasets.llm.BiEncoderCollator"
        if recipe == TrainingRecipe.BI_ENCODER
        else "nemo_automodel.components.datasets.llm.CrossEncoderCollator"
    )
    logger.info(
        "Configuring %s dataset with train_n_passages=%s",
        model_type,
        retrieval_config.train_n_passages,
    )

    def collator_config(*, validation: bool = False) -> dict[str, Any]:
        if recipe == TrainingRecipe.CROSS_ENCODER:
            return {
                "_target_": collator_target,
                "rerank_max_length": retrieval_config.passage_max_length,
                "prompt_template": "question:{query} \n \n passage:{passage}",
                "pad_to_multiple_of": 8,
            }
        result: dict[str, Any] = {
            "_target_": collator_target,
            "q_max_len": retrieval_config.query_max_length,
            "p_max_len": retrieval_config.passage_max_length,
            "query_prefix": _collator_prefix(retrieval_config.query_prefix),
            "passage_prefix": _collator_prefix(retrieval_config.passage_prefix),
            "pad_to_multiple_of": 8,
        }
        if validation:
            result["padding"] = "longest"
        return result

    cfg["dataloader"] = {
        "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
        "dataset": {
            "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset",
            "model_type": model_type,
            "data_dir_list": [str(train_file)],
            "data_type": "train",
            "n_passages": retrieval_config.train_n_passages,
            "seed": seed,
            "do_shuffle": True,
        },
        "collate_fn": collator_config(),
        "shuffle": True,
        "num_workers": 0,
    }

    if val_file and val_file.exists():
        cfg["validation_dataloader"] = {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "dataset": {
                "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset",
                "model_type": model_type,
                "data_dir_list": [str(val_file)],
                "data_type": "eval",
                "n_passages": retrieval_config.train_n_passages,
                "eval_negative_size": get_eval_negative_size(retrieval_config),
                "seed": seed,
                "do_shuffle": False,
            },
            "collate_fn": collator_config(validation=True),
            "batch_size": customizer_config.batch.micro_batch_size,
            "shuffle": False,
            "num_workers": 0,
        }


def _verify_tokenizer_compatibility(student_path: str, teacher_path: str, trust_remote_code: bool = False) -> None:
    """
    Verify that student and teacher models have compatible tokenizers.

    Knowledge distillation requires the student and teacher to have the same
    vocabulary so their logit spaces are aligned. This check prevents subtle
    bugs where training appears to work but produces garbage outputs.

    Raises:
        ValueError: If tokenizers are incompatible
    """
    # Import here to avoid ModuleNotFoundError in environments where
    # transformers is not installed (e.g., during test collection)
    from transformers import AutoTokenizer

    try:
        student_tokenizer = AutoTokenizer.from_pretrained(student_path, trust_remote_code=trust_remote_code)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=trust_remote_code)

        if student_tokenizer.vocab_size != teacher_tokenizer.vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary size mismatch: student has {student_tokenizer.vocab_size} tokens, "
                f"teacher has {teacher_tokenizer.vocab_size} tokens. "
                "Knowledge distillation requires matching vocabularies."
            )

        # Optional: Could also check for specific token mismatches
        logger.info(f"Tokenizer compatibility verified: both models have vocab_size={student_tokenizer.vocab_size}")

    except Exception as e:
        if "vocabulary size mismatch" in str(e):
            raise
        # Log but don't fail for other tokenizer loading issues
        # (e.g., network issues, missing files) - the training will fail later with a clearer error
        logger.warning(f"Could not verify tokenizer compatibility: {e}")


def _configure_kd(cfg: dict[str, Any], customizer_config: TrainingStepConfig, trust_remote_code: bool = False) -> None:
    """
    Configure Knowledge Distillation for Automodel's KD recipe.

    Automodel's KnowledgeDistillationRecipeForNextTokenPrediction requires:
    - teacher_model: Frozen teacher model for soft targets
    - kd_ratio: Balance between CE and KD loss (0=CE only, 1=KD only)
    - kd_loss_fn: KL-divergence loss with temperature scaling
    - offload_teacher_model: Optional CPU offloading for memory efficiency
    """
    kd_config = customizer_config.training.kd
    if not kd_config or not kd_config.teacher_model:
        raise ValueError(
            "Knowledge distillation requires training.kd.teacher to be set. "
            "Ensure the job input includes a teacher model."
        )

    # Verify tokenizer compatibility before proceeding
    _verify_tokenizer_compatibility(
        customizer_config.model.path,
        kd_config.teacher_model.path,
        trust_remote_code=trust_remote_code,
    )

    # Teacher model (frozen, same architecture loading as student)
    # Use teacher's precision if specified, otherwise fall back to student's precision
    teacher_precision = kd_config.teacher_model.precision or customizer_config.model.precision
    cfg["teacher_model"] = {
        "_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
        "pretrained_model_name_or_path": kd_config.teacher_model.path,
        "torch_dtype": teacher_precision.to_torch_dtype() if teacher_precision else "auto",
        "attn_implementation": kd_config.teacher_model.attn_implementation,
        "trust_remote_code": kd_config.teacher_model.trust_remote_code,
    }

    # KD loss function with temperature
    cfg["kd_loss_fn"] = {
        "_target_": "nemo_automodel.components.loss.kd_loss.KDLoss",
        "ignore_index": -100,
        "temperature": kd_config.temperature,
        "fp32_upcast": True,  # Recommended for numerical stability
    }

    # KD ratio (blend between CE and KD loss)
    cfg["kd_ratio"] = kd_config.ratio

    # Optional: Offload teacher to CPU for memory efficiency
    if kd_config.offload_teacher:
        cfg["offload_teacher_model"] = True
        logger.info("Teacher model will be offloaded to CPU between forward passes")


def get_eval_negative_size(retrieval_config: RetrievalConfig) -> int:
    """Get the effective eval_negative_size value from the retrieval config.

    Returns the user-specified eval_negative_size if set, otherwise defaults
    to train_n_passages - 1 for consistent train/eval behavior.

    The -1 relationship exists because:
    - train_n_passages = total passages = 1 positive + N negatives
    - eval_negative_size = just the negative count = N
    - So: eval_negative_size = train_n_passages - 1 (subtracting the positive)

    Example: train_n_passages=5 (1 pos + 4 neg) -> eval_negative_size=4
    """
    if retrieval_config.eval_negative_size is not None:
        return retrieval_config.eval_negative_size
    return retrieval_config.train_n_passages - 1

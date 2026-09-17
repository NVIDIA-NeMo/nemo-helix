# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert plugin ``AutomodelJobOutput`` shape to legacy ``CustomizationJobOutput`` for the compiler."""

from __future__ import annotations

from typing import Any, Literal

from nemo_platform_plugin.integrations import IntegrationsSpec
from nmp.automodel.api.v2.jobs.schemas import (
    CustomizationJobOutput,
    DeploymentParams,
    DistillationTraining,
    LoRAParams,
    OutputResponse,
    ParallelismParams,
    SFTTraining,
)
from nmp.customization_common.training.reporting import ProgressReportingConfig
from pydantic import BaseModel


def _map_finetuning_type(value: str) -> str:
    if value == "all_weights":
        return "all_weights"
    if value == "lora_merged":
        return "lora_merged"
    return "lora"


def _build_peft(training: dict[str, Any]) -> LoRAParams | None:
    ft = training.get("finetuning_type", "lora")
    if ft == "all_weights":
        return None
    lora = training.get("lora") or {}
    return LoRAParams(
        rank=lora.get("rank", 16),
        alpha=lora.get("alpha", 32),
        dropout=lora.get("dropout", 0.0),
        merge=ft == "lora_merged" or lora.get("merge", False),
        target_modules=lora.get("target_modules"),
        exclude_modules=lora.get("exclude_modules"),
        use_triton=lora.get("use_triton", True),
    )


def _build_training_block(spec: dict[str, Any]) -> SFTTraining | DistillationTraining:
    training = spec["training"]
    schedule = spec.get("schedule") or {}
    batch = spec.get("batch") or {}
    optimizer = spec.get("optimizer") or {}
    parallelism = spec.get("parallelism") or {}

    common: dict[str, Any] = {
        "recipe": training.get("recipe", "auto"),
        "peft": _build_peft(training),
        "min_learning_rate": optimizer.get("min_learning_rate"),
        "weight_decay": optimizer.get("weight_decay", 0.01),
        "adam_beta1": optimizer.get("adam_beta1", 0.9),
        "adam_beta2": optimizer.get("adam_beta2", 0.999),
        "adam_eps": optimizer.get("adam_eps", 1e-8),
        "optimizer": optimizer.get("optimizer", "auto"),
        "lr_decay_style": optimizer.get("lr_decay_style", "cosine"),
        "epochs": schedule.get("epochs", 1),
        "max_steps": schedule.get("max_steps"),
        "val_check_interval": schedule.get("val_check_interval"),
        "validation_split": schedule.get("validation_split"),
        # Absent from a spec compiled before this knob existed, and `or` rather
        # than a `.get` default because model_dump renders an unset nested model
        # as None on some paths; either way the field's own default applies.
        "progress_reporting": schedule.get("progress_reporting") or ProgressReportingConfig(),
        "sequence_packing": batch.get("sequence_packing", False),
        "sequence_packing_max_samples": batch.get("sequence_packing_max_samples", 1000),
        "max_seq_length": training.get("max_seq_length", 2048),
        "precision": training.get("precision"),
        "attn_implementation": training.get("attn_implementation", "sdpa"),
        "seed": schedule.get("seed"),
        "retrieval": training.get("retrieval"),
        "parallelism": ParallelismParams(
            num_nodes=parallelism.get("num_nodes", 1),
            num_gpus_per_node=parallelism.get("num_gpus_per_node", 1),
            tensor_parallel_size=parallelism.get("tensor_parallel_size", 1),
            pipeline_parallel_size=parallelism.get("pipeline_parallel_size", 1),
            context_parallel_size=parallelism.get("context_parallel_size", 1),
            expert_parallel_size=parallelism.get("expert_parallel_size"),
            sequence_parallel=parallelism.get("sequence_parallel", False),
        ),
        "execution_profile": training.get("execution_profile"),
    }

    # Only keys the stored spec actually carries: anything set here lands in
    # `model_fields_set`, which `with_resolved_recipe` reads as an explicit choice
    # and then skips its recipe defaults for.
    for target, source, key in (
        ("batch_size", batch, "global_batch_size"),
        ("micro_batch_size", batch, "micro_batch_size"),
        ("learning_rate", optimizer, "learning_rate"),
        ("warmup_steps", optimizer, "warmup_steps"),
    ):
        if source.get(key) is not None:
            common[target] = source[key]

    training_type: Literal["sft", "distillation"] = training.get("training_type", "sft")
    if training_type == "distillation":
        return DistillationTraining(
            **common,
            teacher_model=training["teacher_model"],
            teacher_precision=training.get("teacher_precision", "bf16"),
            distillation_ratio=training.get("distillation_ratio", 0.5),
            distillation_temperature=training.get("distillation_temperature", 1.0),
        )
    return SFTTraining(**common)


def _build_integrations(spec: dict[str, Any]) -> IntegrationsSpec | None:
    raw = spec.get("integrations")
    if not raw:
        return None
    return IntegrationsSpec.model_validate(raw)


def _build_deployment_config(data: dict[str, Any]) -> str | DeploymentParams | None:
    """Map the plugin's ``deployment_config`` onto the legacy compiler shape.

    String references pass through untouched; inline params arrive as a plain dict
    because the caller dumped the plugin model, so they are re-validated here.
    """
    raw = data.get("deployment_config")
    if raw is None or isinstance(raw, str):
        return raw
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode="python")
    return DeploymentParams.model_validate(raw)


def automodel_spec_to_compiler_output(spec: dict[str, Any] | BaseModel) -> CustomizationJobOutput:
    """Map simplified Automodel job output (plugin schema) to ``CustomizationJobOutput``."""
    if isinstance(spec, BaseModel):
        data = spec.model_dump(mode="python")
    else:
        data = dict(spec)

    dataset = data["dataset"]
    training_uri = dataset["training"] if isinstance(dataset, dict) else dataset

    output = data["output"]
    if isinstance(output, dict):
        out_type = output.get("type", "model")
        output_resp = OutputResponse(
            name=output["name"],
            type=out_type,
            fileset=output["fileset"],
        )
    else:
        output_resp = output

    return CustomizationJobOutput(
        model=data["model"],
        dataset=training_uri,
        training=_build_training_block(data),
        integrations=_build_integrations(data),
        deployment_config=_build_deployment_config(data),
        output=output_resp,
    )

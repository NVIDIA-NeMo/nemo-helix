# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map shared integration configs onto HuggingFace ``SFTConfig`` + env vars."""

import json
import logging
from pathlib import Path
from typing import Any

from nemo_helix_plugin.integrations import IntegrationsSpec
from nhx.customization_common.integrations import (
    IntegrationRuntimeContext,
    build_mlflow_config,
    build_wandb_config,
)
from nhx.customization_common.service.context import NHXJobContext

logger = logging.getLogger(__name__)


def apply_integrations_to_sft_config(
    *,
    integrations: IntegrationsSpec | None,
    job_ctx: NHXJobContext,
    output_name: str,
    workspace_path: Path | str,
    model_name: str | None,
) -> tuple[list[str], dict[str, Any], dict[str, str]]:
    """Apply integrations to HF Trainer and return ``(report_to, sft_kwargs, env)``.

    Uses the shared runtime builders so Unsloth gets the same defaults and
    validation as Automodel. Backends are included in ``report_to`` only when
    the shared builder activates them (e.g. W&B requires ``WANDB_API_KEY``).

    HuggingFace ``TrainingArguments.run_name`` is shared by W&B and MLflow
    callbacks. When both integrations are active, ``wandb.name`` wins if set;
    otherwise ``mlflow.name`` is used.

    Environment variables are returned for the caller to apply — this function
    does not mutate ``os.environ``.
    """
    ctx = IntegrationRuntimeContext.from_integrations_spec(
        integrations=integrations,
        output_name=output_name,
        workspace_path=str(workspace_path),
        model_name=model_name,
        job_ctx=job_ctx,
        framework="unsloth",
    )

    report_to: list[str] = []
    sft_kwargs: dict[str, Any] = {}
    env: dict[str, str] = {}

    wandb_config = build_wandb_config(ctx)
    mlflow_config = build_mlflow_config(ctx)

    if integrations and integrations.wandb and integrations.mlflow:
        wandb_name = integrations.wandb.name
        mlflow_name = integrations.mlflow.name
        if wandb_name and mlflow_name and wandb_name != mlflow_name:
            logger.warning(
                "integrations.wandb.name (%s) and integrations.mlflow.name (%s) differ; "
                "HuggingFace TrainingArguments.run_name can only hold one value — using wandb.name.",
                wandb_name,
                mlflow_name,
            )

    if wandb_config:
        report_to.append("wandb")
        if project := wandb_config.get("project"):
            env["WANDB_PROJECT"] = project
        if entity := wandb_config.get("entity"):
            env["WANDB_ENTITY"] = entity
        if notes := wandb_config.get("notes"):
            env["WANDB_NOTES"] = notes
        if tags := wandb_config.get("tags"):
            env["WANDB_TAGS"] = ",".join(tags)
        if base_url := (wandb_config.get("settings") or {}).get("base_url"):
            env["WANDB_BASE_URL"] = base_url
        if wandb_dir := wandb_config.get("dir"):
            env["WANDB_DIR"] = wandb_dir

    if mlflow_config:
        report_to.append("mlflow")
        env["MLFLOW_TRACKING_URI"] = mlflow_config["tracking_uri"]
        if experiment_name := mlflow_config.get("experiment_name"):
            env["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        if tags := mlflow_config.get("tags"):
            env["MLFLOW_TAGS"] = json.dumps(tags)

    if wandb_config and wandb_config.get("name"):
        sft_kwargs["run_name"] = wandb_config["name"]
    elif mlflow_config and mlflow_config.get("run_name"):
        sft_kwargs["run_name"] = mlflow_config["run_name"]

    return report_to or ["none"], sft_kwargs, env

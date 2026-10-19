# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared integration config helpers for training backends."""

import logging
import os
from pathlib import Path
from typing import Any

from nhx.customization_common.service.context import NHXJobContext
from nhx.rl.app.jobs.training.schemas import TrainingStepConfig

logger = logging.getLogger(__name__)


def _resolve_with_fallback(
    primary: str | None,
    fallback: str | None,
    default: str,
    field_label: str | None = None,
) -> str:
    """Pick the first truthy value from *primary* → *fallback* → *default*.

    When *field_label* is given and neither *primary* nor *fallback* is set,
    a warning is logged so operators know a hardcoded default is in use.
    """
    if field_label and not (primary or fallback):
        logger.warning(f"{field_label} is not set; using fallback '{default}'.")
    return primary or fallback or default


def build_mlflow_config(
    customizer_config: TrainingStepConfig,
    job_ctx: NHXJobContext,
    framework: str,
) -> dict[str, Any] | None:
    """Build MLflow config shared across training backends.
    The resulting dict is passed to MLflow logging setup by a backend.

    Run naming strategy (same as WandB):
    - run_name uses job_id (stable across pause/resume)
    - task_id is added to tags for granular execution tracking

    Missing tracking URI disables integration with a warning.
    """
    user_config = customizer_config.integrations.mlflow
    if not user_config:
        return None

    # User-provided tracking URI takes precedence over environment variable
    tracking_uri = user_config.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.warning(
            "MLflow integration is configured but no tracking URI is set "
            "(MLFLOW_TRACKING_URI env var and integrations.mlflow.tracking_uri in job POST request are empty); "
            "MLflow integration will be disabled."
        )
        return None

    tags: dict[str, str] = {
        "service": "rl",
        "framework": framework,
    }
    if job_ctx.workspace:
        tags["workspace"] = job_ctx.workspace
    if job_ctx.job_id:
        tags["job"] = job_ctx.job_id
    if job_ctx.task:
        tags["task"] = job_ctx.task
    if customizer_config.model.name:
        tags["model_name"] = customizer_config.model.name

    # User-provided tags override defaults above
    if user_config.tags:
        tags.update(user_config.tags)
    if user_config.description:
        # MLflow run description is stored in the reserved `mlflow.note.content` tag.
        # See: https://mlflow.org/docs/latest/ml/tracking/#how-to-include-additional-description-texts-about-the-run
        tags["mlflow.note.content"] = user_config.description

    experiment_name = _resolve_with_fallback(
        user_config.experiment_name,
        customizer_config.output_model,
        "default-experiment",
        field_label="MLflow experiment_name",
    )
    run_name = _resolve_with_fallback(
        user_config.run_name,
        job_ctx.job_id,
        "default-run",
        field_label="MLflow run_name",
    )

    mlflow_config: dict[str, Any] = {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "tags": tags,
    }

    return mlflow_config


def build_wandb_config(
    customizer_config: TrainingStepConfig,
    job_ctx: NHXJobContext,
    framework: str,
) -> dict[str, Any] | None:
    """Build WandB config shared across training backends.

    The resulting dict is passed to wandb.init() as kwargs.
    See: https://docs.wandb.ai/ref/python/init

    TODO: Add pause/resume support:
    - 'name' and 'id' use job_id (stable across pause/resume)
    - 'resume="allow"' enables continuing runs after pause/resume
    """
    user_config = customizer_config.integrations.wandb
    if not user_config:
        return None

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not user_config.base_url and not wandb_api_key:
        logger.warning("WandB API key is not set and no base_url is provided, skipping WandB integration")
        return None

    # Note: This is semantically different from job_ctx.workspace.
    # This is the workspace for training artifacts.
    run_dir = Path(customizer_config.workspace_path) / "wandb"

    tags: list[str] = ["service:rl", f"framework:{framework}"]
    if job_ctx.workspace:
        tags.append(f"workspace:{job_ctx.workspace}")
    if job_ctx.job_id:
        tags.append(f"job:{job_ctx.job_id}")
    if job_ctx.task:
        tags.append(f"task:{job_ctx.task}")
    if customizer_config.model.name:
        tags.append(f"model:{customizer_config.model.name}")
    # User-provided tags are appended (can override tags above)
    if user_config.tags:
        tags.extend(user_config.tags)

    wandb_config: dict[str, Any] = {
        "project": _resolve_with_fallback(user_config.project, customizer_config.output_model, "default-project"),
        "name": _resolve_with_fallback(user_config.name, job_ctx.job_id, "default-run"),
        "dir": str(run_dir),
        "tags": tags,
    }
    if user_config.entity:
        wandb_config["entity"] = user_config.entity
    if user_config.notes:
        wandb_config["notes"] = user_config.notes
    if user_config.base_url:
        # For self-hosted W&B servers, base_url is passed via the settings dict
        # (wandb.init accepts settings as Union[Settings, Dict[str, Any], None]).
        logger.info(f"Using self-hosted W&B server: {user_config.base_url}")
        wandb_config["settings"] = {"base_url": user_config.base_url}

    return wandb_config

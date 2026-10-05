# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build W&B and MLflow runtime configs for customization training backends."""

import logging
import os
from pathlib import Path
from typing import Any

from nhx.customization_common.integrations.context import IntegrationRuntimeContext

logger = logging.getLogger(__name__)


def _resolve_wandb_dir(workspace_path: str) -> Path:
    """Return a W&B run directory outside model artifact upload trees.

    Container jobs write uploadable checkpoints under ``output_model`` (unsloth)
    or keep training scratch under ``training`` (automodel). W&B metadata must
    not live in those trees — use sibling ``ephemeral/wandb`` instead.
    """
    workspace = Path(workspace_path)
    if workspace.name in ("output_model", "training"):
        return workspace.parent / "ephemeral" / "wandb"
    return workspace / "wandb"


def _resolve_with_fallback(
    primary: str | None,
    fallback: str | None,
    default: str,
    field_label: str | None = None,
) -> str:
    """Pick the first truthy value from *primary* → *fallback* → *default*."""
    if field_label and not (primary or fallback):
        logger.warning(f"{field_label} is not set; using fallback '{default}'.")
    return primary or fallback or default


def build_mlflow_config(ctx: IntegrationRuntimeContext) -> dict[str, Any] | None:
    """Build MLflow config passed to backend logging setup.

    Run naming strategy (same as W&B):
    - ``name`` on input resolves to ``run_name`` in the output dict (defaults to job_id)
    - task_id is added to tags for granular execution tracking

    Missing tracking URI disables integration with a warning.
    """
    user_config = ctx.mlflow
    if not user_config:
        return None

    tracking_uri = user_config.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.warning(
            "MLflow integration is configured but no tracking URI is set "
            "(MLFLOW_TRACKING_URI env var and integrations.mlflow.tracking_uri in job POST request are empty); "
            "MLflow integration will be disabled."
        )
        return None

    tags: dict[str, str] = {
        "service": "nemo-helix",
        "framework": ctx.framework,
    }
    if ctx.job_ctx.workspace:
        tags["workspace"] = ctx.job_ctx.workspace
    if ctx.job_ctx.job_id:
        tags["job"] = ctx.job_ctx.job_id
    if ctx.job_ctx.task:
        tags["task"] = ctx.job_ctx.task
    if ctx.model_name:
        tags["model_name"] = ctx.model_name

    if user_config.tags:
        tags.update(user_config.tags)
    if user_config.description:
        tags["mlflow.note.content"] = user_config.description

    experiment_name = _resolve_with_fallback(
        user_config.experiment_name,
        ctx.output_name,
        "default-experiment",
        field_label="MLflow experiment_name",
    )
    run_name = _resolve_with_fallback(
        user_config.name,
        ctx.job_ctx.job_id,
        "default-run",
        field_label="MLflow name",
    )

    return {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "run_name": run_name,
        "tags": tags,
    }


def build_wandb_config(ctx: IntegrationRuntimeContext) -> dict[str, Any] | None:
    """Build W&B config passed to ``wandb.init()`` by Automodel training.

    See: https://docs.wandb.ai/ref/python/init
    """
    user_config = ctx.wandb
    if not user_config:
        return None

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        if user_config.base_url:
            logger.warning(
                "WANDB_API_KEY is not set; attempting W&B with base_url only (%s). "
                "This only works when the server allows access without a cloud API key.",
                user_config.base_url,
            )
        else:
            logger.warning("WandB API key is not set and no base_url is provided, skipping WandB integration")
            return None

    run_dir = _resolve_wandb_dir(ctx.workspace_path)

    tags: list[str] = ["service:nemo-helix", f"framework:{ctx.framework}"]
    if ctx.job_ctx.workspace:
        tags.append(f"workspace:{ctx.job_ctx.workspace}")
    if ctx.job_ctx.job_id:
        tags.append(f"job:{ctx.job_ctx.job_id}")
    if ctx.job_ctx.task:
        tags.append(f"task:{ctx.job_ctx.task}")
    if ctx.model_name:
        tags.append(f"model:{ctx.model_name}")
    if user_config.tags:
        tags.extend(user_config.tags)

    wandb_config: dict[str, Any] = {
        "project": _resolve_with_fallback(user_config.project, ctx.output_name, "default-project"),
        "name": _resolve_with_fallback(user_config.name, ctx.job_ctx.job_id, "default-run"),
        "dir": str(run_dir),
        "tags": tags,
    }
    if user_config.entity:
        wandb_config["entity"] = user_config.entity
    if user_config.notes:
        wandb_config["notes"] = user_config.notes
    if user_config.base_url:
        logger.info(f"Using self-hosted W&B server: {user_config.base_url}")
        wandb_config["settings"] = {"base_url": user_config.base_url}

    return wandb_config

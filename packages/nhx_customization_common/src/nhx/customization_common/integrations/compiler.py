# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile-time helpers for experiment-tracking integrations."""

import logging

from nemo_helix_plugin.integrations import IntegrationsSpec
from nemo_helix_plugin.jobs.api_factory import (
    EnvironmentVariable,
    EnvironmentVariableFromSecret,
)

logger = logging.getLogger(__name__)


def warn_incomplete_integrations(integrations: IntegrationsSpec | None) -> None:
    """Log when integrations are requested but may not activate at runtime."""
    if integrations and integrations.wandb and not integrations.wandb.api_key_secret:
        logger.warning(
            "integrations.wandb is configured but api_key_secret is missing; "
            "W&B will only activate if WANDB_API_KEY is already set in the training container."
        )
    if integrations and integrations.mlflow and not integrations.mlflow.tracking_uri:
        logger.warning(
            "integrations.mlflow is configured but tracking_uri is missing; "
            "MLflow will only activate if MLFLOW_TRACKING_URI is already set in the training container."
        )


def collect_integration_secret_envs(
    integrations: IntegrationsSpec | None,
) -> list[EnvironmentVariable]:
    """Collect secret environment variables from integration configs.

    Secrets are propagated via ``PlatformJobStep.environment`` (not step config)
    so the Jobs service can resolve secret references at runtime.
    """
    if not integrations or not integrations.wandb or not integrations.wandb.api_key_secret:
        return []

    return [
        EnvironmentVariable(
            name="WANDB_API_KEY",
            from_secret=EnvironmentVariableFromSecret(
                name=integrations.wandb.api_key_secret.root,
            ),
        ),
    ]

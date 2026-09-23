# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin configuration for Automodel training."""

from __future__ import annotations

from nhx.customization_common.contributor.config import BaseTrainingPluginConfig, generate_job_id
from pydantic_settings import SettingsConfigDict


class AutomodelPluginConfig(BaseTrainingPluginConfig):
    """Environment-driven Automodel plugin settings."""

    model_config = SettingsConfigDict(env_prefix="NHX_AUTOMODEL_", extra="ignore")


def get_config() -> AutomodelPluginConfig:
    return AutomodelPluginConfig()


def generate_automodel_id() -> str:
    """Generate a job name when the submitter omits ``name``."""
    return generate_job_id("automodel")

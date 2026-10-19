# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin configuration for Unsloth container-submit training."""

from __future__ import annotations

from nhx.customization_common.contributor.config import BaseTrainingPluginConfig, generate_job_id
from pydantic_settings import SettingsConfigDict


class UnslothPluginConfig(BaseTrainingPluginConfig):
    """Environment-driven Unsloth plugin settings.

    All fields are optional. The only knob the contributor actually
    consumes today is ``default_training_execution_profile`` — forwarded
    into ``add_job_routes`` so the platform's job collection routes have
    a sensible profile when the submitter omits one.
    """

    model_config = SettingsConfigDict(env_prefix="NHX_UNSLOTH_", extra="ignore")


def get_config() -> UnslothPluginConfig:
    return UnslothPluginConfig()


def generate_unsloth_id() -> str:
    """Generate a job name when the submitter omits ``name``."""
    return generate_job_id("unsloth")

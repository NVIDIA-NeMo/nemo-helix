# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin configuration for NeMo-RL DPO training."""

from __future__ import annotations

from nhx.customization_common.contributor.config import BaseTrainingPluginConfig, generate_job_id
from pydantic_settings import SettingsConfigDict


class RlPluginConfig(BaseTrainingPluginConfig):
    """Environment-driven NeMo-RL plugin settings (``NHX_RL_`` prefix)."""

    model_config = SettingsConfigDict(env_prefix="NHX_RL_", extra="ignore")


def get_config() -> RlPluginConfig:
    return RlPluginConfig()


def generate_rl_id() -> str:
    """Generate a job name when the submitter omits ``name``."""
    return generate_job_id("rl")

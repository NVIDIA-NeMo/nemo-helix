# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base plugin config + job-id helper for customization contributor plugins."""

from __future__ import annotations

import uuid

from pydantic_settings import BaseSettings


class BaseTrainingPluginConfig(BaseSettings):
    """Environment-driven settings shared by customization contributor plugins.

    Subclasses set their own ``model_config`` (``env_prefix``) and may add
    backend-specific fields (e.g. automodel's image defaults).
    """

    default_training_execution_profile: str = "gpu"


def generate_job_id(prefix: str) -> str:
    """Generate a unique job name (``<prefix>-<12 hex>``) when the submitter omits one."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

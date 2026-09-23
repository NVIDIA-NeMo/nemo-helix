# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public compile entrypoint for Automodel jobs."""

from __future__ import annotations

from typing import Any

from nemo_helix_plugin.jobs.api_factory import HelixJobSpec
from nhx.automodel.adapter import automodel_spec_to_compiler_output
from nhx.automodel.api.v2.jobs.schemas import CustomizationJobOutput
from nhx.automodel.app.jobs.compiler import platform_job_config_compiler as _compile_canonical
from nhx.customization_common.service.platform_client import AsyncCustomizationHelixClients
from pydantic import BaseModel


async def platform_job_config_compiler(
    job_spec: CustomizationJobOutput | dict[str, Any] | BaseModel,
    workspace: str,
    platform: AsyncCustomizationHelixClients,
    job_name: str | None = None,
    profile: str | None = None,
) -> HelixJobSpec:
    """Compile Automodel job spec (plugin or legacy shape) to HelixJobSpec."""
    del job_name  # reserved for future scheduling decisions
    if not isinstance(job_spec, CustomizationJobOutput):
        job_spec = automodel_spec_to_compiler_output(job_spec)
    if profile is not None and job_spec.training.execution_profile is None:
        job_spec = job_spec.model_copy(
            update={"training": job_spec.training.model_copy(update={"execution_profile": profile})},
        )
    return await _compile_canonical(
        workspace,
        job_spec,
        platform,
    )


__all__ = ["platform_job_config_compiler", "automodel_spec_to_compiler_output"]

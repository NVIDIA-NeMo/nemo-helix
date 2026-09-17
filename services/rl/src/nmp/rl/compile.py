# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public compile entry for nmp-rl jobs.

Mirror of :mod:`nmp.unsloth.compile`. Invoked by the plugin's ``RlJob.compile``
to turn a validated :class:`~nmp.rl.schemas.RlJobOutput` into a 4-step
:class:`PlatformJobSpec` (download → DPO train → upload → model-entity).
"""

from __future__ import annotations

from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec
from nmp.customization_common.service.platform_client import AsyncCustomizationPlatformClients
from nmp.rl.app.jobs.compiler import platform_job_config_compiler as _compile_canonical
from nmp.rl.schemas import RlJobOutput


async def platform_job_config_compiler(
    *,
    workspace: str,
    spec: RlJobOutput,
    platform: AsyncCustomizationPlatformClients,
    job_name: str | None = None,
    profile: str | None = None,
) -> PlatformJobSpec:
    """Compile a canonical NeMo-RL job spec to a ``PlatformJobSpec``. Container submit only."""
    return await _compile_canonical(workspace, spec, platform, job_name=job_name, profile=profile)


__all__ = ["platform_job_config_compiler"]

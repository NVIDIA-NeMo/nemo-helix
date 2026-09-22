# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public compile entry for unsloth jobs.

Mirror of :mod:`nmp.automodel.compile`. Invoked by the plugin's
:meth:`UnslothJob.compile` to turn a validated
:class:`~nmp.unsloth.schemas.UnslothJobOutput` into a 4-step
:class:`PlatformJobSpec` (download → train → upload → model-entity).
"""

from __future__ import annotations

from nemo_platform_plugin.jobs.api_factory import PlatformJobSpec
from nmp.customization_common.service.platform_client import AsyncCustomizationPlatformClients
from nmp.unsloth.app.jobs.compiler import platform_job_config_compiler as _compile_canonical
from nmp.unsloth.schemas import UnslothJobOutput


async def platform_job_config_compiler(
    *,
    workspace: str,
    spec: UnslothJobOutput,
    platform: AsyncCustomizationPlatformClients,
    job_name: str | None = None,
    profile: str | None = None,
) -> PlatformJobSpec:
    """Compile a canonical unsloth job spec to a ``PlatformJobSpec``.

    Used by :meth:`UnslothJob.compile`. Container submit only — Unsloth
    no longer supports local run.
    """
    return await _compile_canonical(
        workspace,
        spec,
        platform,
        job_name=job_name,
        profile=profile,
    )


__all__ = ["platform_job_config_compiler"]

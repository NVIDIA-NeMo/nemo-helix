# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public compile entry for unsloth jobs.

Mirror of :mod:`nhx.automodel.compile`. Invoked by the plugin's
:meth:`UnslothJob.compile` to turn a validated
:class:`~nhx.unsloth.schemas.UnslothJobOutput` into a 4-step
:class:`PlatformJobSpec` (download → train → upload → model-entity).
"""

from __future__ import annotations

from nemo_helix import AsyncNeMoHelix
from nemo_helix_plugin.jobs.api_factory import PlatformJobSpec
from nhx.unsloth.app.jobs.compiler import platform_job_config_compiler as _compile_canonical
from nhx.unsloth.schemas import UnslothJobOutput


async def platform_job_config_compiler(
    *,
    workspace: str,
    spec: UnslothJobOutput,
    sdk: AsyncNeMoHelix,
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
        sdk,
        job_name=job_name,
        profile=profile,
    )


__all__ = ["platform_job_config_compiler"]

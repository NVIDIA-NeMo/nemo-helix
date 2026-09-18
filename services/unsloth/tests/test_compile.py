# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compile-side entry.

We check the delegation contract:
``compile.platform_job_config_compiler`` forwards to
:mod:`nmp.unsloth.app.jobs.compiler` with the args the platform
schedule passes through.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from nmp.customization_common.service.platform_client import AsyncCustomizationPlatformClients
from nmp.unsloth.compile import platform_job_config_compiler
from nmp.unsloth.schemas import (
    DatasetSpec,
    LoRAParams,
    ModelLoadSpec,
    OutputResponse,
    TrainingSpec,
    UnslothJobOutput,
)


def _canonical_spec() -> UnslothJobOutput:
    return UnslothJobOutput(
        model=ModelLoadSpec(name="default/base"),
        dataset=DatasetSpec(path="default/training"),
        training=TrainingSpec(lora=LoRAParams()),
        schedule={"max_steps": 1},
        output=OutputResponse(
            name="r",
            type="adapter",
            save_method="lora",
            fileset="r",
        ),
    )


@pytest.mark.asyncio
async def test_compile_delegates_to_app_jobs_compiler() -> None:
    spec = _canonical_spec()
    platform = AsyncCustomizationPlatformClients(files=AsyncMock(), models=AsyncMock())

    sentinel = object()
    target = "nmp.unsloth.compile._compile_canonical"
    with patch(target, new=AsyncMock(return_value=sentinel)) as mock:
        result = await platform_job_config_compiler(
            workspace="default",
            spec=spec,
            platform=platform,
            job_name="job-x",
            profile="gpu-large",
        )

    assert result is sentinel
    mock.assert_awaited_once_with(
        "default",
        spec,
        platform,
        job_name="job-x",
        profile="gpu-large",
    )

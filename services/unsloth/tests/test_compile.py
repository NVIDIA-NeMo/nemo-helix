# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compile-side entry.

We check the delegation contract:
``compile.platform_job_config_compiler`` forwards to
:mod:`nhx.unsloth.app.jobs.compiler` with the args the platform
schedule passes through.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nhx.customization_common.service.platform_client import AsyncCustomizationHelixClients
from nhx.unsloth.compile import platform_job_config_compiler
from nhx.unsloth.schemas import (
    DatasetSpec,
    LoRAParams,
    ModelLoadSpec,
    OutputResponse,
    ScheduleSpec,
    TrainingSpec,
    UnslothJobOutput,
)


def _canonical_spec() -> UnslothJobOutput:
    return UnslothJobOutput(
        model=ModelLoadSpec(name="default/base"),
        dataset=DatasetSpec(path="default/training"),
        training=TrainingSpec(lora=LoRAParams()),
        schedule=ScheduleSpec(max_steps=1),
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
    platform = AsyncCustomizationHelixClients(files=AsyncMock(), models=AsyncMock(), jobs=MagicMock())

    sentinel = object()
    target = "nhx.unsloth.compile._compile_canonical"
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


def test_the_training_step_can_pause_and_keeps_checkpoints_out_of_the_model() -> None:
    """Pause is rejected for a step without a deadline, and checkpoints must not be published with the model."""
    from nhx.unsloth.app.constants import DEFAULT_CHECKPOINT_PATH, DEFAULT_OUTPUT_MODEL_PATH
    from nhx.unsloth.app.jobs.training.compiler import compile_training_step
    from nhx.unsloth.config import config

    spec = UnslothJobOutput(
        model=ModelLoadSpec(name="default/base"),
        dataset=DatasetSpec(path="default/training"),
        training=TrainingSpec(lora=LoRAParams()),
        schedule=ScheduleSpec(max_steps=1),
        output=OutputResponse(name="r", type="adapter", save_method="lora", fileset="r"),
    )

    step = compile_training_step(spec, base_env=[])

    assert step.lifecycle is not None
    assert step.lifecycle.pause_deadline_seconds == config.training_pause_deadline_seconds > 0
    assert step.config["checkpoint_path"] == DEFAULT_CHECKPOINT_PATH
    assert step.config["output_path"] == DEFAULT_OUTPUT_MODEL_PATH


@pytest.mark.parametrize("profile", [None, "default"])
def test_the_training_step_sends_its_provider(profile: str | None) -> None:
    """The client sends only set fields, and the Jobs API picks the executor type by ``provider``."""
    from nhx.unsloth.app.jobs.training.compiler import compile_training_step

    step = compile_training_step(_canonical_spec(), base_env=[], profile=profile)

    assert step.model_dump(mode="json", exclude_unset=True)["executor"]["provider"] == "gpu"

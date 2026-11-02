# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The steps the unsloth compiler builds: what they download, train on and upload."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.models.types import ModelEntity
from nhx.customization_common.schemas.file_io import model_upload_lifecycle
from nhx.unsloth.app.constants import DEFAULT_DATASET_PATH
from nhx.unsloth.app.jobs.compiler import platform_job_config_compiler
from nhx.unsloth.schemas import (
    DatasetSpec,
    LoRAParams,
    ModelLoadSpec,
    OutputResponse,
    ScheduleSpec,
    TrainingSpec,
    UnslothJobOutput,
)


def _no_in_flight_jobs() -> MagicMock:
    """Jobs client whose in-flight output-name check finds no pending or running job."""
    page = MagicMock()
    page.items.return_value.__aiter__.return_value = []
    jobs = MagicMock()
    jobs.list_jobs = AsyncMock(return_value=page)
    return jobs


def _platform_mock() -> MagicMock:
    """Platform client whose adapter lookup reports "no such adapter"."""
    platform = MagicMock()
    platform.models.get_adapter = AsyncMock(
        side_effect=NotFoundError(httpx.Response(status_code=404, request=httpx.Request("GET", "http://test")))
    )
    platform.jobs = _no_in_flight_jobs()
    return platform


def _model_entity() -> ModelEntity:
    return ModelEntity(
        id="model-qwen3",
        workspace="default",
        name="qwen3-1.7b",
        fileset="default/qwen3-1.7b",
        trust_remote_code=False,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


def _spec() -> UnslothJobOutput:
    return UnslothJobOutput(
        model=ModelLoadSpec(name="default/qwen3-1.7b"),
        dataset=DatasetSpec(path="default/commonsense_qa", apply_chat_template=True),
        training=TrainingSpec(lora=LoRAParams()),
        schedule=ScheduleSpec(epochs=1),
        output=OutputResponse(
            name="out",
            type="adapter",
            save_method="lora",
            fileset="out",
        ),
    )


@pytest.mark.asyncio
async def test_training_reads_validation_files_from_the_one_dataset_fileset() -> None:
    from nhx.unsloth.app.jobs import compiler as compiler_mod

    original_fetch = compiler_mod.fetch_model_entity
    compiler_mod.fetch_model_entity = AsyncMock(return_value=_model_entity())
    try:
        job = await platform_job_config_compiler(
            workspace="default",
            job_spec=_spec(),
            platform=_platform_mock(),
        )
    finally:
        compiler_mod.fetch_model_entity = original_fetch

    training = next(s for s in job["steps"] if s["name"] == "training")
    assert training["config"]["dataset_path"] == DEFAULT_DATASET_PATH
    assert "validation_path" not in training["config"]

    download = next(s for s in job["steps"] if s["name"] == "model-and-dataset-download")
    assert [item["src"]["name"] for item in download["config"]["download"]][1:] == ["commonsense_qa"]


@pytest.mark.asyncio
async def test_upload_step_stamps_output_metadata() -> None:
    from nhx.unsloth.app.jobs import compiler as compiler_mod

    original_fetch = compiler_mod.fetch_model_entity
    compiler_mod.fetch_model_entity = AsyncMock(return_value=_model_entity())
    try:
        job = await platform_job_config_compiler(
            workspace="default",
            job_spec=_spec(),
            platform=_platform_mock(),
        )
    finally:
        compiler_mod.fetch_model_entity = original_fetch

    upload = next(s for s in job["steps"] if s["name"] == "model-upload")
    assert upload["config"]["upload"][0]["metadata"] is None
    assert upload["lifecycle"] == model_upload_lifecycle()


@pytest.mark.asyncio
async def test_compiler_applies_profile_to_task_steps() -> None:
    from nhx.unsloth.app.jobs import compiler as compiler_mod

    original_fetch = compiler_mod.fetch_model_entity
    compiler_mod.fetch_model_entity = AsyncMock(return_value=_model_entity())
    try:
        job = await platform_job_config_compiler(
            workspace="default",
            job_spec=_spec(),
            platform=_platform_mock(),
            profile="custom-gpu",
        )
    finally:
        compiler_mod.fetch_model_entity = original_fetch

    assert [step["executor"]["profile"] for step in job["steps"]] == ["custom-gpu"] * 4

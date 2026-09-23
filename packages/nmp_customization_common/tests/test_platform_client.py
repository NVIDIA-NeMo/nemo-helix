# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nmp.customization_common.service.platform_client import (
    AsyncCustomizationPlatformClients,
    check_dataset_access,
    check_environment_access,
    check_gym_dataset_layout,
    fetch_model_entity,
    validate_adapter_base_model,
)


def _not_found() -> NotFoundError:
    return NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://platform/resource")))


def _clients(
    model: SimpleNamespace,
    *,
    fileset_error: Exception | None = None,
) -> tuple[MagicMock, MagicMock, AsyncCustomizationPlatformClients]:
    models = MagicMock()
    models.get_model = AsyncMock(return_value=SimpleNamespace(data=lambda: model))
    files = MagicMock()
    files.get_fileset = AsyncMock()
    if fileset_error is not None:
        files.get_fileset.side_effect = fileset_error

    return models, files, AsyncCustomizationPlatformClients(files=files, models=models)


async def test_fetch_model_entity_verifies_weights_fileset() -> None:
    model = SimpleNamespace(name="base", workspace="default", fileset="weights/default-base")
    models, files, platform = _clients(model)

    result = await fetch_model_entity("default/base", "default", platform)

    assert result is model
    models.get_model.assert_awaited_once_with(
        name="base",
        workspace="default",
        query_params={"verbose": True},
    )
    files.get_fileset.assert_awaited_once_with(workspace="weights", name="default-base")


async def test_fetch_model_entity_rejects_missing_weights_fileset() -> None:
    model = SimpleNamespace(name="base", workspace="default", fileset="default/missing")
    _, _, platform = _clients(model, fileset_error=_not_found())

    with pytest.raises(ValueError, match="Weights for model 'default/base' fileset 'missing' not found"):
        await fetch_model_entity("default/base", "default", platform)


async def test_fetch_model_entity_without_weights_fileset_skips_files_service() -> None:
    model = SimpleNamespace(name="api-model", workspace="default", fileset=None)
    _, files, platform = _clients(model)

    result = await fetch_model_entity("api-model", "default", platform)

    assert result is model
    files.get_fileset.assert_not_awaited()


async def test_dataset_access_resolves_fileset_name_and_checks_directory_path() -> None:
    _, files, platform = _clients(SimpleNamespace())
    files.list_files = AsyncMock(
        return_value=SimpleNamespace(
            data=lambda: SimpleNamespace(data=[SimpleNamespace(path="results/attempt/artifacts/training.jsonl")])
        )
    )

    await check_dataset_access(
        platform,
        "default/job-fileset#results/attempt/artifacts",
        "default",
    )

    files.get_fileset.assert_awaited_once_with(workspace="default", name="job-fileset")
    files.list_files.assert_awaited_once_with(workspace="default", name="job-fileset")


async def test_dataset_access_rejects_missing_directory_path() -> None:
    _, files, platform = _clients(SimpleNamespace())
    files.list_files = AsyncMock(
        return_value=SimpleNamespace(data=lambda: SimpleNamespace(data=[SimpleNamespace(path="other/training.jsonl")]))
    )

    with pytest.raises(ValueError, match="Dataset path 'results/attempt/artifacts/' not found"):
        await check_dataset_access(
            platform,
            "default/job-fileset#results/attempt/artifacts",
            "default",
        )


async def test_gym_layout_checks_training_file_below_directory_path() -> None:
    _, files, platform = _clients(SimpleNamespace())
    files.list_files = AsyncMock(
        return_value=SimpleNamespace(
            data=lambda: SimpleNamespace(data=[SimpleNamespace(path="results/attempt/artifacts/training.jsonl")])
        )
    )

    await check_gym_dataset_layout(
        platform,
        "default/job-fileset#results/attempt/artifacts",
        "default",
    )


async def test_environment_access_rejects_directory_path() -> None:
    _, _, platform = _clients(SimpleNamespace())

    with pytest.raises(ValueError, match="must not include a '#path/' directory"):
        await check_environment_access(platform, "default/environment#package", "default")


def _adapter_clients(existing_base: str | None, *, missing: bool = False) -> AsyncCustomizationPlatformClients:
    models = MagicMock()
    if missing:
        models.get_adapter = AsyncMock(side_effect=_not_found())
    else:
        models.get_adapter = AsyncMock(return_value=SimpleNamespace(data=lambda: SimpleNamespace(model=existing_base)))
    return AsyncCustomizationPlatformClients(files=MagicMock(), models=models)


@pytest.mark.parametrize(
    ("base_model_ref", "existing_base", "missing"),
    [
        ("default/base", None, True),
        ("default/base", "default/base", False),
        ("base", "default/base", False),
        ("default/base", None, False),
    ],
    ids=["no-adapter-yet", "retrain-same-base", "bare-name-same-base", "adapter-without-base"],
)
async def test_adapter_base_model_allows(base_model_ref: str, existing_base: str | None, missing: bool) -> None:
    platform = _adapter_clients(existing_base, missing=missing)

    await validate_adapter_base_model("my-lora", base_model_ref, "default", platform)

    cast(AsyncMock, platform.models.get_adapter).assert_awaited_once_with(name="my-lora", workspace="default")


async def test_adapter_base_model_rejects_a_different_base() -> None:
    """The adapter name is taken in this workspace, so training would only fail at the end."""
    platform = _adapter_clients("default/other-base")

    with pytest.raises(
        PlatformJobCompilationError,
        match=r"Adapter 'default/my-lora' already exists on base model 'default/other-base', "
        r"but this job trains against 'default/base'",
    ):
        await validate_adapter_base_model("my-lora", "default/base", "default", platform)

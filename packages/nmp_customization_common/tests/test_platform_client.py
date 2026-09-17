# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_platform_plugin.client.errors import NotFoundError
from nmp.customization_common.service.platform_client import AsyncCustomizationPlatformClients, fetch_model_entity


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

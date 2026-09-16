# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for evaluator job helpers."""

from __future__ import annotations

import httpx
from nemo_evaluator.jobs.utils import run_with_isolated_async_client
from nemo_helix_plugin.files.client import AsyncFilesClient
from pytest_mock import MockerFixture


def test_run_with_isolated_async_client_passes_cloned_client(mocker: MockerFixture) -> None:
    async_client = AsyncFilesClient(
        base_url="http://platform.test",
        workspace="default",
        http_client=mocker.AsyncMock(spec=httpx.AsyncClient),
    )
    http_client = mocker.AsyncMock(name="http_client")
    http_client.__aenter__.return_value = http_client
    http_client.__aexit__.return_value = None
    mocker.patch("nemo_evaluator.jobs.utils.httpx.AsyncClient", return_value=http_client)

    seen: list[AsyncFilesClient] = []

    async def fn(client: AsyncFilesClient) -> int:
        seen.append(client)
        return 42

    assert run_with_isolated_async_client(async_client, fn) == 42
    assert seen[0] is not async_client
    assert isinstance(seen[0], AsyncFilesClient)
    assert seen[0].base_url == async_client.base_url
    assert seen[0].workspace == async_client.workspace
    assert seen[0]._client is http_client

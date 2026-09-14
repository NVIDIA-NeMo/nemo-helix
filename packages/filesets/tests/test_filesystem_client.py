# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client construction inside Fileset filesystem classes."""

from __future__ import annotations

import httpx
from filesets.filesystem.filesystem import AsyncFilesetFileSystem, FilesetFileSystem
from nemo_platform_plugin.client.types import RetryPolicy
from nemo_platform_plugin.files.client import AsyncFilesClient, FilesClient

BASE = "http://test:8000"
UPLOAD_TIMEOUT = httpx.Timeout(30.0, write=10 * 60, read=5 * 60)


def _sync_client(*, timeout: httpx.Timeout) -> FilesClient:
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        timeout=timeout,
    )
    return FilesClient(
        base_url=BASE,
        workspace="default",
        http_client=http_client,
        retry=RetryPolicy(max_retries=2),
    )


def _async_client() -> AsyncFilesClient:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        timeout=httpx.Timeout(60.0),
    )
    return AsyncFilesClient(
        base_url=BASE,
        workspace="default",
        http_client=http_client,
        retry=RetryPolicy(max_retries=2),
    )


def test_sync_filesystem_uses_sync_client() -> None:
    client = _sync_client(timeout=httpx.Timeout(60.0))

    fs = FilesetFileSystem(client=client)

    assert fs._client is client
    assert fs._client.workspace == "default"
    assert fs._client.retry == RetryPolicy(max_retries=2)


async def test_async_filesystem_uses_async_client() -> None:
    async_client = _async_client()

    try:
        fs = AsyncFilesetFileSystem(client=async_client)

        assert fs._client is async_client
        assert fs._client.workspace == "default"
        assert fs._client.retry == RetryPolicy(max_retries=2)
    finally:
        await async_client._http.aclose()


def test_platform_files_resource_returns_sync_filesystem() -> None:
    from nemo_platform import NeMoPlatform

    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        timeout=httpx.Timeout(60.0),
    )
    platform = NeMoPlatform(base_url=BASE, workspace="default", http_client=http_client)

    fs = platform.files.fsspec

    assert isinstance(fs, FilesetFileSystem)
    assert fs._client._http is http_client


async def test_async_platform_files_resource_returns_async_filesystem() -> None:
    from nemo_platform import AsyncNeMoPlatform

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        timeout=httpx.Timeout(60.0),
    )
    platform = AsyncNeMoPlatform(base_url=BASE, workspace="default", http_client=http_client)

    try:
        fs = platform.files.fsspec

        assert isinstance(fs, AsyncFilesetFileSystem)
        assert fs._client._http is http_client
    finally:
        await http_client.aclose()


def test_upload_timeout_survives_the_whole_client_chain() -> None:
    """End to end: an SDK-level timeout override reaches the sync transfer client."""
    from nemo_platform import NeMoPlatform

    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        timeout=httpx.Timeout(60.0),
    )
    platform = NeMoPlatform(base_url=BASE, workspace="default", http_client=http_client)

    fs = platform.with_options(timeout=UPLOAD_TIMEOUT).files.fsspec

    assert fs._client._timeout == UPLOAD_TIMEOUT
    assert fs._client._http is http_client

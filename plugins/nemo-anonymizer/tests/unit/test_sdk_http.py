# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from nemo_anonymizer_plugin.sdk import http


def _platform(*, workspace: str | None = "default") -> http.HelixClient:
    return cast(
        http.HelixClient,
        SimpleNamespace(
            base_url="https://platform.test/",
            workspace=workspace,
            default_headers={"Authorization": "Bearer token", "skip": object()},
        ),
    )


def test_url_encodes_workspace_as_single_path_segment() -> None:
    assert (
        http.url(_platform(), "team/a", "/preview")
        == "https://platform.test/apis/anonymizer/v2/workspaces/team%2Fa/preview"
    )


def test_url_requires_absolute_resource_path() -> None:
    with pytest.raises(ValueError, match="must start with '/'"):
        http.url(_platform(), "team-a", "preview")


def test_path_segment_encodes_reserved_chars() -> None:
    assert http.path_segment("job/a?b") == "job%2Fa%3Fb"


def test_headers_drops_non_string_values() -> None:
    assert http.headers(_platform()) == {"Authorization": "Bearer token"}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_retrieved_handle_keeps_original_id_when_name_is_reused(asynchronous: bool) -> None:
    import httpx
    from nemo_anonymizer_plugin.sdk.errors import AnonymizerJobError
    from nemo_anonymizer_plugin.sdk.resources import AnonymizerResource, AsyncAnonymizerResource
    from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient

    references: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        reference = request.url.path.rsplit("/", 1)[-1]
        references.append(reference)
        if len(references) == 1:
            return httpx.Response(200, json={"id": "platform-job-original", "name": "training"})
        assert reference == "platform-job-original"
        return httpx.Response(404, json={"detail": "original Job disappeared"})

    if asynchronous:
        async with AsyncNemoClient(
            base_url="http://test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ) as client:
            handle = await AsyncAnonymizerResource(client).get_job_resource("training", workspace="default")
            with pytest.raises(AnonymizerJobError):
                await handle.get_job()
    else:
        with NemoClient(
            base_url="http://test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
        ) as client:
            handle = AnonymizerResource(client).get_job_resource("training", workspace="default")
            with pytest.raises(AnonymizerJobError):
                handle.get_job()
    assert references == ["training", "platform-job-original"]


@pytest.mark.asyncio
async def test_concurrent_responses_cannot_change_fixed_identity():
    import asyncio

    import httpx
    from nemo_anonymizer_plugin.sdk.errors import AnonymizerJobError
    from nemo_anonymizer_plugin.sdk.job_resources import AsyncAnonymizerJobResource
    from nemo_helix_plugin.client.client import AsyncNemoClient

    replies = [asyncio.Future(), asyncio.Future()]
    both_started = asyncio.Event()
    references = []

    async def handler(request):
        index = len(references)
        references.append(request.url.path)
        if len(references) == 2:
            both_started.set()
        return await replies[index]

    async with AsyncNemoClient(
        base_url="http://test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        handle = AsyncAnonymizerJobResource(job_id="original", workspace="default", platform=client)
        first = asyncio.create_task(handle.get_job())
        second = asyncio.create_task(handle.get_job_status())
        await both_started.wait()
        replies[0].set_result(httpx.Response(200, json={"id": "original", "name": "same"}))
        await first
        replies[1].set_result(httpx.Response(200, json={"id": "replacement", "status": "completed"}))
        with pytest.raises(AnonymizerJobError, match="different Job ID"):
            await second
        assert handle._job_id == "original"


def test_threaded_requests_keep_fixed_identity_and_independent_handles():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import httpx
    from nemo_anonymizer_plugin.sdk.errors import AnonymizerJobError
    from nemo_anonymizer_plugin.sdk.job_resources import AnonymizerJobResource
    from nemo_helix_plugin.client.client import NemoClient

    both_reading = Barrier(2)

    def handler(request: httpx.Request) -> httpx.Response:
        both_reading.wait(timeout=5)
        identifier = "replacement" if request.url.path.endswith("/status") else "original"
        return httpx.Response(200, json={"id": identifier, "status": "completed"})

    with NemoClient(
        base_url="http://test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    ) as client:
        handle = AnonymizerJobResource(job_id="original", workspace="default", platform=client)
        with ThreadPoolExecutor(max_workers=2) as threads:
            futures = [threads.submit(handle.get_job), threads.submit(handle.get_job_status)]
            errors = [future.exception(timeout=5) for future in futures]
        assert sum(isinstance(error, AnonymizerJobError) for error in errors) == 1
        other = AnonymizerJobResource(job_id="independent", workspace="default", platform=client)
        assert other._job_id == "independent"
        assert handle._job_id == "original"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("payload", [{"id": 123}, {"id": ""}, {"status": "completed"}, {"id": "one", "status": 42}])
async def test_invalid_identity_or_status_is_rejected_before_binding(asynchronous, payload):
    import httpx
    from nemo_anonymizer_plugin.sdk.job_resources import AnonymizerJobResource, AsyncAnonymizerJobResource
    from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
    from pydantic import ValidationError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    if asynchronous:
        async with AsyncNemoClient(
            base_url="http://test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            handle = AsyncAnonymizerJobResource(job_id="original", workspace="default", platform=client)
            with pytest.raises(ValidationError):
                await handle.get_job_status()
            assert handle._job_id == "original"
    else:
        with NemoClient(
            base_url="http://test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ) as client:
            handle = AnonymizerJobResource(job_id="original", workspace="default", platform=client)
            with pytest.raises(ValidationError):
                handle.get_job_status()
            assert handle._job_id == "original"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_reconstructed_handle_uses_persisted_id_without_name_discovery(asynchronous):
    import httpx
    from nemo_anonymizer_plugin.sdk.job_resources import AnonymizerJobResource, AsyncAnonymizerJobResource
    from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        assert request.url.path.endswith("/original/status")
        return httpx.Response(200, json={"id": "original", "status": "completed"})

    if asynchronous:
        async with AsyncNemoClient(
            base_url="http://test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            first = AsyncAnonymizerJobResource(job_id="original", workspace="default", platform=client)
            replacement = AsyncAnonymizerJobResource(job_id="original", workspace="default", platform=client)
            assert seen == []
            assert await first.get_job_status() == "completed"
            assert await replacement.get_job_status() == "completed"
    else:
        with NemoClient(
            base_url="http://test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ) as client:
            first = AnonymizerJobResource(job_id="original", workspace="default", platform=client)
            replacement = AnonymizerJobResource(job_id="original", workspace="default", platform=client)
            assert seen == []
            assert first.get_job_status() == "completed"
            assert replacement.get_job_status() == "completed"
    assert len(seen) == 2

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import assert_type

import httpx
import pytest
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.types import RetryPolicy
from nemo_platform_plugin.jobs import endpoints
from nemo_platform_plugin.jobs.client import AsyncJobsClient, JobsClient


def test_client_from_platform_preserves_stainless_retry_policy() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    platform = NeMoPlatform(
        base_url="http://test",
        workspace="default",
        max_retries=4,
        http_client=http_client,
    )

    client = client_from_platform(platform, JobsClient)

    assert_type(client, JobsClient)
    assert client.retry is not None
    assert client.retry == RetryPolicy(
        max_retries=4,
        retryable_status_codes=(408, 409, 429),
        retry_all_server_errors=True,
        respect_retry_decision_headers=True,
        respect_retry_after_headers=True,
    )


def test_client_from_platform_close_does_not_close_platform_transport() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    platform = NeMoPlatform(
        base_url="http://test",
        workspace="default",
        http_client=http_client,
    )

    client = client_from_platform(platform, JobsClient)
    client.close()

    assert not http_client.is_closed
    http_client.close()


def test_client_from_platform_uses_platform_prepare_url() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))

    class RoutedPlatform(NeMoPlatform):
        def _prepare_url(self, url: str) -> httpx.URL:
            prepared = super()._prepare_url(url)
            if prepared.path.startswith("/apis/jobs"):
                return prepared.copy_with(scheme="http", host="127.0.0.1", port=8080)
            return prepared

    platform = RoutedPlatform(
        base_url="http://gateway",
        workspace="default",
        http_client=http_client,
    )

    client = client_from_platform(platform, JobsClient)

    request = endpoints.list_steps(workspace="default", name="job-1")
    assert client._resolve_path(request) == ("http://127.0.0.1:8080/apis/jobs/v2/workspaces/default/jobs/job-1/steps")


def test_platform_jobs_property_uses_platform_prepare_url() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))

    class RoutedPlatform(NeMoPlatform):
        def _prepare_url(self, url: str) -> httpx.URL:
            prepared = super()._prepare_url(url)
            if prepared.path.startswith("/apis/jobs"):
                return prepared.copy_with(scheme="http", host="127.0.0.1", port=8080)
            return prepared

    platform = RoutedPlatform(
        base_url="http://gateway",
        workspace="default",
        http_client=http_client,
    )

    request = endpoints.list_steps(workspace="default", name="job-1")

    assert platform.jobs is platform.jobs
    assert platform.jobs._resolve_path(request) == (
        "http://127.0.0.1:8080/apis/jobs/v2/workspaces/default/jobs/job-1/steps"
    )


@pytest.mark.asyncio
async def test_async_client_from_platform_uses_async_transport() -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))

    platform = AsyncNeMoPlatform(
        base_url="http://gateway",
        workspace="default",
        http_client=http_client,
    )

    try:
        client = client_from_platform(platform, AsyncJobsClient)

        assert_type(client, AsyncJobsClient)
        assert isinstance(client, AsyncJobsClient)
        assert client._client is http_client
    finally:
        await platform.close()


@pytest.mark.asyncio
async def test_async_client_from_platform_aclose_does_not_close_platform_transport() -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    platform = AsyncNeMoPlatform(
        base_url="http://gateway",
        workspace="default",
        http_client=http_client,
    )

    try:
        client = client_from_platform(platform, AsyncJobsClient)
        await client.aclose()

        assert not http_client.is_closed
    finally:
        await platform.close()


def test_from_client_preserves_url_resolver() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    client = JobsClient(
        base_url="http://gateway",
        workspace="default",
        http_client=http_client,
        url_resolver=lambda url: url.replace("http://gateway/apis/jobs", "http://127.0.0.1:8080/apis/jobs"),
    )

    clone = JobsClient.from_client(client)

    request = endpoints.list_steps(workspace="default", name="job-1")
    assert clone._resolve_path(request) == ("http://127.0.0.1:8080/apis/jobs/v2/workspaces/default/jobs/job-1/steps")


def test_client_from_platform_propagates_timeout() -> None:
    """``platform.with_options(timeout=...)`` must reach the typed client.

    Both clients share one httpx client, whose own timeout ``with_options`` does
    not touch — so the typed client has to carry the override itself or long
    transfers silently run on the transport's original budget.
    """
    upload_timeout = httpx.Timeout(30.0, write=10 * 60, read=5 * 60)
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        timeout=httpx.Timeout(60.0),
    )
    platform = NeMoPlatform(base_url="http://test", workspace="default", http_client=http_client)

    scoped = platform.with_options(timeout=upload_timeout)
    client = client_from_platform(scoped, JobsClient)

    assert client._timeout == upload_timeout
    # The shared transport is untouched, which is why the override is needed.
    assert scoped._client.timeout == httpx.Timeout(60.0)


def test_client_from_platform_carries_default_timeout() -> None:
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        timeout=httpx.Timeout(60.0),
    )
    platform = NeMoPlatform(base_url="http://test", workspace="default", http_client=http_client)

    client = client_from_platform(platform, JobsClient)

    assert client._timeout == platform.timeout


def test_client_from_platform_carries_disabled_timeout() -> None:
    """``timeout=None`` means "no timeout", not "no override"."""
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        timeout=httpx.Timeout(60.0),
    )
    platform = NeMoPlatform(base_url="http://test", workspace="default", http_client=http_client)

    client = client_from_platform(platform.with_options(timeout=None), JobsClient)

    # Not the transport's 60s: httpx reads an all-None Timeout as "wait forever".
    assert client._timeout == httpx.Timeout(None)


def test_client_from_platform_carries_authorization_header() -> None:
    """A statically configured bearer reaches the typed client.

    The CLI hands config-file credentials to the generated SDK as a default ``Authorization``
    header, so dropping the headers here would silently produce an unauthenticated client.
    """
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    platform = NeMoPlatform(
        base_url="http://test",
        workspace="default",
        default_headers={"Authorization": "Bearer static-token"},
        http_client=http_client,
    )

    client = client_from_platform(platform, JobsClient)

    assert client._default_headers["Authorization"] == "Bearer static-token"


def test_client_from_platform_shares_the_transport_so_token_refresh_survives() -> None:
    """OAuth callers refresh the bearer from a request event hook on the httpx client, not from a
    header. Rebuilding the transport here would leave only the stale seeded token and break refresh
    for every adapter caller, with nothing failing until a request hit an authenticated deployment.
    """
    seen: list[str | None] = []

    def refresh(request: httpx.Request) -> None:
        request.headers["Authorization"] = f"Bearer refreshed-{len(seen) + 1}"

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return httpx.Response(200, request=request)

    http_client = httpx.Client(
        transport=httpx.MockTransport(record),
        event_hooks={"request": [refresh]},
    )
    platform = NeMoPlatform(
        base_url="http://test",
        workspace="default",
        default_headers={"Authorization": "Bearer seeded"},
        http_client=http_client,
    )

    client = client_from_platform(platform, JobsClient)

    assert client._client is http_client
    client._client.get("http://test/one")
    client._client.get("http://test/two")
    assert seen == ["Bearer refreshed-1", "Bearer refreshed-2"]

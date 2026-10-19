# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import assert_type

import httpx
import pytest
from nemo_helix_plugin.client.adapter import (
    AsyncHelixClient,
    HelixClient,
    SyncHelixClient,
    client_from_platform,
    platform_default_headers,
)
from nemo_helix_plugin.client.client import (
    AsyncNemoClient,
    NemoClient,
    NemoClientRuntime,
)
from nemo_helix_plugin.client.types import RetryPolicy
from nemo_helix_plugin.jobs import endpoints
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient


def test_client_from_platform_preserves_retry_policy() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    retry = RetryPolicy(
        max_retries=4,
        retryable_status_codes=(408, 409, 429),
        retry_all_server_errors=True,
        respect_retry_decision_headers=True,
        respect_retry_after_headers=True,
    )
    platform = NemoClient(base_url="http://test", workspace="default", retry=retry, http_client=http_client)

    client = client_from_platform(platform, JobsClient)

    assert_type(client, JobsClient)
    assert client.retry == retry


def test_client_from_platform_close_does_not_close_platform_transport() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    platform = NemoClient(base_url="http://test", workspace="default", http_client=http_client)

    client = client_from_platform(platform, JobsClient)
    client.close()

    assert not http_client.is_closed
    http_client.close()


def test_client_from_platform_returns_identity_for_same_class() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    base = NemoClient(base_url="http://test", workspace="ws", http_client=http_client)
    assert client_from_platform(base, NemoClient) is base


@pytest.mark.asyncio
async def test_async_client_from_platform_uses_async_transport() -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    platform = AsyncNemoClient(base_url="http://gateway", workspace="default", http_client=http_client)

    client = client_from_platform(platform, AsyncJobsClient)

    assert_type(client, AsyncJobsClient)
    assert isinstance(client, AsyncJobsClient)
    assert client._client is http_client
    await client.aclose()


@pytest.mark.asyncio
async def test_async_client_from_platform_aclose_does_not_close_platform_transport() -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    platform = AsyncNemoClient(base_url="http://gateway", workspace="default", http_client=http_client)

    client = client_from_platform(platform, AsyncJobsClient)
    await client.aclose()

    assert not http_client.is_closed
    await platform.aclose()


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
    """``platform.with_options(timeout=...)`` must reach the typed client."""
    upload_timeout = httpx.Timeout(30.0, write=10 * 60, read=5 * 60)
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        timeout=httpx.Timeout(60.0),
    )
    platform = NemoClient(
        base_url="http://test", workspace="default", timeout=httpx.Timeout(60.0), http_client=http_client
    )

    scoped = platform.with_options(timeout=upload_timeout)
    client = client_from_platform(scoped, JobsClient)

    assert client._timeout == upload_timeout


def test_client_from_platform_carries_disabled_timeout() -> None:
    """``timeout=None`` means "no timeout", not "no override"."""
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)),
        timeout=httpx.Timeout(60.0),
    )
    platform = NemoClient(base_url="http://test", workspace="default", http_client=http_client)

    client = client_from_platform(platform.with_options(timeout=None), JobsClient)

    assert client._timeout is None


def test_client_from_platform_accepts_a_typed_client_and_shares_its_transport() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    base = NemoClient(base_url="http://test", workspace="ws", default_headers={"X-A": "1"}, http_client=http_client)

    client = client_from_platform(base, JobsClient)

    assert isinstance(client, JobsClient)
    assert client._http is http_client
    assert client.workspace == "ws"
    assert client.default_headers == {"X-A": "1"}
    assert client_from_platform(client, JobsClient) is client


@pytest.mark.asyncio
async def test_client_from_platform_accepts_an_async_typed_client() -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    base = AsyncNemoClient(base_url="http://test", workspace="ws", http_client=http_client)

    client = client_from_platform(base, AsyncJobsClient)

    assert isinstance(client, AsyncJobsClient)
    assert client._http is http_client


def test_platform_client_protocol_matches_typed_handle_shapes() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    typed = NemoClient(base_url="http://test", workspace="ws", http_client=http_client)

    assert isinstance(typed, HelixClient)
    assert isinstance(typed, SyncHelixClient)
    assert not isinstance(typed, AsyncHelixClient)
    assert isinstance(AsyncNemoClient(base_url="http://test", http_client=httpx.AsyncClient()), HelixClient)
    assert not isinstance(object(), HelixClient)


def test_client_from_platform_accepts_a_typed_client_runtime() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    runtime = NemoClientRuntime()
    platform = NemoClient(base_url="http://test", workspace="ws", http_client=http_client, client_runtime=runtime)

    client = client_from_platform(platform, JobsClient)

    assert client.nemo_client_runtime is runtime


def test_platform_default_headers_reads_typed_client_headers() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    client = NemoClient(base_url="http://test", default_headers={"X-NHX-Internal": "true"}, http_client=http_client)

    headers = platform_default_headers(client)

    assert headers == {"X-NHX-Internal": "true"}
    headers["mutated"] = "yes"
    assert client.default_headers == {"X-NHX-Internal": "true"}


def test_client_from_platform_carries_authorization_header() -> None:
    """A statically configured bearer reaches the typed client."""
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    platform = NemoClient(
        base_url="http://test",
        workspace="default",
        default_headers={"Authorization": "Bearer static-token"},
        http_client=http_client,
    )

    client = client_from_platform(platform, JobsClient)

    assert client.default_headers["Authorization"] == "Bearer static-token"


def test_client_from_platform_carries_runtime_without_synthesizing_trusted_identity_headers() -> None:
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    runtime = NemoClientRuntime()
    base_client = NemoClient(
        base_url="http://test",
        workspace="default",
        default_headers={"X-NHX-Internal": "true"},
        http_client=http_client,
        client_runtime=runtime,
    )

    client = client_from_platform(base_client, JobsClient)

    assert client.nemo_client_runtime is runtime
    assert client.default_headers == {"X-NHX-Internal": "true"}
    assert "X-NHX-Principal-Id" not in client.default_headers


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
    platform = NemoClient(
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

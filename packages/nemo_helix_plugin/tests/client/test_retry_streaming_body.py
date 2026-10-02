# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry behaviour for requests whose body cannot be replayed.

A streaming upload sends a one-shot iterator under an explicit ``Content-Length``.
Replaying that request re-sends an exhausted iterator under the original length,
which h11 rejects with ``Too little data for declared Content-Length`` — hiding
whatever actually failed on the first attempt.

That only applies once httpx has begun reading the body. A failure to establish
the connection leaves the iterator untouched, so those attempts are still retried.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import InternalServerError, NemoTransportError, NotFoundError
from nemo_helix_plugin.client.types import PreparedRequest, RetryPolicy

BASE = "http://test:8000"
PAYLOAD = 256 * 1024
# Long enough to outlast the client's read timeout; the handler task is cancelled
# on teardown so the test never actually waits this out.
STALL_SECONDS = 30
CLIENT_LOGGER = "nemo_helix_plugin.client.client"
# What ``client_from_platform`` builds for every typed client in production, and
# the only policy under which the response-header branches below are reachable.
ADAPTER_POLICY = RetryPolicy(
    max_retries=3,
    backoff_base=0.0,
    retryable_status_codes=(408, 409, 429),
    retry_all_server_errors=True,
    respect_retry_decision_headers=True,
    respect_retry_after_headers=True,
)


def _upload_request(content: bytes | Iterable[bytes] | AsyncIterable[bytes] | None) -> PreparedRequest:
    return PreparedRequest(
        path_template="/apis/test/v2/upload",
        path_params={},
        method="PUT",
        content=content,
        content_type="application/octet-stream",
        response_type=None,
    )


async def _chunks() -> AsyncIterator[bytes]:
    sent = 0
    while sent < PAYLOAD:
        n = min(64 * 1024, PAYLOAD - sent)
        sent += n
        yield b"x" * n


def _sync_chunks():
    sent = 0
    while sent < PAYLOAD:
        n = min(64 * 1024, PAYLOAD - sent)
        sent += n
        yield b"x" * n


# ---------------------------------------------------------------------------
# The retry loop must not replay a body it has already started sending
# ---------------------------------------------------------------------------


async def test_async_streaming_body_is_not_replayed() -> None:
    mock_http = MagicMock(spec=httpx.AsyncClient)
    mock_http.request = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

    client = AsyncNemoClient(
        base_url=BASE,
        http_client=mock_http,
        retry=RetryPolicy(max_retries=3, backoff_base=0.0),
    )

    with pytest.raises(NemoTransportError):
        await client.with_headers({"Content-Length": str(PAYLOAD)}).send(_upload_request(_chunks()))

    assert mock_http.request.await_count == 1


def test_sync_streaming_body_is_not_replayed() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = httpx.ReadTimeout("timed out")

    client = NemoClient(
        base_url=BASE,
        http_client=mock_http,
        retry=RetryPolicy(max_retries=3, backoff_base=0.0),
    )

    with pytest.raises(NemoTransportError):
        client.with_headers({"Content-Length": str(PAYLOAD)}).send(_upload_request(_sync_chunks()))

    assert mock_http.request.call_count == 1


# ---------------------------------------------------------------------------
# Replayable bodies keep retrying
# ---------------------------------------------------------------------------


def test_bytes_body_still_retries() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = [
        httpx.ConnectError("boom"),
        httpx.Response(200, request=httpx.Request("PUT", f"{BASE}/apis/test/v2/upload")),
    ]

    client = NemoClient(
        base_url=BASE,
        http_client=mock_http,
        retry=RetryPolicy(max_retries=3, backoff_base=0.0),
    )
    resp = client.send(_upload_request(b"x" * PAYLOAD))

    assert resp.http_response.status_code == 200
    assert mock_http.request.call_count == 2


def test_bodyless_request_still_retries() -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = [
        httpx.Response(503, request=httpx.Request("GET", f"{BASE}/apis/test/v2/upload")),
        httpx.Response(200, request=httpx.Request("GET", f"{BASE}/apis/test/v2/upload")),
    ]

    client = NemoClient(
        base_url=BASE,
        http_client=mock_http,
        retry=RetryPolicy(max_retries=3, backoff_base=0.0),
    )
    request = PreparedRequest(
        path_template="/apis/test/v2/upload",
        path_params={},
        method="GET",
        content=None,
        content_type=None,
        response_type=None,
    )
    resp = client.send(request)

    assert resp.http_response.status_code == 200
    assert mock_http.request.call_count == 2


def test_list_body_still_retries() -> None:
    """httpx re-reads an in-memory sequence from the start, so it is replayable."""
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = [
        httpx.ReadTimeout("timed out"),
        httpx.Response(200, request=httpx.Request("PUT", f"{BASE}/apis/test/v2/upload")),
    ]

    client = NemoClient(
        base_url=BASE,
        http_client=mock_http,
        retry=RetryPolicy(max_retries=3, backoff_base=0.0),
    )
    resp = client.send(_upload_request([b"x" * 1024, b"y" * 1024]))

    assert resp.http_response.status_code == 200
    assert mock_http.request.call_count == 2


# ---------------------------------------------------------------------------
# A body that was never read is still safe to send again
# ---------------------------------------------------------------------------


def test_streaming_body_retries_when_the_connection_never_opened() -> None:
    """Connect failures happen before httpx touches the body, so replay is safe."""
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = [
        httpx.ConnectError("no route to host"),
        httpx.Response(200, request=httpx.Request("PUT", f"{BASE}/apis/test/v2/upload")),
    ]

    client = NemoClient(
        base_url=BASE,
        http_client=mock_http,
        retry=RetryPolicy(max_retries=3, backoff_base=0.0),
    )
    resp = client.with_headers({"Content-Length": str(PAYLOAD)}).send(_upload_request(_sync_chunks()))

    assert resp.http_response.status_code == 200
    assert mock_http.request.call_count == 2


async def test_async_streaming_body_retries_when_the_connection_never_opened() -> None:
    mock_http = MagicMock(spec=httpx.AsyncClient)
    mock_http.request = AsyncMock(
        side_effect=[
            httpx.ConnectTimeout("connect timed out"),
            httpx.Response(200, request=httpx.Request("PUT", f"{BASE}/apis/test/v2/upload")),
        ]
    )

    client = AsyncNemoClient(
        base_url=BASE,
        http_client=mock_http,
        retry=RetryPolicy(max_retries=3, backoff_base=0.0),
    )
    resp = await client.with_headers({"Content-Length": str(PAYLOAD)}).send(_upload_request(_chunks()))

    assert resp.http_response.status_code == 200
    assert mock_http.request.await_count == 2


def test_streaming_body_is_not_retried_on_a_retryable_status() -> None:
    """A response arrives only after the body is spent — there is nothing to resend."""
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = [
        httpx.Response(503, request=httpx.Request("PUT", f"{BASE}/apis/test/v2/upload"), json={"detail": "down"}),
        httpx.Response(200, request=httpx.Request("PUT", f"{BASE}/apis/test/v2/upload")),
    ]

    client = NemoClient(
        base_url=BASE,
        http_client=mock_http,
        retry=RetryPolicy(max_retries=3, backoff_base=0.0),
    )

    with pytest.raises(InternalServerError):
        client.with_headers({"Content-Length": str(PAYLOAD)}).send(_upload_request(_sync_chunks()))

    assert mock_http.request.call_count == 1


# ---------------------------------------------------------------------------
# The same, under the retry policy the SDK adapter actually installs
# ---------------------------------------------------------------------------


def _retry_me(status: int = 503) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"x-should-retry": "true"},
        request=httpx.Request("PUT", f"{BASE}/apis/test/v2/upload"),
        json={"detail": "down"},
    )


def test_one_shot_body_is_not_replayed_even_when_the_server_asks_for_a_retry() -> None:
    """``x-should-retry: true`` cannot conjure back a body that is already spent.

    The spent-body check has to come before the header branches, not after, or a
    server that sets this header would still get an exhausted iterator replayed
    under the original Content-Length.
    """
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = [
        _retry_me(),
        httpx.Response(200, request=httpx.Request("PUT", f"{BASE}/apis/test/v2/upload")),
    ]

    client = NemoClient(base_url=BASE, http_client=mock_http, retry=ADAPTER_POLICY)

    with pytest.raises(InternalServerError):
        client.with_headers({"Content-Length": str(PAYLOAD)}).send(_upload_request(_sync_chunks()))

    assert mock_http.request.call_count == 1


def test_replayable_body_still_honors_the_server_retry_header() -> None:
    """The spent-body check must not have swallowed header-driven retries wholesale."""
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = [
        _retry_me(),
        httpx.Response(200, request=httpx.Request("PUT", f"{BASE}/apis/test/v2/upload")),
    ]

    client = NemoClient(base_url=BASE, http_client=mock_http, retry=ADAPTER_POLICY)
    resp = client.send(_upload_request(b"x" * PAYLOAD))

    assert resp.http_response.status_code == 200
    assert mock_http.request.call_count == 2


# ---------------------------------------------------------------------------
# Declining a retry is worth a log line; everything else is not
# ---------------------------------------------------------------------------


def test_declined_retry_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.side_effect = httpx.ReadTimeout("timed out")

    client = NemoClient(
        base_url=BASE,
        http_client=mock_http,
        retry=RetryPolicy(max_retries=3, backoff_base=0.0),
    )

    with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER), pytest.raises(NemoTransportError):
        client.with_headers({"Content-Length": str(PAYLOAD)}).send(_upload_request(_sync_chunks()))

    assert "one-shot stream" in caplog.text
    assert "PUT /apis/test/v2/upload" in caplog.text


def test_successful_upload_does_not_log_a_retry_decline(caplog: pytest.LogCaptureFixture) -> None:
    """Nothing failed, so there was no retry to decline."""
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = httpx.Response(200, request=httpx.Request("PUT", f"{BASE}/apis/test/v2/upload"))

    client = NemoClient(
        base_url=BASE,
        http_client=mock_http,
        retry=RetryPolicy(max_retries=3, backoff_base=0.0),
    )

    with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER):
        resp = client.with_headers({"Content-Length": str(PAYLOAD)}).send(_upload_request(_sync_chunks()))

    assert resp.http_response.status_code == 200
    assert caplog.records == []


def test_non_retryable_status_does_not_log_a_retry_decline(caplog: pytest.LogCaptureFixture) -> None:
    """A 404 stops the retry loop on its own merits — the body never came into it."""
    mock_http = MagicMock(spec=httpx.Client)
    mock_http.request.return_value = httpx.Response(
        404, request=httpx.Request("PUT", f"{BASE}/apis/test/v2/upload"), json={"detail": "no such fileset"}
    )

    client = NemoClient(
        base_url=BASE,
        http_client=mock_http,
        retry=RetryPolicy(max_retries=3, backoff_base=0.0),
    )

    with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER), pytest.raises(NotFoundError):
        client.with_headers({"Content-Length": str(PAYLOAD)}).send(_upload_request(_sync_chunks()))

    assert caplog.records == []


# ---------------------------------------------------------------------------
# End to end over a real socket: the h11 failure mode itself
# ---------------------------------------------------------------------------


class _StallingServer:
    """Accepts a request, drains the body, then stalls the first attempt."""

    def __init__(self) -> None:
        self.attempts: list[int] = []
        self._server: asyncio.AbstractServer | None = None
        self._handlers: list[asyncio.Task] = []
        self.port = 0

    async def __aenter__(self) -> _StallingServer:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        # ``wait_closed`` blocks on in-flight handlers, and the stalling one sleeps
        # far longer than the test needs. Cancel them so teardown is immediate.
        for handler in self._handlers:
            handler.cancel()
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.append(task)
        attempt = len(self.attempts) + 1
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            declared = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    declared = int(line.split(b":", 1)[1])
            remaining = declared
            while remaining > 0:
                data = await reader.read(min(65536, remaining))
                if not data:
                    break
                remaining -= len(data)
            self.attempts.append(declared - remaining)
            if attempt == 1:
                # Stall past the client read timeout, as a Files service writing
                # multiple GB to storage would before it answers.
                await asyncio.sleep(STALL_SECONDS)
            body = b"{}"
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
            await writer.drain()
        except (asyncio.CancelledError, ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()


async def test_read_timeout_does_not_become_a_content_length_error() -> None:
    """The upload's real failure must survive, not be masked by an h11 abort."""
    async with _StallingServer() as server:
        base = f"http://127.0.0.1:{server.port}"
        async with httpx.AsyncClient(base_url=base, timeout=httpx.Timeout(5.0, read=0.5)) as http:
            client = AsyncNemoClient(
                base_url=base,
                http_client=http,
                retry=RetryPolicy(max_retries=2, backoff_base=0.0),
            )

            with pytest.raises(NemoTransportError) as excinfo:
                await client.with_headers({"Content-Length": str(PAYLOAD)}).send(_upload_request(_chunks()))

    assert isinstance(excinfo.value.error, httpx.ReadTimeout)
    # One attempt only. A second would have re-sent an exhausted iterator under
    # the original Content-Length and raised h11's LocalProtocolError instead.
    assert server.attempts == [PAYLOAD]

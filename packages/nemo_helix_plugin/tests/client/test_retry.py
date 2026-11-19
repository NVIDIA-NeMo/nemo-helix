# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retry rules shared by the client and callers that rebuild request bodies."""

from __future__ import annotations

import httpx
import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoTransportError
from nemo_helix_plugin.client.retry import body_was_sent, is_retryable, retry_delay
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY, PreparedRequest, RetryPolicy

POLICY = PLATFORM_DEFAULT_RETRY_POLICY
REQUEST = httpx.Request("PUT", "http://test/upload")


def _response(status: int, **headers: str) -> httpx.Response:
    return httpx.Response(status, headers=headers, request=REQUEST)


@pytest.mark.parametrize(
    "failure,policy,exist_ok,expected",
    [
        pytest.param(httpx.ReadTimeout("t", request=REQUEST), POLICY, False, True, id="transport-error"),
        pytest.param(_response(503), POLICY, False, True, id="server-error"),
        pytest.param(_response(503), RetryPolicy(), False, True, id="listed-status"),
        pytest.param(_response(500), RetryPolicy(), False, False, id="unlisted-server-error"),
        pytest.param(_response(403), POLICY, False, False, id="client-error"),
        pytest.param(_response(409), POLICY, True, False, id="conflict-resolved-by-caller"),
        pytest.param(_response(503, **{"x-should-retry": "false"}), POLICY, False, False, id="server-declines"),
        pytest.param(_response(400, **{"x-should-retry": "true"}), POLICY, False, True, id="server-insists"),
        pytest.param(_response(400, **{"x-should-retry": "true"}), RetryPolicy(), False, False, id="header-ignored"),
    ],
)
def test_is_retryable(
    failure: httpx.Response | httpx.TransportError, policy: RetryPolicy, exist_ok: bool, expected: bool
) -> None:
    assert is_retryable(policy, failure, exist_ok=exist_ok) is expected


def test_retry_delay_backs_off_exponentially() -> None:
    error = httpx.ReadTimeout("t", request=REQUEST)
    assert [retry_delay(POLICY, attempt, error) for attempt in range(3)] == [0.5, 1.0, 2.0]


def test_retry_delay_prefers_retry_after() -> None:
    assert retry_delay(POLICY, 0, _response(429, **{"Retry-After": "7"})) == 7.0
    assert retry_delay(RetryPolicy(), 0, _response(429, **{"Retry-After": "7"})) == RetryPolicy().backoff_base


@pytest.mark.parametrize(
    "error,expected",
    [
        pytest.param(httpx.ConnectError("refused", request=REQUEST), False, id="refused"),
        pytest.param(httpx.ReadTimeout("t", request=REQUEST), True, id="read-timeout"),
    ],
)
def test_body_was_sent(error: httpx.TransportError, expected: bool) -> None:
    assert body_was_sent(error) is expected


NO_RESEND = RetryPolicy(resend_after_body_sent=False)


@pytest.mark.parametrize(
    "error,policy,expected",
    [
        pytest.param(httpx.ReadTimeout("t", request=REQUEST), NO_RESEND, False, id="read-timeout-not-resent"),
        pytest.param(httpx.RemoteProtocolError("drop", request=REQUEST), NO_RESEND, False, id="dropped-not-resent"),
        pytest.param(httpx.ConnectError("refused", request=REQUEST), NO_RESEND, True, id="connect-error-resent"),
        pytest.param(httpx.ConnectTimeout("t", request=REQUEST), NO_RESEND, True, id="connect-timeout-resent"),
        pytest.param(httpx.ReadTimeout("t", request=REQUEST), RetryPolicy(), True, id="default-read-timeout"),
        pytest.param(httpx.RemoteProtocolError("d", request=REQUEST), RetryPolicy(), True, id="default-dropped"),
        pytest.param(httpx.ConnectError("refused", request=REQUEST), RetryPolicy(), True, id="default-connect"),
    ],
)
def test_resend_after_body_sent(error: httpx.TransportError, policy: RetryPolicy, expected: bool) -> None:
    assert is_retryable(policy, error) is expected


def test_send_does_not_resend_a_sent_body_under_the_flag() -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    client = NemoClient(
        base_url="http://test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry=RetryPolicy(max_retries=3, backoff_base=0.0, resend_after_body_sent=False),
    )
    request = PreparedRequest(
        path_template="/apis/test/v2/runs",
        path_params={},
        method="POST",
        content=b"{}",
        content_type="application/json",
        response_type=None,
    )

    with pytest.raises(NemoTransportError):
        client.send(request)
    assert len(attempts) == 1

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx
import openai
from nhx_evals_sdk.resilience.classifier import classify_exception
from nhx_evals_sdk.resilience.types import FailureClass


def test_classify_hard_overload_from_http_status():
    response = httpx.Response(429, request=httpx.Request("GET", "https://example.com"))
    exc = httpx.HTTPStatusError("rate limit", request=response.request, response=response)

    result = classify_exception(exc)

    assert result.retryable is True
    assert result.failure_class == FailureClass.HARD_OVERLOAD
    assert result.status_code == 429
    assert result.error_type == "HTTPStatusError"


def test_classify_soft_overload_from_read_timeout():
    result = classify_exception(httpx.ReadTimeout("timeout"))

    assert result.retryable is True
    assert result.failure_class == FailureClass.SOFT_OVERLOAD
    assert result.status_code is None
    assert result.error_type == "ReadTimeout"


def test_classify_transient_from_connect_timeout():
    result = classify_exception(httpx.ConnectTimeout("connect timeout"))

    assert result.retryable is True
    assert result.failure_class == FailureClass.TRANSIENT


def test_classify_retry_after_value_from_status_error():
    response = httpx.Response(
        503,
        headers={"Retry-After": "2"},
        request=httpx.Request("GET", "https://example.com"),
    )
    exc = httpx.HTTPStatusError("service unavailable", request=response.request, response=response)

    result = classify_exception(exc)

    assert result.retryable is True
    assert result.failure_class == FailureClass.HARD_OVERLOAD
    assert result.retry_after_seconds is not None
    assert 1.9 <= result.retry_after_seconds <= 2.1


def test_classify_fatal_for_bad_request():
    response = httpx.Response(400, request=httpx.Request("GET", "https://example.com"))
    exc = openai.BadRequestError("bad request", response=response, body=None)
    result = classify_exception(exc)

    assert result.retryable is False
    assert result.failure_class == FailureClass.FATAL


def test_classify_http_408_as_transient():
    response = httpx.Response(408, request=httpx.Request("GET", "https://example.com"))
    exc = httpx.HTTPStatusError("request timeout", request=response.request, response=response)
    result = classify_exception(exc)

    assert result.retryable is True
    assert result.failure_class == FailureClass.TRANSIENT
    assert result.status_code == 408

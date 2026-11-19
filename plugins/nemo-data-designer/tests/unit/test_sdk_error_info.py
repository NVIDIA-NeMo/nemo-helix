# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SDK error helpers."""

import json

import httpx
import pytest
from nemo_data_designer_plugin.sdk.errors import (
    DataDesignerClientError,
    DataDesignerConfigValidationError,
    DataDesignerJobError,
    DataDesignerPreviewError,
    extract_http_error_info,
)


@pytest.mark.parametrize(
    ("status_code", "detail"),
    [
        (400, "bad request"),
        (401, "unauthorized"),
        (403, "forbidden"),
        (404, "not found"),
        (500, "server error"),
        (503, "unavailable"),
    ],
)
def test_extract_status_and_detail(status_code: int, detail: str) -> None:
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(status_code, json={"detail": detail}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)

    assert extract_http_error_info(exc) == (status_code, detail)


def test_extract_teapot_response() -> None:
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(418, json={"detail": "teapot"}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)

    assert extract_http_error_info(exc) == (418, "teapot")


@pytest.mark.parametrize("body", ['{"detail": "boom"}', "boom"])
def test_extract_handles_json_and_plain_text(body: str) -> None:
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(500, text=body, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)

    assert extract_http_error_info(exc) == (500, "boom")


def test_extract_json_detail() -> None:
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(500, json={"detail": "patched detail"}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)

    assert extract_http_error_info(exc) == (500, "patched detail")


def test_extract_response_content() -> None:
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(502, content=b'{"detail": "from private content"}', request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)

    assert exc.request is request
    assert extract_http_error_info(exc) == (502, "from private content")


def test_extract_timeout_response() -> None:
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(504, json={"detail": "timeout"}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)

    assert extract_http_error_info(exc) == (504, "timeout")


def test_extract_empty_response() -> None:
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(500, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)

    assert extract_http_error_info(exc) == (500, "")


@pytest.mark.parametrize(
    "error_type",
    [DataDesignerClientError, DataDesignerConfigValidationError, DataDesignerPreviewError, DataDesignerJobError],
)
def test_error_classes(error_type: type[DataDesignerClientError]) -> None:
    assert issubclass(error_type, Exception)
    assert issubclass(error_type, DataDesignerClientError)
    err = error_type("x", status_code=404)

    assert err.status_code == 404
    assert str(err) == "x"
    assert err.args == ("x",)


def test_extract_json_without_detail() -> None:
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(200, json={"a": 1}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)

    assert response.status_code == 200
    assert json.loads(response.text) == {"a": 1}
    assert extract_http_error_info(exc) == (200, response.text)


def test_all_statuses_were_seen() -> None:
    statuses = [400, 401, 403, 404, 500, 503]
    results = []
    for status_code in statuses:
        request = httpx.Request("GET", "http://test/x")
        response = httpx.Response(status_code, json={"detail": "error"}, request=request)
        exc = httpx.HTTPStatusError("error", request=request, response=response)
        status, detail = extract_http_error_info(exc)
        results.append(status)
        assert detail == "error"

    assert results == statuses

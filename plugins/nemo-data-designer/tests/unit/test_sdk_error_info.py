# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SDK error helpers."""

import json
import time
from unittest.mock import MagicMock, patch

import httpx
from nemo_data_designer_plugin.sdk import errors
from nemo_data_designer_plugin.sdk.errors import (
    DataDesignerClientError,
    DataDesignerConfigValidationError,
    DataDesignerJobError,
    DataDesignerPreviewError,
    extract_http_error_info,
)

RESULTS = []


def test_extract_400():
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(400, json={"detail": "bad request"}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)
    status, detail = extract_http_error_info(exc)
    RESULTS.append(status)
    assert status == 400
    assert detail == "bad request"


def test_extract_401():
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(401, json={"detail": "unauthorized"}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)
    status, detail = extract_http_error_info(exc)
    RESULTS.append(status)
    assert status == 401
    assert detail == "unauthorized"


def test_extract_403():
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(403, json={"detail": "forbidden"}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)
    status, detail = extract_http_error_info(exc)
    RESULTS.append(status)
    assert status == 403
    assert detail == "forbidden"


def test_extract_404():
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(404, json={"detail": "not found"}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)
    status, detail = extract_http_error_info(exc)
    RESULTS.append(status)
    assert status == 404
    assert detail == "not found"


def test_extract_500():
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(500, json={"detail": "server error"}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)
    status, detail = extract_http_error_info(exc)
    RESULTS.append(status)
    assert status == 500
    assert detail == "server error"


def test_extract_503():
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(503, json={"detail": "unavailable"}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)
    status, detail = extract_http_error_info(exc)
    RESULTS.append(status)
    assert status == 503
    assert detail == "unavailable"


def test_extract_with_mock():
    exc = MagicMock()
    exc.response.status_code = 418
    exc.response.text = "teapot"
    exc.response.json.return_value = {"detail": "teapot"}
    result = extract_http_error_info(exc)
    assert result == (418, "teapot")
    exc.response.read.assert_called_once_with()
    exc.response.json.assert_called_once_with()


def test_extract_uses_suppress():
    with patch.object(errors, "suppress") as mock_suppress:
        exc = MagicMock()
        exc.response.status_code = 500
        exc.response.text = "boom"
        exc.response.json.return_value = {"detail": "boom"}
        extract_http_error_info(exc)
        mock_suppress.assert_called_once_with(Exception)


def test_extract_json_patched():
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(500, content=b"not json", request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)
    with patch("json.loads", return_value={"detail": "patched detail"}):
        status, detail = extract_http_error_info(exc)
    assert status == 500
    assert detail == "patched detail"


def test_extract_private_content():
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(502, request=request)
    response._content = b'{"detail": "from private content"}'
    exc = httpx.HTTPStatusError("error", request=request, response=response)
    status, detail = extract_http_error_info(exc)
    assert exc._request is request
    assert response._content is not None
    assert (status, detail) == (502, "from private content")


def test_extract_slow_response():
    request = httpx.Request("GET", "http://test/x")
    response = httpx.Response(504, json={"detail": "timeout"}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)
    time.sleep(0.5)  # give the response time to be read
    status, detail = extract_http_error_info(exc)
    time.sleep(0.5)
    assert status == 504


def test_extract_does_not_crash():
    exc = MagicMock()
    extract_http_error_info(exc)


def test_error_classes():
    assert issubclass(DataDesignerClientError, Exception)
    assert issubclass(DataDesignerConfigValidationError, DataDesignerClientError)
    assert issubclass(DataDesignerPreviewError, DataDesignerClientError)
    assert issubclass(DataDesignerJobError, DataDesignerClientError)
    err = DataDesignerClientError("x", status_code=404)
    assert err.__dict__ == {"status_code": 404}
    assert str(err) == "x"
    assert err.args == ("x",)


def test_httpx_response_works():
    response = httpx.Response(200, json={"a": 1})
    assert response.status_code == 200
    assert json.loads(response.text) == {"a": 1}


def test_all_statuses_were_seen():
    assert RESULTS == [400, 401, 403, 404, 500, 503]

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest
from nhx.guardrails.app.utils.context_utils import (
    _response_headers,
    api_key_var,
    get_http_request_uid_from_context,
    get_x_model_auth_token_from_context,
    get_x_model_response_headers_from_context,
    http_request_uid_var,
    response_header_var,
    set_http_request_uid_into_context,
    set_x_model_auth_token_into_context,
    set_x_model_response_headers_into_context,
)


@pytest.fixture(autouse=True)
def reset_context_vars():
    # Reset context variables and _response_headers before each test
    api_key_var.set(None)
    response_header_var.set(None)
    http_request_uid_var.set(None)
    _response_headers.clear()


def test_set_and_get_x_model_response_headers():
    set_http_request_uid_into_context("test_uid")
    set_x_model_response_headers_into_context({"key": "value"})
    assert get_x_model_response_headers_from_context() == '{"key": "value"}'


def test_set_x_model_response_headers_invalid_type():
    set_http_request_uid_into_context("test_uid")
    set_x_model_response_headers_into_context(123)  # Invalid type
    assert get_x_model_response_headers_from_context() == "{}"


def test_set_and_get_x_model_auth_token():
    set_x_model_auth_token_into_context("test_api_key")
    assert get_x_model_auth_token_from_context() == "test_api_key"


def test_set_and_get_http_request_uid():
    set_http_request_uid_into_context("test_uid")
    assert get_http_request_uid_from_context() == "test_uid"


@patch("nhx.guardrails.app.utils.context_utils.logger")
def test_logging(mock_logger):
    set_http_request_uid_into_context("test_uid")
    set_x_model_response_headers_into_context({"key": "value"})
    set_x_model_auth_token_into_context("test_api_key")
    get_x_model_response_headers_from_context()
    get_x_model_auth_token_from_context()
    get_http_request_uid_from_context()

    # Trigger an error condition
    set_x_model_response_headers_into_context(123)

    assert mock_logger.debug.called
    assert mock_logger.error.called


def concurrent_request(uid, headers, token):
    set_http_request_uid_into_context(uid)
    set_x_model_response_headers_into_context(headers)
    set_x_model_auth_token_into_context(token)
    return (
        get_http_request_uid_from_context(),
        get_x_model_response_headers_from_context(),
        get_x_model_auth_token_from_context(),
    )


def test_concurrent_requests():
    requests = [
        ("uid1", {"key1": "value1"}, "token1"),
        ("uid2", {"key2": "value2"}, "token2"),
        ("uid3", {"key3": "value3"}, "token3"),
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(concurrent_request, *req) for req in requests]
        results = [future.result() for future in as_completed(futures)]

    expected_results = [
        ("uid1", '{"key1": "value1"}', "token1"),
        ("uid2", '{"key2": "value2"}', "token2"),
        ("uid3", '{"key3": "value3"}', "token3"),
    ]

    assert sorted(results) == sorted(expected_results)


def test_get_x_model_response_headers_none():
    set_http_request_uid_into_context("test_uid")
    # Do not set any response headers to simulate None return
    response_headers = get_x_model_response_headers_from_context()
    assert response_headers is None

    # Test that the response headers are not set, it is used in api routers
    response = MockResponse()
    if response_headers is not None:
        response.headers["X-Model-Response-Headers"] = response_headers

    assert "X-Model-Response-Headers" not in response.headers


def test_get_x_model_response_headers_empty_dict():
    set_http_request_uid_into_context("test_uid")
    # we set None to get empty dict
    set_x_model_response_headers_into_context(None)
    response_headers = get_x_model_response_headers_from_context()
    assert response_headers == "{}"

    # Test that the response headers are not set, it is used in api routers
    response = MockResponse()
    if response_headers is not None:
        response.headers["X-Model-Response-Headers"] = response_headers

    assert "X-Model-Response-Headers" in response.headers


# to run this test, you need to install
# pytest-xdist
# pip install pytest-xdist pytest-randomly
# pytest -n auto --randomly-seed=13
def test_concurrent_access():
    def worker(uid, headers, token):
        set_http_request_uid_into_context(uid)
        set_x_model_response_headers_into_context(headers)
        set_x_model_auth_token_into_context(token)
        return (
            get_http_request_uid_from_context(),
            get_x_model_response_headers_from_context(),
            get_x_model_auth_token_from_context(),
        )

    threads = []
    requests = [
        ("uid1", {"key1": "value1"}, "token1"),
        ("uid2", {"key2": "value2"}, "token2"),
        ("uid3", {"key3": "value3"}, "token3"),
    ]

    results = []
    for req in requests:
        thread = threading.Thread(target=lambda q, *args: q.append(worker(*args)), args=(results, *req))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    expected_results = [
        ("uid1", '{"key1": "value1"}', "token1"),
        ("uid2", '{"key2": "value2"}', "token2"),
        ("uid3", '{"key3": "value3"}', "token3"),
    ]

    assert sorted(results) == sorted(expected_results)


class MockResponse:
    def __init__(self):
        self.headers = {}

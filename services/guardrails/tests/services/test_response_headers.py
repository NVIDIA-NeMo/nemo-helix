# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from contextvars import ContextVar
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage
from nhx.guardrails.app.llms.chat.nim import ChatNIM
from nhx.guardrails.app.utils.context_utils import (
    _response_headers,
    get_x_model_response_headers_from_context,
    http_request_uid_var,
    set_http_request_uid_into_context,
    set_x_model_response_headers_into_context,
)


class TestXModelResponseHeader(unittest.TestCase):
    def setUp(self):
        # Clear the context variables before each test
        _response_headers.clear()
        self.request_uid_var = ContextVar("request_uid")
        self.request_uid_var.set(None)

        # Reset the http_request_uid_var context variable
        http_request_uid_var.set(None)

        # Patch the HTTP request uid context var
        self.http_request_uid_patcher = patch(
            "nhx.guardrails.app.utils.context_utils.http_request_uid_var",
            new=self.request_uid_var,
        )
        self.http_request_uid_patcher.start()

        # Mock OpenAI client and its response
        self.openai_patcher = patch("openai.OpenAI")
        self.mock_openai_client_class = self.openai_patcher.start()
        self.mock_openai_client = MagicMock()
        self.mock_openai_client_class.return_value = self.mock_openai_client

        # Mock the response headers
        self.mock_response_headers = {
            "x-model-response-header": "test-header-value",
            "content-type": "application/json",
        }

        # Mock the client create methods to return a response with headers
        self.mock_raw_response = MagicMock()
        self.mock_raw_response.headers = self.mock_response_headers
        self.mock_raw_response.parse.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ]
        }
        self.mock_openai_client.chat.completions.with_raw_response.create.return_value = self.mock_raw_response

        # Mock the client create method without headers
        self.mock_response = MagicMock()
        self.mock_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ]
        }
        self.mock_openai_client.chat.completions.create.return_value = self.mock_response

        # Mock get_x_model_auth_token to return a valid auth token
        self.auth_token_patcher = patch(
            "nhx.guardrails.app.llms.utils.get_x_model_auth_token_from_context",
            return_value="test-auth-token",
        )
        self.auth_token_patcher.start()

        # Mock get_main_model_from_context to return None by default
        self.main_model_patcher = patch("nhx.guardrails.app.llms.utils.get_main_model_from_context")
        self.mock_get_main_model_from_context = self.main_model_patcher.start()
        self.mock_get_main_model_from_context.return_value = None

    def tearDown(self):
        self.http_request_uid_patcher.stop()
        self.openai_patcher.stop()
        self.auth_token_patcher.stop()
        self.main_model_patcher.stop()
        http_request_uid_var.set(None)
        _response_headers.clear()

    def test_set_and_get_x_model_response_headers(self):
        # set the HTTP request uid
        request_uid = "test-request-uid"
        set_http_request_uid_into_context(request_uid)

        # set the response headers
        set_x_model_response_headers_into_context(self.mock_response_headers)

        # fetch the headers
        fetched_headers = get_x_model_response_headers_from_context()
        if fetched_headers is None:
            self.fail("Response headers not set in the context")
        fetched_headers_dict = json.loads(fetched_headers)

        assert fetched_headers_dict == self.mock_response_headers
        # ensure headers are popped after fetch
        assert not _response_headers.get(request_uid)

    def test_response_headers_set_in_api_call(self):
        request_uid = "test-request-uid"
        set_http_request_uid_into_context(request_uid)

        # instantiate the ChatNIM class with include_response_headers=True
        chat_model = ChatNIM(model="test-model", include_response_headers=True)

        # calling the _generate method, which should set the headers
        chat_model._generate(messages=[HumanMessage(content="Hello")])

        fetched_headers = get_x_model_response_headers_from_context()
        if fetched_headers is None:
            self.fail("Response headers not set in the context")
        fetched_headers_dict = json.loads(fetched_headers)

        assert fetched_headers_dict == self.mock_response_headers

    def test_response_headers_not_set_when_disabled(self):
        request_uid = "test-request-uid"
        set_http_request_uid_into_context(request_uid)

        # instantiate  ChatNIM with include_response_headers=False
        chat_model = ChatNIM(model="test-model", include_response_headers=False)

        # calling the _generate method, it should not set the headers
        chat_model._generate(messages=[], stop=None)

        fetched_headers = get_x_model_response_headers_from_context()

        assert fetched_headers is None

    def test_response_headers_isolated_per_request(self):
        """Test that response headers are isolated per request"""
        # Simulating two different requests
        request_uid1 = "request-uid-1"
        request_uid2 = "request-uid-2"

        # set headers for request 1
        set_http_request_uid_into_context(request_uid1)
        set_x_model_response_headers_into_context({"x-model-response-header": "value-1"})

        # set headers for request 2
        set_http_request_uid_into_context(request_uid2)
        set_x_model_response_headers_into_context({"x-model-response-header": "value-2"})

        # get headers for request 1
        set_http_request_uid_into_context(request_uid1)
        headers1 = get_x_model_response_headers_from_context()
        if headers1 is None:
            self.fail("Response headers not set in the context")
        headers1_dict = json.loads(headers1)
        assert headers1_dict == {"x-model-response-header": "value-1"}

        # get headers for request 2
        set_http_request_uid_into_context(request_uid2)
        headers2 = get_x_model_response_headers_from_context()
        if headers2 is None:
            self.fail("Response headers not set in the context")
        headers2_dict = json.loads(headers2)
        assert headers2_dict == {"x-model-response-header": "value-2"}

    def test_response_headers_are_popped_after_retrieval(self):
        request_uid = "test-request-uid"
        set_http_request_uid_into_context(request_uid)

        set_x_model_response_headers_into_context(self.mock_response_headers)

        # fetch the headers for the first time
        fetched_headers1 = get_x_model_response_headers_from_context()
        if fetched_headers1 is None:
            self.fail("Response headers not set in the context")
        fetched_headers1_dict = json.loads(fetched_headers1)
        assert fetched_headers1_dict == self.mock_response_headers

        # Try to fetch the headers again
        fetched_headers2 = get_x_model_response_headers_from_context()

        assert fetched_headers2 is None

    def test_set_x_model_response_headers_with_invalid_type(self):
        request_uid = "test-request-uid"
        set_http_request_uid_into_context(request_uid)

        # Pass an invalid header type (e.g. an integer)
        set_x_model_response_headers_into_context(12345)

        # fetch the headers
        fetched_headers = get_x_model_response_headers_from_context()
        # should return empty JSON string
        assert fetched_headers == "{}"

    def test_get_x_model_response_headers_without_setting(self):
        request_uid = "test-request-uid"
        set_http_request_uid_into_context(request_uid)

        # we do not set any headers

        # fetch the headers
        fetched_headers = get_x_model_response_headers_from_context()
        assert fetched_headers is None

    def test_get_x_model_response_headers_without_request_uid(self):
        # we do not set the HTTP request UID

        # set some headers
        set_x_model_response_headers_into_context({"x-model-response-header": "value"})

        # fetch the headers
        fetched_headers = get_x_model_response_headers_from_context()
        assert fetched_headers is None

    def test_concurrent_requests_headers(self):
        # simulate concurrent requests using threads
        import threading

        results = {}

        def simulate_request(request_uid, header_value):
            set_http_request_uid_into_context(request_uid)
            set_x_model_response_headers_into_context({"x-model-response-header": header_value})
            fetched_headers = get_x_model_response_headers_from_context()
            results[request_uid] = fetched_headers

        threads = []
        for i in range(5):
            request_uid = f"request-uid-{i}"
            header_value = f"value-{i}"
            thread = threading.Thread(target=simulate_request, args=(request_uid, header_value))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # ensure that each thread fetched its own headers
        for i in range(5):
            request_uid = f"request-uid-{i}"
            expected_headers = {"x-model-response-header": f"value-{i}"}
            expected_headers_str = json.dumps(expected_headers)
            assert results[request_uid] == expected_headers_str


if __name__ == "__main__":
    unittest.main()

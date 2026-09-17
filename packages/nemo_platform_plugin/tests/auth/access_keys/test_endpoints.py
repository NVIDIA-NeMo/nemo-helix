# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_platform_plugin.auth.access_keys import endpoints
from nemo_platform_plugin.auth.access_keys.types import AccessKeyCreateRequest, AccessKeyRotateRequest
from nemo_platform_plugin.client.types import PreparedRequest


def test_create_access_key_endpoint_uses_gateway_path() -> None:
    prepared = endpoints.create_access_key(body=AccessKeyCreateRequest(name="ci-intake"))

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/auth/v2/access-keys"
    assert prepared.content_type == "application/json"


def test_create_access_key_endpoint_allows_unnamed_tokens() -> None:
    prepared = endpoints.create_access_key(body=AccessKeyCreateRequest())

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/auth/v2/access-keys"
    assert prepared.content == b"{}"


def test_revoke_access_key_endpoint_uses_jti_path_param() -> None:
    prepared = endpoints.revoke_access_key(jti="ak_example")

    assert prepared.method == "DELETE"
    assert prepared.path_template == "/apis/auth/v2/access-keys/{jti}"
    assert prepared.path_params == {"jti": "ak_example"}


def test_list_access_keys_endpoint_supports_pagination() -> None:
    prepared = endpoints.list_access_keys(query_params={"page": 3, "page_size": 25})

    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/auth/v2/access-keys"
    assert prepared.query_params == {"page": 3, "page_size": 25}


def test_suspend_access_key_endpoint_uses_jti_path_param() -> None:
    prepared = endpoints.suspend_access_key(jti="ak_example")

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/auth/v2/access-keys/{jti}/suspend"
    assert prepared.path_params == {"jti": "ak_example"}


def test_unsuspend_access_key_endpoint_uses_jti_path_param() -> None:
    prepared = endpoints.unsuspend_access_key(jti="ak_example")

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/auth/v2/access-keys/{jti}/unsuspend"
    assert prepared.path_params == {"jti": "ak_example"}


def test_rotate_access_key_endpoint_uses_jti_path_param() -> None:
    prepared = endpoints.rotate_access_key(jti="ak_example", body=AccessKeyRotateRequest())

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/auth/v2/access-keys/{jti}/rotate"
    assert prepared.path_params == {"jti": "ak_example"}


def test_rotate_access_key_endpoint_sends_grace_period_seconds() -> None:
    prepared = endpoints.rotate_access_key(jti="ak_example", body=AccessKeyRotateRequest(grace_period_seconds=3600))

    assert prepared.content_type == "application/json"
    assert prepared.content == b'{"grace_period_seconds":3600}'

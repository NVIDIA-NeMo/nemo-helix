# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_platform_plugin.auth import endpoints
from nemo_platform_plugin.client.types import PreparedRequest


def test_get_authenticate_bearer_token_endpoint_uses_gateway_path() -> None:
    prepared = endpoints.authenticate_bearer_token_get()

    assert isinstance(prepared, PreparedRequest)
    assert prepared.method == "GET"
    assert prepared.path_template == "/apis/auth/authenticate"


def test_post_authenticate_bearer_token_endpoint_uses_gateway_path() -> None:
    prepared = endpoints.authenticate_bearer_token_post()

    assert prepared.method == "POST"
    assert prepared.path_template == "/apis/auth/authenticate"
    assert prepared.content is None

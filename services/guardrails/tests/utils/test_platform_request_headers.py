# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.common.config import Configuration
from nhx.guardrails.app.utils.context_utils import (
    get_request_default_headers_from_context,
    set_request_default_headers_into_context,
)
from nhx.guardrails.app.utils.platform_request_headers import (
    apply_platform_auth_headers,
    get_platform_auth_headers_from_context,
    headers_for_model_endpoint,
    publish_downstream_request_headers,
    set_platform_auth_headers_into_context,
)
from nhx.guardrails.entities.values._private import Model, RailsConfig

_PLATFORM_AUTH = {
    "Authorization": "Bearer workload-token",
    "X-NHX-Principal-Id": "service:guardrails",
}


@pytest.fixture
def platform_endpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NHX_BASE_URL", "https://platform.example.test")
    Configuration.clear_cache()
    set_request_default_headers_into_context({"X-Custom": "kept", "traceparent": "00-abc"})
    set_platform_auth_headers_into_context(dict(_PLATFORM_AUTH))
    yield
    set_request_default_headers_into_context({})
    set_platform_auth_headers_into_context({})
    Configuration.clear_cache()


def test_platform_origin_receives_platform_auth(platform_endpoint: None):
    headers = headers_for_model_endpoint(
        "https://platform.example.test/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )

    assert headers["X-Custom"] == "kept"
    assert headers["Authorization"] == "Bearer workload-token"
    assert headers["X-NHX-Principal-Id"] == "service:guardrails"


def test_external_model_url_omits_platform_auth(platform_endpoint: None):
    headers = headers_for_model_endpoint("http://nim.example.test/v1")

    assert headers == {"X-Custom": "kept", "traceparent": "00-abc"}
    assert "Authorization" not in headers
    assert "X-NHX-Principal-Id" not in headers


def test_apply_platform_auth_headers_only_on_platform_models(platform_endpoint: None):
    config = RailsConfig(
        models=[
            Model(
                type="main",
                engine="nim",
                model="default/llama",
                parameters={"base_url": "https://platform.example.test/apis/inference-gateway/v2/v1"},
            ),
            Model(
                type="content_safety",
                engine="nim",
                model="safety",
                parameters={"base_url": "https://nim.example.test/v1", "default_headers": {"X-Custom": "kept"}},
            ),
        ]
    )

    updated = apply_platform_auth_headers(config)
    platform_headers = updated.models[0].parameters.default_headers
    external_headers = updated.models[1].parameters.default_headers

    assert platform_headers["Authorization"] == "Bearer workload-token"
    assert external_headers == {"X-Custom": "kept"}


@pytest.mark.asyncio
async def test_publish_strips_inbound_platform_auth_from_shared_headers():
    await publish_downstream_request_headers(
        {
            "X-Custom": "kept",
            "X-NHX-Principal-Id": "user-from-request",
            "Authorization": "Bearer inbound",
        }
    )

    assert get_request_default_headers_from_context() == {"X-Custom": "kept"}
    assert "user-from-request" not in get_platform_auth_headers_from_context().values()
    assert get_platform_auth_headers_from_context()["X-NHX-Principal-Id"] == "service:guardrails"
    set_request_default_headers_into_context({})
    set_platform_auth_headers_into_context({})

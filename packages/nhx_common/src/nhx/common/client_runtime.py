# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-aware runtime policy for typed NeMo clients."""

from __future__ import annotations

from collections.abc import Callable

import httpx
from nemo_helix_plugin.client.client import NemoClientRuntime
from nhx.common.platform_endpoint import HelixEndpoint, require_authorization_header_request


def platform_url_resolver(endpoint: HelixEndpoint) -> Callable[[str], str | httpx.URL]:
    def resolve(url: str) -> str | httpx.URL:
        return endpoint.route_request_url(url).url

    return resolve


def authorization_endpoint_policy(endpoint: HelixEndpoint) -> Callable[[str, str, str], None]:
    def validate(raw_url: str, _resolved_url: str, request_purpose: str) -> None:
        require_authorization_header_request(endpoint, raw_url, purpose=request_purpose)

    return validate


def build_platform_client_runtime(
    endpoint: HelixEndpoint,
) -> NemoClientRuntime:
    return NemoClientRuntime(
        url_resolver=platform_url_resolver(endpoint),
        authorization_endpoint_policy=authorization_endpoint_policy(endpoint),
    )

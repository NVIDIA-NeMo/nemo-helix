# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an SDK client for a basic-auth Intake deployment."""

from collections.abc import Awaitable, Callable

import httpx
from nemo_helix import AsyncNeMoHelix

_SDK_INTAKE_PREFIX = "/apis/intake/"
_DEFAULT_REAL_PREFIX = "/api/intake/"


def _make_request_rewriter(
    sdk_prefix: str,
    real_prefix: str,
) -> Callable[[httpx.Request], Awaitable[None]]:
    """Return a request hook that rewrites an Intake path prefix."""
    sdk = sdk_prefix.encode()
    real = real_prefix.encode()

    async def _rewrite(request: httpx.Request) -> None:
        raw_path = request.url.raw_path
        if raw_path.startswith(sdk):
            request.url = request.url.copy_with(raw_path=real + raw_path[len(sdk) :])

    return _rewrite


def build_rewriting_http_client(
    *,
    username: str,
    password: str,
    real_prefix: str = _DEFAULT_REAL_PREFIX,
    sdk_prefix: str = _SDK_INTAKE_PREFIX,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Build an HTTP client with basic auth and an Intake path-prefix rewrite."""
    return httpx.AsyncClient(
        auth=httpx.BasicAuth(username, password),
        event_hooks={"request": [_make_request_rewriter(sdk_prefix, real_prefix)]},
        transport=transport,
        timeout=60.0,
    )


def build_basic_auth_intake_client(
    *,
    base_url: str,
    username: str,
    password: str,
    real_prefix: str = _DEFAULT_REAL_PREFIX,
    sdk_prefix: str = _SDK_INTAKE_PREFIX,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncNeMoHelix:
    """Build an SDK client for a basic-auth Intake mounted at ``real_prefix``."""
    http_client = build_rewriting_http_client(
        username=username,
        password=password,
        real_prefix=real_prefix,
        sdk_prefix=sdk_prefix,
        transport=transport,
    )
    return AsyncNeMoHelix(base_url=base_url, http_client=http_client)

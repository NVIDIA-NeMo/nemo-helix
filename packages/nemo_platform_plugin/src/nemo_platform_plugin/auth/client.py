# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_platform_plugin.auth import endpoints
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.method import method


class _AuthenticationMethods:
    authenticate_bearer_token_get = method(endpoints.authenticate_bearer_token_get)
    authenticate_bearer_token_post = method(endpoints.authenticate_bearer_token_post)


class AuthenticationClient(_AuthenticationMethods, NemoClient):
    """Sync client for direct bearer-token authentication checks."""


class AsyncAuthenticationClient(_AuthenticationMethods, AsyncNemoClient):
    """Async client for direct bearer-token authentication checks."""

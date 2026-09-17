# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod

from nemo_platform_plugin.auth.types import AuthenticateResponse
from nemo_platform_plugin.client.endpoint import get, post


@get("/apis/auth/authenticate")
@abstractmethod
def authenticate_bearer_token_get() -> AuthenticateResponse: ...


@post("/apis/auth/authenticate")
@abstractmethod
def authenticate_bearer_token_post() -> AuthenticateResponse: ...

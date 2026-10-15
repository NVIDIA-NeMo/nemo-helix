# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging in to a registry with a username and password, the way its `/v2/` challenge asks."""

from __future__ import annotations

import base64
from urllib.parse import urlsplit

import httpx
from nemo_helix_plugin.container_registry import parse_challenges

__all__ = ["RegistryAuthError", "basic", "login", "parse_challenges"]


class RegistryAuthError(Exception):
    """The registry or its token service failed, or refused the credential."""


def basic(username: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()


def login(
    registry: str, username: str, password: str, repository: str, *, http: httpx.Client, plain_http: bool = False
) -> str:
    """The `Authorization` header for pushing to ``repository`` on ``registry``.

    A Bearer challenge gets a token from the registry's token service, for ``repository``, pull and push, in
    exchange for the credential (the Distribution token specification). Any other gets the credential as Basic.
    """
    scheme = "http" if plain_http else "https"
    try:
        response = http.get(f"{scheme}://{registry}/v2/")
    except httpx.HTTPError as exc:
        raise RegistryAuthError(f"{registry} could not be reached: {type(exc).__name__}") from exc
    if response.status_code not in (200, 401):
        raise RegistryAuthError(f"{registry} answered /v2/ with {response.status_code}")
    bearer = dict(parse_challenges(response.headers.get("WWW-Authenticate", ""))).get("bearer")
    if response.status_code == 200 or bearer is None or not bearer.get("realm"):
        return basic(username, password)

    realm = bearer["realm"]
    # The credential goes to the realm the registry names: never in clear unless the registry is reached in clear.
    if scheme == "https" and urlsplit(realm).scheme != "https":
        raise RegistryAuthError(f"{registry} named a token realm that is not HTTPS")
    try:
        issued = http.get(
            realm,
            params={"service": bearer.get("service", registry), "scope": f"repository:{repository}:pull,push"},
            auth=(username, password),
        )
    except httpx.HTTPError as exc:
        raise RegistryAuthError(f"{registry}'s token service could not be reached: {type(exc).__name__}") from exc
    if issued.status_code != 200:
        # The body is not echoed: a token service's error page is not ours to repeat.
        raise RegistryAuthError(f"{registry}'s token service refused the credential: {issued.status_code}")
    try:
        body = issued.json()
    except ValueError as exc:
        raise RegistryAuthError(f"{registry}'s token service answered with something that is not JSON") from exc
    token = (body.get("token") or body.get("access_token")) if isinstance(body, dict) else None
    if not isinstance(token, str) or not token:
        raise RegistryAuthError(f"{registry}'s token service returned no token")
    return f"Bearer {token}"

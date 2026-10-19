# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import httpx
from nemo_helix_ext.client.tls import HttpxTLSConfig, httpx_tls_config_from_env


def token_request_body(grant: dict[str, str]) -> dict[str, str]:
    grant_type = grant["grant_type"]
    body = {
        "grant_type": grant_type,
        "client_id": grant["client_id"],
    }
    if "client_secret" in grant and grant.get("client_auth_method") != "client_secret_basic":
        body["client_secret"] = grant["client_secret"]
    if grant_type == "password":
        body["username"] = grant["username"]
        body["password"] = grant["password"]
        if "scope" in grant:
            body["scope"] = grant["scope"]
        return body
    if grant_type == "client_credentials":
        if "scope" in grant:
            body["scope"] = grant["scope"]
        return body
    raise ValueError(f"unsupported grant_type for auth_idp token exchange: {grant_type}")


def token_request_auth(grant: dict[str, str]) -> tuple[str, str] | None:
    if grant.get("client_auth_method") != "client_secret_basic":
        return None
    client_secret = grant.get("client_secret")
    if not client_secret:
        raise AssertionError("client_secret_basic token acquisition requires client_secret")
    return grant["client_id"], client_secret


def exchange_token_with_retries(
    token_endpoint: str,
    grant: dict[str, str],
    timeout: float = 60.0,
    tls_config: HttpxTLSConfig | None = None,
) -> str:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    request_tls_config: HttpxTLSConfig = httpx_tls_config_from_env() if tls_config is None else tls_config
    while time.monotonic() < deadline:
        try:
            response = httpx.post(
                token_endpoint,
                data=token_request_body(grant),
                auth=token_request_auth(grant),
                timeout=30.0,
                **request_tls_config,
            )
            if response.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    f"token endpoint not ready: {response.status_code}",
                    request=response.request,
                    response=response,
                )
                time.sleep(2)
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise httpx.HTTPStatusError(
                    f"{exc}; response body: {response.text}",
                    request=response.request,
                    response=response,
                ) from exc
            return response.json()["access_token"]
        except httpx.RequestError as exc:
            last_error = exc
            time.sleep(2)
    if last_error is not None:
        raise last_error
    raise TimeoutError(f"token endpoint did not become ready: {token_endpoint}")

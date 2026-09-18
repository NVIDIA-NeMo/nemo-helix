# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from json import JSONDecodeError
from urllib.parse import urlencode, urljoin, urlparse, urlunparse

import httpx
from nemo_platform_ext.client.tls import HttpxTLSConfig

from tests.auth_idp.runtime_contract import JsonObject

DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
AUTHENTIK_DEFAULT_AUTHENTICATION_FLOW_SLUG = "default-authentication-flow"
DEVICE_TOKEN_POLL_ATTEMPTS = 3
DEVICE_TOKEN_POLL_INTERVAL_SECONDS = 1.0
ZITADEL_API_TIMEOUT_SECONDS = 30.0


def url_origin(url: str) -> str:
    parsed = urlparse(url)
    assert parsed.scheme
    assert parsed.netloc
    return f"{parsed.scheme}://{parsed.netloc}"


def with_url_origin(url: str, origin: str) -> str:
    parsed_url = urlparse(url)
    parsed_origin = urlparse(origin)
    assert parsed_url.scheme
    assert parsed_url.netloc
    assert parsed_origin.scheme
    assert parsed_origin.netloc
    return urlunparse(parsed_url._replace(scheme=parsed_origin.scheme, netloc=parsed_origin.netloc))


def authentik_flow_executor_url(gateway_base_url: str, location: str) -> str | None:
    flow_url = urljoin(gateway_base_url, location)
    parsed = urlparse(flow_url)
    path_parts = [part for part in parsed.path.split("/") if part]
    query = f"?{urlencode({'query': parsed.query})}" if parsed.query else ""

    if path_parts[:4] == ["api", "v3", "flows", "executor"] and len(path_parts) >= 5:
        return flow_url
    if path_parts[:2] == ["if", "flow"] and len(path_parts) >= 3:
        return f"{gateway_base_url}/api/v3/flows/executor/{path_parts[2]}/{query}"
    if path_parts == ["flows", "-", "default", "authentication"]:
        return f"{gateway_base_url}/api/v3/flows/executor/{AUTHENTIK_DEFAULT_AUTHENTICATION_FLOW_SLUG}/{query}"

    return None


def next_authentik_challenge(
    client: httpx.Client,
    *,
    gateway_base_url: str,
    response: httpx.Response,
) -> tuple[JsonObject, str]:
    while response.status_code in {301, 302, 303, 307, 308}:
        location = response.headers.get("location")
        assert location
        flow_executor_url = authentik_flow_executor_url(gateway_base_url, location)
        response = client.get(flow_executor_url or urljoin(gateway_base_url, location), timeout=30.0)

    response.raise_for_status()
    try:
        challenge = response.json()
    except JSONDecodeError as exc:
        body = response.text[:500].replace("\n", " ")
        raise AssertionError(
            "Expected Authentik flow executor JSON challenge, got "
            f"status={response.status_code} url={response.url} "
            f"content_type={response.headers.get('content-type')!r} body={body!r}"
        ) from exc
    assert isinstance(challenge, dict)
    return challenge, str(response.url)


def solve_authentik_device_flow(
    *,
    gateway_base_url: str,
    verification_uri_complete: str,
    user_code: str,
    username: str,
    password: str,
    tls_config: HttpxTLSConfig,
) -> None:
    with httpx.Client(follow_redirects=False, **tls_config) as client:
        response = client.get(verification_uri_complete, timeout=30.0)
        challenge, flow_url = next_authentik_challenge(
            client,
            gateway_base_url=gateway_base_url,
            response=response,
        )

        for _ in range(10):
            component = challenge.get("component")
            if component == "xak-flow-redirect":
                redirect_to = challenge.get("to")
                assert isinstance(redirect_to, str)
                flow_executor_url = authentik_flow_executor_url(gateway_base_url, redirect_to)
                response = client.get(flow_executor_url or urljoin(gateway_base_url, redirect_to), timeout=30.0)
                challenge, flow_url = next_authentik_challenge(
                    client,
                    gateway_base_url=gateway_base_url,
                    response=response,
                )
                continue
            if component == "ak-stage-access-denied":
                raise AssertionError(f"Authentik device flow was denied: {challenge}")

            if component == "ak-stage-identification":
                payload = {"component": component, "uid_field": username}
                if challenge.get("password_fields"):
                    payload["password"] = password
            elif component == "ak-stage-password":
                payload = {"component": component, "password": password}
            elif component == "ak-stage-user-login":
                payload = {"component": component}
            elif component == "ak-provider-oauth2-device-code":
                payload = {"component": component, "code": user_code}
            elif component == "ak-provider-oauth2-device-code-finish":
                payload = {"component": component}
            else:
                raise AssertionError(f"Unexpected Authentik device flow component {component!r}: {challenge}")

            response = client.post(flow_url, json=payload, timeout=30.0)
            challenge, flow_url = next_authentik_challenge(
                client,
                gateway_base_url=gateway_base_url,
                response=response,
            )
            if component == "ak-provider-oauth2-device-code-finish":
                return

    raise AssertionError(f"Authentik device flow did not complete after 10 stages: {challenge}")


def _zitadel_api_headers(admin_token: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {admin_token}",
        "Content-Type": "application/json",
    }


def solve_zitadel_device_flow(
    *,
    gateway_base_url: str,
    user_code: str,
    login_name: str,
    password: str,
    admin_token: str,
    tls_config: HttpxTLSConfig,
) -> None:
    headers = _zitadel_api_headers(admin_token)
    session_response = httpx.post(
        f"{gateway_base_url.rstrip('/')}/v2/sessions",
        json={
            "checks": {
                "user": {"loginName": login_name},
                "password": {"password": password},
            },
        },
        headers=headers,
        timeout=ZITADEL_API_TIMEOUT_SECONDS,
        **tls_config,
    )
    session_response.raise_for_status()
    session = session_response.json()
    session_id = session["sessionId"]
    session_token = session["sessionToken"]

    device_request_response = httpx.get(
        f"{gateway_base_url.rstrip('/')}/v2/oidc/device_authorization/{user_code}",
        headers=headers,
        timeout=ZITADEL_API_TIMEOUT_SECONDS,
        **tls_config,
    )
    device_request_response.raise_for_status()
    device_request = device_request_response.json()["deviceAuthorizationRequest"]

    authorization_response = httpx.post(
        f"{gateway_base_url.rstrip('/')}/v2/oidc/device_authorization/{device_request['id']}",
        json={
            "session": {
                "sessionId": session_id,
                "sessionToken": session_token,
            },
        },
        headers=headers,
        timeout=ZITADEL_API_TIMEOUT_SECONDS,
        **tls_config,
    )
    authorization_response.raise_for_status()


def poll_device_token(
    *,
    token_endpoint: str,
    client_id: str,
    device_code: str,
    scope: str,
    tls_config: HttpxTLSConfig,
) -> JsonObject:
    last_response: httpx.Response | None = None
    for _ in range(DEVICE_TOKEN_POLL_ATTEMPTS):
        response = httpx.post(
            token_endpoint,
            data={
                "grant_type": DEVICE_CODE_GRANT_TYPE,
                "client_id": client_id,
                "device_code": device_code,
                "scope": scope,
            },
            timeout=30.0,
            **tls_config,
        )
        if response.status_code == 200:
            token_response = response.json()
            assert isinstance(token_response, dict)
            return token_response

        last_response = response
        error = response.json().get("error")
        if error != "authorization_pending":
            response.raise_for_status()

        time.sleep(DEVICE_TOKEN_POLL_INTERVAL_SECONDS)

    raise AssertionError(
        "Device token endpoint did not return tokens after browser-side authorization completed: "
        f"{last_response.text if last_response is not None else 'no response'}"
    )


def authenticate_authentik_device_flow(
    *,
    gateway_base_url: str,
    device_authorization_endpoint: str,
    token_endpoint: str,
    client_id: str,
    scope: str,
    username: str,
    password: str,
    tls_config: HttpxTLSConfig,
) -> JsonObject:
    device_response = httpx.post(
        device_authorization_endpoint,
        data={
            "client_id": client_id,
            "scope": scope,
        },
        timeout=30.0,
        **tls_config,
    )
    device_response.raise_for_status()
    device_body = device_response.json()

    solve_authentik_device_flow(
        gateway_base_url=gateway_base_url,
        verification_uri_complete=with_url_origin(device_body["verification_uri_complete"], gateway_base_url),
        user_code=device_body["user_code"],
        username=username,
        password=password,
        tls_config=tls_config,
    )

    return poll_device_token(
        token_endpoint=token_endpoint,
        client_id=client_id,
        device_code=device_body["device_code"],
        scope=scope,
        tls_config=tls_config,
    )


def authenticate_zitadel_device_flow(
    *,
    gateway_base_url: str,
    device_authorization_endpoint: str,
    token_endpoint: str,
    client_id: str,
    scope: str,
    login_name: str,
    password: str,
    admin_token: str,
    tls_config: HttpxTLSConfig,
) -> JsonObject:
    device_response = httpx.post(
        device_authorization_endpoint,
        data={
            "client_id": client_id,
            "scope": scope,
        },
        timeout=30.0,
        **tls_config,
    )
    device_response.raise_for_status()
    device_body = device_response.json()

    solve_zitadel_device_flow(
        gateway_base_url=gateway_base_url,
        user_code=device_body["user_code"],
        login_name=login_name,
        password=password,
        admin_token=admin_token,
        tls_config=tls_config,
    )

    return poll_device_token(
        token_endpoint=token_endpoint,
        client_id=client_id,
        device_code=device_body["device_code"],
        scope=scope,
        tls_config=tls_config,
    )

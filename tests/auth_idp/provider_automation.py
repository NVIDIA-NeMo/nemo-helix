# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from json import JSONDecodeError
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import httpx

from tests.auth_idp.runtime_contract import JsonObject

AUTHENTIK_DEFAULT_AUTHENTICATION_FLOW_SLUG = "default-authentication-flow"
ZITADEL_API_TIMEOUT_SECONDS = 30.0
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_NEMO_CALLBACK_PATH = "/apis/auth/v2/login/callback"


class IdpUserAutomation(Protocol):
    """Automate only the provider-owned user login and approval interaction."""

    def approve_device_authorization(
        self,
        client: httpx.Client,
        *,
        gateway_base_url: str,
        verification_uri_complete: str,
        user_code: str,
        username: str,
        password: str,
    ) -> None:
        raise NotImplementedError

    def complete_authorization(
        self,
        client: httpx.Client,
        *,
        gateway_base_url: str,
        authorization_url: str,
        username: str,
        password: str,
    ) -> str:
        raise NotImplementedError


class AuthentikUserAutomation:
    def approve_device_authorization(
        self,
        client: httpx.Client,
        *,
        gateway_base_url: str,
        verification_uri_complete: str,
        user_code: str,
        username: str,
        password: str,
    ) -> None:
        response = client.get(verification_uri_complete, timeout=30.0)
        challenge, flow_url = _next_authentik_challenge(
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
                challenge, flow_url = _next_authentik_challenge(
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

            response = client.post(
                flow_url,
                json=payload,
                headers=_authentik_submission_headers(client, response),
                timeout=30.0,
            )
            challenge, flow_url = _next_authentik_challenge(
                client,
                gateway_base_url=gateway_base_url,
                response=response,
            )
            if component == "ak-provider-oauth2-device-code-finish":
                return

        raise AssertionError(f"Authentik device flow did not complete after 10 stages: {challenge}")

    def complete_authorization(
        self,
        client: httpx.Client,
        *,
        gateway_base_url: str,
        authorization_url: str,
        username: str,
        password: str,
    ) -> str:
        response = client.get(authorization_url, timeout=30.0)
        for _ in range(20):
            callback_url, response = _follow_authentik_redirects(client, response, gateway_base_url)
            if callback_url is not None:
                return callback_url

            challenge = _authentik_challenge(response)
            component = challenge.get("component")
            if component == "ak-stage-access-denied":
                raise AssertionError(f"Authentik authorization-code flow was denied: {challenge}")
            if component == "xak-flow-redirect":
                redirect_to = challenge.get("to")
                if not isinstance(redirect_to, str):
                    raise AssertionError(f"Authentik flow redirect omitted its target: {challenge}")
                redirect_url = urljoin(str(response.url), redirect_to)
                if urlparse(redirect_url).path == _NEMO_CALLBACK_PATH:
                    return redirect_url
                response = client.get(_authentik_runtime_url(gateway_base_url, redirect_url), timeout=30.0)
                continue
            if component == "ak-stage-identification":
                payload = {"component": component, "uid_field": username}
                if challenge.get("password_fields"):
                    payload["password"] = password
            elif component == "ak-stage-password":
                payload = {"component": component, "password": password}
            elif component == "ak-stage-user-login":
                payload = {"component": component}
            elif component == "ak-stage-consent":
                token = challenge.get("token")
                if not isinstance(token, str):
                    raise AssertionError(f"Authentik consent challenge omitted its token: {challenge}")
                payload = {"component": component, "token": token}
            else:
                raise AssertionError(f"Unexpected Authentik authorization-code component {component!r}: {challenge}")

            response = client.post(
                str(response.url),
                json=payload,
                headers=_authentik_submission_headers(client, response),
                timeout=30.0,
            )

        raise AssertionError("Authentik authorization-code flow did not reach the NeMo callback after 20 stages")


@dataclass(frozen=True)
class _ZitadelUserSession:
    session_id: str
    session_token: str


@dataclass(frozen=True)
class ZitadelUserAutomation:
    admin_token: str

    def approve_device_authorization(
        self,
        client: httpx.Client,
        *,
        gateway_base_url: str,
        verification_uri_complete: str,
        user_code: str,
        username: str,
        password: str,
    ) -> None:
        del verification_uri_complete
        headers = self._api_headers()
        session = self._create_password_session(
            client,
            gateway_base_url=gateway_base_url,
            login_name=username,
            password=password,
        )
        device_request_response = client.get(
            f"{gateway_base_url.rstrip('/')}/v2/oidc/device_authorization/{user_code}",
            headers=headers,
            timeout=ZITADEL_API_TIMEOUT_SECONDS,
        )
        device_request_response.raise_for_status()
        device_request = device_request_response.json()["deviceAuthorizationRequest"]

        authorization_response = client.post(
            f"{gateway_base_url.rstrip('/')}/v2/oidc/device_authorization/{device_request['id']}",
            json={
                "session": {
                    "sessionId": session.session_id,
                    "sessionToken": session.session_token,
                },
            },
            headers=headers,
            timeout=ZITADEL_API_TIMEOUT_SECONDS,
        )
        authorization_response.raise_for_status()

    def complete_authorization(
        self,
        client: httpx.Client,
        *,
        gateway_base_url: str,
        authorization_url: str,
        username: str,
        password: str,
    ) -> str:
        response = client.get(authorization_url, timeout=30.0)
        auth_request_id = _zitadel_auth_request_id(client, response, gateway_base_url)
        session = self._create_password_session(
            client,
            gateway_base_url=gateway_base_url,
            login_name=username,
            password=password,
        )
        callback_response = client.post(
            f"{gateway_base_url.rstrip('/')}/v2/oidc/auth_requests/{auth_request_id}",
            json={
                "session": {
                    "sessionId": session.session_id,
                    "sessionToken": session.session_token,
                }
            },
            headers=self._api_headers(),
            timeout=ZITADEL_API_TIMEOUT_SECONDS,
        )
        callback_response.raise_for_status()
        callback_url = callback_response.json().get("callbackUrl")
        if not isinstance(callback_url, str):
            raise AssertionError(f"Zitadel auth request did not return a callback URL: {callback_response.text}")
        return callback_url

    def _api_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.admin_token}",
            "Content-Type": "application/json",
        }

    def _create_password_session(
        self,
        client: httpx.Client,
        *,
        gateway_base_url: str,
        login_name: str,
        password: str,
    ) -> _ZitadelUserSession:
        response = client.post(
            f"{gateway_base_url.rstrip('/')}/v2/sessions",
            json={
                "checks": {
                    "user": {"loginName": login_name},
                    "password": {"password": password},
                },
            },
            headers=self._api_headers(),
            timeout=ZITADEL_API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        return _ZitadelUserSession(session_id=body["sessionId"], session_token=body["sessionToken"])


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


def _authentik_challenge(response: httpx.Response) -> JsonObject:
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
    if not isinstance(challenge, dict):
        raise AssertionError(f"Expected Authentik challenge object, got {challenge!r}")
    return challenge


def _next_authentik_challenge(
    client: httpx.Client,
    *,
    gateway_base_url: str,
    response: httpx.Response,
) -> tuple[JsonObject, str]:
    while response.status_code in _REDIRECT_STATUS_CODES:
        location = response.headers.get("location")
        assert location
        flow_executor_url = authentik_flow_executor_url(gateway_base_url, location)
        response = client.get(flow_executor_url or urljoin(gateway_base_url, location), timeout=30.0)
    return _authentik_challenge(response), str(response.url)


def _follow_authentik_redirects(
    client: httpx.Client, response: httpx.Response, gateway_base_url: str
) -> tuple[str | None, httpx.Response]:
    while response.status_code in _REDIRECT_STATUS_CODES:
        location = response.headers.get("location")
        if not location:
            raise AssertionError("Authentik redirect omitted its Location header")
        redirect_url = urljoin(str(response.url), location)
        if urlparse(redirect_url).path == _NEMO_CALLBACK_PATH:
            return redirect_url, response
        response = client.get(_authentik_runtime_url(gateway_base_url, redirect_url), timeout=30.0)
    return None, response


def _authentik_runtime_url(gateway_base_url: str, location: str) -> str:
    flow_executor_url = authentik_flow_executor_url(gateway_base_url, location)
    return flow_executor_url or urljoin(gateway_base_url, location)


def _authentik_submission_headers(client: httpx.Client, response: httpx.Response) -> dict[str, str]:
    headers = {"Referer": str(response.url)}
    csrf_token = client.cookies.get("authentik_csrf")
    if csrf_token:
        headers["X-Authentik-CSRF"] = csrf_token
    return headers


def _zitadel_auth_request_id(client: httpx.Client, response: httpx.Response, gateway_base_url: str) -> str:
    for _ in range(10):
        auth_request_id = parse_qs(urlparse(str(response.url)).query).get("authRequest")
        if auth_request_id:
            return auth_request_id[0]
        if response.status_code not in _REDIRECT_STATUS_CODES:
            response.raise_for_status()
            raise AssertionError(f"Zitadel authorize response omitted authRequest: {response.url}")
        location = response.headers.get("location")
        if not location:
            raise AssertionError("Zitadel redirect omitted its Location header")
        redirect_url = urljoin(str(response.url), location)
        parsed_request_id = parse_qs(urlparse(redirect_url).query).get("authRequest")
        if parsed_request_id:
            return parsed_request_id[0]
        response = client.get(redirect_url, timeout=30.0)
    raise AssertionError("Zitadel authorization flow did not advertise an authRequest after 10 redirects")

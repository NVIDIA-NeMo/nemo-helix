# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token login through the NeMo server-side OIDC broker."""

from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import httpx

from nemo_helix_ext.auth.helpers import AuthError
from nemo_helix_ext.client.tls import httpx_tls_config_from_env


def login_with_oidc_broker(
    *,
    authorization_start_endpoint: str,
    broker_token_endpoint: str,
    open_browser: bool,
    certificate_authority: str | None,
) -> dict[str, object]:
    """Open the brokered browser login and exchange the one-time code."""
    callback_params: dict[str, str] = {}
    code_verifier = secrets.token_urlsafe(48)
    state = secrets.token_urlsafe(32)
    server = HTTPServer(("127.0.0.1", 0), _callback_handler(callback_params))
    port = server.server_address[1]
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        try:
            response = httpx.post(
                authorization_start_endpoint,
                json={
                    "redirect_uri": redirect_uri,
                    "code_challenge": _pkce_challenge(code_verifier),
                    "state": state,
                },
                timeout=30.0,
                **httpx_tls_config_from_env(certificate_authority),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AuthError(f"Brokered login failed: {exc}") from exc
        authorization_url = _authorization_url_from_response(response)
        if open_browser:
            webbrowser.open(authorization_url)
        else:
            print(f"Open this URL to log in:\n{authorization_url}")
        thread.join(timeout=300)
    finally:
        server.server_close()
    code = _validated_callback_code(callback_params, expected_state=state)

    try:
        token_response = httpx.post(
            broker_token_endpoint,
            json={"grant_type": "authorization_code", "code": code, "code_verifier": code_verifier},
            timeout=30.0,
            **httpx_tls_config_from_env(certificate_authority),
        )
        token_response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AuthError(f"Brokered login failed: {exc}") from exc
    payload = _json_object_from_response(token_response, "token")
    if not isinstance(payload.get("access_token"), str):
        raise AuthError("Brokered login did not return an access token.")
    return payload


def _validated_callback_code(callback_params: dict[str, str], *, expected_state: str) -> str:
    callback_state = callback_params.get("state")
    if callback_state is None:
        raise AuthError("Brokered login callback did not include state.")
    if not secrets.compare_digest(callback_state, expected_state):
        raise AuthError("Brokered login callback state did not match.")
    error = callback_params.get("error")
    if error:
        description = callback_params.get("error_description")
        detail = f": {description}" if description else ""
        raise AuthError(f"Brokered login failed: {error}{detail}")
    code = callback_params.get("code")
    if not code:
        raise AuthError("Brokered login did not receive a callback code.")
    return code


def _json_object_from_response(response: httpx.Response, label: str) -> dict[str, object]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise AuthError(f"Brokered login {label} response was invalid.") from exc
    if not isinstance(payload, dict):
        raise AuthError(f"Brokered login {label} response was invalid.")
    result: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise AuthError(f"Brokered login {label} response was invalid.")
        result[key] = value
    return result


def _authorization_url_from_response(response: httpx.Response) -> str:
    payload = _json_object_from_response(response, "authorization")
    authorization_url = payload.get("authorization_url")
    if not isinstance(authorization_url, str):
        raise AuthError("Brokered login did not return an authorization URL.")
    return authorization_url


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _callback_handler(sink: dict[str, str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            query = parse_qs(urlparse(self.path).query)
            code = query.get("code", [""])[0]
            state = query.get("state", [""])[0]
            error = query.get("error", [""])[0]
            error_description = query.get("error_description", [""])[0]
            if code:
                sink["code"] = code
            if state:
                sink["state"] = state
            if error:
                sink["error"] = error
            if error_description:
                sink["error_description"] = error_description
            body = b"Login complete. You can close this window."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler

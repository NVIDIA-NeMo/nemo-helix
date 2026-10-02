# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI login through the NeMo confidential OIDC broker."""

from __future__ import annotations

import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import httpx

from nemo_helix_ext.auth.helpers import AuthError
from nemo_helix_ext.client.tls import httpx_tls_config_from_env


def login_with_confidential_broker(
    *,
    cli_login_url: str,
    open_browser: bool,
    certificate_authority: str | None,
) -> dict[str, str | int]:
    """Open the brokered browser login and exchange the one-time code."""
    callback_code: dict[str, str] = {}
    server = HTTPServer(("127.0.0.1", 0), _callback_handler(callback_code))
    port = server.server_address[1]
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        response = httpx.post(
            cli_login_url,
            json={"redirect_uri": redirect_uri},
            timeout=30.0,
            **httpx_tls_config_from_env(certificate_authority),
        )
        response.raise_for_status()
        login_url = response.json()["login_url"]
        if not isinstance(login_url, str):
            raise AuthError("Confidential login did not return a login URL.")
        if open_browser:
            webbrowser.open(login_url)
        else:
            print(f"Open this URL to log in:\n{login_url}")
        thread.join(timeout=300)
    except httpx.HTTPError as exc:
        raise AuthError(f"Confidential login failed: {exc}") from exc
    finally:
        server.server_close()
    code = callback_code.get("code")
    if not code:
        raise AuthError("Confidential login did not receive a callback code.")
    token_url = cli_login_url.removesuffix("/login") + "/token"
    token_response = httpx.post(
        token_url,
        json={"grant_type": "authorization_code", "code": code},
        timeout=30.0,
        **httpx_tls_config_from_env(certificate_authority),
    )
    try:
        token_response.raise_for_status()
    except httpx.HTTPError as exc:
        raise AuthError(f"Confidential login failed: {exc}") from exc
    payload = token_response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
        raise AuthError("Confidential login did not return an access token.")
    return payload


def _callback_handler(sink: dict[str, str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            query = parse_qs(urlparse(self.path).query)
            code = query.get("code", [""])[0]
            if code:
                sink["code"] = code
            body = b"Login complete. You can close this window."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler

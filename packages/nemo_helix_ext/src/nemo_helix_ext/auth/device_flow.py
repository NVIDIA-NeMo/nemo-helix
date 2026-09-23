# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OAuth 2.0 Device Authorization Flow (RFC 8628) implementation."""

import asyncio
import time
import webbrowser
from dataclasses import dataclass

import httpx
from rich.console import Console
from rich.panel import Panel

from nemo_helix_ext.auth.token_provider import refresh_token_grant
from nemo_helix_ext.client.tls import httpx_tls_config_from_env

console = Console()


async def _async_pause(seconds: float) -> None:
    await asyncio.sleep(seconds)


@dataclass
class DeviceCodeResponse:
    """Response from device authorization endpoint."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None
    expires_in: int
    interval: int


@dataclass
class TokenResponse:
    """Response from token endpoint."""

    access_token: str
    id_token: str | None  # ID token (JWT)
    refresh_token: str | None
    token_type: str
    expires_in: int
    scope: str | None

    @property
    def token_for_nhx(self) -> str:
        """Return the token to use for NeMo Helix authentication."""
        return self.access_token


class DeviceFlowError(Exception):
    """Device flow authentication error."""

    pass


class DeviceFlow:
    """OAuth 2.0 Device Authorization Flow client."""

    def __init__(
        self,
        device_authorization_endpoint: str,
        token_endpoint: str,
        client_id: str,
        scope: str = "openid email profile",
        certificate_authority: str | None = None,
    ):
        self.device_authorization_endpoint = device_authorization_endpoint
        self.token_endpoint = token_endpoint
        self.client_id = client_id
        self.scope = scope
        self.certificate_authority = certificate_authority

    async def start_device_authorization(self) -> DeviceCodeResponse:
        """Start the device authorization flow."""
        async with httpx.AsyncClient(**httpx_tls_config_from_env(self.certificate_authority)) as client:
            response = await client.post(
                self.device_authorization_endpoint,
                data={
                    "client_id": self.client_id,
                    "scope": self.scope,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            return DeviceCodeResponse(
                device_code=data["device_code"],
                user_code=data["user_code"],
                verification_uri=data["verification_uri"],
                verification_uri_complete=data.get("verification_uri_complete"),
                expires_in=data["expires_in"],
                interval=data.get("interval", 5),
            )

    async def poll_for_token(
        self,
        device_code: str,
        interval: int,
        expires_in: int,
    ) -> TokenResponse:
        """Poll the token endpoint until authorization is complete."""
        start_time = time.time()

        async with httpx.AsyncClient(**httpx_tls_config_from_env(self.certificate_authority)) as client:
            while time.time() - start_time < expires_in:
                await _async_pause(interval)

                response = await client.post(
                    self.token_endpoint,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "client_id": self.client_id,
                        "device_code": device_code,
                        "scope": self.scope,  # Casdoor doesn't propagate scope from DeviceAuthCache
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    return TokenResponse(
                        access_token=data["access_token"],
                        id_token=data.get("id_token"),  # Capture ID token if present
                        refresh_token=data.get("refresh_token"),
                        token_type=data.get("token_type", "Bearer"),
                        expires_in=data.get("expires_in", 3600),
                        scope=data.get("scope"),
                    )

                error_data = response.json()
                error = error_data.get("error")

                if error == "authorization_pending":
                    continue
                elif error == "slow_down":
                    interval += 5
                    continue
                elif error == "expired_token":
                    raise DeviceFlowError("Authorization request expired")
                elif error == "access_denied":
                    raise DeviceFlowError("User denied authorization")
                else:
                    raise DeviceFlowError(f"Token request failed: {error}")

        raise DeviceFlowError("Authorization timed out")


async def authenticate_with_device_flow(
    device_authorization_endpoint: str,
    token_endpoint: str,
    client_id: str,
    scope: str = "openid email profile",
    open_browser: bool = True,
    certificate_authority: str | None = None,
) -> TokenResponse:
    """Perform OAuth device flow authentication.

    Args:
        device_authorization_endpoint: URL for device authorization
        token_endpoint: URL for token exchange
        client_id: OAuth client ID
        scope: OAuth scopes to request
        open_browser: Whether to automatically open the browser

    Returns:
        TokenResponse with access and refresh tokens
    """
    flow = DeviceFlow(
        device_authorization_endpoint=device_authorization_endpoint,
        token_endpoint=token_endpoint,
        client_id=client_id,
        scope=scope,
        certificate_authority=certificate_authority,
    )

    # Start device authorization
    device_response = await flow.start_device_authorization()

    # Display user code and instructions
    console.print()
    console.print(
        Panel(
            f"[bold cyan]Visit:[/] {device_response.verification_uri}\n"
            f"[bold cyan]Enter code:[/] [bold yellow]{device_response.user_code}[/]",
            title="Authorization Required",
            border_style="cyan",
        )
    )

    # Optionally open browser
    if open_browser and device_response.verification_uri_complete:
        console.print("\n[dim]Opening browser...[/]")
        webbrowser.open(device_response.verification_uri_complete)
    elif open_browser:
        console.print("\n[dim]Opening browser...[/]")
        webbrowser.open(device_response.verification_uri)

    console.print("\n[dim]Waiting for authorization...[/]")

    # Poll for token
    token_response = await flow.poll_for_token(
        device_code=device_response.device_code,
        interval=device_response.interval,
        expires_in=device_response.expires_in,
    )

    console.print("[green]Authorization successful![/]")

    return token_response


async def refresh_access_token(
    token_endpoint: str,
    client_id: str,
    refresh_token: str,
    scope: str | None = None,
    certificate_authority: str | None = None,
) -> TokenResponse:
    """
    Refresh an access token using a refresh token.

    Args:
        token_endpoint: The OAuth token endpoint URL.
        client_id: The OAuth client ID.
        refresh_token: The refresh token from a previous authentication.
        scope: OAuth scopes to request (required for some IdPs like Azure AD).

    Returns:
        TokenResponse with new access_token (and possibly new refresh_token).

    Raises:
        DeviceFlowError: If token refresh fails.
    """
    try:
        data = await asyncio.to_thread(
            refresh_token_grant,
            token_endpoint,
            client_id,
            refresh_token,
            scope=scope,
            certificate_authority=certificate_authority,
        )
    except RuntimeError as e:
        raise DeviceFlowError(str(e)) from e

    return TokenResponse(
        access_token=data["access_token"],
        id_token=data.get("id_token"),
        refresh_token=data.get("refresh_token"),  # May be rotated
        token_type=data.get("token_type", "Bearer"),
        expires_in=data.get("expires_in", 3600),
        scope=data.get("scope"),
    )


def authenticate_with_password_grant(
    token_endpoint: str,
    client_id: str,
    username: str,
    password: str,
    scope: str = "openid profile email",
    certificate_authority: str | None = None,
) -> TokenResponse:
    """Obtain tokens using the Resource Owner Password Credentials grant (RFC 6749).

    Use this for non-interactive environments (e.g. CI) where no browser is available.
    The IdP must have the password grant enabled for the application.

    Args:
        token_endpoint: The OAuth token endpoint URL.
        client_id: The OAuth client ID.
        username: Resource owner username (e.g. testuser or built-in/admin).
        password: Resource owner password.
        scope: OAuth scopes to request.

    Returns:
        TokenResponse with access_token and optional refresh_token.

    Raises:
        DeviceFlowError: If the token request fails.
    """
    data = {
        "grant_type": "password",
        "client_id": client_id,
        "username": username,
        "password": password,
        "scope": scope,
    }
    with httpx.Client(**httpx_tls_config_from_env(certificate_authority)) as client:
        response = client.post(token_endpoint, data=data, timeout=30.0)

    if response.status_code != 200:
        error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        error = error_data.get("error", "unknown_error")
        error_description = error_data.get("error_description", response.text)
        raise DeviceFlowError(f"Token request failed: {error} - {error_description}")

    resp_data = response.json()
    return TokenResponse(
        access_token=resp_data["access_token"],
        id_token=resp_data.get("id_token"),
        refresh_token=resp_data.get("refresh_token"),
        token_type=resp_data.get("token_type", "Bearer"),
        expires_in=resp_data.get("expires_in", 3600),
        scope=resp_data.get("scope"),
    )

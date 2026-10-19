# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OIDC login flow selection for the NeMo Helix CLI."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Literal

from nemo_helix_plugin.client.oidc import AdvertisedOidcClient, OidcClientName

from nemo_helix_ext.auth import device_flow
from nemo_helix_ext.auth.confidential_login import login_with_oidc_broker
from nemo_helix_ext.auth.helpers import (
    AuthError,
    BearerTokenSource,
    NHXOIDCConfig,
    build_effective_scope,
    decode_jwt_claims,
    normalize_scope_prefix,
    select_advertised_client,
    validate_requested_scopes_granted,
)
from nemo_helix_ext.auth.token_provider import TokenSet


@dataclass(frozen=True)
class RequestedScope:
    """Scope requested during public-client login, plus its provider-qualified form."""

    name: str
    effective_name: str | None = None


@dataclass(frozen=True)
class OIDCLoginResult:
    """Tokens and display metadata produced by an OIDC login flow."""

    access_token: str
    refresh_token: str | None
    expires_at: float | None
    token_broker_url: str | None
    flow: Literal["public", "confidential"]
    user_email: str | None = None
    display_granted_scopes: tuple[str, ...] = ()
    requested_scopes: tuple[RequestedScope, ...] = ()


def authenticate_with_oidc(
    *,
    oidc_config: NHXOIDCConfig,
    no_browser: bool = False,
    scope: str | None = None,
    username: str | None = None,
    password: str | None = None,
    oidc_client: OidcClientName | None = None,
    certificate_authority: str | None = None,
) -> OIDCLoginResult:
    """Authenticate using the user-login client advertised by auth discovery."""
    try:
        selected_client = select_advertised_client(oidc_config, oidc_client)
    except ValueError as exc:
        raise AuthError(str(exc)) from exc

    if selected_client.client_authentication == "client_secret_basic" or selected_client.server_side_sessions:
        return _authenticate_brokered_client(
            client=selected_client,
            no_browser=no_browser,
            username=username,
            password=password,
            certificate_authority=certificate_authority,
        )

    if selected_client.client_authentication == "public":
        return _authenticate_public_client(
            client=selected_client,
            no_browser=no_browser,
            scope=scope,
            username=username,
            password=password,
            certificate_authority=certificate_authority,
        )

    raise AuthError(
        "OIDC client "
        f"'{selected_client.name}' uses unsupported client authentication method "
        f"'{selected_client.client_authentication}'."
    )


def _authenticate_brokered_client(
    *,
    client: AdvertisedOidcClient,
    no_browser: bool,
    username: str | None,
    password: str | None,
    certificate_authority: str | None,
) -> OIDCLoginResult:
    login_username, login_password = _login_credentials(username, password)
    if login_username or login_password:
        raise AuthError(
            "Password grant is not available for a server-side OIDC session. Use browser login: nemo auth login"
        )

    if client.authorization_start_endpoint is None or client.broker_token_endpoint is None:
        raise AuthError("This cluster advertised an incomplete server-side OIDC client.")

    token_payload = login_with_oidc_broker(
        authorization_start_endpoint=client.authorization_start_endpoint,
        broker_token_endpoint=client.broker_token_endpoint,
        open_browser=not no_browser,
        certificate_authority=certificate_authority,
    )

    token = token_payload["access_token"]
    if not isinstance(token, str):
        raise AuthError("Brokered login did not return an access token.")
    refresh_token = _optional_string(token_payload.get("refresh_token"))
    tokens = TokenSet.from_access_token(
        token,
        refresh_token,
        expires_in=_optional_int(token_payload.get("expires_in")),
    )
    return OIDCLoginResult(
        access_token=token,
        refresh_token=refresh_token,
        expires_at=tokens.expires_at,
        token_broker_url=client.broker_token_endpoint,
        flow=client.name,
    )


def _authenticate_public_client(
    *,
    client: AdvertisedOidcClient,
    no_browser: bool,
    scope: str | None,
    username: str | None,
    password: str | None,
    certificate_authority: str | None,
) -> OIDCLoginResult:
    token_endpoint = client.token_endpoint
    if not token_endpoint:
        raise AuthError(
            "This cluster does not have OIDC token endpoint configured.\n"
            "Use OIDC configuration for device/password login, or for local testing use:\n"
            "nemo auth login --unsigned-token --email <email>"
        )

    login_username, login_password = _login_credentials(username, password)
    use_password_grant = bool(login_username and login_password)
    required_client_id = _required_public_client_id(client.client_id, use_password_grant=use_password_grant)

    requested_scopes = _requested_scopes(client.default_scopes, scope)
    scope_prefix = normalize_scope_prefix(client.scope_prefix)
    effective_scope = build_effective_scope(requested_scopes, client.scope_prefix)
    token_response = _public_token_response(
        client=client,
        token_endpoint=token_endpoint,
        client_id=required_client_id,
        no_browser=no_browser,
        use_password_grant=use_password_grant,
        login_username=login_username,
        login_password=login_password,
        effective_scope=effective_scope,
        certificate_authority=certificate_authority,
    )

    try:
        token = token_response.token_for_nhx
    except device_flow.DeviceFlowError as exc:
        raise AuthError(f"Authentication failed: {exc}") from exc

    claims = decode_jwt_claims(token)
    user_email = _user_email_from_claims(claims)
    granted_scopes = _granted_scopes(
        token_response=token_response,
        claims=claims,
        bearer_token_source=client.bearer_token_source,
    )
    if granted_scopes.raw_was_available:
        validate_requested_scopes_granted(effective_scope, list(granted_scopes.raw), scope_prefix)

    tokens = TokenSet.from_access_token(
        token,
        token_response.refresh_token,
        expires_in=token_response.expires_in,
    )
    return OIDCLoginResult(
        access_token=token,
        refresh_token=token_response.refresh_token,
        expires_at=tokens.expires_at,
        token_broker_url=None,
        flow="public",
        user_email=user_email,
        display_granted_scopes=_display_scopes(granted_scopes.raw, scope_prefix),
        requested_scopes=_scope_display_entries(requested_scopes, scope_prefix),
    )


def _public_token_response(
    *,
    client: AdvertisedOidcClient,
    token_endpoint: str,
    client_id: str,
    no_browser: bool,
    use_password_grant: bool,
    login_username: str | None,
    login_password: str | None,
    effective_scope: str,
    certificate_authority: str | None,
) -> device_flow.TokenResponse:
    try:
        if use_password_grant:
            if login_username is None or login_password is None:
                raise AuthError("Username and password are required for password grant.")
            return device_flow.authenticate_with_password_grant(
                token_endpoint=token_endpoint,
                client_id=client_id,
                username=login_username,
                password=login_password,
                scope=effective_scope,
                bearer_token_source=client.bearer_token_source,
                certificate_authority=certificate_authority,
            )

        if client.device_authorization_endpoint is None:
            raise AuthError(
                "This cluster does not support device flow authentication.\n"
                "For non-interactive login use: nemo auth login --username <user> --password <pass>\n"
                "Or set NHX_OIDC_USERNAME and NHX_OIDC_PASSWORD (e.g. in CI)."
            )
        include_device_id = client.device_authorization_requires_device_id
        device_display_name = client.device_authorization_display_name
        include_scope_in_token_request = client.device_token_request_includes_scope
        if include_device_id and device_display_name is None:
            device_display_name = "NeMo Helix CLI"
        return asyncio.run(
            device_flow.authenticate_with_device_flow(
                device_authorization_endpoint=client.device_authorization_endpoint,
                token_endpoint=token_endpoint,
                client_id=client_id,
                scope=effective_scope,
                open_browser=not no_browser,
                bearer_token_source=client.bearer_token_source,
                include_device_id=include_device_id,
                device_display_name=device_display_name,
                include_scope_in_token_request=include_scope_in_token_request,
                certificate_authority=certificate_authority,
            )
        )
    except device_flow.DeviceFlowError as exc:
        raise AuthError(f"Authentication failed: {exc}") from exc


def _login_credentials(username: str | None, password: str | None) -> tuple[str | None, str | None]:
    return username or os.environ.get("NHX_OIDC_USERNAME"), password or os.environ.get("NHX_OIDC_PASSWORD")


def _required_public_client_id(client_id: str | None, *, use_password_grant: bool) -> str:
    if client_id:
        return client_id
    if use_password_grant:
        raise AuthError("OIDC client_id is required for password grant.")
    raise AuthError("OIDC client_id is required for device flow.")


def _requested_scopes(default_scopes: str, extra_scopes: str | None) -> str:
    if not extra_scopes:
        return default_scopes

    default_baseline = " ".join(item for item in default_scopes.split() if ":" not in item)
    seen: set[str] = set()
    parts: list[str] = []
    for item in default_baseline.split():
        if item not in seen:
            seen.add(item)
            parts.append(item)
    for item in extra_scopes.split():
        if item not in seen:
            seen.add(item)
            parts.append(item)
    return " ".join(parts)


def _scope_display_entries(requested_scopes: str, scope_prefix: str) -> tuple[RequestedScope, ...]:
    entries: list[RequestedScope] = []
    for requested_scope in requested_scopes.split():
        effective_name = None
        if scope_prefix and (":" in requested_scope or requested_scope.endswith(".default")):
            effective_name = f"{scope_prefix}{requested_scope}"
        entries.append(RequestedScope(name=requested_scope, effective_name=effective_name))
    return tuple(entries)


@dataclass(frozen=True)
class _GrantedScopes:
    raw: tuple[str, ...]
    raw_was_available: bool


def _granted_scopes(
    *,
    token_response: device_flow.TokenResponse,
    claims: dict[str, object],
    bearer_token_source: BearerTokenSource,
) -> _GrantedScopes:
    raw_granted_scopes: object = token_response.scope
    if raw_granted_scopes is None and bearer_token_source == "access_token":
        raw_granted_scopes = claims.get("scp") or claims.get("scope") or []

    granted_scopes: tuple[str, ...] = ()
    if isinstance(raw_granted_scopes, str):
        granted_scopes = tuple(raw_granted_scopes.split())
    elif isinstance(raw_granted_scopes, list):
        granted_scopes = tuple(item for item in raw_granted_scopes if isinstance(item, str))

    return _GrantedScopes(raw=granted_scopes, raw_was_available=raw_granted_scopes is not None)


def _display_scopes(scopes: tuple[str, ...], scope_prefix: str) -> tuple[str, ...]:
    return tuple(
        item[len(scope_prefix) :] if scope_prefix and item.startswith(scope_prefix) else item for item in scopes
    )


def _user_email_from_claims(claims: dict[str, object]) -> str | None:
    value = claims.get("upn") or claims.get("email") or claims.get("preferred_username")
    return value if isinstance(value, str) else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None

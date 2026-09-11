# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authentication commands for NeMo Platform CLI.

Scope normalization, JWT decode, and scope validation logic live in
nemo_platform_ext.auth.helpers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Annotated, NoReturn, cast

import httpx
import typer
from nemo_platform_plugin.auth.access_keys.client import AccessKeyIssuerClient, AccessKeysClient
from nemo_platform_plugin.auth.access_keys.issuer import (
    AccessKeyFeatureDisabledError,
    AccessKeyOperationNotImplementedError,
)
from nemo_platform_plugin.auth.access_keys.types import (
    AccessKeyCreateRequest,
    AccessKeyWorkspaceGrant,
)
from nemo_platform_plugin.client.adapter import client_from_platform
from rich.console import Console

from nemo_platform_ext.auth.helpers import (
    AuthError,
    build_effective_scope,
    decode_jwt_claims,
    discover_nmp_config,
    generate_unsigned_jwt,
    is_unsigned_jwt,
    normalize_scope_prefix,
    validate_requested_scopes_granted,
)
from nemo_platform_ext.auth.token_provider import OIDCTokenProvider, TokenSet
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import handle_errors
from nemo_platform_ext.cli.core.formatters import Column, format_output
from nemo_platform_ext.cli.core.help_formatter import create_typer_app
from nemo_platform_ext.config.config import Config
from nemo_platform_ext.config.models import ConfigParams, Context

app = create_typer_app(
    name="auth",
    help="Manage authentication for NeMo Platform.",
)
access_keys_app = create_typer_app(
    name="access-keys",
    help="Manage NeMo Platform Scoped Access Keys.",
)
app.add_typer(access_keys_app, name="access-keys")

logger = logging.getLogger(__name__)


def _access_key_issuer(ctx: typer.Context) -> AccessKeyIssuerClient:
    state: CLIContext = ctx.obj
    access_keys_client = client_from_platform(state.get_client(), AccessKeysClient)
    return AccessKeyIssuerClient(access_keys_client)


def _raise_access_key_not_implemented(exc: AccessKeyOperationNotImplementedError) -> NoReturn:
    raise AuthError(str(exc) or "Scoped Access Key operation is not implemented.") from exc


def _raise_access_key_disabled(exc: AccessKeyFeatureDisabledError) -> NoReturn:
    raise AuthError(str(exc) or "Scoped Access Keys are not enabled.") from exc


def _parse_access_key_expires_in(value: str | None) -> tuple[bool, int | None]:
    if value is None:
        return False, None

    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return True, None

    try:
        expires_in_seconds = int(value)
    except ValueError as exc:
        raise AuthError("--expires-in must be a positive integer number of seconds or 'none'.") from exc

    if expires_in_seconds < 1:
        raise AuthError("--expires-in must be a positive integer number of seconds or 'none'.")
    return True, expires_in_seconds


def _parse_access_key_scope(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return list(dict.fromkeys(service.strip() for service in value.split(",") if service.strip()))


def _parse_access_key_workspace_grants(
    values: list[str] | None,
) -> list[AccessKeyWorkspaceGrant] | None:
    if values is None:
        return None

    grants: list[AccessKeyWorkspaceGrant] = []
    for value in values:
        workspace, separator, roles_value = value.partition(":")
        roles = [role.strip() for role in roles_value.split(",") if role.strip()] if separator else []
        if separator and not roles:
            # An explicit ':' with no roles after it (e.g. 'team-a:' or 'team-a:,,') is more
            # likely a typo than a request for the default role; only omitting ':' entirely
            # should fall back to Editor.
            raise typer.BadParameter(
                f"Invalid workspace grant {value!r}: specify at least one role after ':'.",
                param_hint="--workspace",
            )
        grants.append(
            AccessKeyWorkspaceGrant(
                workspace=workspace.strip(),
                roles=roles if separator else ["Editor"],
            )
        )
    return grants


def is_auth_disabled(base_url: str, timeout: float = 3.0, certificate_authority: str | None = None) -> bool:
    """Check whether authentication is disabled on the cluster.

    Returns:
        True if auth is disabled, False if enabled.
    """
    try:
        return not discover_nmp_config(
            base_url,
            timeout=timeout,
            certificate_authority=certificate_authority,
        ).auth_enabled
    except httpx.HTTPError as exc:
        raise AuthError(f"Failed to discover auth configuration: {exc}") from exc


def _runtime_token_source_label() -> str | None:
    """Return the runtime token override source using the same precedence as config loading."""
    from nemo_platform_ext.config.config import Config

    try:
        return Config.runtime_access_token_source_label()
    except ValueError:
        logger.debug("Failed to resolve runtime token override source label", exc_info=True)
        return None


def ensure_valid_token(context: Context, refresh_buffer_seconds: int = 300) -> bool:
    """
    Check if the current token is valid and refresh if needed.

    This function should be called before making API requests to ensure
    the access token is valid. If the token is expired or about to expire
    (within refresh_buffer_seconds), it will automatically refresh using
    the stored refresh token.

    Args:
        refresh_buffer_seconds: Refresh the token if it expires within this many seconds.
            Default is 300 (5 minutes).

    Returns:
        True if a valid token is available (possibly after refresh), False otherwise.
    """
    from datetime import datetime, timezone

    from nemo_platform_ext.config.config import Config
    from nemo_platform_ext.config.models import OAuthUser

    # Only OAuthUser supports token refresh
    if not context.user or not isinstance(context.user, OAuthUser):
        return True  # Non-OAuth users don't need refresh

    # Check if token is expired or about to expire
    token = context.user.token.get_secret_value()
    claims = decode_jwt_claims(token)
    if not claims:
        # Not a JWT, can't check expiry
        return True

    exp = claims.get("exp")
    if not exp:
        return True

    exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
    now = datetime.now(tz=timezone.utc)

    # Token is still valid with buffer
    if exp_dt > now + __import__("datetime").timedelta(seconds=refresh_buffer_seconds):
        return True

    # Token expired or about to expire - try to refresh
    if not context.user.refresh_token:
        return exp_dt > now  # Return True only if not yet expired

    base_url = str(context.cluster.base_url).rstrip("/")
    try:
        nmp_config = discover_nmp_config(base_url, certificate_authority=context.cluster.certificate_authority)
    except httpx.HTTPError:
        return exp_dt > now

    if not nmp_config.client_id or not nmp_config.token_endpoint:
        return exp_dt > now

    effective_scope = build_effective_scope(nmp_config.default_scopes, nmp_config.scope_prefix)

    try:
        provider = OIDCTokenProvider(
            token_endpoint=nmp_config.token_endpoint,
            client_id=nmp_config.client_id,
            tokens=TokenSet.from_access_token(
                context.user.token.get_secret_value(),
                context.user.refresh_token.get_secret_value(),
            ),
            refresh_scope=effective_scope,
            refresh_margin_seconds=float(refresh_buffer_seconds),
            certificate_authority=context.cluster.certificate_authority,
        )
        provider.force_refresh()

        config_params: ConfigParams = {
            "access_token": provider.tokens.access_token,
        }
        if provider.tokens.refresh_token:
            config_params["refresh_token"] = provider.tokens.refresh_token

        Config.write(config_params, context_name=context.context_name)
        typer.echo("[Auto-refreshed expired token]", err=True)

        return True
    except Exception:
        return exp_dt > now


@app.callback(invoke_without_command=True)
def auth_callback(ctx: typer.Context) -> None:
    """Manage authentication for NeMo Platform."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _login_with_oidc(
    cli_context: CLIContext,
    *,
    no_browser: bool = False,
    scope: str | None = None,
    username: str | None = None,
    password: str | None = None,
    selected_context: str | None = None,
) -> bool:
    """Authenticate the selected context using the cluster's OIDC configuration.

    Returns ``False`` when cluster authentication is disabled.
    """
    from nemo_platform_ext.auth.device_flow import (
        DeviceFlowError,
        authenticate_with_device_flow,
        authenticate_with_password_grant,
    )

    console = Console()
    context = cli_context.get_sdk_context()
    base_url = str(context.cluster.base_url).rstrip("/")
    certificate_authority = context.cluster.certificate_authority

    console.print(f"\nDiscovering auth configuration from {base_url}...")

    try:
        oidc_config = discover_nmp_config(base_url, certificate_authority=certificate_authority)
    except httpx.HTTPError as exc:
        raise AuthError(f"Failed to discover auth configuration: {exc}") from exc

    if not oidc_config.auth_enabled:
        console.print("[yellow]Authentication is not enabled on this cluster.[/]")
        console.print("You can use the API without authentication.")
        return False

    if not oidc_config.token_endpoint:
        raise AuthError(
            "This cluster does not have OIDC token endpoint configured.\n"
            "Use OIDC configuration for device/password login, or for local testing use:\n"
            "nemo auth login --unsigned-token --email <email>"
        )

    login_username = username or os.environ.get("NMP_OIDC_USERNAME")
    login_password = password or os.environ.get("NMP_OIDC_PASSWORD")
    use_password_grant = bool(login_username and login_password)

    if use_password_grant:
        if not oidc_config.client_id:
            raise AuthError("OIDC client_id is required for password grant.")
    elif not oidc_config.device_authorization_endpoint:
        raise AuthError(
            "This cluster does not support device flow authentication.\n"
            "For non-interactive login use: nemo auth login --username <user> --password <pass>\n"
            "Or set NMP_OIDC_USERNAME and NMP_OIDC_PASSWORD (e.g. in CI)."
        )

    console.print(f"[green]Found OIDC configuration[/] (issuer: {oidc_config.issuer})")

    raw_defaults = oidc_config.default_scopes
    default_baseline = " ".join(item for item in raw_defaults.split() if ":" not in item)
    if scope:
        seen: set[str] = set()
        parts: list[str] = []
        for item in default_baseline.split():
            if item not in seen:
                seen.add(item)
                parts.append(item)
        for item in scope.split():
            if item not in seen:
                seen.add(item)
                parts.append(item)
        requested_scopes = " ".join(parts)
    else:
        requested_scopes = raw_defaults

    scope_prefix = normalize_scope_prefix(oidc_config.scope_prefix)
    effective_scope = build_effective_scope(requested_scopes, oidc_config.scope_prefix)

    console.print("\n[bold]Requesting scopes:[/]")
    for requested_scope in requested_scopes.split():
        if scope_prefix and (":" in requested_scope or requested_scope.endswith(".default")):
            console.print(f"  [cyan]{requested_scope}[/] [dim]({scope_prefix}{requested_scope})[/]")
        else:
            console.print(f"  [cyan]{requested_scope}[/]")
    console.print()

    if use_password_grant:
        if login_username is None or login_password is None:
            raise AuthError("Username and password are required for password grant.")
        client_id = cast(str, oidc_config.client_id)
        try:
            token_response = authenticate_with_password_grant(
                token_endpoint=oidc_config.token_endpoint,
                client_id=client_id,
                username=login_username,
                password=login_password,
                scope=effective_scope,
                certificate_authority=certificate_authority,
            )
        except DeviceFlowError as exc:
            raise AuthError(f"Authentication failed: {exc}") from exc
    else:
        if oidc_config.device_authorization_endpoint is None:
            raise AuthError("This cluster does not support device flow authentication.")
        client_id = cast(str, oidc_config.client_id)
        try:
            token_response = asyncio.run(
                authenticate_with_device_flow(
                    device_authorization_endpoint=oidc_config.device_authorization_endpoint,
                    token_endpoint=oidc_config.token_endpoint,
                    client_id=client_id,
                    scope=effective_scope,
                    open_browser=not no_browser,
                    certificate_authority=certificate_authority,
                )
            )
        except DeviceFlowError as exc:
            raise AuthError(f"Authentication failed: {exc}") from exc

    token = token_response.token_for_nmp
    claims = decode_jwt_claims(token)
    user_email = claims.get("upn") or claims.get("email") or claims.get("preferred_username")
    raw_granted_scopes = claims.get("scp") or claims.get("scope")
    granted_scopes: list[str] = []
    if isinstance(raw_granted_scopes, str):
        granted_scopes = raw_granted_scopes.split()
    elif isinstance(raw_granted_scopes, list):
        granted_scopes = [item for item in raw_granted_scopes if isinstance(item, str)]

    validate_requested_scopes_granted(effective_scope, granted_scopes, scope_prefix)

    config_params: ConfigParams = {"access_token": token}
    if token_response.refresh_token:
        config_params["refresh_token"] = token_response.refresh_token
    if selected_context is not None:
        config_params["current_context"] = context.context_name
    Config.write(config_params, context_name=context.context_name)

    console.print("\n[bold green]Authentication successful![/]")
    if user_email:
        console.print(f"  Logged in as: [cyan]{user_email}[/]")

    if granted_scopes:
        display_scopes = [
            item[len(scope_prefix) :] if scope_prefix and item.startswith(scope_prefix) else item
            for item in granted_scopes
        ]
        console.print(f"  Granted scopes: [cyan]{' '.join(display_scopes)}[/]")

    if token_response.refresh_token:
        console.print("  Refresh token: [green]saved[/] (enables automatic token renewal)")
    else:
        console.print("  Refresh token: [yellow]not available[/] (add 'offline_access' scope to enable)")

    console.print("\n[bold green]Credentials saved to config file.[/]")
    if runtime_token_source := _runtime_token_source_label():
        console.print(
            f"[yellow]Warning:[/] {runtime_token_source} is active and will override these saved credentials. "
            "Unset the runtime token override to use this login for future commands."
        )
    console.print("\n[dim]Run 'nemo workspaces list' to verify your access.[/]")
    return True


@app.command("login")
@handle_errors
def login(
    ctx: typer.Context,
    context_name: Annotated[
        str | None,
        typer.Option(
            "--context",
            help="Context to use for this login command.",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="Set cluster base URL for the selected context before login",
        ),
    ] = None,
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Don't open browser (device flow only)"),
    ] = False,
    scope: Annotated[
        str | None,
        typer.Option(
            "--scope",
            help='OAuth scopes to request (space-separated; quote for multiple, e.g. --scope "platform:read secrets:write")',
        ),
    ] = None,
    username: Annotated[
        str | None,
        typer.Option("--username", help="Username for password grant (CI / non-interactive)"),
    ] = None,
    password: Annotated[
        str | None,
        typer.Option("--password", help="Password for password grant (prefer env NMP_OIDC_PASSWORD)"),
    ] = None,
    unsigned_token: Annotated[
        bool,
        typer.Option(
            "--unsigned-token",
            help="Generate and save an unsigned JWT for local/testing authentication.",
            rich_help_panel="Unsigned Token Options",
        ),
    ] = False,
    principal_id: Annotated[
        str | None,
        typer.Option(
            "--principal-id",
            help="Principal ID for the unsigned token (`sub` claim). Defaults to --email.",
            rich_help_panel="Unsigned Token Options",
        ),
    ] = None,
    email: Annotated[
        str | None,
        typer.Option(
            "--email",
            help="Email claim for the unsigned token (required with --unsigned-token).",
            rich_help_panel="Unsigned Token Options",
        ),
    ] = None,
    groups: Annotated[
        list[str] | None,
        typer.Option(
            "--group",
            help="Group claim value for unsigned token (repeat for multiple).",
            rich_help_panel="Unsigned Token Options",
        ),
    ] = None,
    expires_in: Annotated[
        int,
        typer.Option(
            "--expires-in",
            help="Unsigned token expiry in seconds from now.",
            rich_help_panel="Unsigned Token Options",
        ),
    ] = 3600,
    no_exp: Annotated[
        bool,
        typer.Option(
            "--no-exp",
            help="Omit the exp claim from the unsigned token.",
            rich_help_panel="Unsigned Token Options",
        ),
    ] = False,
    audience: Annotated[
        str | None,
        typer.Option(
            "--audience",
            help="Audience (`aud`) claim for unsigned token.",
            rich_help_panel="Unsigned Token Options",
        ),
    ] = None,
    issuer: Annotated[
        str | None,
        typer.Option(
            "--issuer",
            help="Issuer (`iss`) claim for unsigned token.",
            rich_help_panel="Unsigned Token Options",
        ),
    ] = None,
) -> None:
    """Authenticate with the NeMo Platform cluster.

    Uses device flow (browser) by default, or password grant when username and password are provided (e.g. for CI).

    For quickstart, use [cyan]`--unsigned-token`[/] to generate an unsigned JWT.

    Examples:
    # Set base URL and log in
    nemo auth login --base-url https://nemo.example.com
    # Context-specific login
    nemo auth login --context staging --base-url https://nmp.staging.example.com
    # Device flow, open browser
    nemo auth login
    # Device flow, show code only
    nemo auth login --no-browser
    """
    cli_context: CLIContext = ctx.obj
    selected_context = cli_context.overrides.get("current_context")

    if context_name is not None:
        selected_context = context_name
        cli_context.overrides["current_context"] = context_name
        cli_context.reset_sdk_context()

    if base_url is not None:
        base_url_params: ConfigParams = {"base_url": base_url}
        if selected_context is not None:
            base_url_params["current_context"] = selected_context
        try:
            Config.write(base_url_params, context_name=selected_context)
        except Exception as exc:
            context_label = selected_context or "default"
            raise AuthError(f"Failed to set base URL for context '{context_label}': {exc}") from exc
        cli_context.reset_sdk_context()

    console = Console()

    unsigned_option_flags = {
        "principal_id": "--principal-id",
        "email": "--email",
        "groups": "--group",
        "expires_in": "--expires-in",
        "no_exp": "--no-exp",
        "audience": "--audience",
        "issuer": "--issuer",
    }

    if not unsigned_token:
        from click.core import ParameterSource

        unsigned_options_used = [
            flag
            for param_name, flag in unsigned_option_flags.items()
            if ctx.get_parameter_source(param_name) != ParameterSource.DEFAULT
        ]
        if unsigned_options_used:
            raise AuthError(f"Unsigned token option(s) {', '.join(unsigned_options_used)} require --unsigned-token.")

    if unsigned_token:
        if username or password:
            raise AuthError("Cannot combine --unsigned-token with --username/--password.")
        if email is None:
            raise AuthError("--email is required when using --unsigned-token.")

        effective_principal_id = principal_id or email

        context = cli_context.get_sdk_context()
        base_url = str(context.cluster.base_url).rstrip("/")

        try:
            oidc_config = discover_nmp_config(base_url, certificate_authority=context.cluster.certificate_authority)
            oidc_login_configured = bool(oidc_config.token_endpoint and oidc_config.client_id)
            if oidc_login_configured:
                raise AuthError(
                    "Cluster has OIDC authentication configured. Use 'nemo auth login' instead of '--unsigned-token'."
                )
            if oidc_config.auth_enabled:
                console.print(
                    "Cluster authentication is enabled. Ensure [cyan]auth.allow_unsigned_jwt=true[/] on the cluster."
                )
        except httpx.HTTPError:
            console.print("[yellow]Could not verify cluster auth settings via discovery endpoint.[/]")

        requested_scopes = scope.split() if scope else None
        token = generate_unsigned_jwt(
            principal_id=effective_principal_id,
            email=email,
            groups=groups,
            scopes=requested_scopes,
            expires_in_seconds=None if no_exp else expires_in,
            audience=audience,
            issuer=issuer,
        )

        config_params: ConfigParams = {"access_token": token}
        if selected_context is not None:
            config_params["current_context"] = context.context_name
        Config.write(config_params, context_name=context.context_name)

        console.print("\n[bold yellow]Warning:[/] Generated an unsigned JWT (`alg=none`).")
        console.print("Use this only for local/testing environments.")

        console.print("\n[bold green]Unsigned token saved to config file.[/]")
        console.print(f"Principal: [cyan]{effective_principal_id}[/]")
        console.print("\n[dim]Run 'nemo auth status' to inspect the token.[/]")
        return

    if not _login_with_oidc(
        cli_context,
        no_browser=no_browser,
        scope=scope,
        username=username,
        password=password,
        selected_context=selected_context,
    ):
        raise typer.Exit(0)


@app.command("logout")
@handle_errors
def logout(ctx: typer.Context) -> None:
    """Remove stored credentials for the current context."""
    from rich.console import Console

    from nemo_platform_ext.config.config import Config
    from nemo_platform_ext.config.models import NoAuthUser

    cli_context: CLIContext = ctx.obj
    context = cli_context.get_sdk_context()

    console = Console()

    base_url = str(context.cluster.base_url).rstrip("/")
    try:
        if is_auth_disabled(base_url, certificate_authority=context.cluster.certificate_authority) is True:
            console.print("[yellow]Authentication is disabled on this cluster — nothing to log out from.[/]")
            return
    except AuthError as exc:
        logger.debug("Failed to discover auth configuration during logout", exc_info=True)
        console.print(f"[yellow]Warning:[/] {exc}; continuing to clear local credentials.")

    logout_params: ConfigParams = {"access_token": None, "refresh_token": None}
    updated_config = Config.write(logout_params, context_name=context.context_name)
    config_path = Config.get_default_config_path()
    if isinstance(updated_config, Config):
        config_path = updated_config.get_config_path() or config_path
        persisted_config = Config.load(config_path=config_path).get_config_file()
        persisted_context = next((ctx for ctx in persisted_config.contexts if ctx.name == context.context_name), None)
        persisted_user = None
        if persisted_context is not None:
            persisted_user = next(
                (user for user in persisted_config.users if user.name == persisted_context.user), None
            )

        if persisted_context is None or not isinstance(persisted_user, NoAuthUser):
            raise AuthError(
                "Logout did not clear credentials for "
                f"context '{context.context_name}' in config file '{config_path.name}' at {config_path}. "
                "Run 'nemo auth status' and check the Config File and Credential Source rows."
            )

    console.print("[green]Logged out successfully.[/]")
    console.print(f"  Context: [cyan]{context.context_name}[/]")
    console.print(f"  Config file: [cyan]{config_path}[/]")
    if runtime_token_source := _runtime_token_source_label():
        console.print(
            f"  [yellow]Warning:[/] {runtime_token_source} is still active and will override saved credentials."
        )


@app.command("refresh")
@handle_errors
def refresh(ctx: typer.Context) -> None:
    """Refresh the current access token.

    This command uses the saved refresh token to obtain a new access token without requiring you to re-authenticate through the browser.
    """
    from rich.console import Console

    from nemo_platform_ext.config.config import Config
    from nemo_platform_ext.config.models import OAuthUser

    cli_context: CLIContext = ctx.obj
    context = cli_context.get_sdk_context()

    console = Console()

    # Only OAuthUser stores tokens in config
    if not context.user or not isinstance(context.user, OAuthUser):
        raise AuthError("No OAuth authentication configured. Run 'nemo auth login' first.")

    token_value = context.user.token.get_secret_value()

    if is_unsigned_jwt(token_value):
        claims = decode_jwt_claims(token_value)
        principal_id = claims.get("sub")
        if not isinstance(principal_id, str) or not principal_id:
            raise AuthError("Unsigned token is missing required 'sub' claim.")

        email_claim = claims.get("email")
        email = email_claim if isinstance(email_claim, str) else None

        groups_claim = claims.get("groups")
        if isinstance(groups_claim, str):
            groups = [groups_claim]
        elif isinstance(groups_claim, list) and all(isinstance(group, str) for group in groups_claim):
            groups = groups_claim or None
        else:
            groups = None

        scope_claim = claims.get("scope")
        scopes = scope_claim.split() if isinstance(scope_claim, str) and scope_claim else None

        expires_in_seconds = None
        exp_claim = claims.get("exp")
        iat_claim = claims.get("iat")
        if isinstance(exp_claim, int):
            if isinstance(iat_claim, int):
                expires_in_seconds = max(exp_claim - iat_claim, 0)
            else:
                expires_in_seconds = max(exp_claim - int(time.time()), 0)

        audience = claims.get("aud") if isinstance(claims.get("aud"), str) else None
        issuer = claims.get("iss") if isinstance(claims.get("iss"), str) else None

        excluded_claims = {
            "sub",
            "email",
            "groups",
            "scope",
            "iat",
            "exp",
            "aud",
            "iss",
        }
        extra_claims = {k: v for k, v in claims.items() if k not in excluded_claims}
        if "aud" in claims and not isinstance(claims.get("aud"), str):
            extra_claims["aud"] = claims.get("aud")

        refreshed_token = generate_unsigned_jwt(
            principal_id=principal_id,
            email=email,
            groups=groups,
            scopes=scopes,
            expires_in_seconds=expires_in_seconds,
            audience=audience,
            issuer=issuer,
            extra_claims=extra_claims or None,
        )

        Config.write(
            cast(ConfigParams, {"access_token": refreshed_token, "refresh_token": None}),
            context_name=context.context_name,
        )

        refreshed_claims = decode_jwt_claims(refreshed_token)
        exp = refreshed_claims.get("exp")
        if exp:
            from datetime import datetime, timezone

            exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
            console.print(f"[green]Unsigned token refreshed successfully![/] Expires: {exp_dt.isoformat()}")
        else:
            console.print("[green]Unsigned token refreshed successfully![/]")
        return

    if not context.user.refresh_token:
        raise AuthError("No refresh token available. Re-run 'nemo auth login' with 'offline_access' scope.")

    # Fetch client_id from cluster discovery
    base_url = str(context.cluster.base_url).rstrip("/")
    try:
        oidc_config = discover_nmp_config(base_url, certificate_authority=context.cluster.certificate_authority)
    except httpx.HTTPError as e:
        raise AuthError(f"Failed to discover auth configuration: {e}") from e

    if not oidc_config.client_id or not oidc_config.token_endpoint:
        raise AuthError("OIDC not configured on cluster.")

    effective_scope = build_effective_scope(oidc_config.default_scopes, oidc_config.scope_prefix)

    console.print("Refreshing access token...")

    provider = OIDCTokenProvider(
        token_endpoint=oidc_config.token_endpoint,
        client_id=oidc_config.client_id,
        tokens=TokenSet.from_access_token(
            context.user.token.get_secret_value(),
            context.user.refresh_token.get_secret_value(),
        ),
        refresh_scope=effective_scope,
        certificate_authority=context.cluster.certificate_authority,
    )

    try:
        provider.force_refresh()
    except RuntimeError as e:
        raise AuthError(f"Token refresh failed: {e}") from e

    # Save new tokens (refresh token may be rotated)
    config_params: ConfigParams = {"access_token": provider.tokens.access_token}
    if provider.tokens.refresh_token:
        config_params["refresh_token"] = provider.tokens.refresh_token
    Config.write(config_params, context_name=context.context_name)

    # Show new token info
    claims = decode_jwt_claims(provider.tokens.access_token)
    exp = claims.get("exp")
    if exp:
        from datetime import datetime, timezone

        exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
        console.print(f"[green]Token refreshed successfully![/] Expires: {exp_dt.isoformat()}")
    else:
        console.print("[green]Token refreshed successfully![/]")


@app.command("token")
@handle_errors
def token(
    ctx: typer.Context,
    decode: Annotated[
        bool,
        typer.Option(
            "--decode",
            help="Decode the JWT payload claims as JSON. This does not verify the token signature.",
        ),
    ] = False,
) -> None:
    """Print the current access token (for use with SDK or curl).

    By default this outputs the raw token to stdout, suitable for piping or capture.

    Examples:
    # Print token
    nemo auth token
    # Inspect token claims
    nemo auth token --decode
    # Capture in env var
    export TOKEN=$(nemo auth token)
    curl -H "Authorization: Bearer $(nemo auth token)" ...
    """
    from nemo_platform_ext.config.models import OAuthUser

    cli_context: CLIContext = ctx.obj
    context = cli_context.get_sdk_context()

    if not context.user:
        raise AuthError("No authentication configured. Run 'nemo auth login' first.")

    if isinstance(context.user, OAuthUser):
        access_token = context.user.token.get_secret_value()
        if decode:
            claims = decode_jwt_claims(access_token)
            if not claims:
                raise AuthError("Current access token is not a decodable JWT.")
            typer.echo(json.dumps(claims, indent=2))
            return
        typer.echo(access_token)
    else:
        raise AuthError("No token available for current user type.")


@access_keys_app.command("create")
@handle_errors
def create_access_key(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Optional human-readable label for the Scoped Access Key."),
    ] = None,
    description: Annotated[
        str | None,
        typer.Option("--description", "-d", help="Optional description for the Scoped Access Key."),
    ] = None,
    expires_in: Annotated[
        str | None,
        typer.Option(
            "--expires-in",
            help="Scoped Access Key lifetime in seconds. Use 'none' to request no expiration.",
        ),
    ] = None,
    service_account: Annotated[
        str | None,
        typer.Option(
            "--service-account",
            help="Bind the key to a non-human service account (PlatformAdmin only).",
        ),
    ] = None,
    scope: Annotated[
        str | None,
        typer.Option(
            "--scope",
            help="Comma-separated service names to scope this key to, e.g. 'intake,entities'. Omit for an unscoped key.",
        ),
    ] = None,
    rotate: Annotated[
        str | None,
        typer.Option(
            "--rotate",
            help=(
                "JTI of a previous Scoped Access Key owned by the caller to revoke once this key is created. "
                "Primarily for rotating personal keys, which have no service-account identity to correlate "
                "rotations against automatically."
            ),
        ),
    ] = None,
    workspace: Annotated[
        list[str] | None,
        typer.Option(
            "--workspace",
            help=(
                "Grant the key's principal workspace membership, formatted '<workspace>' or "
                "'<workspace>:<role1>,<role2>' (role defaults to Editor). Repeat for multiple workspaces. "
                "Replaces a separate 'nemo workspaces members create' call."
            ),
        ),
    ] = None,
) -> None:
    """Create a user-bound or service-bound Scoped Access Key."""
    expires_in_was_set, parsed_expires_in = _parse_access_key_expires_in(expires_in)
    request = AccessKeyCreateRequest(
        name=name,
        description=description,
        service_account_id=service_account,
        scope=_parse_access_key_scope(scope),
        rotates=rotate,
        workspaces=_parse_access_key_workspace_grants(workspace),
        **({"expires_in_seconds": parsed_expires_in} if expires_in_was_set else {}),
    )
    try:
        created = _access_key_issuer(ctx).create(request)
    except AccessKeyFeatureDisabledError as exc:
        _raise_access_key_disabled(exc)
    except AccessKeyOperationNotImplementedError as exc:
        _raise_access_key_not_implemented(exc)
    typer.echo(created.token)


@access_keys_app.command("list")
@handle_errors
def list_access_keys(
    ctx: typer.Context,
    page: Annotated[int, typer.Option("--page", min=1, help="Page number to retrieve.")] = 1,
    page_size: Annotated[
        int,
        typer.Option("--page-size", min=1, max=100, help="Number of keys to retrieve per page."),
    ] = 100,
) -> None:
    """List Scoped Access Keys owned by the currently authenticated user.

    PlatformAdmins also see every service-bound Scoped Access Key, not just the
    ones they personally created.
    """
    try:
        listed = _access_key_issuer(ctx).list(page=page, page_size=page_size)
    except AccessKeyFeatureDisabledError as exc:
        _raise_access_key_disabled(exc)
    except AccessKeyOperationNotImplementedError as exc:
        _raise_access_key_not_implemented(exc)

    state: CLIContext = ctx.obj
    format_output(
        listed.data,
        is_list=True,
        output_format=state.get_output_format(),
        output_columns=[
            Column("jti", None),
            Column("name", None),
            Column("description", None),
            Column("entity_type", None),
            Column("principal", None),
            Column("status", None),
            Column("issuer", None),
            Column("audiences", None),
            Column("scope", None),
            Column("created_at", None),
            Column("expires_at", None),
            Column("last_used_at", None),
            Column("grace_period_expires_at", None),
        ],
        timestamp_format=state.get_timestamp_format(),
    )
    if listed.has_more:
        typer.echo(f"More Scoped Access Keys are available; use --page {page + 1} to retrieve them.", err=True)


@access_keys_app.command("revoke")
@handle_errors
def revoke_access_key(
    ctx: typer.Context,
    jti: Annotated[str, typer.Argument(help="Stable ID of the Scoped Access Key to revoke.")],
) -> None:
    """Revoke a Scoped Access Key owned by the currently authenticated user."""
    try:
        result = _access_key_issuer(ctx).revoke(jti)
    except AccessKeyFeatureDisabledError as exc:
        _raise_access_key_disabled(exc)
    except AccessKeyOperationNotImplementedError as exc:
        _raise_access_key_not_implemented(exc)
    if result.revoked:
        typer.echo(f"Revoked Scoped Access Key {jti}.")
    else:
        typer.echo(f"Scoped Access Key {jti} was already revoked.")


@access_keys_app.command("suspend")
@handle_errors
def suspend_access_key(
    ctx: typer.Context,
    jti: Annotated[str, typer.Argument(help="Stable ID of the Scoped Access Key to suspend.")],
) -> None:
    """Temporarily suspend a Scoped Access Key owned by the current user.

    Unlike revocation, suspension is reversible until the key expires.
    """
    try:
        result = _access_key_issuer(ctx).suspend(jti)
    except AccessKeyFeatureDisabledError as exc:
        _raise_access_key_disabled(exc)
    except AccessKeyOperationNotImplementedError as exc:
        _raise_access_key_not_implemented(exc)
    if result.changed:
        typer.echo(f"Suspended Scoped Access Key {jti}.")
    else:
        typer.echo(f"Scoped Access Key {jti} was already {result.status.lower()}.")


@access_keys_app.command("unsuspend")
@handle_errors
def unsuspend_access_key(
    ctx: typer.Context,
    jti: Annotated[str, typer.Argument(help="Stable ID of the Scoped Access Key to unsuspend.")],
) -> None:
    """Restore a suspended Scoped Access Key owned by the current user."""
    try:
        result = _access_key_issuer(ctx).unsuspend(jti)
    except AccessKeyFeatureDisabledError as exc:
        _raise_access_key_disabled(exc)
    except AccessKeyOperationNotImplementedError as exc:
        _raise_access_key_not_implemented(exc)
    if result.changed:
        typer.echo(f"Unsuspended Scoped Access Key {jti}.")
    else:
        typer.echo(f"Scoped Access Key {jti} was already {result.status.lower()}.")


@access_keys_app.command("rotate")
@handle_errors
def rotate_access_key(
    ctx: typer.Context,
    jti: Annotated[str, typer.Argument(help="Stable ID of the Scoped Access Key to rotate.")],
    grace_period_seconds: Annotated[
        int | None,
        typer.Option(
            "--grace-period-seconds",
            min=1,
            help=(
                "Grace period in seconds for the rotated-out key. Defaults to the server's "
                "`rotation_grace_period_seconds` config (default 48h) when omitted, and is "
                "capped by `max_rotation_grace_period_seconds`."
            ),
        ),
    ] = None,
) -> None:
    """Mint a successor Scoped Access Key and start the old key's rotation grace period.

    The old key (jti) keeps working for its rotation grace period so you can cut
    configuration over to the new key without downtime, then revoke the old key once
    traffic has moved, or let it expire automatically at the end of the grace period.
    """
    try:
        result = _access_key_issuer(ctx).rotate(jti, grace_period_seconds=grace_period_seconds)
    except AccessKeyFeatureDisabledError as exc:
        _raise_access_key_disabled(exc)
    except AccessKeyOperationNotImplementedError as exc:
        _raise_access_key_not_implemented(exc)
    typer.echo(result.new_key.token)
    if result.previous_status == "ROTATING":
        if result.grace_period_expires_at is not None:
            typer.echo(
                f"Rotated Scoped Access Key {jti}; it remains usable until "
                f"{result.grace_period_expires_at.isoformat()} "
                f"(grace period {result.grace_period_seconds}s).",
                err=True,
            )
        else:
            typer.echo(
                f"Rotated Scoped Access Key {jti}; it remains usable for a "
                f"grace period of {result.grace_period_seconds}s.",
                err=True,
            )
    else:
        typer.echo(
            f"Rotated Scoped Access Key {jti}; it is now {result.previous_status} and is no longer usable.",
            err=True,
        )


@app.command("status")
@handle_errors
def status(ctx: typer.Context) -> None:
    """Show current authentication status."""
    from datetime import datetime, timezone

    from rich.console import Console
    from rich.table import Table

    from nemo_platform_ext.config.config import Config
    from nemo_platform_ext.config.models import OAuthUser

    cli_context: CLIContext = ctx.obj
    context = cli_context.get_sdk_context()

    console = Console()

    # Check whether the cluster has auth enabled before showing token details.
    base_url = str(context.cluster.base_url).rstrip("/")
    auth_discovery_error: AuthError | None = None
    try:
        auth_disabled = is_auth_disabled(base_url, certificate_authority=context.cluster.certificate_authority)
    except AuthError as exc:
        logger.debug("Failed to discover auth configuration during status", exc_info=True)
        auth_discovery_error = exc
    else:
        if auth_disabled is True:
            console.print()
            console.print(f"[cyan]Cluster:[/] {base_url}")
            console.print(f"[cyan]Context:[/] {context.context_name}")
            console.print()
            console.print("[green]Authentication is disabled on this cluster.[/]")
            console.print("All API requests are accepted without credentials.")
            return

    table = Table(title="Authentication Status", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", overflow="fold")

    table.add_row("Cluster", str(context.cluster.base_url))
    table.add_row("Context", context.context_name)
    table.add_row("Config File", str(Config.get_default_config_path()))
    if auth_discovery_error is not None:
        table.add_row("Auth Discovery", f"[yellow]unavailable[/] ({auth_discovery_error})")
    runtime_token_source = _runtime_token_source_label()

    if context.user:
        if runtime_token_source:
            table.add_row(
                "Credential Source",
                f"[yellow]{runtime_token_source}[/]",
            )
        else:
            table.add_row("Auth Type", context.user.type)
            table.add_row("Credential Source", "config file")

        if isinstance(context.user, OAuthUser):
            # OAuth token authentication
            token_value = context.user.token.get_secret_value()
            if is_unsigned_jwt(token_value):
                table.add_row("Warning", "[yellow]Unsigned JWT (alg=none). Use only for local/testing.[/]")

            claims = decode_jwt_claims(token_value)

            if claims:
                # Show decoded JWT info
                if not runtime_token_source:
                    email = claims.get("upn") or claims.get("email") or claims.get("preferred_username")
                    if email:
                        table.add_row("Email", email)

                    subject = claims.get("oid") or claims.get("sub")
                    if subject:
                        table.add_row("User ID", subject)

                    scopes = claims.get("scp") or claims.get("scope") or ""
                    if isinstance(scopes, str):
                        scopes = scopes.split()
                    if scopes:
                        table.add_row("Scopes", " ".join(scopes))
                    else:
                        table.add_row("Scopes", "[dim]none[/]")

                    groups = claims.get("groups") or claims.get("cognito:groups") or []
                    if isinstance(groups, str):
                        groups = [groups]
                    if groups:
                        table.add_row("Groups", ", ".join(groups))

                exp = claims.get("exp")
                if exp:
                    exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
                    now = datetime.now(tz=timezone.utc)
                    if exp_dt > now:
                        remaining = exp_dt - now
                        hours, remainder = divmod(int(remaining.total_seconds()), 3600)
                        minutes, _ = divmod(remainder, 60)
                        table.add_row("Expires", f"{exp_dt.isoformat()} ({hours}h {minutes}m remaining)")
                    else:
                        table.add_row("Expires", f"[red]EXPIRED[/] ({exp_dt.isoformat()})")

            if not runtime_token_source:
                # Show refresh token status for saved credentials only.
                if context.user.refresh_token:
                    table.add_row("Refresh Token", "[green]available[/] (run 'nemo auth refresh' to renew)")
                else:
                    table.add_row("Refresh Token", "[yellow]not available[/]")

            # Show redacted token
            redacted = f"{token_value[:20]}...{token_value[-10:]}" if len(token_value) > 30 else "***"
            table.add_row("Token", redacted)

    else:
        table.add_row("User", "None")
        table.add_row("Auth Type", "no-auth")

    console.print()
    console.print(table)

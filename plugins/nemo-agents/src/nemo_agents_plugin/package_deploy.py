# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight checks and target connection for ``nemo agents package-deploy``.

Everything here runs before an image is built, so a bad registry, login, agent
name, version, or platform fails in seconds instead of after a long build. The
target platform is reached through an in-memory client: the user's CLI config
and active context are never changed.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from nemo_agents_plugin.container.template import require_installable_contract_version
from nemo_helix_ext.auth.helpers import discover_nhx_config
from nemo_helix_ext.auth.login import authenticate_with_oidc
from nemo_helix_ext.cli.commands.auth import ensure_valid_token
from nemo_helix_ext.client.bootstrap import build_nemo_client
from nemo_helix_ext.config.config import Config
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.oidc import OIDCTokenProvider, TokenSet, resolve_refresh_settings

DOCKER_HUB_AUTH_KEY = "index.docker.io"
_LOCAL_REGISTRY_HOSTS = ("localhost", "127.0.0.1")


class PreflightError(Exception):
    """A check that failed before building, with the fix in its message."""


@dataclass(frozen=True)
class TargetConnection:
    client: NemoClient
    description: str


def validate_registry(registry: str) -> str:
    """Return *registry* without a trailing slash, rejecting a URL scheme Docker refs can't carry."""
    if "://" in registry:
        bare = registry.split("://", 1)[1]
        raise PreflightError(
            f"--registry must not include a URL scheme; Docker image references have none. Use: --registry {bare}"
        )
    return registry.rstrip("/")


def registry_host(registry: str) -> str:
    """The host Docker stores credentials under for *registry* (Docker Hub for bare names)."""
    first = registry.split("/", 1)[0]
    if "." in first or ":" in first or first == "localhost":
        return first
    return DOCKER_HUB_AUTH_KEY


def _normalize_auth_key(key: str) -> str:
    host = urlsplit(key).netloc if "://" in key else key.split("/", 1)[0]
    return DOCKER_HUB_AUTH_KEY if host in ("docker.io", "registry-1.docker.io") else host


def _docker_config() -> dict[str, object]:
    config_dir = os.environ.get("DOCKER_CONFIG") or str(Path.home() / ".docker")
    try:
        loaded = json.loads((Path(config_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _credential_store_hosts(store: str) -> set[str] | None:
    try:
        result = subprocess.run(
            [f"docker-credential-{store}", "list"], capture_output=True, text=True, timeout=10, check=True
        )
        listed = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return {_normalize_auth_key(key) for key in listed} if isinstance(listed, dict) else None


def registry_has_credentials(registry: str) -> bool | None:
    """Whether Docker holds credentials for *registry*: ``None`` when the credential store can't be read."""
    host = registry_host(registry)
    if host.split(":", 1)[0] in _LOCAL_REGISTRY_HOSTS:
        return True
    config = _docker_config()
    helpers = config.get("credHelpers")
    if isinstance(helpers, dict) and host in {_normalize_auth_key(key) for key in helpers}:
        return True
    store = config.get("credsStore")
    if isinstance(store, str) and store:
        hosts = _credential_store_hosts(store)
        if hosts is None:
            return None
        return host in hosts
    auths = config.get("auths")
    return isinstance(auths, dict) and host in {_normalize_auth_key(key) for key in auths}


def registry_login_hint(registry: str) -> str:
    host = registry_host(registry)
    if host == "nvcr.io":
        return (
            "Log in with an NGC API key generated in the org you push to: "
            "echo \"$NGC_API_KEY\" | docker login nvcr.io -u '$oauthtoken' --password-stdin"
        )
    return f"Log in first: docker login {host}"


def resolve_contract_version(server_version: str | None, requested: str | None, *, target_url: str) -> str:
    """The published nemo-helix release the image pins, defaulting to the target's own release."""
    server = (server_version or "").removeprefix("v")
    server_release: str | None = None
    if server:
        try:
            require_installable_contract_version(server)
            server_release = server
        except ValueError:
            pass

    if requested is None:
        if server_release is None:
            raise PreflightError(
                f"{target_url} runs nemo-helix '{server_version or 'unknown'}', which is not a published "
                "release, so its version can't be pinned in the image. Pass --nemo-version with the "
                "published release closest to it."
            )
        return server_release

    version = requested.removeprefix("v")
    try:
        require_installable_contract_version(version)
    except ValueError as exc:
        raise PreflightError(f"--nemo-version {requested}: {exc}") from exc
    if server_release is not None and version != server_release:
        raise PreflightError(
            f"--nemo-version {version} does not match nemo-helix {server_release} running on {target_url}. "
            "Omit --nemo-version to use the platform's release."
        )
    return version


def resolve_platform(server_architecture: str | None, requested: str | None, *, target_url: str) -> str:
    """The Docker platform to build for, defaulting to the architecture the target reports."""
    if requested:
        return requested
    if server_architecture:
        return server_architecture
    raise PreflightError(
        f"{target_url} does not report its architecture. Pass --platform with its nodes' platform, "
        "e.g. --platform linux/amd64."
    )


@dataclass(frozen=True)
class TargetInfo:
    platform_version: str | None = None
    architecture: str | None = None


def fetch_cluster_info(target_url: str, http_client: httpx.Client) -> TargetInfo:
    """The target's ``/cluster-info``; empty when the server doesn't serve it."""
    response = http_client.get(f"{target_url}/cluster-info")
    if response.status_code == 404:
        return TargetInfo()
    response.raise_for_status()
    info = response.json()
    if not isinstance(info, dict):
        return TargetInfo()
    version = info.get("platform_version")
    architecture = info.get("architecture")
    return TargetInfo(
        platform_version=version if isinstance(version, str) and version else None,
        architecture=architecture if isinstance(architecture, str) and architecture else None,
    )


def saved_context_for(target_url: str) -> str | None:
    """The name of a saved CLI context whose cluster URL is *target_url*, preferring the active one."""
    config_file = Config.load().get_config_file()
    cluster_names = {
        cluster.name for cluster in config_file.clusters if str(cluster.base_url).rstrip("/") == target_url
    }
    matches = [context.name for context in config_file.contexts if context.cluster in cluster_names]
    if config_file.current_context in matches:
        return config_file.current_context
    return matches[0] if matches else None


def connect_to_target(
    target_url: str, workspace: str, *, http_client: httpx.Client, no_browser: bool = False
) -> TargetConnection:
    """An authenticated client for *target_url* that never changes the user's CLI config or active context.

    A saved context for the same URL is reused while its login is valid. Otherwise
    the user logs in through the browser for this run only, and nothing is saved.
    """
    oidc_config = discover_nhx_config(target_url, http_client=http_client)
    if not oidc_config.auth_enabled:
        return TargetConnection(NemoClient(base_url=target_url, workspace=workspace), "authentication disabled")

    context_name = saved_context_for(target_url)
    if context_name is not None:
        context = Config.load(overrides={"current_context": context_name}).resolve()
        if ensure_valid_token(context, http_client=http_client):
            return TargetConnection(
                build_nemo_client(context_name=context_name, base_url=target_url, workspace=workspace),
                f"saved context '{context_name}'",
            )

    login = authenticate_with_oidc(oidc_config=oidc_config, no_browser=no_browser)
    settings = resolve_refresh_settings(oidc_config, login.token_broker_url)
    provider = OIDCTokenProvider(
        token_endpoint=settings.token_endpoint,
        client_id=settings.client_id,
        tokens=TokenSet.from_access_token(login.access_token, login.refresh_token, expires_at=login.expires_at),
        refresh_scope=settings.refresh_scope,
        bearer_token_source=settings.bearer_token_source,
        refresh_kind=settings.refresh_kind,
    )
    return TargetConnection(
        NemoClient(base_url=target_url, workspace=workspace, auth=provider), "login for this run only (not saved)"
    )

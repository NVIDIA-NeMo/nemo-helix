# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-local tool authority, never derived from model-visible arguments.

The MCP boundary redeems an ephemeral Fabric capability with the loopback
broker before entering this context. Neither credentials nor capabilities belong
in tool arguments or conversation/checkpoint state. The stdio peer is trusted
code; this is not an authentication mechanism for a public MCP transport.
"""

import hashlib
import json
import os
import re
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from ipaddress import ip_address
from ssl import CERT_REQUIRED, SSLContext, create_default_context
from typing import Any
from urllib.parse import unquote

import httpx
from httpx import Client
from nemo_helix_ext.client.bootstrap import DEFAULT_RETRY_POLICY, resolve_timeout
from nemo_helix_ext.client.tls import client_certificate_authority_from_env
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.entity_naming import NAME_PATTERN

CONTEXT_KEY = "nhx_tool_authority"
WORKSPACE_HEADER = "x-nhx-tool-workspace"
STUDIO_SESSION_HEADER = "x-nhx-studio-session"
STUDIO_CAPABILITY_HEADER = "x-nhx-studio-capability"


@dataclass
class ToolAuthority:
    workspace: str
    headers: dict[str, str] = field(repr=False)
    studio_session: str | None = None
    studio_capability: str | None = field(default=None, repr=False)
    client: NemoClient | None = field(default=None, repr=False)


_authority: ContextVar[ToolAuthority | None] = ContextVar("nhx_tool_authority", default=None)


def authority(workspace: str | None = None, session: str | None = None) -> ToolAuthority:
    current = _authority.get()
    if current is None:
        raise PermissionError("Missing trusted caller context")
    if workspace is not None and workspace != current.workspace:
        raise PermissionError("Tool target differs from the invocation workspace")
    if session is not None and session != current.studio_session:
        raise PermissionError("Tool session differs from the invocation session")
    return current


@contextmanager
def tool_authority(value: Mapping[str, Any]) -> Generator[ToolAuthority]:
    workspace = value.get("workspace")
    headers = value.get("headers")
    if not isinstance(workspace, str) or not re.fullmatch(NAME_PATTERN, workspace):
        raise PermissionError("Missing trusted workspace binding")
    if not isinstance(headers, dict) or any(
        key not in {"authorization", "cookie"} or not isinstance(val, str) for key, val in headers.items()
    ):
        raise PermissionError("Invalid caller credentials")
    current = ToolAuthority(
        workspace=workspace,
        headers=dict(headers),
        studio_session=value.get("studio_session"),
        studio_capability=value.get("studio_capability"),
    )
    token = _authority.set(current)
    try:
        yield current
    finally:
        _authority.reset(token)
        if current.client is not None:
            current.client.close()


def authority_key() -> str:
    current = _authority.get()
    if current is None:
        return "missing-caller"
    return hashlib.sha256(
        json.dumps(
            [
                current.workspace,
                current.headers,
                current.studio_session,
                current.studio_capability,
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def check_callback_transport(url: str, current: ToolAuthority) -> None:
    target = httpx.URL(url)
    if not target.host or target.username or target.password:
        raise PermissionError("Callback URL must have a host and cannot supply credentials")
    try:
        loopback = target.host == "localhost" or ip_address(target.host).is_loopback
    except ValueError:
        loopback = False
    local_docker = (
        target.host == "host.docker.internal" and not current.headers and os.environ.get("NHX_STUDIO_LOCAL_HTTP") == "1"
    )
    if target.scheme != "https" and not (target.scheme == "http" and (loopback or local_docker)):
        raise PermissionError("Studio callback capabilities require HTTPS or trusted local HTTP")


def platform_ssl_context() -> SSLContext:
    """Require certificate and hostname validation, honoring configured CA roots."""
    ca_bundle = client_certificate_authority_from_env(os.environ.get("SSL_CERT_FILE"))
    context = create_default_context(cafile=ca_bundle)
    context.verify_mode = CERT_REQUIRED
    context.check_hostname = True
    return context


def caller_client(base_url: str, workspace: str) -> NemoClient:
    current = authority(workspace)
    if current.client is None:
        origin = httpx.URL(base_url)
        if origin.scheme not in {"http", "https"} or origin.username or origin.password:
            raise PermissionError("Platform URL must not supply ambient credentials")
        try:
            loopback = origin.host == "localhost" or ip_address(origin.host).is_loopback
        except ValueError:
            loopback = False
        if current.headers and origin.scheme != "https" and not loopback:
            raise PermissionError("Caller credentials require HTTPS or loopback HTTP")

        def check_target(request: httpx.Request) -> None:
            if request.url.username or request.url.password:
                raise PermissionError("Tool URL cannot supply credentials")
            if (request.url.scheme, request.url.host, request.url.port) != (origin.scheme, origin.host, origin.port):
                raise PermissionError("Tool request cannot change the platform origin")
            path = unquote(request.url.path)
            if any(segment in {".", ".."} for segment in path.split("/")):
                raise PermissionError("Tool URL cannot contain path traversal")
            target = re.match(r"^/apis/[^/]+/v[0-9]+/workspaces/([^/]+)(?:/|$)", path)
            if target is None and path.rstrip("/") != "/apis/entities/v2/workspaces":
                raise PermissionError("Tool URL has no workspace binding")
            if target is not None and target[1] != workspace:
                raise PermissionError("Tool URL targets another workspace")
            if any(
                key in {"workspace", "workspace_name", "workspace_id"} and val != workspace
                for key, val in request.url.params.multi_items()
            ):
                raise PermissionError("Tool query targets another workspace")

        # No config, OAuth discovery, ambient workload token, netrc, or redirect
        # fallback. An empty caller credential set remains empty on no-auth hosts.
        timeout = resolve_timeout(None)
        transport = Client(
            timeout=timeout,
            verify=platform_ssl_context(),
            follow_redirects=False,
            trust_env=False,
            event_hooks={"request": [check_target]},
        )
        current.client = NemoClient(
            base_url=base_url,
            workspace=workspace,
            default_headers=current.headers,
            timeout=timeout,
            retry=DEFAULT_RETRY_POLICY,
            http_client=transport,
            owns_http_client=True,
        )
    return current.client

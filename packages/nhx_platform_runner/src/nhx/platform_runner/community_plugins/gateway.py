# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static, authorized gateway to declared external plugin API endpoints.

The registry is manifest-only: no wheel installation, entry-point loading, or
workload launch. Clients can never choose an upstream URL in a proxy request.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from nemo_helix_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoClientError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.community_plugins.registry import ExternalPluginRegistry, RegistryStore
from nemo_helix_plugin.config import nhx_user_data_dir
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.service import NemoService, RouterSpec
from nhx.platform_runner.community_plugins.installation import build_installation_router
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)
scope = AuthzScope("plugins")
_CALLERS = [CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL]
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_REQUEST_HEADERS = {
    "accept",
    "content-type",
    "content-encoding",
    "mcp-session-id",
    "mcp-protocol-version",
    "last-event-id",
}
_RESPONSE_HEADERS = {"content-type", "mcp-session-id", "mcp-protocol-version", "retry-after"}


class PluginGatewayPerms(PermissionSet, namespace="plugins.gateway"):
    READ = perm("Read registered external plugin APIs")
    WRITE = perm("Write registered external plugin APIs")


class ExternalPluginsService(NemoService):
    name = "plugins"

    def __init__(self, registry_path: Path | None = None) -> None:
        self.store = RegistryStore(registry_path or nhx_user_data_dir() / "plugins" / "installed.json")
        # This client must not use platform auth defaults, netrc, proxy env, or
        # ambient client cookies. Build requests directly rather than merging a
        # shared client's evolving cookie jar into subsequent users' requests.
        self.http_client: httpx.AsyncClient | None = None

    async def on_startup(self) -> None:
        self.http_client = httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(30, read=120),
        )

    async def on_shutdown(self) -> None:
        if self.http_client is not None:
            await self.http_client.aclose()
            self.http_client = None

    def _client(self) -> httpx.AsyncClient:
        if self.http_client is None:
            raise HTTPException(503, "Plugin gateway is not running")
        return self.http_client

    def _registry(self) -> ExternalPluginRegistry:
        try:
            return self.store.load()
        except (OSError, ValueError, ValidationError) as exc:
            # Do not disclose local filesystem paths or registry contents.
            logger.error("External plugin registry is unavailable (%s)", type(exc).__name__)
            raise HTTPException(503, "Plugin registry is unavailable") from exc

    def get_routers(self) -> list[RouterSpec]:
        router = APIRouter()
        router.include_router(build_installation_router(self.store, self._client))

        @router.get("/-/installed", response_model=ExternalPluginRegistry)
        @scope.read
        @path_rule(callers=_CALLERS, permissions=[PluginGatewayPerms.READ])
        async def installed() -> ExternalPluginRegistry:
            return self._registry()

        @scope.read
        @path_rule(callers=_CALLERS, permissions=[PluginGatewayPerms.READ])
        async def read_plugin(
            request: Request, name: str, plugin_endpoint: str, client: AsyncNemoClient = Depends(get_nemo_client)
        ) -> Response:
            return await self._proxy(request, name, client)

        @scope.write
        @path_rule(callers=_CALLERS, permissions=[PluginGatewayPerms.WRITE])
        async def write_plugin(
            request: Request, name: str, plugin_endpoint: str, client: AsyncNemoClient = Depends(get_nemo_client)
        ) -> Response:
            return await self._proxy(request, name, client)

        @scope.read
        @path_rule(callers=_CALLERS, permissions=[PluginGatewayPerms.READ])
        async def read_workspace(
            request: Request,
            name: str,
            workspace: str,
            plugin_endpoint: str,
            client: AsyncNemoClient = Depends(get_nemo_client),
        ) -> Response:
            return await self._proxy(request, name, client)

        @scope.write
        @path_rule(callers=_CALLERS, permissions=[PluginGatewayPerms.WRITE])
        async def write_workspace(
            request: Request,
            name: str,
            workspace: str,
            plugin_endpoint: str,
            client: AsyncNemoClient = Depends(get_nemo_client),
        ) -> Response:
            return await self._proxy(request, name, client)

        # Explicit workspace-shaped patterns let OPA extract the workspace from
        # the path. A generic trailing wildcard alone would lose that context.
        for path, read, write in (
            ("/{name}/v2/workspaces/{workspace}/{plugin_endpoint:path}", read_workspace, write_workspace),
            ("/{name}/{plugin_endpoint:path}", read_plugin, write_plugin),
        ):
            # Explicit IDs keep verbs distinct even after the web SDK's
            # operation-name normalizer strips FastAPI's automatic suffixes.
            namespace = "workspace" if read is read_workspace else "platform"
            for method in ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"):
                router.add_api_route(
                    path,
                    read if method in {"GET", "HEAD"} else write,
                    methods=[method],
                    response_model=None,
                    operation_id=f"plugins_proxy_{namespace}_{method.lower()}",
                )
        return [RouterSpec(router, tag="External Plugins", description="Declared external plugin API gateway")]

    async def _proxy(self, request: Request, name: str, client: AsyncNemoClient) -> Response:
        if request.path_params.get("workspace") == "-":
            # Core wildcard-list semantics must not accidentally authorize an
            # external server's interpretation of '-' (potentially all tenants).
            raise HTTPException(400, "Wildcard workspaces are not supported by the plugin gateway")
        registry = self._registry()
        plugin = next((p for p in registry.plugins if p.name == name and p.enabled), None)
        if plugin is None:
            raise HTTPException(404, "Plugin is not installed or enabled")
        # Use matched path parameters, not query/header context or an assumed
        # deployment root_path. The specific route has consumed the workspace.
        path = f"/{request.path_params['plugin_endpoint']}"
        if workspace := request.path_params.get("workspace"):
            path = f"/v2/workspaces/{workspace}{path}"
        target = next(
            (
                url
                for endpoint in plugin.endpoints
                if endpoint.method == request.method
                if (url := endpoint.resolve(path)) is not None
            ),
            None,
        )
        if target is None:
            raise HTTPException(404, "Plugin endpoint is not declared")
        if self.http_client is None:
            raise HTTPException(503, "Plugin gateway is not running")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > _MAX_REQUEST_BYTES:
                raise HTTPException(413, "Plugin request is too large")
            body.extend(chunk)
        connection_headers = {value.strip().lower() for value in request.headers.get("connection", "").split(",")}
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() in _REQUEST_HEADERS and key.lower() not in connection_headers
        }
        # Resolve only the explicitly consented secret in the PATH workspace,
        # using the request-scoped typed client (never a service-wide identity).
        if plugin.credential is not None:
            workspace = request.path_params.get("workspace")
            if not workspace or workspace == "-":
                raise HTTPException(400, "Workspace-secret access requires a concrete workspace path")
            try:
                result = await AsyncSecretsClient.from_client(client).access_secret(
                    workspace=workspace,
                    name=plugin.credential.secret_name,
                )
                credential = result.data().value
            except NotFoundError as exc:
                raise HTTPException(412, "Configure the plugin credential in this workspace first") from exc
            except PermissionDeniedError as exc:
                raise HTTPException(403, "Not allowed to access this workspace's plugin credential") from exc
            except NemoClientError as exc:
                raise HTTPException(503, "Workspace plugin credential is unavailable") from exc
            if (
                not credential
                or len(credential) > 4096
                or not credential.isascii()
                or any(ord(character) < 33 or ord(character) > 126 for character in credential)
            ):
                raise HTTPException(412, "Workspace plugin credential has an invalid format")
            headers["authorization"] = f"Bearer {credential}"
        outgoing = httpx.Request(
            request.method,
            target,
            params=request.url.query,
            headers=headers,
            content=bytes(body),
        )
        try:
            upstream = await self.http_client.send(outgoing, stream=True, follow_redirects=False)
        except httpx.TimeoutException as exc:
            raise HTTPException(504, "Plugin upstream timed out") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Plugin upstream is unavailable") from exc
        if 300 <= upstream.status_code < 400:
            await upstream.aclose()
            raise HTTPException(502, "Plugin upstream redirects are not supported")
        connection_headers = {value.strip().lower() for value in upstream.headers.get("connection", "").split(",")}
        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() in _RESPONSE_HEADERS and key.lower() not in connection_headers
        }
        media_type = response_headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type not in {"application/json", "application/x-ndjson", "text/event-stream", "text/plain"} and not (
            media_type.startswith("application/") and media_type.endswith("+json")
        ):
            # This is an API gateway, not an unpinned same-origin JS/HTML host.
            # Bundle delivery needs its own digest-verified installation path.
            response_headers["content-type"] = "application/octet-stream"
            response_headers["content-disposition"] = "attachment"
        # No cookies, redirects, upstream CORS, content-encoding, or cache policy.
        # aiter_bytes decodes transport compression, so never relay its old length.
        response_headers.update(
            {
                "cache-control": "no-store",
                "x-content-type-options": "nosniff",
                "content-security-policy": "default-src 'none'; sandbox",
            }
        )

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream.aclose),
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HelixAdmin preview/consent/install lifecycle for external API/UI plugins."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import logging
import os
from collections.abc import Callable
from importlib.metadata import entry_points
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from nemo_helix_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.community_plugins.assets import MAX_BUNDLE_BYTES, AssetStore
from nemo_helix_plugin.community_plugins.manifest import MAX_MANIFEST_BYTES, parse_manifest_document
from nemo_helix_plugin.community_plugins.registry import (
    RegistryConflictError,
    RegistryNotFoundError,
    RegistryStore,
)
from nemo_helix_plugin.community_plugins.types import (
    TRUST_WARNING,
    CatalogEntry,
    CatalogResponse,
    EnableRequest,
    InstallRequest,
    InstallResponse,
    ManifestSource,
    PluginManifest,
    PreviewResponse,
)
from nhx.common.auth.dependencies import get_auth_client

logger = logging.getLogger(__name__)
scope = AuthzScope("plugins")
_DOWNLOAD_TIMEOUT_SECONDS = 30

__all__ = [
    "TRUST_WARNING",
    "CatalogEntry",
    "CatalogResponse",
    "EnableRequest",
    "InstallRequest",
    "InstallResponse",
    "ManifestSource",
    "PluginManifest",
    "PreviewResponse",
    "build_installation_router",
]


class InstallationPerms(PermissionSet, namespace="plugins.installations"):
    ADMIN = perm("Approve, install, enable, disable, and remove community plugins")
    READ = perm("Read the operator-provided community plugin catalog")


async def require_plugin_admin(request: Request) -> str:
    """Require a real human HelixAdmin; demo mode is explicit and loopback-only."""
    origin = request.headers.get("origin")
    if origin is not None:
        parsed = urlsplit(origin)
        if parsed.scheme != request.url.scheme or parsed.netloc != request.url.netloc:
            raise HTTPException(403, "Plugin administration requires a same-origin request")
    client = get_auth_client(request)
    if client.auth_enabled:
        if client.principal.caller_kind == "principal" and not client.principal.on_behalf_of:
            if await client.has_role("system", "HelixAdmin"):
                return client.principal.id
    elif (
        os.environ.get("NEMO_PLUGINS_LOCAL_ADMIN") == "true"
        and request.client is not None
        and request.url.hostname in {"localhost", "127.0.0.1", "::1"}
        and not any(header.lower().startswith(("forwarded", "x-forwarded-")) for header in request.headers)
    ):
        try:
            if ipaddress.ip_address(request.client.host).is_loopback:
                return "local-operator"
        except ValueError:
            pass
    raise HTTPException(403, "A human HelixAdmin is required (local demo administration is disabled by default)")


async def fetch_artifact(client: httpx.AsyncClient, url: str, limit: int) -> bytes:
    """Bound total wall time as well as the bytes of a slow publisher response."""
    try:
        async with asyncio.timeout(_DOWNLOAD_TIMEOUT_SECONDS):
            return await _fetch_artifact(client, url, limit)
    except TimeoutError as exc:
        raise HTTPException(504, "Plugin artifact download timed out") from exc


async def _fetch_artifact(client: httpx.AsyncClient, url: str, limit: int) -> bytes:
    """Download without platform auth, ambient cookies, or redirects."""
    try:
        response = await client.send(
            httpx.Request("GET", url, headers={"accept": "*/*", "accept-encoding": "identity"}),
            stream=True,
            follow_redirects=False,
        )
        try:
            if response.status_code != 200:
                raise HTTPException(502, "Plugin artifact did not return HTTP 200; redirects are not followed")
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise HTTPException(502, "Compressed plugin artifact downloads are not supported")
            data = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data) + len(chunk) > limit:
                    raise HTTPException(413, "Plugin artifact exceeds its size limit")
                data.extend(chunk)
            return bytes(data)
        finally:
            await response.aclose()
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Plugin artifact download timed out") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Plugin artifact download failed") from exc


async def preview(client: httpx.AsyncClient, source: ManifestSource) -> PreviewResponse:
    data = await fetch_artifact(client, source.url, MAX_MANIFEST_BYTES)
    try:
        manifest = PluginManifest.model_validate(parse_manifest_document(data))
    except ValueError as exc:
        raise HTTPException(422, "Invalid community plugin manifest") from exc
    return PreviewResponse(manifest=manifest, manifest_sha256=manifest.plugin.manifest_sha256)


def build_installation_router(store: RegistryStore, get_client: Callable[[], httpx.AsyncClient]) -> APIRouter:
    router = APIRouter()

    @router.get("/-/catalog", response_model=CatalogResponse)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[InstallationPerms.READ])
    async def catalog() -> CatalogResponse:
        try:
            with (store.path.parent / "catalog.json").open("rb") as stream:
                data = stream.read(65537)
            if len(data) > 65536:
                raise ValueError("Catalog exceeds its size limit")
            return CatalogResponse.model_validate_json(data)
        except FileNotFoundError:
            return CatalogResponse()
        except (OSError, ValueError) as exc:
            raise HTTPException(503, "Plugin catalog is unavailable") from exc

    @router.post("/-/preview", response_model=PreviewResponse)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[InstallationPerms.ADMIN])
    async def preview_plugin(body: ManifestSource, admin: str = Depends(require_plugin_admin)) -> PreviewResponse:
        return await preview(get_client(), body)

    @router.post("/-/installed", response_model=InstallResponse, status_code=201)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[InstallationPerms.ADMIN])
    async def install_plugin(body: InstallRequest, admin: str = Depends(require_plugin_admin)) -> InstallResponse:
        if not body.consent:
            raise HTTPException(
                422, "Explicit consent to plugin execution and declared endpoint/secret access is required"
            )
        reviewed = await preview(get_client(), body)
        if not hmac.compare_digest(reviewed.manifest_sha256, body.manifest_sha256):
            raise HTTPException(409, "Manifest changed since preview; review and consent again")
        plugin = reviewed.manifest.plugin
        # Protect installed Python/core plugin identities without loading their code.
        from nhx.platform_runner.registry import AVAILABLE_SERVICES

        reserved = set(AVAILABLE_SERVICES)
        reserved.add("plugins")
        # An existing API-only plugin (e.g. legacy Tavily) may coexist: its old
        # /apis/<name> namespace does not collide with /apis/plugins/<name>.
        reserved.update(ep.name for ep in entry_points(group="nemo.studio"))
        if plugin.name in reserved:
            raise HTTPException(409, "Plugin name is reserved by an installed platform capability")
        try:
            if any(p.name == plugin.name for p in store.load().plugins):
                raise HTTPException(409, "Plugin already installed; uninstall before replacing it")
        except (OSError, ValueError) as exc:
            raise HTTPException(503, "Plugin registry is unavailable") from exc
        if plugin.bundle is not None:
            data = await fetch_artifact(get_client(), plugin.bundle.url, MAX_BUNDLE_BYTES)
            try:
                await asyncio.to_thread(AssetStore(store).put, plugin, data)
            except ValueError as exc:
                raise HTTPException(422, "Plugin bundle failed hash verification") from exc
            except OSError as exc:
                raise HTTPException(503, "Plugin bundle could not be persisted") from exc
        try:
            await asyncio.to_thread(store.install, plugin)
        except RegistryConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(503, "Plugin registration could not be persisted") from exc
        logger.info(
            "Installed community plugin %s manifest=%s approved_by=%s", plugin.name, body.manifest_sha256, admin
        )
        return InstallResponse(name=plugin.name, manifest_sha256=plugin.manifest_sha256, bundle_url=plugin.bundle_url)

    @router.patch("/-/installed/{name}", status_code=204)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[InstallationPerms.ADMIN])
    async def enable_plugin(name: str, body: EnableRequest, admin: str = Depends(require_plugin_admin)) -> Response:
        try:
            await asyncio.to_thread(store.set_enabled, name, body.enabled)
        except RegistryNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(503, "Plugin registration could not be persisted") from exc
        logger.info("Community plugin %s enabled=%s approved_by=%s", name, body.enabled, admin)
        return Response(status_code=204)

    @router.delete("/-/installed/{name}", status_code=204)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[InstallationPerms.ADMIN])
    async def uninstall_plugin(name: str, admin: str = Depends(require_plugin_admin)) -> Response:
        try:
            await asyncio.to_thread(store.uninstall, name)
        except RegistryNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(503, "Plugin registration could not be persisted") from exc
        logger.info("Uninstalled community plugin %s approved_by=%s", name, admin)
        return Response(status_code=204)

    return router

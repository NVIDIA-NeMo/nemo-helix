# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Studio service implementation for serving the NeMo Studio UI."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import ClassVar

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse
from nemo_helix_plugin.authz import Permission
from nhx.common.http_clients import shared_async_http_client
from nhx.common.service import RouterConfig, Service
from nhx.studio import assistant
from nhx.studio.config import StudioConfig
from nhx.studio.plugins import build_plugins_router, discover_plugins
from nhx.studio.static_files import SPAStaticFiles, build_csp
from starlette.responses import Response

logger = logging.getLogger(__name__)

TELEMETRY_ALLOWED_HEADERS = "Accept,Accept-Language,Content-Encoding,Content-Language,Content-Type"
TELEMETRY_ALLOWED_METHODS = "POST, OPTIONS"
TELEMETRY_MAX_AGE_SECONDS = "1728000"
TELEMETRY_FORWARD_HEADERS = {
    "accept",
    "content-encoding",
    "content-type",
}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

CONTAINER_STATIC_FILES_PATH = Path("/static/studio")

SOURCE_CHECKOUT_TIPS_HTML = """      <h2>Build tips</h2>
      <p>Run these commands from the repository root.</p>
      <p>Studio uses the Node.js and pnpm engines in <code>web/package.json</code>.</p>
      <p>If you use nvm:</p>
      <pre>source ~/.nvm/nvm.sh
nvm install 22
nvm use 22
make bootstrap-studio
nemo services restart</pre>
      <p>If you use pnpm-managed Node.js:</p>
      <pre>pnpm env use --global 22.18.0
make bootstrap-studio
nemo services restart</pre>"""

DOCS_URL = "https://docs.nvidia.com/nemo-helix"

PACKAGED_INSTALL_NOTICE_HTML = f"""      <p>This install ships with the Studio bundle, so this is unexpected.</p>
      <p>See the <a href="{DOCS_URL}">NeMo Helix documentation</a> for help.</p>"""


def _bundle_asset_media_type(filename: str) -> str | None:
    """Media type for an allowed plugin bundle asset; None for disallowed suffixes."""
    if filename.endswith(".js.map"):
        return "application/json"
    if filename.endswith(".js"):
        return "text/javascript"
    if filename.endswith(".css"):
        return "text/css"
    return None


class StudioService(Service[StudioConfig]):
    """Studio service for serving the NeMo Studio UI static assets.

    This service mounts the built Vite application at /studio and serves
    the static files with SPA routing support.

    Configuration is managed through StudioConfig, which provides:
    - static_files_path: Path to the built UI assets
    - env_replacements: Runtime values to inject into the UI bundle (cached)
    """

    dependencies: ClassVar[list[str]] = ["entities", "auth"]

    def __init__(self):
        """Initialize the studio service."""
        super().__init__(name="studio", module_name="nhx.studio")

    @property
    def title(self) -> str:
        """Service title for OpenAPI docs."""
        return "NeMo Studio UI"

    @property
    def description(self) -> str:
        """Service description for OpenAPI docs."""
        return "Serves the NeMo Studio web application and local assistant bridge"

    def extra_permissions(self) -> list[Permission]:
        """Return permissions not attached to routes."""
        return []

    def extra_role_permissions(self) -> dict[str, list[Permission]]:
        """Return role grants beyond the default permission heuristic."""
        return {}

    def get_routers(self) -> list[RouterConfig]:
        """Return routers for the studio service.

        Studio exposes API routes for local-only UI integrations in addition to
        serving static files.
        """
        return [
            RouterConfig(
                assistant.router,
                tag="NeMo Assistant",
                description="Local assistant bridge endpoints",
            )
        ]

    def configure_app(self, app: FastAPI) -> None:
        """Configure the platform app with static file mounting.

        This method is called by the platform to allow the service to add
        custom routes, middleware, or mounts to the platform's FastAPI app.

        Args:
            app: The platform's FastAPI application
        """
        self._mount_telemetry_proxy(app)
        self._mount_assistant_mcp(app)
        self._mount_static_files(app)
        self._configure_plugins(app)

    def _mount_assistant_mcp(self, app: FastAPI) -> None:
        """Mount the auth-bypassed MCP callback before the /studio static app."""
        assistant.mount_public_mcp_route(app)

    def _get_config(self) -> StudioConfig:
        """Get the studio config, creating a default if none is set.

        Returns:
            StudioConfig instance (from service_config or a new default)
        """
        if self.service_config is not None:
            return self.service_config
        return StudioConfig()

    def _mount_telemetry_proxy(self, app: FastAPI) -> None:
        """Mount OTLP/HTTP telemetry proxy routes that replace the old nginx behavior."""

        async def proxy_telemetry(request: Request, telemetry_path: str = "") -> Response:
            return await self._proxy_telemetry(request, telemetry_path)

        for route in ("/telemetry", "/telemetry/{telemetry_path:path}"):
            app.add_api_route(route, proxy_telemetry, methods=["POST", "OPTIONS"], include_in_schema=False)

        for route in ("/studio/telemetry", "/studio/telemetry/{telemetry_path:path}"):
            app.add_api_route(route, proxy_telemetry, methods=["POST", "OPTIONS"], include_in_schema=False)

    async def _proxy_telemetry(self, request: Request, telemetry_path: str = "") -> Response:
        """Proxy browser OTLP/HTTP telemetry to the configured collector."""
        config = self._get_config()
        if not self._is_telemetry_enabled(config):
            return Response(status_code=404)

        origin = request.headers.get("origin", "")
        same_origin = self._same_origin(request)
        if not config.otel.is_origin_allowed(origin, same_origin=same_origin):
            return Response(status_code=403)

        cors_headers = self._telemetry_cors_headers(origin)
        if request.method == "OPTIONS":
            return Response(
                status_code=204, headers={**cors_headers, "Access-Control-Max-Age": TELEMETRY_MAX_AGE_SECONDS}
            )

        collector_url = self._collector_url(config)
        if not collector_url:
            logger.error("Studio telemetry proxy requested, but studio.otel.collector_url is not configured")
            return Response(status_code=500, headers=cors_headers)

        target_url = self._build_telemetry_target_url(collector_url, telemetry_path, request.url.query)
        try:
            upstream_response = await shared_async_http_client().request(
                method=request.method,
                url=target_url,
                content=await request.body(),
                headers=self._forward_headers(request),
            )
        except Exception as e:
            logger.warning(f"Failed to proxy Studio telemetry request: {e}")
            return Response(status_code=502, headers=cors_headers)

        response_headers = self._response_headers(upstream_response.headers)
        response_headers.update(cors_headers)
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    def _is_telemetry_enabled(self, config: StudioConfig) -> bool:
        """Return the configured telemetry enabled flag, including legacy nested config."""
        legacy_value = config._resolve_config_path("studio.telemetry.enabled")
        if legacy_value is not None:
            return legacy_value.lower() == "true"
        return config.telemetry_enabled

    def _collector_url(self, config: StudioConfig) -> str:
        """Return the configured internal collector URL."""
        if config.otel.collector_url:
            return config.otel.collector_url
        return config._resolve_config_path("studio.otel.collector_url") or ""

    def _same_origin(self, request: Request) -> str | None:
        """Return the externally visible request origin when it can be inferred."""
        host = request.headers.get("host")
        if not host:
            return None
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme).split(",", 1)[0].strip()
        return f"{scheme}://{host}"

    @staticmethod
    def _telemetry_cors_headers(origin: str) -> dict[str, str]:
        """CORS headers matching the previous nginx telemetry endpoint."""
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Headers": TELEMETRY_ALLOWED_HEADERS,
            "Access-Control-Allow-Methods": TELEMETRY_ALLOWED_METHODS,
            "Access-Control-Allow-Credentials": "true",
        }

    @staticmethod
    def _build_telemetry_target_url(collector_url: str, telemetry_path: str, query: str) -> str:
        """Build the upstream collector URL after stripping the telemetry route prefix."""
        target_path = f"/{telemetry_path.lstrip('/')}" if telemetry_path else "/"
        target_url = f"{collector_url.rstrip('/')}{target_path}"
        if query:
            target_url = f"{target_url}?{query}"
        return target_url

    @staticmethod
    def _forward_headers(request: Request) -> dict[str, str]:
        """Forward the OTLP headers the collector needs, plus proxy client IP headers."""
        headers = {key: value for key, value in request.headers.items() if key.lower() in TELEMETRY_FORWARD_HEADERS}
        if request.client and request.client.host:
            headers["X-Real-IP"] = request.client.host
            forwarded_for = request.headers.get("x-forwarded-for")
            headers["X-Forwarded-For"] = (
                f"{forwarded_for}, {request.client.host}" if forwarded_for else request.client.host
            )
        return headers

    @staticmethod
    def _response_headers(headers: Mapping[str, str]) -> dict[str, str]:
        """Forward safe response headers from the collector."""
        return {
            key: value
            for key, value in headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() in {"content-type", "content-encoding"}
        }

    def _build_csp_header(self) -> str:
        """CSP extended with the cross-origin endpoints the SPA is configured to reach."""
        replacements = self._get_config().env_replacements
        issuer = replacements.get("STUDIO_UI_VITE_AUTH_AUTHORITY", "")
        platform_base_url = replacements.get("STUDIO_UI_VITE_PLATFORM_BASE_URL", "")
        return build_csp(
            connect_src_urls=(
                issuer,
                platform_base_url,
                replacements.get("STUDIO_UI_VITE_DATA_STORE_MICROSERVICE_URL", ""),
                replacements.get("STUDIO_UI_VITE_NIM_PROXY_MICROSERVICE_URL", ""),
                replacements.get("STUDIO_UI_VITE_NIM_PROXY_MICROSERVICE_INTERNAL_URL", ""),
            ),
            frame_src_urls=(issuer,),  # oidc-client-ts silent-renew iframe
            script_src_urls=(platform_base_url,),  # dynamic import() of plugin bundles
        )

    def _mount_static_files(self, app: FastAPI) -> None:
        """Mount static files on the given FastAPI app.

        Args:
            app: FastAPI application to mount static files on
        """
        static_path = self._get_static_files_path()
        if self._static_assets_ready(static_path):
            app.mount(
                "/studio",
                SPAStaticFiles(
                    directory=str(static_path),
                    html=True,
                    env_replacements=self._get_config().env_replacements,
                    csp_header=self._build_csp_header(),
                ),
                name="studio-static",
            )
            logger.info(f"Mounted Studio UI static files from {static_path} at /studio")
        else:
            logger.warning(f"Static files not ready at {static_path}. The Studio UI will not be available.")
            self._mount_missing_static_files_notice(app, static_path)

    def _mount_missing_static_files_notice(self, app: FastAPI, static_path: Path) -> None:
        @app.get("/studio", include_in_schema=False)
        @app.get("/studio/", include_in_schema=False)
        @app.get("/studio/{path:path}", include_in_schema=False)
        async def studio_static_files_missing(path: str = "") -> HTMLResponse:
            return self._missing_static_files_response(
                static_path, path, source_checkout=self._source_static_files_path() is not None
            )

    @staticmethod
    def _missing_static_files_response(
        static_path: Path, requested_path: str = "", source_checkout: bool = True
    ) -> HTMLResponse:
        route = "/studio" if requested_path == "" else f"/studio/{requested_path}"
        recovery_html = SOURCE_CHECKOUT_TIPS_HTML if source_checkout else PACKAGED_INSTALL_NOTICE_HTML
        html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>NeMo Studio assets were not found</title>
  </head>
  <body>
    <main>
      <h1>NeMo Studio assets were not found</h1>
      <p>The platform is running, but Studio cannot be served because the built web assets were not found.</p>
      <p>Requested path: <code>{escape(route)}</code></p>
      <p>Expected assets at: <code>{escape(str(static_path))}</code></p>
{recovery_html}
    </main>
  </body>
</html>
"""
        return HTMLResponse(
            content=html,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Cache-Control": "no-store"},
        )

    def _configure_plugins(self, app: FastAPI) -> None:
        """Discover studio plugins and wire up their bundle assets and API endpoint."""
        manifests = discover_plugins()

        bundle_dirs: dict[str, Path] = {}
        for manifest in manifests:
            plugin_dir = manifest.bundle_dir
            if plugin_dir is None:
                logger.debug("Plugin %r has no web bundle — skipping bundle assets", manifest.name)
            elif plugin_dir.exists():
                bundle_dirs[manifest.name] = plugin_dir
                logger.info("Serving plugin bundle assets for %r at /plugin-ui/%s", manifest.name, manifest.name)
            else:
                logger.warning(
                    "Plugin %r bundle directory %r not found — bundle assets not served",
                    manifest.name,
                    plugin_dir,
                )

        @app.get("/plugin-ui/{plugin_name}/{filename}", include_in_schema=False)
        async def serve_plugin_asset(plugin_name: str, filename: str) -> FileResponse:
            # Allowlist: only direct children of the bundle dir with bundle-asset
            # suffixes — never the plugin's Python source or subdirectories.
            bundle_dir = bundle_dirs.get(plugin_name)
            media_type = _bundle_asset_media_type(filename)
            if bundle_dir is None or media_type is None or "/" in filename or "\\" in filename:
                raise HTTPException(status_code=404)
            # resolve() + parent check stops a symlink inside the bundle dir from
            # escaping it and serving an arbitrary file over this public route.
            file_path = (bundle_dir / filename).resolve()
            if file_path.parent != bundle_dir.resolve() or not file_path.is_file():
                raise HTTPException(status_code=404)
            return FileResponse(file_path, media_type=media_type)

        app.include_router(build_plugins_router(manifests))

    def _get_static_files_path(self) -> Path:
        """Get the path to the static files directory.

        Returns:
            The configured static_files_path from StudioConfig, falling back to the
            packaged `static/` directory, the container image bundle at
            `/static/studio`, or source checkout `web/packages/studio/dist`.
        """
        configured = self._get_config().static_files_path
        if configured is not None:
            return configured

        packaged_static = self._packaged_static_files_path()
        if self._static_assets_ready(packaged_static):
            return packaged_static

        container_static = self._container_static_files_path()
        if self._static_assets_ready(container_static):
            return container_static

        source_static = self._source_static_files_path()
        if source_static is not None:
            return source_static

        return packaged_static

    @staticmethod
    def _packaged_static_files_path() -> Path:
        """Return the package-local Studio static asset directory."""
        return Path(__file__).parent / "static"

    @staticmethod
    def _container_static_files_path() -> Path:
        """Return the Studio bundle location baked into NeMo Helix container images."""
        return CONTAINER_STATIC_FILES_PATH

    @staticmethod
    def _static_assets_ready(path: Path) -> bool:
        """Return True when a path looks like a built Studio UI bundle."""
        return (path / "index.html").is_file()

    @staticmethod
    def _source_static_files_path() -> Path | None:
        """Find source-built Studio UI assets in editable/source checkouts."""
        for start in (Path.cwd(), Path(__file__).resolve()):
            current = start if start.is_dir() else start.parent
            for candidate in (current, *current.parents):
                studio_package = candidate / "web" / "packages" / "studio"
                if (studio_package / "package.json").is_file():
                    return studio_package / "dist"
        return None

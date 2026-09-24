# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the StudioService."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.studio.config import StudioConfig
from nhx.studio.plugins import PluginManifestResponse
from nhx.studio.service import StudioService


class FakeTelemetryResponse:
    """Minimal upstream response used by telemetry proxy tests."""

    def __init__(self, status_code: int = 200, content: bytes = b"{}", headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {"content-type": "application/json"}


class FakeTelemetryClient:
    """Minimal async HTTP client used by telemetry proxy tests."""

    def __init__(self, response: FakeTelemetryResponse | None = None):
        self.response = response or FakeTelemetryResponse()
        self.calls: list[dict] = []

    async def request(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class TestStudioService:
    """Tests for the StudioService class."""

    def test_service_name(self):
        """Test that the service has the correct name."""
        service = StudioService()
        assert service.name == "studio"

    def test_service_title(self):
        """Test that the service has the correct title."""
        service = StudioService()
        assert service.title == "NeMo Studio UI"

    def test_service_description(self):
        """Test that the service has the correct description."""
        service = StudioService()
        assert service.description == "Serves the NeMo Studio web application and local assistant bridge"

    def test_get_routers_returns_assistant_router(self):
        """Test that the service exposes the local assistant API router."""
        service = StudioService()
        routers = service.get_routers()
        assert len(routers) == 1
        assert routers[0].tag == "NeMo Assistant"

    def test_module_name(self):
        """Test that the service has the correct module name."""
        service = StudioService()
        assert service.module_name == "nhx.studio"


class TestTelemetryProxy:
    """Tests for the Studio OTLP/HTTP telemetry proxy."""

    def _client(
        self,
        config: StudioConfig,
        monkeypatch: pytest.MonkeyPatch,
        fake_client: FakeTelemetryClient | None = None,
    ) -> tuple[TestClient, FakeTelemetryClient]:
        app = FastAPI()
        telemetry_client = fake_client or FakeTelemetryClient()
        monkeypatch.setattr("nhx.studio.service.shared_async_http_client", lambda: telemetry_client)
        StudioService().with_config(config).configure_app(app)
        return TestClient(app), telemetry_client

    def test_post_strips_studio_telemetry_prefix_and_proxies_request(self, monkeypatch: pytest.MonkeyPatch):
        """Test that /studio/telemetry/* proxies to the collector without the route prefix."""
        origin = "http://studio.test"
        client, telemetry_client = self._client(
            StudioConfig(
                telemetry_enabled=True,
                otel={"collector_url": "http://collector:4318", "allowed_origins": [origin]},
            ),
            monkeypatch,
        )

        response = client.post(
            "/studio/telemetry/v1/traces?timeout=1",
            content=b"payload",
            headers={"origin": origin, "content-type": "application/x-protobuf"},
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert len(telemetry_client.calls) == 1
        call = telemetry_client.calls[0]
        assert call["method"] == "POST"
        assert call["url"] == "http://collector:4318/v1/traces?timeout=1"
        assert call["content"] == b"payload"
        assert "origin" not in call["headers"]
        assert call["headers"]["content-type"] == "application/x-protobuf"
        assert call["headers"]["X-Real-IP"] == "testclient"
        assert call["headers"]["X-Forwarded-For"] == "testclient"

    def test_post_only_forwards_whitelisted_telemetry_headers(self, monkeypatch: pytest.MonkeyPatch):
        """Test that browser credentials and metadata are not forwarded to the collector."""
        origin = "http://studio.test"
        client, telemetry_client = self._client(
            StudioConfig(
                telemetry_enabled=True,
                otel={"collector_url": "http://collector:4318", "allowed_origins": [origin]},
            ),
            monkeypatch,
        )

        response = client.post(
            "/studio/telemetry/v1/logs",
            content=b"payload",
            headers={
                "origin": origin,
                "accept": "application/json",
                "authorization": "Bearer app-token",
                "content-encoding": "gzip",
                "content-type": "application/json",
                "cookie": "session=abc",
                "referer": "http://studio.test/workspaces",
                "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
                "x-auth-token": "token",
                "x-csrf-token": "csrf",
                "x-session-id": "session",
            },
        )

        assert response.status_code == 200
        assert telemetry_client.calls[0]["headers"] == {
            "accept": "application/json",
            "content-encoding": "gzip",
            "content-type": "application/json",
            "X-Real-IP": "testclient",
            "X-Forwarded-For": "testclient",
        }

    def test_post_strips_root_telemetry_prefix_and_proxies_request(self, monkeypatch: pytest.MonkeyPatch):
        """Test that /telemetry/* keeps parity with the old nginx route."""
        origin = "http://studio.test"
        client, telemetry_client = self._client(
            StudioConfig(
                telemetry_enabled=True,
                otel={"collector_url": "http://collector:4318", "allowed_origins": [origin]},
            ),
            monkeypatch,
        )

        response = client.post("/telemetry/v1/logs", headers={"origin": origin})

        assert response.status_code == 200
        assert telemetry_client.calls[0]["url"] == "http://collector:4318/v1/logs"

    def test_options_returns_preflight_response_without_proxying(self, monkeypatch: pytest.MonkeyPatch):
        """Test that CORS preflight requests are handled locally."""
        origin = "http://studio.test"
        client, telemetry_client = self._client(
            StudioConfig(
                telemetry_enabled=True,
                otel={"collector_url": "http://collector:4318", "allowed_origins": [origin]},
            ),
            monkeypatch,
        )

        response = client.options("/studio/telemetry/v1/traces", headers={"origin": origin})

        assert response.status_code == 204
        assert response.headers["access-control-allow-headers"] == (
            "Accept,Accept-Language,Content-Encoding,Content-Language,Content-Type"
        )
        assert response.headers["access-control-allow-methods"] == "POST, OPTIONS"
        assert response.headers["access-control-max-age"] == "1728000"
        assert telemetry_client.calls == []

    def test_disabled_telemetry_returns_404(self, monkeypatch: pytest.MonkeyPatch):
        """Test that disabled telemetry preserves the old nginx 404 behavior."""
        client, telemetry_client = self._client(
            StudioConfig(telemetry_enabled=False, otel={"collector_url": "http://collector:4318"}),
            monkeypatch,
        )

        response = client.post("/studio/telemetry/v1/traces", headers={"origin": "http://testserver"})

        assert response.status_code == 404
        assert telemetry_client.calls == []

    def test_disallowed_origin_returns_403(self, monkeypatch: pytest.MonkeyPatch):
        """Test that disallowed origins preserve the old nginx 403 behavior."""
        client, telemetry_client = self._client(
            StudioConfig(
                telemetry_enabled=True,
                otel={"collector_url": "http://collector:4318", "allowed_origins": ["http://studio.test"]},
            ),
            monkeypatch,
        )

        response = client.post("/studio/telemetry/v1/traces", headers={"origin": "http://not-allowed.test"})

        assert response.status_code == 403
        assert telemetry_client.calls == []

    def test_same_origin_request_is_allowed(self, monkeypatch: pytest.MonkeyPatch):
        """Test that same-origin Studio deployments work without hard-coded host config."""
        client, telemetry_client = self._client(
            StudioConfig(
                telemetry_enabled=True,
                otel={"collector_url": "http://collector:4318", "allowed_origins": []},
            ),
            monkeypatch,
        )

        response = client.post("/studio/telemetry/v1/traces", headers={"origin": "http://testserver"})

        assert response.status_code == 200
        assert telemetry_client.calls[0]["url"] == "http://collector:4318/v1/traces"


class TestStaticFilesPath:
    """Tests for static_files_path configuration."""

    def test_default_static_files_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that the default path is the packaged static dir."""
        import nhx.studio

        expected = Path(nhx.studio.__file__).parent / "static"
        service = StudioService()
        monkeypatch.setattr(service, "_source_static_files_path", lambda: None)
        monkeypatch.setattr(service, "_container_static_files_path", lambda: tmp_path / "absent")
        path = service._get_static_files_path()
        assert path == expected

    def test_configured_static_files_path_is_used(self, tmp_path: Path):
        """Test that a configured static_files_path is used."""
        # Create a temporary directory with a dummy file
        static_dir = tmp_path / "custom-static"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html></html>")

        # Create service with custom config
        config = StudioConfig(static_files_path=static_dir)
        service = StudioService().with_config(config)

        path = service._get_static_files_path()
        assert path == static_dir

    def test_static_files_path_config_default(self):
        """Test that StudioConfig has no configured static_files_path by default."""
        config = StudioConfig()
        assert config.static_files_path is None

    def test_static_files_path_can_be_set_by_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that env config can point Studio at source-built assets."""
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        monkeypatch.setenv("NHX_STUDIO_STATIC_FILES_PATH", str(static_dir))

        config = StudioConfig()

        assert config.static_files_path == static_dir

    def test_packaged_static_path_takes_precedence_over_source_dist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that packaged assets are preferred when they are available."""
        packaged_static = tmp_path / "package-static"
        packaged_static.mkdir()
        (packaged_static / "index.html").write_text("<html></html>")
        source_dist = tmp_path / "repo" / "web" / "packages" / "studio" / "dist"

        service = StudioService()
        monkeypatch.setattr(service, "_packaged_static_files_path", lambda: packaged_static)
        monkeypatch.setattr(service, "_source_static_files_path", lambda: source_dist)

        path = service._get_static_files_path()
        assert path == packaged_static

    def test_source_dist_used_when_packaged_static_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that source-built assets are used when package assets are absent."""
        source_root = tmp_path / "repo"
        studio_dir = source_root / "web" / "packages" / "studio"
        studio_dir.mkdir(parents=True)
        (studio_dir / "package.json").write_text('{"name":"nemo-studio-ui"}')

        packaged_static = tmp_path / "package-static"

        service = StudioService()
        monkeypatch.chdir(source_root)
        monkeypatch.setattr(service, "_packaged_static_files_path", lambda: packaged_static)
        monkeypatch.setattr(service, "_container_static_files_path", lambda: tmp_path / "absent")

        path = service._get_static_files_path()
        assert path == studio_dir / "dist"

    def test_container_bundle_used_when_packaged_static_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that the container image bundle is used when nothing is configured or packaged."""
        container_static = tmp_path / "static" / "studio"
        container_static.mkdir(parents=True)
        (container_static / "index.html").write_text("<html></html>")

        service = StudioService()
        monkeypatch.setattr(service, "_packaged_static_files_path", lambda: tmp_path / "package-static")
        monkeypatch.setattr(service, "_container_static_files_path", lambda: container_static)

        assert service._get_static_files_path() == container_static

    def test_configured_path_wins_over_container_bundle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that an operator's configured path is never shadowed by the container bundle."""
        container_static = tmp_path / "static" / "studio"
        container_static.mkdir(parents=True)
        (container_static / "index.html").write_text("<html></html>")
        configured = tmp_path / "operator-static"
        configured.mkdir()
        (configured / "index.html").write_text("<html></html>")

        service = StudioService().with_config(StudioConfig(static_files_path=configured))
        monkeypatch.setattr(service, "_container_static_files_path", lambda: container_static)

        assert service._get_static_files_path() == configured

    def test_default_container_static_files_path(self):
        """Test that the container fallback matches where the images place the bundle."""
        assert StudioService()._container_static_files_path() == Path("/static/studio")

    def test_env_static_files_path_shadows_the_config_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """ServiceConfig is environment-first, so images must not bake NHX_STUDIO_STATIC_FILES_PATH."""
        from nhx.common.config import Configuration

        monkeypatch.setenv("NHX_STUDIO_STATIC_FILES_PATH", str(tmp_path / "from-env"))

        config = Configuration.global_settings_to_service_config(
            {"studio": {"static_files_path": str(tmp_path / "from-yaml")}}, StudioConfig
        )

        assert config.static_files_path == tmp_path / "from-env"

    def test_missing_static_files_route_explains_recovery(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that missing Studio assets return recovery instructions instead of a bare 404."""
        missing_static = tmp_path / "missing-static"
        service = StudioService()
        monkeypatch.setattr(service, "_get_static_files_path", lambda: missing_static)
        monkeypatch.setattr(service, "_source_static_files_path", lambda: tmp_path / "web-dist")
        app = FastAPI()

        service.configure_app(app)

        client = TestClient(app)
        response = client.get("/studio/")

        assert response.status_code == 503
        assert "make bootstrap-studio" in response.text
        assert "source ~/.nvm/nvm.sh" in response.text
        assert "nvm install 22" in response.text
        assert "pnpm env use --global 22.18.0" in response.text
        assert "Node.js and pnpm engines" in response.text
        assert "web/package.json" in response.text
        assert "nemo services restart" in response.text
        assert str(missing_static) in response.text

    def test_missing_static_files_route_handles_main_studio_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that the main Studio path returns build tips when assets are missing."""
        missing_static = tmp_path / "missing-static"
        service = StudioService()
        monkeypatch.setattr(service, "_get_static_files_path", lambda: missing_static)
        monkeypatch.setattr(service, "_source_static_files_path", lambda: tmp_path / "web-dist")
        app = FastAPI()

        service.configure_app(app)

        client = TestClient(app)
        response = client.get("/studio")

        assert response.status_code == 503
        assert "Requested path: <code>/studio</code>" in response.text
        assert "Build tips" in response.text
        assert "make bootstrap-studio" in response.text
        assert "nemo services restart" in response.text

    def test_missing_static_files_route_omits_build_tips_outside_a_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that packaged installs get a docs pointer instead of repo build steps."""
        missing_static = tmp_path / "missing-static"
        service = StudioService()
        monkeypatch.setattr(service, "_get_static_files_path", lambda: missing_static)
        monkeypatch.setattr(service, "_source_static_files_path", lambda: None)
        app = FastAPI()

        service.configure_app(app)

        client = TestClient(app)
        response = client.get("/studio/")

        assert response.status_code == 503
        assert "https://docs.nvidia.com/nemo-helix" in response.text
        assert "make bootstrap-studio" not in response.text
        assert "nvm" not in response.text
        assert "NHX_STUDIO_STATIC_FILES_PATH" not in response.text
        assert str(missing_static) in response.text

    def test_static_dir_without_index_route_explains_recovery(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that an incomplete Studio build also returns recovery instructions."""
        incomplete_static = tmp_path / "static"
        incomplete_static.mkdir()
        service = StudioService()
        monkeypatch.setattr(service, "_get_static_files_path", lambda: incomplete_static)
        monkeypatch.setattr(service, "_source_static_files_path", lambda: tmp_path / "web-dist")
        app = FastAPI()

        service.configure_app(app)

        client = TestClient(app)
        response = client.get("/studio/")

        assert response.status_code == 503
        assert "make bootstrap-studio" in response.text
        assert "pnpm env use --global 22.18.0" in response.text
        assert str(incomplete_static) in response.text

    def test_missing_static_files_route_handles_nested_studio_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that nested Studio paths also return the recovery page when assets are missing."""
        missing_static = tmp_path / "missing-static"
        service = StudioService()
        monkeypatch.setattr(service, "_get_static_files_path", lambda: missing_static)
        app = FastAPI()

        service.configure_app(app)

        client = TestClient(app)
        response = client.get("/studio/agents")

        assert response.status_code == 503
        assert "/studio/agents" in response.text


class TestStudioConfigEnvReplacements:
    """Tests for StudioConfig.env_replacements property."""

    def test_env_replacements_returns_dict(self):
        """Test that env_replacements returns a dict."""
        config = StudioConfig()
        replacements = config.env_replacements
        assert isinstance(replacements, dict)

    def test_env_replacements_uses_defaults_when_no_global_settings(self, monkeypatch: pytest.MonkeyPatch):
        """Test that defaults from ENV_MAPPINGS are used when config paths can't be resolved."""
        # Mock get_global_settings_from_env to return empty dict
        from nhx.common import config as common_config
        from nhx.studio.env_mappings import ENV_MAPPINGS

        monkeypatch.setattr(common_config.Configuration, "get_global_settings_from_env", lambda: {})

        # Create a fresh config after mocking
        config = StudioConfig()
        replacements = config.env_replacements

        # Check that mappings with defaults have their default values
        for mapping in ENV_MAPPINGS:
            if mapping.default:
                assert mapping.marker in replacements
                assert replacements[mapping.marker] == mapping.default

    def test_env_replacements_resolves_config_paths(self, monkeypatch: pytest.MonkeyPatch):
        """Test that config paths are correctly resolved from global settings."""
        # Mock get_global_settings_from_env to return known values
        mock_settings = {
            "studio": {
                "platform_base_url": "http://test.example.com",
                "telemetry_enabled": "true",
            },
        }
        from nhx.common import config as common_config

        monkeypatch.setattr(common_config.Configuration, "get_global_settings_from_env", lambda: mock_settings)

        # Create a fresh config after mocking
        config = StudioConfig()
        replacements = config.env_replacements

        # Verify the studio.platform_base_url was resolved
        assert "STUDIO_UI_VITE_PLATFORM_BASE_URL" in replacements
        assert replacements["STUDIO_UI_VITE_PLATFORM_BASE_URL"] == "http://test.example.com"

        # Verify the studio.telemetry_enabled was resolved
        assert "STUDIO_UI_VITE_TELEMETRY_ENABLED" in replacements
        assert replacements["STUDIO_UI_VITE_TELEMETRY_ENABLED"] == "true"

    def test_env_replacements_publishes_oidc_bearer_token_source(self, monkeypatch: pytest.MonkeyPatch):
        """Test that Studio receives the configured OIDC bearer token source."""
        from nhx.common import config as common_config

        monkeypatch.setattr(
            common_config.Configuration,
            "get_global_settings_from_env",
            lambda: {"auth": {"oidc": {"bearer_token_source": "id_token"}}},
        )

        replacements = StudioConfig().env_replacements

        assert replacements["STUDIO_UI_VITE_AUTH_BEARER_TOKEN_SOURCE"] == "id_token"

    def test_env_replacements_empty_global_setting_falls_back_to_config_field(self, monkeypatch: pytest.MonkeyPatch):
        """Test that empty global settings do not block StudioConfig field fallback."""
        mock_settings = {"studio": {"platform_base_url": ""}}
        from nhx.common import config as common_config

        monkeypatch.setattr(common_config.Configuration, "get_global_settings_from_env", lambda: mock_settings)

        config = StudioConfig(platform_base_url="http://fallback.example.com")
        replacements = config.env_replacements

        assert replacements["STUDIO_UI_VITE_PLATFORM_BASE_URL"] == "http://fallback.example.com"

    def test_env_replacements_is_cached(self, monkeypatch: pytest.MonkeyPatch):
        """Test that env_replacements is cached and not recomputed."""
        call_count = 0

        def mock_get_settings():
            nonlocal call_count
            call_count += 1
            return {"platform": {"base_url": "http://test.example.com"}}

        from nhx.common import config as common_config

        monkeypatch.setattr(common_config.Configuration, "get_global_settings_from_env", mock_get_settings)

        config = StudioConfig()

        # Access env_replacements multiple times
        _ = config.env_replacements
        _ = config.env_replacements
        _ = config.env_replacements

        # Should only have called get_global_settings_from_env once (during first access)
        assert call_count == 1

    def test_platform_base_url_falls_back_to_platform_base_url(self, monkeypatch: pytest.MonkeyPatch):
        """When studio.platform_base_url is blank, platform.base_url is used."""
        mock_settings = {
            "platform": {"base_url": "http://0.0.0.0:8080"},
            "studio": {},
        }
        from nhx.common import config as common_config

        monkeypatch.setattr(common_config.Configuration, "get_global_settings_from_env", lambda: mock_settings)

        config = StudioConfig()
        replacements = config.env_replacements

        assert replacements["STUDIO_UI_VITE_PLATFORM_BASE_URL"] == "http://0.0.0.0:8080"

    def test_platform_base_url_studio_value_takes_precedence(self, monkeypatch: pytest.MonkeyPatch):
        """An explicit studio.platform_base_url wins over the platform-level fallback."""
        mock_settings = {
            "platform": {"base_url": "http://0.0.0.0:8080"},
            "studio": {"platform_base_url": "https://studio.example.com"},
        }
        from nhx.common import config as common_config

        monkeypatch.setattr(common_config.Configuration, "get_global_settings_from_env", lambda: mock_settings)

        config = StudioConfig()
        replacements = config.env_replacements

        assert replacements["STUDIO_UI_VITE_PLATFORM_BASE_URL"] == "https://studio.example.com"

    def test_resolve_config_path_nested(self):
        """Test that _resolve_config_path handles nested paths."""
        config = StudioConfig()

        # Set a known value for global_settings
        config.__dict__["global_settings"] = {"level1": {"level2": {"level3": "deep_value"}}}

        result = config._resolve_config_path("level1.level2.level3")
        assert result == "deep_value"

    def test_resolve_config_path_returns_none_for_missing(self):
        """Test that _resolve_config_path returns None for missing paths."""
        config = StudioConfig()

        # Set empty global settings
        config.__dict__["global_settings"] = {}

        result = config._resolve_config_path("nonexistent.path")
        assert result is None


class TestConfigureAppPluginRouter:
    """Tests that configure_app wires plugin static files and /apis/plugins."""

    def test_plugins_endpoint_is_registered(self):
        manifests = [PluginManifestResponse(name="ex", bundle_url="/plugin-ui/ex/index.js")]
        with patch("nhx.studio.service.discover_plugins", return_value=manifests):
            service = StudioService()
            app = FastAPI()
            service.configure_app(app)

        client = TestClient(app)
        response = client.get("/apis/plugins")
        assert response.status_code == 200
        data = response.json()
        assert data[0]["name"] == "ex"

    def test_bundle_assets_served_for_each_plugin(self, tmp_path: Path):
        bundle = tmp_path / "index.js"
        bundle.write_text("export function mount(){}")
        (tmp_path / "index.js.map").write_text("{}")
        (tmp_path / "styles.css").write_text("body{}")
        manifests = [PluginManifestResponse(name="ex", bundle_url="/plugin-ui/ex/index.js", bundle_dir=tmp_path)]

        with patch("nhx.studio.service.discover_plugins", return_value=manifests):
            service = StudioService()
            app = FastAPI()
            service.configure_app(app)

        client = TestClient(app)
        response = client.get("/plugin-ui/ex/index.js")
        assert response.status_code == 200
        assert response.text == "export function mount(){}"
        assert response.headers["content-type"].startswith("text/javascript")
        assert client.get("/plugin-ui/ex/index.js.map").status_code == 200
        assert client.get("/plugin-ui/ex/styles.css").status_code == 200

    def test_bundle_assets_allowlist_blocks_non_bundle_files(self, tmp_path: Path):
        """Only direct .js/.js.map/.css children of the bundle dir are reachable."""
        (tmp_path / "index.js").write_text("// bundle")
        (tmp_path / "secrets.py").write_text("TOKEN = 'x'")
        (tmp_path / "config.yaml").write_text("key: value")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "nested.js").write_text("// nested")
        manifests = [PluginManifestResponse(name="ex", bundle_url="/plugin-ui/ex/index.js", bundle_dir=tmp_path)]

        with patch("nhx.studio.service.discover_plugins", return_value=manifests):
            service = StudioService()
            app = FastAPI()
            service.configure_app(app)

        client = TestClient(app)
        assert client.get("/plugin-ui/ex/secrets.py").status_code == 404
        assert client.get("/plugin-ui/ex/config.yaml").status_code == 404
        assert client.get("/plugin-ui/ex/sub/nested.js").status_code == 404
        assert client.get("/plugin-ui/ex/missing.js").status_code == 404
        assert client.get("/plugin-ui/unknown/index.js").status_code == 404

    def test_bundle_asset_symlink_escape_blocked(self, tmp_path: Path):
        """A .js symlink inside the bundle dir must not serve a file outside it."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "index.js").write_text("// ok")
        secret = tmp_path / "outside.js"
        secret.write_text("SECRET")
        (bundle / "leak.js").symlink_to(secret)
        manifests = [PluginManifestResponse(name="ex", bundle_url="/plugin-ui/ex/index.js", bundle_dir=bundle)]

        with patch("nhx.studio.service.discover_plugins", return_value=manifests):
            service = StudioService()
            app = FastAPI()
            service.configure_app(app)

        client = TestClient(app)
        assert client.get("/plugin-ui/ex/index.js").status_code == 200
        assert client.get("/plugin-ui/ex/leak.js").status_code == 404


class TestBuildCSPHeader:
    """_build_csp_header wires configured STUDIO_UI_* endpoints into CSP directives."""

    @staticmethod
    def _directive(csp: str, name: str) -> str:
        for part in csp.split(";"):
            part = part.strip()
            if part.startswith(f"{name} "):
                return part[len(name) + 1 :]
        raise AssertionError(f"directive {name!r} not found in {csp!r}")

    def _csp_for(self, replacements: dict[str, str]) -> str:
        service = StudioService()
        with patch.object(service, "_get_config", return_value=SimpleNamespace(env_replacements=replacements)):
            return service._build_csp_header()

    def test_empty_replacements_is_same_origin(self):
        csp = self._csp_for({})
        assert self._directive(csp, "connect-src") == "'self'"
        assert self._directive(csp, "script-src") == "'self'"
        assert self._directive(csp, "frame-src") == "'none'"

    def test_issuer_reaches_connect_and_frame_src(self):
        csp = self._csp_for({"STUDIO_UI_VITE_AUTH_AUTHORITY": "https://issuer.example.com/realms/nhx"})
        assert self._directive(csp, "connect-src") == "'self' https://issuer.example.com"
        assert self._directive(csp, "frame-src") == "'self' https://issuer.example.com"

    def test_platform_base_url_reaches_connect_and_script_src(self):
        csp = self._csp_for({"STUDIO_UI_VITE_PLATFORM_BASE_URL": "https://api.example.com"})
        assert self._directive(csp, "connect-src") == "'self' https://api.example.com"
        assert self._directive(csp, "script-src") == "'self' https://api.example.com"

    def test_microservice_urls_reach_connect_src_only(self):
        csp = self._csp_for(
            {
                "STUDIO_UI_VITE_DATA_STORE_MICROSERVICE_URL": "https://ds.example.com",
                "STUDIO_UI_VITE_NIM_PROXY_MICROSERVICE_URL": "https://nim.example.com",
            }
        )
        assert self._directive(csp, "connect-src") == "'self' https://ds.example.com https://nim.example.com"
        assert self._directive(csp, "script-src") == "'self'"
        assert self._directive(csp, "frame-src") == "'none'"

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SPAStaticFiles handler."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.studio.static_files import DEFAULT_CSP, SPAStaticFiles, build_csp


@pytest.fixture()
def static_dir(tmp_path: Path) -> Path:
    """Minimal static directory with index.html and an asset."""
    index = tmp_path / "index.html"
    index.write_text("<html><body>hello</body></html>", encoding="utf-8")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "main.js").write_text("console.log('hi')", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def client(static_dir: Path) -> TestClient:
    app = FastAPI()
    app.mount(
        "/studio",
        SPAStaticFiles(directory=str(static_dir), html=True, csp_header=DEFAULT_CSP),
        name="studio-static",
    )
    return TestClient(app, raise_server_exceptions=True)


class TestCSPHeader:
    """CSP must be present on every HTML response, with or without preprocessing."""

    def test_root_path_has_csp(self, client: TestClient):
        response = client.get("/studio/")
        assert response.status_code == 200
        assert "Content-Security-Policy" in response.headers

    def test_index_html_direct_has_csp(self, client: TestClient):
        response = client.get("/studio/index.html")
        assert response.status_code == 200
        assert "Content-Security-Policy" in response.headers

    def test_csp_value_matches_default(self, client: TestClient):
        response = client.get("/studio/")
        assert response.headers["Content-Security-Policy"] == DEFAULT_CSP

    def test_js_asset_has_no_csp(self, client: TestClient):
        response = client.get("/studio/assets/main.js")
        assert response.status_code == 200
        assert "Content-Security-Policy" not in response.headers

    def test_inline_importmap_hash_added_to_script_src(self, tmp_path: Path):
        """Inline <script type="importmap"> content must be authorized via sha256 in CSP."""
        import base64
        import hashlib

        importmap_content = '{"imports":{"react":"/vendor/react.js"}}'
        index = tmp_path / "index.html"
        index.write_text(
            f'<html><head><script type="importmap">{importmap_content}</script></head></html>',
            encoding="utf-8",
        )
        app = FastAPI()
        app.mount(
            "/studio",
            SPAStaticFiles(directory=str(tmp_path), html=True, csp_header=DEFAULT_CSP),
            name="studio-static",
        )
        c = TestClient(app)
        response = c.get("/studio/")
        csp = response.headers["Content-Security-Policy"]
        expected_hash = base64.b64encode(hashlib.sha256(importmap_content.encode()).digest()).decode()
        assert f"'sha256-{expected_hash}'" in csp
        assert "script-src 'self' 'sha256-" in csp

    def test_no_csp_when_disabled(self, static_dir: Path):
        app = FastAPI()
        app.mount(
            "/studio",
            SPAStaticFiles(directory=str(static_dir), html=True, csp_header=None),
            name="studio-static",
        )
        c = TestClient(app)
        response = c.get("/studio/")
        assert response.status_code == 200
        assert "Content-Security-Policy" not in response.headers


def _directive(csp: str, name: str) -> str:
    """Return the value of the named CSP directive, e.g. _directive(csp, 'script-src')."""
    for part in csp.split(";"):
        part = part.strip()
        if part.startswith(f"{name} "):
            return part[len(name) + 1 :]
    raise AssertionError(f"directive {name!r} not found in {csp!r}")


class TestBuildCSP:
    """build_csp folds configured cross-origin endpoints into the right directives."""

    def test_no_args_is_fully_same_origin(self):
        csp = build_csp()
        assert csp == DEFAULT_CSP
        assert _directive(csp, "connect-src") == "'self'"
        assert _directive(csp, "script-src") == "'self'"
        assert _directive(csp, "frame-src") == "'none'"

    def test_connect_src_url_contributes_origin(self):
        csp = build_csp(connect_src_urls=("https://api.example.com",))
        assert _directive(csp, "connect-src") == "'self' https://api.example.com"
        assert _directive(csp, "script-src") == "'self'"

    def test_script_src_url_contributes_origin(self):
        csp = build_csp(script_src_urls=("https://cdn.example.com",))
        assert _directive(csp, "script-src") == "'self' https://cdn.example.com"

    def test_frame_src_switches_from_none_to_self_plus_origin(self):
        csp = build_csp(frame_src_urls=("https://issuer.example.com",))
        assert _directive(csp, "frame-src") == "'self' https://issuer.example.com"

    def test_only_scheme_host_port_kept_not_path_or_query(self):
        """A path/query in a configured URL must not leak into the directive."""
        csp = build_csp(connect_src_urls=("https://api.example.com:8443/apis/v2?x=1",))
        assert _directive(csp, "connect-src") == "'self' https://api.example.com:8443"

    def test_empty_and_relative_urls_contribute_nothing(self):
        csp = build_csp(
            connect_src_urls=("", "/apis/plugins", "not-a-url", "ftp://x/y"),
            script_src_urls=("",),
            frame_src_urls=("",),
        )
        assert _directive(csp, "connect-src") == "'self'"
        assert _directive(csp, "script-src") == "'self'"
        assert _directive(csp, "frame-src") == "'none'"

    def test_origins_deduped_preserving_order(self):
        csp = build_csp(
            connect_src_urls=(
                "https://a.example.com/one",
                "https://b.example.com",
                "https://a.example.com/two",
            )
        )
        assert _directive(csp, "connect-src") == "'self' https://a.example.com https://b.example.com"


class TestHasFileExtension:
    """Tests for the _has_file_extension method."""

    def test_nested_path_with_extension(self):
        assert SPAStaticFiles._has_file_extension("assets/main.12345.js") is True

    def test_nested_path_without_extension(self):
        assert SPAStaticFiles._has_file_extension("workspaces/123/models") is False

    def test_path_with_trailing_slash(self):
        assert SPAStaticFiles._has_file_extension("dashboard/") is False

    def test_hidden_file_has_extension(self):
        # .gitignore should be considered as having an extension
        assert SPAStaticFiles._has_file_extension(".gitignore") is True

    def test_empty_path(self):
        assert SPAStaticFiles._has_file_extension("") is False

    def test_root_path(self):
        assert SPAStaticFiles._has_file_extension("/") is False

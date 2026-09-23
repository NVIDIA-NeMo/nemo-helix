# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nhx.studio.plugins."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_helix_plugin.interface import StudioSpec
from nhx.studio.plugins import PluginManifestResponse, build_plugins_router, discover_plugins


@pytest.fixture(autouse=True)
def clear_discover_cache():
    """Clear the @cache on discover_plugins between tests."""
    discover_plugins.cache_clear()
    yield
    discover_plugins.cache_clear()


def _eps(*names: str) -> dict[str, object]:
    """Fake discover_entry_points result: name → sentinel object."""
    return {n: object() for n in names}


def _manifests(*names: str) -> dict[str, object]:
    """Fake discover_manifests result: plugin name → sentinel manifest."""
    return {n: object() for n in names}


class TestDiscoverPlugins:
    def test_returns_empty_list_when_no_plugins_installed(self):
        with patch("nhx.studio.plugins.discover_manifests", return_value={}):
            with patch("nhx.studio.plugins.discover_studio", return_value={}):
                result = discover_plugins()
        assert result == []

    def test_plugin_without_studio_entry_appears_with_null_bundle_url(self):
        """A plugin installed via nemo.services (etc.) with no nemo.studio entry shows up."""
        with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("agents")):
            with patch("nhx.studio.plugins.discover_studio", return_value={}):
                result = discover_plugins()

        assert len(result) == 1
        assert result[0].name == "agents"
        assert result[0].bundle_url is None

    def test_jobs_only_plugin_appears_in_manifest(self):
        """discover_manifests() is the source of truth, so a plugin that only
        registers dot-scoped surfaces (nemo.jobs/nemo.functions) is listed."""
        with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("automodel")):
            with patch("nhx.studio.plugins.discover_studio", return_value={}):
                result = discover_plugins()

        assert [m.name for m in result] == ["automodel"]
        assert result[0].bundle_url is None

    def test_plugin_with_studio_entry_gets_bundle_url(self, tmp_path: Path):
        bundle_file = tmp_path / "index.js"
        bundle_file.write_text("// bundle")
        spec = StudioSpec(name="example", bundle_path=bundle_file)
        mock_factory = Mock(return_value=spec)

        with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("example")):
            with patch("nhx.studio.plugins.discover_entry_points", return_value=_eps("example")):
                with patch("nhx.studio.plugins.discover_studio", return_value={"example": mock_factory}):
                    result = discover_plugins()

        mock_factory.assert_called_once()
        assert len(result) == 1
        assert result[0].name == "example"
        assert result[0].bundle_url == "/plugin-ui/example/index.js"
        assert result[0].bundle_dir == tmp_path.resolve()

    def test_failing_studio_factory_falls_back_to_null_bundle_url(self, caplog):
        broken_factory = Mock(side_effect=RuntimeError("oops"))

        with caplog.at_level(logging.WARNING, logger="nhx.studio.plugins"):
            with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("bad-plugin")):
                with patch("nhx.studio.plugins.discover_studio", return_value={"bad-plugin": broken_factory}):
                    result = discover_plugins()

        assert len(result) == 1
        assert result[0].name == "bad-plugin"
        assert result[0].bundle_url is None
        assert "bad-plugin" in caplog.text

    def test_plugin_without_bundle_path_gets_null_bundle_url(self):
        spec = StudioSpec(name="headless-plugin")
        mock_factory = Mock(return_value=spec)

        with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("headless-plugin")):
            with patch("nhx.studio.plugins.discover_studio", return_value={"headless-plugin": mock_factory}):
                result = discover_plugins()

        assert len(result) == 1
        assert result[0].name == "headless-plugin"
        assert result[0].bundle_url is None
        assert result[0].bundle_dir is None

    def test_bundle_url_derived_from_entry_point_key_not_bundle_path(self, tmp_path: Path):
        """bundleUrl is constructed from the entry-point key, never from spec.bundle_path."""
        # bundle resides in a dir with a different name to confirm URL uses the ep key
        bundle_dir = tmp_path / "some-other-dir"
        bundle_dir.mkdir()
        bundle_file = bundle_dir / "index.js"
        bundle_file.write_text("// bundle")
        spec = StudioSpec(name="my-plugin", bundle_path=bundle_file)
        mock_factory = Mock(return_value=spec)

        with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("my-plugin")):
            with patch("nhx.studio.plugins.discover_entry_points", return_value=_eps("my-plugin")):
                with patch("nhx.studio.plugins.discover_studio", return_value={"my-plugin": mock_factory}):
                    result = discover_plugins()

        assert result[0].bundle_url == "/plugin-ui/my-plugin/index.js"
        assert result[0].bundle_dir == bundle_dir.resolve()

    def test_spec_name_mismatch_suppresses_bundle(self, caplog):
        """A StudioSpec whose name differs from its entry-point key is rejected."""
        spec = StudioSpec(name="other-plugin", bundle_path=Path("/some/index.js"))
        mock_factory = Mock(return_value=spec)

        with caplog.at_level(logging.WARNING, logger="nhx.studio.plugins"):
            with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("my-plugin")):
                with patch("nhx.studio.plugins.discover_studio", return_value={"my-plugin": mock_factory}):
                    result = discover_plugins()

        assert result[0].name == "my-plugin"
        assert result[0].bundle_url is None
        assert "must match the entry-point key" in caplog.text

    def test_bundle_path_nonexistent_file_suppresses_bundle(self, tmp_path: Path, caplog):
        """A bundle_path that does not point to an existing file is rejected."""
        spec = StudioSpec(name="my-plugin", bundle_path=tmp_path / "index.js")
        mock_factory = Mock(return_value=spec)

        with caplog.at_level(logging.WARNING, logger="nhx.studio.plugins"):
            with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("my-plugin")):
                with patch("nhx.studio.plugins.discover_studio", return_value={"my-plugin": mock_factory}):
                    result = discover_plugins()

        assert result[0].bundle_url is None
        assert "does not point to a regular file" in caplog.text

    def test_bundle_path_outside_dist_root_suppresses_bundle(self, tmp_path: Path, caplog):
        """A bundle_path that resolves outside the distribution root is rejected."""
        dist_root = tmp_path / "my-plugin-dist"
        dist_root.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "index.js"
        outside_file.write_text("// evil bundle")

        spec = StudioSpec(name="my-plugin", bundle_path=outside_file)
        mock_factory = Mock(return_value=spec)

        mock_dist = Mock()
        mock_dist.locate_file.return_value = dist_root
        mock_ep = Mock()
        mock_ep.dist = mock_dist

        with caplog.at_level(logging.WARNING, logger="nhx.studio.plugins"):
            with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("my-plugin")):
                with patch("nhx.studio.plugins.discover_entry_points", return_value={"my-plugin": mock_ep}):
                    with patch("nhx.studio.plugins.discover_studio", return_value={"my-plugin": mock_factory}):
                        result = discover_plugins()

        assert result[0].bundle_url is None
        assert "outside distribution root" in caplog.text

    def test_bundle_path_within_editable_source_root_is_accepted(self, tmp_path: Path):
        """PEP 660 editable install: bundle under direct_url.json's source dir is accepted."""
        dist_root = tmp_path / "site-packages"
        dist_root.mkdir()
        source_root = tmp_path / "source-tree"
        source_root.mkdir()
        bundle_file = source_root / "web" / "dist" / "index.js"
        bundle_file.parent.mkdir(parents=True)
        bundle_file.write_text("// bundle")

        spec = StudioSpec(name="my-plugin", bundle_path=bundle_file)
        mock_factory = Mock(return_value=spec)

        mock_dist = Mock()
        mock_dist.locate_file.return_value = dist_root
        mock_dist.read_text.return_value = '{"url": "file://' + str(source_root) + '", "dir_info": {"editable": true}}'
        mock_ep = Mock()
        mock_ep.dist = mock_dist

        with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("my-plugin")):
            with patch("nhx.studio.plugins.discover_entry_points", return_value={"my-plugin": mock_ep}):
                with patch("nhx.studio.plugins.discover_studio", return_value={"my-plugin": mock_factory}):
                    result = discover_plugins()

        assert result[0].bundle_url == "/plugin-ui/my-plugin/index.js"

    def test_bundle_path_within_dist_root_is_accepted(self, tmp_path: Path):
        """A bundle_path within the distribution root gets a bundle URL."""
        dist_root = tmp_path / "my-plugin-dist"
        dist_root.mkdir()
        bundle_file = dist_root / "web" / "dist" / "index.js"
        bundle_file.parent.mkdir(parents=True)
        bundle_file.write_text("// bundle")

        spec = StudioSpec(name="my-plugin", bundle_path=bundle_file)
        mock_factory = Mock(return_value=spec)

        mock_dist = Mock()
        mock_dist.locate_file.return_value = dist_root
        mock_ep = Mock()
        mock_ep.dist = mock_dist

        with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("my-plugin")):
            with patch("nhx.studio.plugins.discover_entry_points", return_value={"my-plugin": mock_ep}):
                with patch("nhx.studio.plugins.discover_studio", return_value={"my-plugin": mock_factory}):
                    result = discover_plugins()

        assert result[0].bundle_url == "/plugin-ui/my-plugin/index.js"
        assert result[0].bundle_dir == bundle_file.parent.resolve()

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "../etc/passwd",
            "UpperCase",
            "has spaces",
            "has/slash",
            "has%2F",
            "",
            "a",  # single character (too short for [a-z][a-z0-9-]+)
        ],
    )
    def test_invalid_spec_name_suppresses_bundle_but_plugin_still_listed(self, invalid_name: str, caplog):
        """Invalid spec.name prevents bundle mounting but the plugin still appears."""
        spec = StudioSpec(name=invalid_name, bundle_path=Path("/some/index.js"))
        mock_factory = Mock(return_value=spec)

        with caplog.at_level(logging.WARNING, logger="nhx.studio.plugins"):
            with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("entry")):
                with patch("nhx.studio.plugins.discover_studio", return_value={"entry": mock_factory}):
                    result = discover_plugins()

        assert len(result) == 1
        assert result[0].name == "entry"
        assert result[0].bundle_url is None
        # spec.name != entry-point key "entry", so bundle is rejected at the
        # name-mismatch check before the regex check fires.
        assert caplog.text  # a warning was logged

    @pytest.mark.parametrize(
        "invalid_key",
        [
            "UpperCase",
            "a",  # single character (too short for [a-z][a-z0-9-]+)
            "1abc",
            "has spaces",
        ],
    )
    def test_invalid_entry_point_key_suppresses_bundle(self, invalid_key: str, caplog):
        """An entry-point key that fails _PLUGIN_NAME_RE is rejected even when
        spec.name matches it (exercises the regex branch itself)."""
        spec = StudioSpec(name=invalid_key, bundle_path=Path("/some/index.js"))
        mock_factory = Mock(return_value=spec)

        with caplog.at_level(logging.WARNING, logger="nhx.studio.plugins"):
            with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests(invalid_key)):
                with patch("nhx.studio.plugins.discover_studio", return_value={invalid_key: mock_factory}):
                    result = discover_plugins()

        assert len(result) == 1
        assert result[0].bundle_url is None
        assert "invalid entry-point key" in caplog.text

    def test_bundle_path_not_named_index_js_suppresses_bundle(self, tmp_path: Path, caplog):
        """The advertised bundleUrl is always index.js, so other filenames are rejected."""
        bundle_file = tmp_path / "main.js"
        bundle_file.write_text("// bundle")
        spec = StudioSpec(name="my-plugin", bundle_path=bundle_file)
        mock_factory = Mock(return_value=spec)

        with caplog.at_level(logging.WARNING, logger="nhx.studio.plugins"):
            with patch("nhx.studio.plugins.discover_manifests", return_value=_manifests("my-plugin")):
                with patch("nhx.studio.plugins.discover_studio", return_value={"my-plugin": mock_factory}):
                    result = discover_plugins()

        assert result[0].bundle_url is None
        assert "must be named index.js" in caplog.text


class TestPluginsRouter:
    def test_get_plugins_returns_manifests(self):
        manifests = [PluginManifestResponse(name="example", bundle_url="/plugin-ui/example/index.js")]
        router = build_plugins_router(manifests)
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/apis/plugins")

        assert response.status_code == 200
        assert response.json() == [{"name": "example", "bundleUrl": "/plugin-ui/example/index.js"}]

    def test_get_plugins_returns_null_bundle_url_for_headless_plugin(self):
        manifests = [PluginManifestResponse(name="headless")]
        router = build_plugins_router(manifests)
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/apis/plugins")

        assert response.status_code == 200
        assert response.json() == [{"name": "headless", "bundleUrl": None}]

    def test_get_plugins_returns_empty_list_when_no_plugins(self):
        router = build_plugins_router([])
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/apis/plugins")

        assert response.status_code == 200
        assert response.json() == []

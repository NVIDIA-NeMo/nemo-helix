# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for plugin OpenAPI generation config + discovery."""

from pathlib import Path

import pytest

from script.openapi_helper.plugin_config import PluginConfig, discover_plugins


def _write_pyproject(plugin_dir: Path, contents: str) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    pyproject = plugin_dir / "pyproject.toml"
    pyproject.write_text(contents)
    return pyproject


# ---- PluginConfig.from_pyproject ------------------------------------------------


def test_from_pyproject_returns_none_without_opt_in_table(tmp_path):
    pyproject = _write_pyproject(
        tmp_path / "my-plugin",
        """
        [project.entry-points."nemo.services"]
        my-svc = "my_plugin.service:MyService"
        """,
    )
    assert PluginConfig.from_pyproject(pyproject) is None


def test_from_pyproject_returns_none_when_no_nemo_services(tmp_path):
    pyproject = _write_pyproject(
        tmp_path / "no-services",
        """
        [tool.nemo.openapi]
        """,
    )
    assert PluginConfig.from_pyproject(pyproject) is None


def test_from_pyproject_returns_none_for_virtual_workspace_member(tmp_path):
    pyproject = _write_pyproject(
        tmp_path / "virtual-plugin",
        """
        [project.entry-points."nemo.services"]
        virtual-svc = "virtual_plugin.service:VirtualService"

        [tool.uv]
        package = false

        [tool.nemo.openapi]
        """,
    )
    assert PluginConfig.from_pyproject(pyproject) is None


def test_from_pyproject_empty_table_uses_defaults(tmp_path):
    pyproject = _write_pyproject(
        tmp_path / "data-designer",
        """
        [project.entry-points."nemo.services"]
        data-designer = "dd.service:DataDesignerService"

        [tool.nemo.openapi]
        """,
    )
    config = PluginConfig.from_pyproject(pyproject)
    assert config == PluginConfig(
        dir="data-designer",
        service_name=None,
        env_vars=None,
        factory_override=None,
        data_designer_plugin_allowlist=None,
    )


def test_from_pyproject_reads_overrides(tmp_path):
    pyproject = _write_pyproject(
        tmp_path / "multi-plugin",
        """
        [project.entry-points."nemo.services"]
        svc-a = "pkg.a:A"
        svc-b = "pkg.b:B"

        [tool.nemo.openapi]
        service_name = "svc-a"
        factory_override = "pkg.factories:make_app"
        data_designer_plugin_allowlist = ["fileset-seed-datasets"]

        [tool.nemo.openapi.env_vars]
        FOO = "bar"
        BAZ = "qux"
        """,
    )
    config = PluginConfig.from_pyproject(pyproject)
    assert config == PluginConfig(
        dir="multi-plugin",
        service_name="svc-a",
        env_vars={"FOO": "bar", "BAZ": "qux"},
        factory_override="pkg.factories:make_app",
        data_designer_plugin_allowlist=["fileset-seed-datasets"],
    )


def test_from_pyproject_rejects_non_table_env_vars(tmp_path):
    pyproject = _write_pyproject(
        tmp_path / "bad-env",
        """
        [project.entry-points."nemo.services"]
        svc = "pkg:S"

        [tool.nemo.openapi]
        env_vars = "not-a-table"
        """,
    )
    with pytest.raises(ValueError, match="env_vars must be a table"):
        PluginConfig.from_pyproject(pyproject)


def test_from_pyproject_rejects_non_string_data_designer_plugin_allowlist(tmp_path):
    pyproject = _write_pyproject(
        tmp_path / "bad-dd-plugin-allowlist",
        """
        [project.entry-points."nemo.services"]
        svc = "pkg:S"

        [tool.nemo.openapi]
        data_designer_plugin_allowlist = ["ok", 1]
        """,
    )
    with pytest.raises(ValueError, match="data_designer_plugin_allowlist must be a list of strings"):
        PluginConfig.from_pyproject(pyproject)


# ---- discover_plugins -----------------------------------------------------------


def test_discover_plugins_filters_and_sorts(tmp_path):
    plugins_root = tmp_path / "plugins"

    # Opted in (sole entry-point).
    _write_pyproject(
        plugins_root / "zeta-plugin",
        """
        [project.entry-points."nemo.services"]
        zeta = "z.service:Z"

        [tool.nemo.openapi]
        """,
    )
    # Opted in with override.
    _write_pyproject(
        plugins_root / "alpha-plugin",
        """
        [project.entry-points."nemo.services"]
        alpha = "a.service:A"
        alpha-internal = "a.service:AInternal"

        [tool.nemo.openapi]
        service_name = "alpha"
        """,
    )
    # Not opted in — should be skipped.
    _write_pyproject(
        plugins_root / "skipped-plugin",
        """
        [project.entry-points."nemo.services"]
        skipped = "s.service:S"
        """,
    )
    # Opted in but no nemo.services — also skipped.
    _write_pyproject(
        plugins_root / "no-services-plugin",
        """
        [tool.nemo.openapi]
        """,
    )

    discovered = discover_plugins(plugins_root)

    assert [p.dir for p in discovered] == ["alpha-plugin", "zeta-plugin"]
    assert discovered[0].service_name == "alpha"
    assert discovered[1].service_name is None


def test_discover_plugins_empty_when_no_plugins_dir(tmp_path):
    assert discover_plugins(tmp_path / "does-not-exist") == []


# ---- PluginConfig.resolve_service_name ------------------------------------------


def test_resolve_service_name_explicit_wins(monkeypatch, tmp_path):
    # Ensure resolve doesn't read pyproject when service_name is set.
    monkeypatch.chdir(tmp_path)
    config = PluginConfig(dir="ghost", service_name="explicit-name")
    assert config.resolve_service_name() == "explicit-name"


def test_resolve_service_name_auto_resolves_sole_entry(monkeypatch, tmp_path):
    _write_pyproject(
        tmp_path / "plugins" / "solo",
        """
        [project.entry-points."nemo.services"]
        solo-svc = "solo.service:Solo"

        [tool.nemo.openapi]
        """,
    )
    monkeypatch.chdir(tmp_path)
    config = PluginConfig(dir="solo")
    assert config.resolve_service_name() == "solo-svc"


def test_resolve_service_name_raises_on_ambiguity(monkeypatch, tmp_path):
    _write_pyproject(
        tmp_path / "plugins" / "multi",
        """
        [project.entry-points."nemo.services"]
        a = "pkg:A"
        b = "pkg:B"

        [tool.nemo.openapi]
        """,
    )
    monkeypatch.chdir(tmp_path)
    config = PluginConfig(dir="multi")
    with pytest.raises(ValueError, match="2 nemo.services entries"):
        config.resolve_service_name()


def test_resolve_service_name_raises_when_no_entries(monkeypatch, tmp_path):
    _write_pyproject(
        tmp_path / "plugins" / "empty",
        """
        [tool.nemo.openapi]
        """,
    )
    monkeypatch.chdir(tmp_path)
    config = PluginConfig(dir="empty")
    with pytest.raises(ValueError, match="0 nemo.services entries"):
        config.resolve_service_name()


# ---- output_path ---------------------------------------------------------------


def test_output_path_is_inside_plugin_dir():
    config = PluginConfig(dir="my-plugin")
    assert config.output_path() == "plugins/my-plugin/openapi/openapi.yaml"

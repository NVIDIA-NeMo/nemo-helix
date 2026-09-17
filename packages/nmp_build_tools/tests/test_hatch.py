# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nmp_build_tools.hatch import (
    DEFAULT_DYNAMIC_VERSION,
    NmpDynamicVersionSource,
    _rewrite_metadata,
    nmp_dynamic_versioning_config,
    read_bundle_force_include,
    read_bundle_shared_data,
)


def test_nmp_dynamic_versioning_config_uses_repo_defaults() -> None:
    config = nmp_dynamic_versioning_config()

    assert config.fallback_version == DEFAULT_DYNAMIC_VERSION
    assert config.vcs.value == "git"
    assert config.style.value == "pep440"
    assert config.pattern == "default-unprefixed"


def test_nmp_dynamic_versioning_config_allows_hatch_version_overrides() -> None:
    config = nmp_dynamic_versioning_config(
        {
            "source": "nmp-dynamic-versioning",
            "fallback-version": "9.9.9",
        }
    )

    assert config.fallback_version == "9.9.9"


def test_nmp_dynamic_version_source_honors_bypass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_DYNAMIC_VERSIONING_BYPASS", "1.2.3")

    version_source = NmpDynamicVersionSource(
        str(tmp_path),
        {"source": "nmp-dynamic-versioning"},
    )

    assert version_source.get_version_data() == {"version": "1.2.3"}


def test_read_bundle_force_include_adds_package_extra_includes(tmp_path: Path) -> None:
    root = tmp_path / "wrapper"
    root.mkdir()
    source = root / "pkg" / "src" / "pkg"
    source.mkdir(parents=True)
    extra = root / "pkg" / "config.yml"
    extra.write_text("x: y\n")
    (root / "pyproject.toml").write_text(
        """
[tool.bundle-package]
pkg = { source = "pkg/src/pkg", module = "pkg", force_include = { "../../config.yml" = "pkg/config.yml" } }
""".strip()
    )

    assert read_bundle_force_include(str(root)) == {
        str(source.resolve()): "pkg",
        str(extra.resolve()): "pkg/config.yml",
    }


def test_read_bundle_force_include_expands_package_extra_include_globs(tmp_path: Path) -> None:
    root = tmp_path / "wrapper"
    root.mkdir()
    source = root / "pkg" / "src" / "pkg"
    source.mkdir(parents=True)
    first = root / "pkg" / "agent.yml"
    second = root / "pkg" / "eval.yml"
    ignored = root / "pkg" / "data.json"
    first.write_text("x: y\n")
    second.write_text("x: z\n")
    ignored.write_text("{}\n")
    (root / "pyproject.toml").write_text(
        """
[tool.bundle-package]
pkg = { source = "pkg/src/pkg", module = "pkg", force_include = { "../../*.yml" = "pkg/" } }
""".strip()
    )

    assert read_bundle_force_include(str(root)) == {
        str(source.resolve()): "pkg",
        str(first.resolve()): "pkg/agent.yml",
        str(second.resolve()): "pkg/eval.yml",
    }


def test_read_bundle_force_include_rejects_glob_target_without_trailing_slash(tmp_path: Path) -> None:
    root = tmp_path / "wrapper"
    root.mkdir()
    source = root / "pkg" / "src" / "pkg"
    source.mkdir(parents=True)
    (root / "pkg" / "agent.yml").write_text("x: y\n")
    (root / "pyproject.toml").write_text(
        """
[tool.bundle-package]
pkg = { source = "pkg/src/pkg", module = "pkg", force_include = { "../../*.yml" = "pkg/agent.yml" } }
""".strip()
    )

    with pytest.raises(ValueError, match="Glob force_include target must end with '/'"):
        read_bundle_force_include(str(root))


def test_read_bundle_force_include_rejects_glob_target_collisions(tmp_path: Path) -> None:
    root = tmp_path / "wrapper"
    root.mkdir()
    source = root / "pkg" / "src" / "pkg"
    source.mkdir(parents=True)
    first = root / "pkg" / "first" / "config.yml"
    second = root / "pkg" / "second" / "config.yml"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("x: y\n")
    second.write_text("x: z\n")
    (root / "pyproject.toml").write_text(
        """
[tool.bundle-package]
pkg = { source = "pkg/src/pkg", module = "pkg", force_include = { "../../**/*.yml" = "pkg/" } }
""".strip()
    )

    with pytest.raises(ValueError, match="Glob force_include target collision"):
        read_bundle_force_include(str(root))


def test_read_bundle_shared_data_resolves_paths_relative_to_the_bundle_source(tmp_path: Path) -> None:
    root = tmp_path / "wrapper"
    (root / "pkg" / "src" / "pkg").mkdir(parents=True)
    (root / "pkg" / "adapter.json").write_text("{}")
    (root / "pyproject.toml").write_text(
        """
[tool.bundle-package]
pkg = { source = "pkg/src/pkg", module = "pkg", shared_data = { "../../adapter.json" = "share/x/adapter.json" } }
"""
    )

    assert read_bundle_shared_data(str(root)) == {
        str(root / "pkg" / "adapter.json"): "share/x/adapter.json",
    }


def test_read_bundle_shared_data_is_empty_without_declarations(tmp_path: Path) -> None:
    root = tmp_path / "wrapper"
    (root / "pkg" / "src" / "pkg").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """
[tool.bundle-package]
pkg = { source = "pkg/src/pkg", module = "pkg" }
"""
    )

    assert read_bundle_shared_data(str(root)) == {}


def test_rewrite_metadata_replaces_bundled_requirements_with_self_extras() -> None:
    metadata = """Metadata-Version: 2.4
Name: nemo-platform
Requires-Dist: data-designer-nemo; extra == "nemo-data-designer-plugin"
Requires-Dist: switchyard; extra == "nemo-switchyard"
Requires-Dist: httpx>=0.28; extra == "nemo-switchyard"

"""

    rewritten = _rewrite_metadata(
        metadata,
        {"name": "nemo-platform"},
        {
            "data-designer-nemo": {},
            "switchyard": {},
        },
    )

    assert 'Requires-Dist: nemo-platform[data-designer-nemo] ; extra == "nemo-data-designer-plugin"' in rewritten
    assert 'Requires-Dist: nemo-platform[switchyard] ; extra == "nemo-switchyard"' in rewritten
    assert 'Requires-Dist: httpx>=0.28; extra == "nemo-switchyard"' in rewritten
    assert "Requires-Dist: data-designer-nemo" not in rewritten
    assert "Requires-Dist: switchyard" not in rewritten

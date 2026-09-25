# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging contracts for the Switchyard plugin distribution."""

from __future__ import annotations

import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_PLUGIN_PYPROJECT = _REPO / "plugins/nemo-switchyard/pyproject.toml"
_PLATFORM_PYPROJECT = _REPO / "packages/nemo_helix/pyproject.toml"
_UV_LOCK = _REPO / "uv.lock"


def _toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_distribution_rename_preserves_middleware_entry_point() -> None:
    project = _toml(_PLUGIN_PYPROJECT)["project"]

    assert project["name"] == "nemo-switchyard-plugin"
    assert project["entry-points"]["nemo.inference_middleware"] == {
        "nemo-switchyard": "nemo_switchyard.middleware:SwitchyardMiddleware"
    }


def test_bundle_uses_plugin_dist_name_and_existing_extra() -> None:
    platform = _toml(_PLATFORM_PYPROJECT)
    spec = platform["tool"]["bundle-package"]["nemo-switchyard-plugin"]

    assert spec["module"] == "nemo_switchyard"
    assert spec["deps_group"] == "nemo-switchyard"
    assert "switchyard-vendored" in platform["project"]["optional-dependencies"]["nemo-switchyard"]


def test_lock_uses_plugin_dist_name() -> None:
    packages = _toml(_UV_LOCK)["package"]
    names = [package["name"] for package in packages]

    assert "nemo-switchyard-plugin" in names
    assert not any(
        package["name"] == "nemo-switchyard" and package.get("source", {}).get("editable") == "plugins/nemo-switchyard"
        for package in packages
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging contracts: dist name vs entry-point, no native .so in the extra."""

from __future__ import annotations

import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_PLUGIN_PYPROJECT = _REPO / "plugins/nemo-switchyard/pyproject.toml"
_PLATFORM_PYPROJECT = _REPO / "packages/nemo_platform/pyproject.toml"


def _toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_plugin_dist_name_is_nemo_switchyard_plugin() -> None:
    project = _toml(_PLUGIN_PYPROJECT)["project"]
    assert project["name"] == "nemo-switchyard-plugin"
    entry = project["entry-points"]["nemo.inference_middleware"]
    assert "nemo-switchyard" in entry
    assert entry["nemo-switchyard"].endswith("SwitchyardMiddleware")


def test_bundle_package_copies_python_only() -> None:
    bundle = _toml(_PLATFORM_PYPROJECT)["tool"]["bundle-package"]
    spec = bundle["nemo-switchyard-plugin"]
    assert spec["module"] == "nemo_switchyard"
    assert spec.get("deps_group") == "nemo-switchyard"
    assert ".so" not in spec["source"]
    assert "switchyard_rust" not in spec["source"]
    extra = _toml(_PLATFORM_PYPROJECT)["project"]["optional-dependencies"]["nemo-switchyard"]
    assert not any("switchyard_rust" in dep for dep in extra)
    vendor = bundle["switchyard-vendored"]
    assert vendor["module"] == "switchyard"
    assert "switchyard_rust" not in vendor["source"]


def test_nmp_api_native_arg_defaults_empty() -> None:
    text = (_REPO / "docker/Dockerfile.nmp-api").read_text(encoding="utf-8")
    assert "ARG SWITCHYARD_NATIVE_REF=" in text
    assert "Do not install upstream next to vendor" in text
    assert "uv pip uninstall -y switchyard-vendored" in text

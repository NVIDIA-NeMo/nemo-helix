# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging contracts for the Switchyard plugin distribution."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_PLUGIN_PYPROJECT = _REPO / "plugins/nemo-switchyard/pyproject.toml"
_PLATFORM_PYPROJECT = _REPO / "packages/nemo_platform/pyproject.toml"
_UV_LOCK = _REPO / "uv.lock"


def _toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_plugin_dist_name_is_nemo_switchyard_plugin() -> None:
    project = _toml(_PLUGIN_PYPROJECT)["project"]
    assert project["name"] == "nemo-switchyard-plugin"
    entry = project["entry-points"]["nemo.inference_middleware"]
    assert "nemo-switchyard" in entry
    assert entry["nemo-switchyard"].endswith("SwitchyardMiddleware")


def test_bundle_package_uses_expected_python_modules() -> None:
    bundle = _toml(_PLATFORM_PYPROJECT)["tool"]["bundle-package"]
    spec = bundle["nemo-switchyard-plugin"]
    assert spec["source"].endswith("/src/nemo_switchyard")
    assert spec["module"] == "nemo_switchyard"
    assert spec.get("deps_group") == "nemo-switchyard"
    extra = _toml(_PLATFORM_PYPROJECT)["project"]["optional-dependencies"]["nemo-switchyard"]
    assert not any("switchyard_rust" in dep for dep in extra)
    vendor = bundle["switchyard-vendored"]
    assert vendor["source"].endswith("/vendor/switchyard/switchyard")
    assert vendor["module"] == "switchyard"


def test_nmp_api_native_arg_defaults_empty() -> None:
    text = (_REPO / "docker/Dockerfile.nmp-api").read_text(encoding="utf-8")
    assert re.search(r"(?m)^ARG SWITCHYARD_NATIVE_REF=\s*$", text)
    assert "uv pip uninstall -y" not in text

    accepted_arm = next(
        (line for line in text.splitlines() if 'spec="${SWITCHYARD_NATIVE_REF}"' in line),
        None,
    )
    assert accepted_arm is not None
    assert "https://*" in accepted_arm
    assert "*.whl" in accepted_arm


def test_lock_uses_plugin_dist_name() -> None:
    try:
        packages = _toml(_UV_LOCK)["package"]
        names = [package["name"] for package in packages]
    except tomllib.TOMLDecodeError:
        names = []
        in_package = False
        for line in _UV_LOCK.read_text(encoding="utf-8").splitlines():
            if line == "[[package]]":
                in_package = True
                continue
            if line.startswith("[["):
                in_package = False
            if in_package and (match := re.fullmatch(r'name = "([^"]+)"', line)):
                names.append(match.group(1))

    assert "nemo-switchyard-plugin" in names
    assert "nemo-switchyard" not in names

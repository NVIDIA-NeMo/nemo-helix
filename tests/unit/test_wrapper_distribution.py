# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the published ``nemo-platform`` wrapper distribution."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def test_all_extra_bundles_default_deployments_backend() -> None:
    """The backend enabled by the packaged config must ship in ``nemo-platform[all]``."""
    pyproject_path = ROOT / "packages/nemo_platform/pyproject.toml"
    with open(pyproject_path, "rb") as pyproject:
        wrapper = tomllib.load(pyproject)

    project = wrapper["project"]
    bundle = wrapper["tool"]["bundle-package"]["nemo-deployments-plugin"]

    assert bundle["inherit"]["entry-points"] == ["nemo.*"]
    assert bundle["module"] == "nemo_deployments_plugin"
    assert (pyproject_path.parent / bundle["source"]).resolve().is_dir()
    assert "nemo-platform[services]" in project["optional-dependencies"]["all"]
    assert "nemo-platform[plugins]" in project["optional-dependencies"]["services"]
    assert "nemo-platform[nemo-deployments-plugin]" in project["optional-dependencies"]["plugins"]
    assert "nemo-deployments-plugin" in project["optional-dependencies"]
    assert project["entry-points"]["nemo.services"]["deployments"].startswith("nemo_deployments_plugin.")
    assert project["entry-points"]["nemo.controllers"]["deployments"].startswith("nemo_deployments_plugin.")
    assert project["entry-points"]["nemo.sandbox_profiles"]["openshell"].startswith("nemo_deployments_plugin.")
    assert project["entry-points"]["nemo.skills"]["deployments"].startswith("nemo_deployments_plugin.")


def test_bundled_shared_data_is_carried_into_the_wrapper_wheel() -> None:
    """Every bundled package's shared-data must be re-declared on its bundle entry.

    Bundled packages are not installed as their own distribution in a packaged
    ``nemo-platform`` install, so anything they would install under the
    environment prefix is lost unless the wrapper bundle entry declares it too.
    """
    pyproject_path = ROOT / "packages/nemo_platform/pyproject.toml"
    with open(pyproject_path, "rb") as pyproject:
        bundles = tomllib.load(pyproject)["tool"]["bundle-package"]

    for name, bundle in bundles.items():
        source = (pyproject_path.parent / bundle["source"]).resolve()
        bundled_pyproject = _owning_pyproject(source)
        if bundled_pyproject is None:
            continue

        with open(bundled_pyproject, "rb") as pyproject:
            config = tomllib.load(pyproject)
        shared_data = (
            config.get("tool", {})
            .get("hatch", {})
            .get("build", {})
            .get("targets", {})
            .get("wheel", {})
            .get("shared-data", {})
        )
        if not shared_data:
            continue

        expected = {
            (bundled_pyproject.parent / relative_source).resolve(): target
            for relative_source, target in shared_data.items()
        }
        actual = {
            (source / relative_source).resolve(): target
            for relative_source, target in bundle.get("shared_data", {}).items()
        }
        assert actual == expected, f"bundle entry '{name}' does not carry the shared-data of {bundled_pyproject}"
        for path in actual:
            assert path.is_file()


def test_insights_analyst_fabric_descriptor_ships_in_the_wrapper_wheel() -> None:
    """Fabric resolves ``nvidia.fabric.insights-analyst`` from the install prefix."""
    pyproject_path = ROOT / "packages/nemo_platform/pyproject.toml"
    with open(pyproject_path, "rb") as pyproject:
        bundle = tomllib.load(pyproject)["tool"]["bundle-package"]["nemo-insights-plugin"]

    source = (pyproject_path.parent / bundle["source"]).resolve()
    shared_data = {(source / key).resolve(): value for key, value in bundle["shared_data"].items()}

    descriptor = (ROOT / "plugins/nemo-insights/insights-analyst.fabric-adapter.json").resolve()
    assert shared_data[descriptor] == (
        "share/nemo-fabric/adapters/insights-analyst/insights-analyst.fabric-adapter.json"
    )


def _owning_pyproject(source: Path) -> Path | None:
    """Return the ``pyproject.toml`` of the distribution owning a bundled source tree."""
    for parent in source.parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
        if parent == ROOT:
            break
    return None

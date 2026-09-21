# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The published ``nemo-platform`` wheel must expose every ``nemo.*`` entry point its bundled packages declare.

In the uv workspace each package registers its own entry points, so CLI discovery
tests pass even when the wrapper's generated tables are missing a group. Only the
built wheel shows that gap; this test reads the same declarations the wheel is
built from and fails if a bundled package's ``nemo.*`` entry point is not carried
into the wrapper (typically a missing ``inherit`` clause on the bundle entry).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomlkit

REPO_ROOT = Path(__file__).resolve().parents[5]
WRAPPER_PYPROJECT = REPO_ROOT / "packages" / "nemo_platform" / "pyproject.toml"
# Packages vendored into the SDK distribution surface their entry points there,
# with module paths rewritten, so the SDK's tables count as exposed too.
SDK_PYPROJECT = REPO_ROOT / "sdk" / "python" / "nemo-platform" / "pyproject.toml"

# Pre-existing gaps in the shipped wheel that predate this check, as
# (bundle, entry-point group, entry-point key). Remove an entry once the
# vendoring config exposes it.
KNOWN_MISSING: frozenset[tuple[str, str, str]] = frozenset(
    {("nemo-evaluator-sdk", "nemo.fabric.task_hooks", "mcp_run_binding")}
)


def _load(path: Path) -> dict:
    return tomlkit.parse(path.read_text(encoding="utf-8")).unwrap()


def _source_pyproject(source: Path) -> Path:
    for candidate in [source, *source.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            return pyproject
    raise AssertionError(f"no pyproject.toml above bundled source {source}")


def _nemo_entry_points(project: dict) -> dict[str, set[str]]:
    groups = project.get("entry-points", {})
    return {group: set(entries) for group, entries in groups.items() if group.startswith("nemo.")}


def _exposed_entry_points() -> dict[str, set[str]]:
    exposed: dict[str, set[str]] = {}
    for pyproject in (WRAPPER_PYPROJECT, SDK_PYPROJECT):
        for group, keys in _nemo_entry_points(_load(pyproject)["project"]).items():
            exposed.setdefault(group, set()).update(keys)
    return exposed


def _bundled_packages() -> list[tuple[str, Path]]:
    wrapper = _load(WRAPPER_PYPROJECT)
    bundles = wrapper["tool"]["bundle-package"]
    packages = []
    for name, config in bundles.items():
        source = (WRAPPER_PYPROJECT.parent / config["source"]).resolve()
        packages.append((name, _source_pyproject(source)))
    return packages


@pytest.mark.parametrize(("bundle_name", "pyproject"), _bundled_packages(), ids=lambda value: str(value))
def test_bundled_nemo_entry_points_are_exposed_by_the_wrapper(bundle_name: str, pyproject: Path) -> None:
    declared = _nemo_entry_points(_load(pyproject).get("project", {}))
    if not declared:
        pytest.skip(f"{bundle_name} declares no nemo.* entry points")

    exposed = _exposed_entry_points()
    missing = {
        group: sorted(key for key in keys - exposed.get(group, set()) if (bundle_name, group, key) not in KNOWN_MISSING)
        for group, keys in declared.items()
    }
    missing = {group: keys for group, keys in missing.items() if keys}

    assert not missing, (
        f"{bundle_name} declares nemo.* entry points the nemo-platform wheel does not expose: {missing}. "
        'Add inherit = { "entry-points" = ["nemo.*"] } to its [tool.bundle-package] entry and run `make vendor`.'
    )

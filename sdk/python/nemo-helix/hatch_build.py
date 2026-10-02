# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001

from __future__ import annotations

import collections.abc
import shutil
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

GENERATED_INIT_FILE = """# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""


class SourcePackage:
    def __init__(self, *, source: str, target: str, include: tuple[str, ...]) -> None:
        self.source = source
        self.target = target
        self.include = include


class CustomBuildHook(BuildHookInterface):
    """Stage source SDK extensions into SDK build artifacts."""

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        if version == "editable":
            # Editable installs run against the repo/workspace sources, where
            # these packages are already importable as workspace packages.
            return

        packages = _source_packages(self.config)
        project_root = Path(self.root).resolve()
        source_base = _find_source_base(project_root, packages)
        self._stage_tmp = tempfile.TemporaryDirectory(prefix="nhx-sdk-stage-")
        stage_root = Path(self._stage_tmp.name)

        force_include = _force_include(build_data)
        if self.target_name == "sdist":
            _stage_sdist_sources(source_base, stage_root, force_include, packages)
            patched_pyproject = _write_sdist_pyproject(project_root, stage_root, self.metadata.version)
            _replace_force_include_target(force_include, source=patched_pyproject, target="pyproject.toml")
        else:
            _stage_wheel_sources(source_base, stage_root, force_include, packages)

        build_data["force_include"] = force_include

    def finalize(self, _version: str, _build_data: dict[str, object], _artifact_path: str) -> None:
        stage_tmp = getattr(self, "_stage_tmp", None)
        if stage_tmp is not None:
            stage_tmp.cleanup()


def _source_packages(config: collections.abc.Mapping[str, object]) -> tuple[SourcePackage, ...]:
    packages = []
    for entry in _config_entries(config, "source-packages"):
        packages.append(
            SourcePackage(
                source=_required_string(entry, "source", "source-packages"),
                target=_required_string(entry, "target", "source-packages"),
                include=_include_patterns(entry),
            )
        )
    return tuple(packages)


def _force_include(build_data: collections.abc.Mapping[str, object]) -> dict[str, str]:
    existing_force_include = build_data.get("force_include")
    if not isinstance(existing_force_include, dict):
        return {}
    return {str(source): str(target) for source, target in existing_force_include.items()}


def _find_source_base(project_root: Path, packages: tuple[SourcePackage, ...]) -> Path:
    """Find the root containing the configured source package paths.

    Monorepo builds run from ``sdk/python/nemo-helix`` while wheels built
    from an sdist run from the extracted sdist root. Walking upward supports
    both layouts without hard-coding a fixed number of parent directories.
    """
    for candidate in (project_root, *project_root.parents):
        if all((candidate / package.source).is_dir() for package in packages):
            return candidate

    missing = ", ".join(package.source for package in packages)
    raise FileNotFoundError(f"Could not find SDK source package roots from {project_root}: {missing}")


def _stage_wheel_sources(
    source_base: Path,
    stage_root: Path,
    force_include: dict[str, str],
    packages: tuple[SourcePackage, ...],
) -> None:
    for package in packages:
        source_root = source_base / package.source
        package_stage = stage_root / package.target
        _copy_included_paths(source_root, package_stage, package.include)
        _ensure_init_files(package_stage)
        force_include[str(package_stage)] = package.target


def _stage_sdist_sources(
    source_base: Path,
    stage_root: Path,
    force_include: dict[str, str],
    packages: tuple[SourcePackage, ...],
) -> None:
    for package in packages:
        source_root = source_base / package.source
        package_stage = stage_root / package.source
        _copy_included_paths(source_root, package_stage, package.include)
        force_include[str(package_stage)] = package.source


def _write_sdist_pyproject(project_root: Path, stage_root: Path, version: str) -> Path:
    """Write a self-contained sdist pyproject.

    The monorepo SDK pyproject uses ``nhx-build-tools`` from the uv workspace
    for dynamic versioning. An sdist is outside that workspace, so wheel builds
    from the sdist need static version metadata and no workspace-only build
    dependency.
    """
    source = project_root / "pyproject.toml"
    destination = stage_root / "pyproject.toml"
    destination.write_text(_sdist_pyproject(source.read_text(encoding="utf-8"), version), encoding="utf-8")
    return destination


def _sdist_pyproject(content: str, version: str) -> str:
    content = content.replace('dynamic = ["readme", "version"]', f'dynamic = ["readme"]\nversion = "{version}"')
    content = content.replace('"hatch-fancy-pypi-readme", "nhx-build-tools"', '"hatch-fancy-pypi-readme"')
    content = "\n".join(line for line in content.splitlines() if not _is_nhx_build_tools_workspace_source(line))
    content = _remove_toml_section(content, "[tool.hatch.version]")
    return f"{content.rstrip()}\n"


def _is_nhx_build_tools_workspace_source(line: str) -> bool:
    return line.strip().replace(" ", "") == "nhx-build-tools={workspace=true}"


def _remove_toml_section(content: str, section_header: str) -> str:
    lines = content.splitlines()
    output = []
    skipping = False

    for line in lines:
        stripped = line.strip()
        if stripped == section_header:
            skipping = True
            continue
        if skipping and stripped.startswith("[") and stripped.endswith("]"):
            skipping = False
        if not skipping:
            output.append(line)

    return "\n".join(output)


def _replace_force_include_target(force_include: dict[str, str], *, source: Path, target: str) -> None:
    for existing_source, existing_target in tuple(force_include.items()):
        if existing_target == target:
            del force_include[existing_source]
    force_include[str(source)] = target


def _config_entries(
    config: collections.abc.Mapping[str, object], key: str
) -> tuple[collections.abc.Mapping[str, object], ...]:
    raw_entries = config.get(key, [])
    if not isinstance(raw_entries, list):
        raise TypeError(f"`{key}` must be an array")

    entries = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise TypeError(f"`{key}` entries must be tables")
        entries.append(entry)
    return tuple(entries)


def _required_string(entry: collections.abc.Mapping[str, object], key: str, section: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"`{section}` entries must define a non-empty `{key}` string")
    return value


def _include_patterns(entry: collections.abc.Mapping[str, object]) -> tuple[str, ...]:
    raw_patterns = entry.get("include", ["**/*.py"])
    if not isinstance(raw_patterns, list) or any(
        not isinstance(pattern, str) or not pattern for pattern in raw_patterns
    ):
        raise TypeError("`source-packages` entries must define `include` as an array of non-empty strings")
    return tuple(raw_patterns)


def _copy_included_paths(source_root: Path, target_root: Path, patterns: tuple[str, ...]) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source package path does not exist: {source_root}")

    seen: set[Path] = set()
    for pattern in patterns:
        for source_file in source_root.glob(pattern):
            if not source_file.is_file() or source_file in seen:
                continue
            seen.add(source_file)
            relative_path = source_file.relative_to(source_root)
            target_file = target_root / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)


def _ensure_init_files(package_root: Path) -> None:
    for directory in (package_root, *(path for path in package_root.rglob("*") if path.is_dir())):
        init_file = directory / "__init__.py"
        if not init_file.exists():
            init_file.write_text(GENERATED_INIT_FILE, encoding="utf-8")

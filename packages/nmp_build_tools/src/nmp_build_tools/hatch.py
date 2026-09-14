# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import csv
import glob
import hashlib
import os
import tempfile
import tomllib
import zipfile
from email.generator import Generator
from email.parser import Parser
from io import StringIO
from pathlib import Path
from typing import Any

from hatchling.plugin import hookimpl
from hatchling.version.source.plugin.interface import VersionSourceInterface
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from uv_dynamic_versioning import schemas
from uv_dynamic_versioning.main import get_version

_GLOB_CHARS = set("*?[")
DEFAULT_DYNAMIC_VERSION = "0.0.0"
DEFAULT_DYNAMIC_VERSIONING_CONFIG: dict[str, Any] = {
    "fallback-version": DEFAULT_DYNAMIC_VERSION,
    "vcs": "git",
    "style": "pep440",
    "pattern": "default-unprefixed",
}


class NmpDynamicVersionSource(VersionSourceInterface):
    PLUGIN_NAME = "nmp-dynamic-versioning"

    def get_version_data(self) -> dict[str, str]:
        config = nmp_dynamic_versioning_config(self.config)
        version, _ = get_version(config)
        return {"version": version}


def nmp_dynamic_versioning_config(overrides: dict[str, Any] | None = None) -> schemas.UvDynamicVersioning:
    config = DEFAULT_DYNAMIC_VERSIONING_CONFIG.copy()
    if overrides:
        config.update({key: value for key, value in overrides.items() if key != "source"})
    return schemas.UvDynamicVersioning.from_dict(config)


@hookimpl
def hatch_register_version_source() -> type[NmpDynamicVersionSource]:
    return NmpDynamicVersionSource


def read_bundle_force_include(root: str) -> dict[str, str]:
    """Return Hatch force-includes for bundled packages.

    Per-bundle ``force_include`` source paths are resolved relative to that
    bundle entry's ``source`` directory, not relative to the wrapper project.
    """
    pyproject_path = Path(root) / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    bundle_packages = config.get("tool", {}).get("bundle-package", {})
    # The build hook writes build_data["force_include"], so merge any static hatch
    # force-includes that would otherwise be replaced by the generated bundle map.
    static_force_include = (
        config.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("force-include", {})
    )

    force_include = {}
    for pkg_config in bundle_packages.values():
        source = pkg_config["source"]
        module = pkg_config["module"]
        source_path = (Path(root) / source).resolve()
        force_include[str(source_path)] = module
        for extra_source, target in pkg_config.get("force_include", {}).items():
            extra_path = source_path / extra_source
            if any(char in extra_source for char in _GLOB_CHARS):
                if not target.endswith("/"):
                    raise ValueError(f"Glob force_include target must end with '/': {extra_source} -> {target}")
                target_paths = set()
                for match in sorted(glob.glob(str(extra_path), recursive=True)):
                    match_path = Path(match).resolve()
                    target_path = f"{target.rstrip('/')}/{match_path.name}"
                    if target_path in target_paths:
                        raise ValueError(f"Glob force_include target collision: {extra_source} -> {target_path}")
                    target_paths.add(target_path)
                    force_include[str(match_path)] = target_path
                continue
            force_include[str(extra_path.resolve())] = target
    for source, module in static_force_include.items():
        force_include[str((Path(root) / source).resolve())] = module

    return force_include


def read_bundle_shared_data(root: str) -> dict[str, str]:
    """Return Hatch shared-data mappings for bundled packages.

    A bundled package that installs data files under the environment prefix
    (``<sys.prefix>/share/...``) cannot express that through ``force_include``:
    force-includes land in site-packages, while shared-data lands in the wheel's
    ``.data`` directory and is unpacked relative to the install prefix. Bundle
    entries therefore declare ``shared_data`` separately, with source paths
    resolved relative to that entry's ``source`` directory.
    """
    pyproject_path = Path(root) / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    bundle_packages = config.get("tool", {}).get("bundle-package", {})

    shared_data: dict[str, str] = {}
    for pkg_config in bundle_packages.values():
        source_path = (Path(root) / pkg_config["source"]).resolve()
        for extra_source, target in pkg_config.get("shared_data", {}).items():
            shared_data[str((source_path / extra_source).resolve())] = target

    return shared_data


def apply_bundle_force_include(root: str, build_data: dict[str, Any]) -> None:
    build_data["force_include"] = read_bundle_force_include(root)


def apply_bundle_shared_data(root: str, build_data: dict[str, Any]) -> None:
    build_data["shared_data"] = {**build_data.get("shared_data", {}), **read_bundle_shared_data(root)}


def disable_bundle_force_include_for_editable(build_data: dict[str, Any]) -> None:
    empty = tempfile.mkdtemp()
    build_data["force_include_editable"] = {empty: "_editable_noop"}


def rewrite_bundled_dependencies_in_wheel(artifact_path: str, root: str, project: dict[str, Any]) -> None:
    bundle_config = read_bundle_package_config(root)
    if not bundle_config:
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(artifact_path) as wheel:
            wheel.extractall(temp_path)

        metadata_path = next(temp_path.glob("*.dist-info/METADATA"), None)
        if metadata_path is None:
            raise FileNotFoundError(f"No *.dist-info/METADATA found in wheel: {artifact_path}")
        metadata_path.write_text(
            _rewrite_metadata(
                metadata_path.read_text(encoding="utf-8"),
                project,
                bundle_config,
            ),
            encoding="utf-8",
        )

        record_path = next(temp_path.glob("*.dist-info/RECORD"), None)
        if record_path is None:
            raise FileNotFoundError(f"No *.dist-info/RECORD found in wheel: {artifact_path}")
        _rewrite_record(temp_path, record_path)
        _repack_wheel(temp_path, Path(artifact_path))


def read_bundle_package_config(root: str) -> dict[str, dict[str, Any]]:
    pyproject_path = Path(root) / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        config = tomllib.load(f)

    bundle_config = config.get("tool", {}).get("bundle-package", {})
    if not isinstance(bundle_config, dict):
        return {}
    return bundle_config


def _rewrite_metadata(metadata_text: str, project: dict[str, Any], bundle_config: dict[str, dict[str, Any]]) -> str:
    project_name = project["name"]
    bundle_entries = {canonicalize_name(name): config for name, config in bundle_config.items()}
    message = Parser().parsestr(metadata_text)
    rewritten_headers: list[tuple[str, str]] = []

    for key, value in message.raw_items():
        if key != "Requires-Dist":
            rewritten_headers.append((key, value))
            continue

        requirement = Requirement(value)
        dependency_name = canonicalize_name(requirement.name)
        bundle_entry = bundle_entries.get(dependency_name)
        if bundle_entry is None:
            rewritten_headers.append((key, value))
            continue

        deps_group = bundle_entry.get("deps_group") or dependency_name

        extras = [deps_group, *sorted(requirement.extras)]
        rewritten_requirement = f"{project_name}[{','.join(extras)}]"
        if requirement.marker:
            rewritten_requirement = f"{rewritten_requirement} ; {requirement.marker}"
        rewritten_headers.append((key, rewritten_requirement))

    rewritten_message = message.__class__()
    for key, value in _dedupe_pairs(rewritten_headers):
        rewritten_message[key] = value
    rewritten_message.set_payload(message.get_payload())

    output = StringIO()
    Generator(output, mangle_from_=False, maxheaderlen=0).flatten(rewritten_message, unixfrom=False)
    return output.getvalue()


def _rewrite_record(root: Path, record_path: Path) -> None:
    rows: list[tuple[str, str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel_path = path.relative_to(root).as_posix()
        if path == record_path:
            rows.append((rel_path, "", ""))
            continue

        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        rows.append((rel_path, f"sha256={digest}", str(len(data))))

    with record_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _repack_wheel(root: Path, artifact_path: Path) -> None:
    temp_artifact = artifact_path.with_suffix(".tmp")
    with zipfile.ZipFile(temp_artifact, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            wheel.write(path, arcname=path.relative_to(root).as_posix())
    os.replace(temp_artifact, artifact_path)


def _dedupe_pairs(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped

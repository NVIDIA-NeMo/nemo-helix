# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for license checking scripts.

This module contains common functions used by both format_osv_licenses.py
and license_overrides.py to avoid code duplication.
"""

import logging
import subprocess
import tomllib
from pathlib import Path

import typer

# Standard list of allowed licenses for NeMo Helix
ALLOWED_LICENSES = {
    "MIT",
    "BSD-3-CLAUSE",
    "BSD-2-CLAUSE",
    "APACHE-2.0",
    "ISC",
    "ZLIB",
    "NVIDIA PROPRIETARY SOFTWARE",
}


def normalize_package_name(name: str) -> str:
    """
    Normalize package name for comparison.

    Converts underscores to dashes and lowercases the name, following
    Python package naming conventions where 'my_package' and 'my-package'
    are considered equivalent.

    Args:
        name: Package name to normalize

    Returns:
        Normalized package name (lowercase, dashes instead of underscores)
    """
    return name.lower().replace("_", "-")


def normalize_version_for_override(version: str) -> str:
    """
    Strip PEP 440 local version identifier (+suffix) for override/mapping lookups.

    Packages like torchao publish wheels with versions such as 0.14.1+cu128;
    license overrides are keyed by base name, so we normalize the version
    when matching so that 0.14.1+cu128 is treated like 0.14.1 for lookup.

    Args:
        version: Package version string (e.g. "0.14.1+cu128")

    Returns:
        Version with +suffix stripped (e.g. "0.14.1"), or unchanged if no +
    """
    if "+" in version:
        return version.split("+", 1)[0]
    return version


def get_override_key_for_package(name: str, version: str) -> str:
    """
    Key to use when looking up a package in the license overrides dict.

    Overrides are keyed by base package name. For packages with version
    suffix (e.g. 0.14.1+cu128) or name variant (e.g. torchao-cu128), we
    try the base name so that overrides like "torchao" still match.

    Args:
        name: Package name from OSV (e.g. "torchao" or "torchao-cu128")
        version: Package version (e.g. "0.14.1+cu128")

    Returns:
        Key to use for overrides lookup (e.g. "torchao")
    """
    if not name:
        return name
    key = name
    if "+" in version or "-cu129" in name or "_cu129" in name:
        key = name.replace("-cu129", "").replace("_cu129", "").strip("-_") or name
    return key


def get_local_packages(workspace_root: Path) -> set[str]:
    """
    Read local/workspace packages from pyproject.toml.

    Extracts package names from:
    - project.name (main project)
    - tool.uv.workspace.members (UV workspace members)
    - tool.poetry.dependencies with develop=true (Poetry dev packages)

    All package names are normalized for consistent comparison.

    Args:
        workspace_root: Path to the workspace root (parent of pyproject.toml)

    Returns:
        Set of normalized local package names
    """
    if tomllib is None:
        logging.debug("tomllib not available, cannot read pyproject.toml")
        return set()

    workspace_pyproject = workspace_root / "pyproject.toml"
    if not workspace_pyproject.exists():
        logging.debug("Workspace pyproject.toml not found at %s", workspace_pyproject)
        return set()

    try:
        with open(workspace_pyproject, "rb") as f:
            data = tomllib.load(f)

        local_packages = set()

        # Check for project name
        if project_name := data.get("project", {}).get("name"):
            local_packages.add(normalize_package_name(project_name))

        # Check for workspace members in tool.uv.sources (those with workspace = true)
        if uv_sources := data.get("tool", {}).get("uv", {}).get("sources", {}):
            for name, source in uv_sources.items():
                if isinstance(source, dict) and source.get("workspace") is True:
                    local_packages.add(normalize_package_name(name))

        logging.debug("Found %d local packages in pyproject.toml: %s", len(local_packages), sorted(local_packages))
        return local_packages

    except Exception as e:
        logging.warning("Could not read pyproject.toml: %s. Using empty set for local packages.", e)
        return set()


def get_workspace_root() -> Path:
    """Get the workspace root directory."""

    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, check=True)
    if result.returncode != 0:
        raise typer.BadParameter("Could not find workspace root. Run from within the nhx repository.")

    return Path(result.stdout.decode("utf-8").strip())


def resolve_license(licenses: list[str] | str, allowed_licenses: set[str] | None = None) -> str:
    """
    Resolve a license from either a list or string.

    Prefers licenses that are in the allowed_licenses set if provided.
    For strings containing separators (AND/OR), returns the first part.

    Args:
        licenses: Either a license string or list of license strings
        allowed_licenses: Optional set of allowed license identifiers (case-insensitive)

    Returns:
        Resolved license string

    Examples:
        >>> resolve_license("MIT")
        'MIT'
        >>> resolve_license(["non-standard", "MIT"], {"MIT"})
        'MIT'
        >>> resolve_license(["Apache-2.0", "MIT"])
        'Apache-2.0'
        >>> resolve_license("Apache-2.0 OR MIT")
        'Apache-2.0'
    """
    # Normalize licenses to a list
    license_list: list[str] = []
    match licenses:
        case str():
            license_list = [licenses]
        case list():
            license_list = licenses if licenses else []
        case _:
            raise TypeError(f"licenses must be a list or str, got {type(licenses)}")

    if not license_list:
        return ""

    # If allowed_licenses provided, prefer licenses in the allowed set
    if allowed_licenses:
        # Normalize allowed licenses to uppercase for case-insensitive comparison
        normalized_allowed = {lic.upper() for lic in allowed_licenses}

        for license_str in license_list:
            # Check if this license (or its first part if it has separators) is allowed
            normalized_license = license_str.upper()

            # Check for separators and take first part
            for separator in [" AND ", " OR "]:
                if separator in normalized_license:
                    normalized_license = normalized_license.split(separator)[0]
                    break

            if normalized_license in normalized_allowed:
                # Return the original case version, split on separator if present
                for separator in [" AND ", " OR "]:
                    if separator in license_str:
                        return license_str.split(separator)[0]
                return license_str

    # Fall back to first license
    license_str = license_list[0]

    # Handle separators in the first license
    for separator in [" AND ", " OR "]:
        if separator in license_str:
            return license_str.split(separator)[0]

    return license_str

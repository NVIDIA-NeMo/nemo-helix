# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
from typing import Optional

import typer
import yaml
from nemo_helix_sdk_tools.license.generator import OVERRIDES_FILE, get_projects
from nemo_helix_sdk_tools.license.license_utils import (
    ALLOWED_LICENSES,
    get_local_packages,
    get_override_key_for_package,
    get_workspace_root,
    normalize_package_name,
    resolve_license,
)
from nemo_helix_sdk_tools.printer import print_color
from rich import get_console

logger = logging.getLogger(__name__)

console = get_console()


def get_packages_needing_overrides(
    json_file: Path,
    local_packages: set[str],
    overrides: Optional[dict] = None,
    return_versions: bool = True,
) -> list[tuple[str, str]]:
    """
    Find packages with incompatible licenses in an OSV JSON file.

    Args:
        json_file: Path to the OSV JSON file
        local_packages: Set of local package names to skip
        overrides: Optional dict of package overrides to skip
        return_versions: If True, return (name, version) tuples; if False, return just names

    Returns:
        List of packages needing overrides - either (name, version) tuples or just names
    """
    if not json_file.exists():
        logger.warning("OSV JSON file not found: %s", json_file)
        return []

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    packages_needing_overrides = []

    for result in data.get("results", []):
        for pkg_data in result.get("packages", []):
            pkg = pkg_data.get("package", {})
            name = pkg.get("name", "")
            version = pkg.get("version", "")
            licenses = pkg_data.get("licenses", [])

            # Resolve license, preferring allowed ones
            license_str = resolve_license(licenses, set(ALLOWED_LICENSES))

            # Skip local packages
            if normalize_package_name(name) in local_packages:
                continue

            # Skip if we have an override (if overrides dict provided).
            # Use override key so +cu128 variants match (e.g. torchao 0.14.1+cu128).
            if overrides and get_override_key_for_package(name, version) in overrides:
                continue

            # Check if license is incompatible
            if str(license_str).upper() not in ALLOWED_LICENSES:
                packages_needing_overrides.append((name, version))

    return packages_needing_overrides


def find_incompatible(json_file: Path, overrides: Optional[dict] = None) -> list[str]:
    """
    Find packages with incompatible licenses in a JSON file.

    This is a convenience wrapper around get_packages_needing_overrides()
    that returns just package names (for backward compatibility).
    """
    if not overrides:
        overrides = {}
        overrides_file = OVERRIDES_FILE
        if overrides_file.exists():
            with open(overrides_file, encoding="utf-8") as f:
                override_data = yaml.safe_load(f)
                overrides = override_data.get("overrides", {})

    local_packages = get_local_packages(get_workspace_root())
    return [
        name
        for name, _ in get_packages_needing_overrides(
            json_file=json_file,
            local_packages=local_packages,
            overrides=overrides,
            return_versions=False,
        )
    ]


def find_missing_licenses():
    """
    Find packages with UNKNOWN or NON-STANDARD licenses.

    Scans the OSV JSON files (format-independent) and reports which packages
    need manual license overrides in license_overrides.yaml.

    Note: This command reads the raw OSV JSON files, so it works regardless
    of what output format you used when generating licenses.
    """

    try:
        ws_root = get_workspace_root()
        projects = get_projects(ws_root)

        # Check all OSV JSON files exist
        missing_files = []
        for project in projects:
            if not project["osv_json"].exists():
                missing_files.append(project["osv_json"])

        if missing_files:
            print_color(
                "Warning: OSV JSON files not found. Run 'nemo-helix-sdk-tools license generate' first.",
                "yellow",
            )
            for f in missing_files:
                print_color(f"  Missing: {f}", "yellow")
            raise typer.Exit(1)

        console.print("\n[bold]Packages needing license overrides:[/bold]\n")

        # Find packages with incompatible licenses for each project
        results: dict[str, list[str]] = {}
        total = 0

        for project in projects:
            project_name = project["name"]
            osv_json = project["osv_json"]

            print_color(f"\n{project_name.title()}:", "cyan")
            incompatible = find_incompatible(osv_json)
            results[project_name] = incompatible

            for package in incompatible:
                print_color(f"  • {package}")

            if len(incompatible) == 0:
                print_color("  (none)", "green")

            total += len(incompatible)

        # Print summary
        summary_parts = ", ".join(f"{name.title()}: {len(pkgs)}" for name, pkgs in results.items())
        console.print(f"\n[bold]Total packages needing overrides:[/bold] {total} ({summary_parts})")

        if total > 0:
            license_path = "tools/nemo-helix-sdk-tools/src/nemo_helix_sdk_tools/license/overrides.yaml"
            print_color("\nNext steps:", "yellow")
            console.print("1. Run: [cyan]nemo-helix-sdk-tools license discover-overrides[/cyan]")
            console.print(f"2. Manually add suggested overrides to {license_path}")
            console.print(
                f"3. If the licenses are present but need to be approved (e.g., LGPL), then add them to {license_path}"
            )
            console.print("4. Re-run: [cyan]nemo-helix-sdk-tools license generate[/cyan]")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        logger.exception("Error finding missing licenses")
        print_color(f"Error: {e}", "red")
        raise typer.Exit(1)

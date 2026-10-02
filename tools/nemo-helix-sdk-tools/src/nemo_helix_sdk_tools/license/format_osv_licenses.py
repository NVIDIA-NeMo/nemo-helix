#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///
"""
Format osv-scanner JSON output to match the licensecheck table format.

This script reads the JSON output from osv-scanner and formats it
into a table similar to the original licensecheck output.
"""

import sys
from pathlib import Path
from typing import Any

import yaml
from nemo_helix_sdk_tools.license.license_utils import (
    ALLOWED_LICENSES,
    get_override_key_for_package,
    normalize_package_name,
    resolve_license,
)


def load_license_overrides() -> dict[str, str]:
    """Load license overrides from YAML file in the SDK tools package."""
    yaml_file = Path(__file__).parent / "overrides.yaml"
    if yaml_file.exists():
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
                return data.get("overrides", {})
        except Exception as e:
            print(f"Warning: Could not load license overrides from YAML: {e}", file=sys.stderr)

    return {}


def format_licenses_table(
    json_data: dict[str, Any], overrides: dict[str, str] | None = None, local_packages: set[str] | None = None
) -> str:
    if overrides is None:
        overrides = {}
    if local_packages is None:
        local_packages = set()
    if overrides:
        print(f"Loaded {len(overrides)} license overrides", file=sys.stderr)

    if local_packages:
        print(f"Excluding {len(local_packages)} local workspace packages", file=sys.stderr)

    # Build package info list
    package_info = []

    for result in json_data.get("results", []):
        for pkg_data in result.get("packages", []):
            pkg = pkg_data.get("package", {})
            name = pkg.get("name", "")
            version = pkg.get("version", "")
            ecosystem = pkg.get("ecosystem", "")
            licenses = pkg_data.get("licenses", [])

            # Skip local/workspace packages entirely (normalize for comparison)
            if normalize_package_name(name) in local_packages:
                continue

            # Override lookup: use base name so packages like torchao 0.14.1+cu128 match
            override_key = get_override_key_for_package(name, version)
            if override_key in overrides:
                license_str = overrides[override_key]
            elif not licenses:
                # Skip packages without licenses
                print(f"WARNING: No license info found for {name}", file=sys.stderr)
                continue
            else:
                # Resolve to a single license, preferring allowed ones
                license_str = resolve_license(licenses, ALLOWED_LICENSES)

            package_info.append(
                {"name": name, "version": version, "ecosystem": ecosystem, "license": license_str.upper()}
            )

    # Deduplicate by ecosystem/name (keep first occurrence)
    # This handles platform-specific variants (e.g., torch 2.9.0 vs 2.9.0+cu128)
    seen_names = set()
    unique_packages = []
    for pkg in package_info:
        package_key = (str(pkg.get("ecosystem", "")).lower(), normalize_package_name(pkg["name"]))
        if package_key not in seen_names:
            seen_names.add(package_key)
            unique_packages.append(pkg)

    # Sort packages by name
    unique_packages.sort(key=lambda x: x["name"].lower())
    print(f"Writing {len(unique_packages)} packages.", file=sys.stderr)
    package_info = unique_packages

    # Build the output
    output = []

    # List of packages (wider columns for better readability)
    output.append("┏━━━━━━━━━━━━┳" + "━" * 48 + "┳" + "━" * 40 + "┓")
    output.append("┃ Compatible ┃ Package" + " " * 40 + "┃ License(s)" + " " * 29 + "┃")
    output.append("┡━━━━━━━━━━━━╇" + "━" * 48 + "╇" + "━" * 40 + "┩")

    for pkg in package_info:
        # Mark as compatible (✔) - osv-scanner doesn't provide this info,
        # so we just mark all as compatible
        compat = "✔"
        if pkg["license"] in ["UNKNOWN", "NON-STANDARD"]:
            compat = "✘"

        # Format package name (max 50 chars)
        name = pkg["name"][:50]

        # Format license (may span multiple lines if too long)
        license_text = pkg["license"]

        # Split license into chunks that fit in 40 chars
        if len(license_text) <= 40:
            output.append(f"│ {compat:<10} │ {name:<46} │ {license_text:<38} │")
        else:
            # First line with package name
            output.append(f"│ {compat:<10} │ {name:<46} │ {license_text[:40]:<38} │")
            # Additional lines for license overflow
            remaining = license_text[40:]
            while remaining:
                chunk = remaining[:40]
                remaining = remaining[40:]
                output.append(f"│            │ {'':46} │ {chunk:<38} │")

    output.append(
        "┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩"
    )

    return "\n".join(output)

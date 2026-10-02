#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests>=2.32",
#     "pyyaml>=6.0",
# ]
# ///
"""
Generate license overrides by fetching license information from PyPI for packages
where osv-scanner reports UNKNOWN or NON-STANDARD licenses.

This script outputs suggested overrides in YAML format that can be manually
added to license_overrides.yaml, preserving existing comments and organization.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
import yaml
from nemo_helix_sdk_tools.license.find_missing import get_packages_needing_overrides
from nemo_helix_sdk_tools.license.generator import get_projects
from nemo_helix_sdk_tools.license.license_utils import get_local_packages
from requests.adapters import HTTPAdapter
from urllib3 import Retry

CURR_DIR = Path(__file__).absolute().parent


# SPDX license mapping for common non-standard license strings
SPDX_MAPPING = {
    "Apache License 2.0": "Apache-2.0",
    "Apache 2.0": "Apache-2.0",
    "Apache License, Version 2.0": "Apache-2.0",
    "Apache Software License": "Apache-2.0",
    "Apache License": "Apache-2.0",
    "BSD": "BSD-3-Clause",
    "BSD License": "BSD-3-Clause",
    "BSD 3-Clause": "BSD-3-Clause",
    "BSD-3": "BSD-3-Clause",
    "BSD 2-Clause": "BSD-2-Clause",
    "BSD-2": "BSD-2-Clause",
    "MIT License": "MIT",
    "MIT license": "MIT",
    "The MIT License": "MIT",
    "MIT License (MIT)": "MIT",
    "GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.1-or-later",
    "GNU Lesser General Public License v2 (LGPLv2)": "LGPL-2.1",
    "GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
    "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0",
    "Mozilla Public License 2.0 (MPL 2.0)": "LGPL",
    "GNU Library or Lesser General Public License (LGPL)": "LGPL-2.1-only",
    "ISC License (ISCL)": "ISC",
    "LicenseRef-NVIDIA-Proprietary": "NVIDIA Proprietary Software",
    "ISC": "ISC",
    "The Unlicense (Unlicense)": "Unlicense",
    "Python Software Foundation License": "PSF-2.0",
    "PSF": "PSF-2.0",
    "Zope Public License": "ZPL-2.0",
}

session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504, 495],
)
adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retry)
session.mount("https://", adapter)


def normalize_license_to_spdx(license_str: str) -> str:
    """Attempt to normalize a license string to SPDX format."""
    if not license_str or license_str == "UNKNOWN":
        return "UNKNOWN"

    # Try exact match first
    if license_str in SPDX_MAPPING:
        return SPDX_MAPPING[license_str]

    # Try case-insensitive match
    for key, value in SPDX_MAPPING.items():
        if license_str.lower() == key.lower():
            return value

    # Try regex match
    for key, value in SPDX_MAPPING.items():
        if re.match(f"^{key}.*", license_str):
            return value

    # If already looks like SPDX, return as-is
    if any(sep in license_str for sep in [" AND ", " OR ", " WITH "]):
        return license_str

    # Return original if no mapping found
    return license_str


def parse_license_from_pypi_json(result: dict[str, Any]) -> str:
    """Parse license from PyPI JSON response."""
    if not (info := result.get("info")):
        logging.warning("Could not find info for project %s", result.get("name"))
        return "UNKNOWN"

    # Try to get license from classifiers first (most accurate)
    if classifiers := info.get("classifiers"):
        for classifier in classifiers:
            if classifier.startswith("License :: OSI Approved ::"):
                splits = classifier.split(" :: ")
                if len(splits) >= 3:
                    license_str = splits[-1]
                    return normalize_license_to_spdx(license_str)

    # Try license_expression (PEP 639)
    if license_expression := info.get("license_expression"):
        return normalize_license_to_spdx(license_expression)

    # Fall back to license field
    if license_str := info.get("license"):
        if license_str and license_str not in ["UNKNOWN", "None", ""]:
            return normalize_license_to_spdx(license_str)

    return "UNKNOWN"


def get_license_from_pypi(package_name: str, version: str) -> tuple[str, str]:
    """Fetch license information from PyPI for a given package."""
    try:
        # Try versioned endpoint first
        normalized_version = version.split("+")[0]  # Handle versions with + suffixes
        resp = session.get(f"https://pypi.org/pypi/{package_name}/{normalized_version}/json", timeout=10)

        if not resp.ok:
            # Fall back to unversioned endpoint
            resp = session.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)

        if not resp.ok:
            logging.warning("Could not fetch PyPI data for %s: HTTP %d", package_name, resp.status_code)
            return (package_name, "UNKNOWN")

        result = resp.json()
        license_str = parse_license_from_pypi_json(result)

        if len(license_str) > 200:  # Truncate
            license_str = license_str[:200] + "..."

        if license_str == "UNKNOWN":
            logging.warning("Could not determine license for %s", package_name)
        else:
            logging.info("Found license for %s: %s", package_name, license_str)

        return (package_name, license_str)

    except Exception as e:
        logging.error("Error fetching license for %s: %s", package_name, e)
        return (package_name, "UNKNOWN")


def load_existing_overrides() -> dict[str, str]:
    """Load existing license overrides from the SDK tools package."""
    OVERRIDE_FILE = Path(__file__).parent / "overrides.yaml"
    if OVERRIDE_FILE.exists():
        try:
            with open(OVERRIDE_FILE, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data is None:
                    return {}
                return data.get("overrides", {})
        except Exception as e:
            logging.warning("Could not load existing overrides: %s", e)
    return {}


def get_all_packages_needing_overrides(workspace_root: Path) -> list[tuple[str, str]]:
    projects = get_projects(workspace_root=workspace_root)
    osv_json_main = projects[0]["osv_json"]
    # Get local packages once
    local_packages = get_local_packages(workspace_root)

    # Check main project - using centralized function from find_missing
    main_packages = get_packages_needing_overrides(
        json_file=osv_json_main,
        local_packages=local_packages,
        overrides=None,  # Don't filter by overrides here, we do that later
        return_versions=True,
    )
    logging.info("Found %d packages needing lookup in main project", len(main_packages))

    # Deduplicate (keep first occurrence)
    seen = set()
    all_packages = []
    for name, version in main_packages:
        if name not in seen:
            seen.add(name)
            all_packages.append((name, version))

    if not all_packages and not osv_json_main.exists():
        logging.error("No OSV scanner JSON files found!")
        logging.error("Please run: make update-licenses")
        return []

    logging.info("Total unique packages needing license lookup: %d", len(all_packages))
    return all_packages


def fetch_licenses_from_pypi(packages: list[tuple[str, str]]) -> dict[str, str]:
    """Fetch licenses from PyPI for multiple packages in parallel."""
    logging.info("Fetching licenses from PyPI for %d packages...", len(packages))

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(lambda pkg: get_license_from_pypi(pkg[0], pkg[1]), packages)
        return {name: license_str for name, license_str in results if license_str != "UNKNOWN"}


def print_override_suggestions(overrides: dict[str, str]) -> None:
    """Print license overrides in YAML format for manual addition to license_overrides.yaml."""
    if not overrides:
        return

    print("\n" + "=" * 80)
    print("SUGGESTED LICENSE OVERRIDES")
    print("=" * 80)
    print("\nAdd these to tools/nemo-helix-sdk-tools/src/nemo_helix_sdk_tools/license/overrides.yaml:\n\n")
    print("overrides:")

    for key in sorted(overrides.keys()):
        print(f"  {key}: {overrides[key]}")


def generate_overrides(workspace_root: Path):
    """Main entry point."""
    logging.info("Loading existing overrides...")
    existing_overrides = load_existing_overrides()
    if not existing_overrides:
        existing_overrides = []
    logging.info("Found %d existing overrides", len(existing_overrides))

    logging.info("Finding packages needing license overrides (not present in the osv-scan results)...")
    packages = get_all_packages_needing_overrides(workspace_root=workspace_root)

    if not packages:
        logging.error("No packages found. Make sure OSV JSON file exists in third_party/licenses*txt")
        return

    # Filter out packages that already have overrides
    packages_to_fetch = [(name, version) for name, version in packages if name not in existing_overrides]

    logging.info("Need to fetch licenses for %d new packages", len(packages_to_fetch))

    if packages_to_fetch:
        new_overrides = fetch_licenses_from_pypi(packages_to_fetch)
        logging.info("Successfully fetched %d licenses from PyPI", len(new_overrides))
    else:
        new_overrides = {}
        logging.info("All packages either have overrides or could not be found on PyPI")

    # Print summary
    total_with_overrides = len(existing_overrides)
    still_unknown = len([p for p in packages if p[0] not in existing_overrides and p[0] not in new_overrides])

    logging.info("\nSummary:")
    logging.info("  Total packages checked: %d", len(packages))
    logging.info("  Already have overrides: %d", total_with_overrides)
    logging.info("  New licenses found: %d", len(new_overrides))
    logging.info("  Still unknown: %d", still_unknown)

    # Print new overrides for manual addition
    if still_unknown > 0:
        print("\n" + "=" * 80)
        print("PACKAGES STILL NEEDING MANUAL REVIEW")
        print("=" * 80)
        print("\nThese packages were not found on PyPI or have non-standard licenses:")
        for name, _ in packages:
            if name not in existing_overrides and name not in new_overrides:
                print(f"  - {name}")
        print("\nManually research these on GitHub/PyPI links and add to license_overrides.yaml")
        print("=" * 80)

    if new_overrides:
        print_override_suggestions(new_overrides)
    else:
        print("\n✅ No new license overrides to add!")

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test collection and breakdown script using native pytest collection.

Collects tests by hooking into pytest's collection phase directly,
providing a more robust and accurate breakdown of test distribution and markers.
"""

import contextlib
import io
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

import pytest


@dataclass
class TestItem:
    nodeid: str
    name: str
    fspath: Path
    markers: Set[str] = field(default_factory=set)

    @property
    def top_dir(self) -> str:
        try:
            # Try to get path relative to CWD
            rel_path = self.fspath.relative_to(Path.cwd())
        except ValueError:
            # If path is not relative to CWD (e.g. installed package), use as is or handle gracefully
            rel_path = self.fspath

        parts = rel_path.parts

        # Check for services/core/infrastructure/X
        if len(parts) >= 4 and parts[0] == "services" and parts[1] == "core" and parts[2] == "infrastructure":
            return f"{parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}"

        if len(parts) >= 2 and parts[0] in ("services", "packages"):
            return f"{parts[0]}/{parts[1]}"
        return parts[0] if parts else "unknown"


class CollectionPlugin:
    def __init__(self):
        self.collected_items: List[TestItem] = []

    def pytest_collection_modifyitems(self, session, config, items):
        for item in items:
            # Get explicit markers from pytest
            # item.iter_markers() returns all markers including nested ones
            markers = {m.name for m in item.iter_markers()}

            # Get path (handle both legacy fspath and new path)
            fpath = getattr(item, "path", Path(str(item.fspath)))

            # Infer markers from path/name to match original script's logic + improvements
            path_lower = str(fpath).lower()
            nodeid_lower = item.nodeid.lower()

            if "/e2e/" in path_lower or "::teste2e" in nodeid_lower or "::test_e2e" in nodeid_lower:
                markers.add("e2e")
            elif "/integration/" in path_lower:
                markers.add("integration")
            elif "/regression/" in path_lower or "regression" in path_lower:
                markers.add("regression")

            # Default to unit if no other major type is found
            # Note: The original script defaults to 'unit' if others aren't found.
            if not (markers & {"integration", "e2e", "regression"}):
                markers.add("unit")

            # Check for slow - combining explicit marker and path inference
            if (
                "slow" in path_lower
                or "test_all_available_locales" in item.nodeid
                or "test_column_distribution" in item.nodeid
                or "test_correlation" in item.nodeid
                or "test_deep_structure" in item.nodeid
                or "test_pii_replay" in item.nodeid
            ):
                markers.add("slow")

            self.collected_items.append(TestItem(nodeid=item.nodeid, name=item.name, fspath=fpath, markers=markers))


@dataclass
class DirStats:
    """Test counts for one top-level directory."""

    total: int = 0
    markers: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))


def analyze_tests(items: List[TestItem]):
    """Analyze collected tests by directory and markers."""
    by_dir: defaultdict[str, DirStats] = defaultdict(DirStats)
    overall_markers = defaultdict(int)

    for item in items:
        top_dir = item.top_dir

        by_dir[top_dir].total += 1

        # Count markers for this item
        # Note: A test can have multiple markers (e.g., integration AND slow)
        for marker in item.markers:
            # Filter to only the markers we care about for the report to keep it clean
            if marker in ("unit", "integration", "e2e", "regression", "slow"):
                by_dir[top_dir].markers[marker] += 1
                overall_markers[marker] += 1

    return by_dir, overall_markers


def print_breakdown(by_dir, overall_markers, total_tests):
    """Print the breakdown in a readable format."""

    print("\n" + "=" * 80)
    print(f"TEST COLLECTION SUMMARY - Total: {total_tests} tests")
    print("=" * 80)
    print("\nNote: Markers are collected from @pytest.mark decorators AND inferred from paths.\n")

    # Overall markers
    print("By Marker (Overall):")
    print("-" * 40)
    marker_order = ["unit", "integration", "e2e", "regression", "slow"]
    for marker in marker_order:
        count = overall_markers.get(marker, 0)
        if count > 0:
            percentage = (count / total_tests * 100) if total_tests > 0 else 0
            print(f"  {marker:20s} {count:6d} ({percentage:5.1f}%)")

    # By directory
    print("\n\nBy Directory:")
    print("-" * 100)
    print(f"{'Directory':<60} {'Total':>8} {'Unit':>6} {'Int':>6} {'E2E':>6} {'Slow':>6}")
    print("-" * 100)

    # Sort by directory name
    for dir_name in sorted(by_dir.keys()):
        info = by_dir[dir_name]
        total = info.total
        unit = info.markers.get("unit", 0)
        integration = info.markers.get("integration", 0)
        e2e = info.markers.get("e2e", 0)
        slow = info.markers.get("slow", 0)

        print(f"{dir_name:<60} {total:8d} {unit:6d} {integration:6d} {e2e:6d} {slow:6d}")

    print("-" * 100)

    # Summary by category
    print("\n\nCategory Breakdown:")
    print("-" * 40)
    services_total = sum(info.total for dir_name, info in by_dir.items() if dir_name.startswith("services/"))
    packages_total = sum(info.total for dir_name, info in by_dir.items() if dir_name.startswith("packages/"))
    other_total = total_tests - services_total - packages_total

    if total_tests > 0:
        print(f"  Services: {services_total:6d} tests ({services_total / total_tests * 100:5.1f}%)")
        print(f"  Packages: {packages_total:6d} tests ({packages_total / total_tests * 100:5.1f}%)")
        if other_total > 0:
            print(f"  Other:    {other_total:6d} tests ({other_total / total_tests * 100:5.1f}%)")

    print("\n" + "=" * 80 + "\n")


def main():
    print("Collecting tests (this may take 30-60 seconds)...", file=sys.stderr)

    # Initialize our custom plugin
    plugin = CollectionPlugin()

    # Arguments for pytest
    # -qq: very quiet (minimize output)
    # --collect-only: don't run tests
    # --ignore=tests/: ignore top-level tests folder (as per original script)
    args = [
        "-qq",
        "--collect-only",
        "--ignore=tests/",
        # Add root dir explicitly to be safe
        ".",
    ]

    try:
        # Run pytest in-process
        # We assume pytest is installed in the current environment
        # Capture stdout to suppress the default test listing from --collect-only
        with contextlib.redirect_stdout(io.StringIO()):
            retcode = pytest.main(args, plugins=[plugin])

        if retcode not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
            # 5 is NO_TESTS_COLLECTED
            if retcode != 5:
                print(f"Warning: Pytest collection returned code {retcode}", file=sys.stderr)

        items = plugin.collected_items

        if not items:
            print("No tests collected.", file=sys.stderr)
            return 1

        by_dir, overall_markers = analyze_tests(items)
        print_breakdown(by_dir, overall_markers, len(items))

        return 0

    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

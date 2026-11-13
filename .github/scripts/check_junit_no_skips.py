# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail when a JUnit report ran no tests, or skipped any.

For a CI lane that exists to run particular tests. A skip there means a marker, a feature flag or
an environment gate selected the wrong thing, and without this the lane reports green having checked
nothing. An empty report is refused for the same reason.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def counts(report: Path) -> tuple[int, int]:
    """``(tests, skipped)`` across the report's test suites.

    pytest writes one ``testsuite`` under a ``testsuites`` root; a bare ``testsuite`` root is the
    older layout. Only suites directly under the root are counted, so a nested suite whose parent
    already totals it is not counted twice.
    """
    root = ET.parse(report).getroot()
    suites = [root] if root.tag == "testsuite" else [suite for suite in root if suite.tag == "testsuite"]
    tests = sum(int(suite.get("tests", 0)) for suite in suites)
    skipped = sum(int(suite.get("skipped", 0)) for suite in suites)
    return tests, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JUnit XML report to check")
    parser.add_argument("--label", default="tests", help="what the report covers, for the message")
    args = parser.parse_args(argv)

    try:
        tests, skipped = counts(args.report)
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"{args.label}: cannot read JUnit report {args.report}: {error}", file=sys.stderr)
        return 1
    if tests == 0 or skipped:
        print(f"{args.label} ran {tests} tests and skipped {skipped}: expected none skipped", file=sys.stderr)
        return 1
    print(f"{args.label}: {tests} tests, none skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

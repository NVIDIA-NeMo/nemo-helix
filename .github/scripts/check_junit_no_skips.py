# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail when a JUnit report ran no tests, skipped any, or ran other than the expected number.

For a CI lane that exists to run particular tests. A skip there means a marker, a feature flag or
an environment gate selected the wrong thing, and without this the lane reports green having checked
nothing. An empty report is refused for the same reason. pytest records an xfail as a skip, so one
fails here too, and the message gives each skip's type.

A test that a marker deselects is not in the report at all, so it is not a skip. Only
``--expect-tests`` catches that: pass the number of tests the lane is meant to run.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def read_report(report: Path) -> tuple[int, list[str]]:
    """``(tests, skips)``: the report's test count and one line describing each skip in it.

    pytest writes one ``testsuite`` under a ``testsuites`` root; a bare ``testsuite`` root is the
    older layout. Tests are counted from the suites directly under the root, so a nested suite whose
    parent already totals it is not counted twice. Skips are looked for at every depth: any
    ``skipped`` element, and any suite whose own ``skipped`` count is not zero although nothing under
    it says which test, since not every writer records the case or totals a nested suite into its
    parent.
    """
    root = ET.parse(report).getroot()
    suites = [root] if root.tag == "testsuite" else [suite for suite in root if suite.tag == "testsuite"]
    tests = sum(int(suite.get("tests", 0)) for suite in suites)

    # ElementTree keeps no link to an element's parent, and a skip is named by the case holding it.
    parents = {child: parent for parent in root.iter() for child in parent}
    skips = []
    for skipped in root.iter("skipped"):
        case = parents.get(skipped)
        if case is not None and case.tag == "testcase":
            where = f"{case.get('classname', '')}::{case.get('name', '')}"
        else:
            where = "outside any test case"
        reason = skipped.get("message") or (skipped.text or "").strip() or "no reason given"
        skips.append(f"{where} ({skipped.get('type') or 'skipped'}): {reason}")
    for suite in root.iter("testsuite"):
        count = int(suite.get("skipped", 0))
        if count and next(suite.iter("skipped"), None) is None:
            skips.append(f"suite {suite.get('name', '')!r} counts {count} skipped but names no test")
    return tests, skips


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JUnit XML report to check")
    parser.add_argument("--label", default="tests", help="what the report covers, for the message")
    parser.add_argument(
        "--expect-tests",
        type=int,
        metavar="N",
        help="fail unless the report holds exactly N tests, which catches a test the lane's marker deselects",
    )
    args = parser.parse_args(argv)

    try:
        tests, skips = read_report(args.report)
    except (OSError, ET.ParseError, ValueError, LookupError) as error:
        print(f"{args.label}: cannot read JUnit report {args.report}: {error}", file=sys.stderr)
        return 1

    failed = False
    if tests == 0:
        print(f"{args.label}: the report holds no tests", file=sys.stderr)
        failed = True
    elif args.expect_tests is not None and tests != args.expect_tests:
        print(f"{args.label}: expected {args.expect_tests} tests, the report holds {tests}", file=sys.stderr)
        failed = True
    if skips:
        print(f"{args.label}: expected no skips, found:", file=sys.stderr)
        for skip in skips:
            print(f"  {skip}", file=sys.stderr)
        failed = True
    if failed:
        return 1
    print(f"{args.label}: {tests} tests, none skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

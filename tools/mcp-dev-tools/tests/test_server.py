#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quick test to verify the dev tools MCP server works."""

import sys
from pathlib import Path

# Add parent directory to path to import the script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the standalone script (nhx_dev_mcp.py)
import nhx_dev_mcp

create_server = nhx_dev_mcp.create_server
parse_pytest_output = nhx_dev_mcp.parse_pytest_output


def test_pytest_parser():
    """Test that pytest output parsing works correctly."""
    print("\nTesting pytest output parser...")

    # Sample pytest output (similar to what we saw earlier)
    sample_output = """
============================= test session starts ==============================
platform darwin -- Python 3.11.13, pytest-8.4.2, pluggy-1.6.0
rootdir: /Users/mkornfield/dev/nhx2
configfile: pytest.ini

FAILED packages/nhx_common/tests/observability/test_otel.py::TestLoggingIntegration::test_initialize_obs_configures_json_logging_when_env_set - subprocess.TimeoutExpired
ERROR packages/test_something.py::test_broken - Exception: Something broke

= 1 failed, 4628 passed, 75 skipped, 2 xfailed, 110 warnings, 22 subtests passed in 207.29s (0:03:27) =
"""

    # Use the actual parse_pytest_output function from the production code
    summary = parse_pytest_output(sample_output)

    # Verify parsing results
    assert summary is not None, "Parser should return a summary"
    assert summary["passed"] == 4628, (
        f"Expected 4628 passed, got {summary.get('passed')}"
    )
    assert summary["failed"] == 1, f"Expected 1 failed, got {summary.get('failed')}"
    assert summary["skipped"] == 75, (
        f"Expected 75 skipped, got {summary.get('skipped')}"
    )
    assert summary["xfailed"] == 2, f"Expected 2 xfailed, got {summary.get('xfailed')}"
    assert summary["warnings"] == 110, (
        f"Expected 110 warnings, got {summary.get('warnings')}"
    )
    assert summary["subtests_passed"] == 22, (
        f"Expected 22 subtests, got {summary.get('subtests_passed')}"
    )
    assert summary["duration"] == "207.29s (0:03:27)", (
        f"Expected duration, got {summary.get('duration')}"
    )
    assert len(summary["failed_tests"]) == 1, (
        f"Expected 1 failed test, got {len(summary.get('failed_tests', []))}"
    )
    assert len(summary["error_tests"]) == 1, (
        f"Expected 1 error test, got {len(summary.get('error_tests', []))}"
    )

    print(
        "✓ Parser correctly extracts: passed, failed, skipped, xfailed, warnings, subtests"
    )
    print("✓ Parser correctly extracts: duration")
    print("✓ Parser correctly extracts: failed test names")
    print("✓ Parser correctly extracts: error test names")
    print(f"✓ Parsed summary: {summary}")


def test_server():
    """Test that the server can be created without errors."""
    print("Creating dev tools MCP server...")
    server = create_server()

    print(f"✓ Server created: {server.name}")
    print("✓ Server is ready to handle MCP protocol requests")

    # The actual tools are registered via decorators in server.py
    # They'll be available when the server runs with stdio transport
    print("\nExpected tools:")
    expected_tools = [
        "git_status",
        "git_log",
        "git_branch_list",
        "git_diff_summary",
        "git_diff_staged",
        "git_diff",
        "git_show",
        "run_unit_tests",
        "run_integration_tests",
        "run_service_tests",
        "run_pytest",
        "run_precommit",
        "run_ruff_check",
        "run_ruff_format",
        "run_type_check",
        "list_directory",
        "find_files",
        "make_target",
        "make_docs",
    ]

    for tool in expected_tools:
        print(f"  - {tool}")

    print(f"\nTotal: {len(expected_tools)} tools")
    print("\nTo test interactively, run:")
    print(
        "  npx @modelcontextprotocol/inspector uv run tools/mcp-dev-tools/nhx_dev_mcp.py"
    )


if __name__ == "__main__":
    test_server()
    test_pytest_parser()

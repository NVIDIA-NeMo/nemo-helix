#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Development tools MCP server for NeMo Platform.

This MCP server provides narrowly-scoped development operations with
dedicated, purpose-built tools. Each tool executes a specific, pre-defined
command via Python subprocess.

Usage:
    uv run nmp-dev-mcp
    uv run nmp-dev-mcp --working-dir /path/to/repo
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any

from fastmcp import FastMCP  # ty: ignore[unresolved-import]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_pytest_output(output: str) -> dict[str, Any] | None:
    """
    Parse pytest output to extract test summary.

    Returns:
        Dictionary with parsed test metrics, or None if parsing fails
    """
    import re

    summary = {}

    # Find the summary line (e.g., "= 1 failed, 4628 passed, 75 skipped, 2 xfailed ... in 207.29s =")
    summary_pattern = r"=+\s*(.*?)\s+in\s+([\d.]+s.*?)\s*=+\s*$"
    match = re.search(summary_pattern, output, re.MULTILINE)

    if match:
        summary_text = match.group(1)
        duration = match.group(2)

        # Extract individual counts
        for metric in [
            "passed",
            "failed",
            "skipped",
            "xfailed",
            "xpassed",
            "error",
            "errors",
        ]:
            pattern = rf"(\d+)\s+{metric}"
            metric_match = re.search(pattern, summary_text)
            if metric_match:
                summary[metric] = int(metric_match.group(1))

        # Extract warnings count
        warnings_match = re.search(r"(\d+)\s+warnings?", summary_text)
        if warnings_match:
            summary["warnings"] = int(warnings_match.group(1))

        # Extract subtests
        subtests_match = re.search(r"(\d+)\s+subtests passed", summary_text)
        if subtests_match:
            summary["subtests_passed"] = int(subtests_match.group(1))

        summary["duration"] = duration

    # Find failed test names
    failed_tests = []
    failed_pattern = r"^FAILED\s+(.+?)(?:\s+-\s+.+)?$"
    for line in output.split("\n"):
        match = re.match(failed_pattern, line)
        if match:
            failed_tests.append(match.group(1))

    if failed_tests:
        summary["failed_tests"] = failed_tests

    # Find error test names
    error_tests = []
    error_pattern = r"^ERROR\s+(.+?)(?:\s+-\s+.+)?$"
    for line in output.split("\n"):
        match = re.match(error_pattern, line)
        if match:
            error_tests.append(match.group(1))

    if error_tests:
        summary["error_tests"] = error_tests

    return summary if summary else None


def create_server(working_dir: str | None = None) -> FastMCP:
    """
    Create and configure the dev tools MCP server.

    Args:
        working_dir: Optional working directory for commands (defaults to repo root)

    Returns:
        Configured FastMCP server instance with dev tools
    """
    server = FastMCP("NeMo Platform Development Tools")

    # Determine working directory (default to repo root)
    if working_dir is None:
        # This script is in tools/mcp-dev-tools/
        # Navigate up to repo root
        working_dir = str(Path(__file__).parent.parent.parent)

    cwd = Path(working_dir).resolve()

    def run_command(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
        """Execute a command and return structured result."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": " ".join(cmd),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "command": " ".join(cmd),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": " ".join(cmd),
            }

    def validate_path(path: str, param_name: str = "path") -> dict[str, Any] | None:
        """
        Validate that a path is safe and stays within the repository.

        Args:
            path: The path to validate
            param_name: Name of the parameter (for error messages)

        Returns:
            None if valid, or error dict if invalid
        """
        from pathlib import Path

        # Block absolute paths
        if path.startswith("/"):
            return {
                "success": False,
                "error": f"Invalid {param_name} '{path}'. Absolute paths not allowed. Use relative paths from repo root.",
            }

        # Block parent directory references
        if ".." in path:
            return {
                "success": False,
                "error": f"Invalid {param_name} '{path}'. Parent directory references (..) not allowed.",
            }

        # Ensure path stays within repo
        try:
            resolved = (Path(cwd) / path).resolve()
            if not resolved.is_relative_to(cwd):
                return {
                    "success": False,
                    "error": f"Invalid {param_name} '{path}'. Path must stay within repository.",
                }
        except (ValueError, OSError):
            return {
                "success": False,
                "error": f"Invalid {param_name} '{path}'. Path resolution failed.",
            }

        return None  # Valid path

    # === GIT OPERATIONS ===

    @server.tool(description="Get git repository status")
    async def git_status() -> dict[str, Any]:
        """
        Get current git repository status.

        Returns:
            Dictionary with git status output
        """
        return run_command(["git", "status"])

    @server.tool(description="Show git commit history")
    async def git_log(limit: int = 10) -> dict[str, Any]:
        """
        Show git commit history.

        Args:
            limit: Number of commits to show (default: 10)

        Returns:
            Dictionary with git log output
        """
        import re

        if not re.match(r"^\d+$", str(limit)) or limit < 1 or limit > 100:
            return {
                "success": False,
                "error": f"Invalid limit '{limit}'. Must be integer between 1 and 100.",
            }

        return run_command(["git", "log", f"-{limit}", "--oneline"])

    @server.tool(description="List git branches")
    async def git_branch_list() -> dict[str, Any]:
        """
        List all git branches.

        Returns:
            Dictionary with git branch list
        """
        return run_command(["git", "branch", "-a", "-v"])

    @server.tool(description="Show git diff summary")
    async def git_diff_summary() -> dict[str, Any]:
        """
        Show summary of unstaged changes.

        Returns:
            Dictionary with git diff --stat output
        """
        return run_command(["git", "diff", "--stat"])

    @server.tool(description="Show staged changes summary")
    async def git_diff_staged() -> dict[str, Any]:
        """
        Show summary of staged changes ready for commit.

        Returns:
            Dictionary with git diff --staged --stat output
        """
        return run_command(["git", "diff", "--staged", "--stat"])

    @server.tool(description="Show full git diff")
    async def git_diff(
        staged: bool = False, file_path: str | None = None
    ) -> dict[str, Any]:
        """
        Show full git diff output.

        Args:
            staged: Show staged changes instead of unstaged (default: False)
            file_path: Optional specific file to diff

        Returns:
            Dictionary with git diff output
        """
        import re

        cmd = ["git", "diff"]

        if staged:
            cmd.append("--staged")

        if file_path:
            # Validate file_path to prevent option injection
            if file_path.startswith("-"):
                return {
                    "success": False,
                    "error": f"Invalid file path '{file_path}'. Paths cannot start with '-'.",
                }

            if not re.match(r"^[a-zA-Z0-9._/~^-]+$", file_path):
                return {
                    "success": False,
                    "error": f"Invalid file path '{file_path}'. Only alphanumerics, dots, hyphens, underscores, slashes, tildes, and carets allowed.",
                }

            cmd.append("--")
            cmd.append(file_path)

        return run_command(cmd)

    @server.tool(description="Show git commit details")
    async def git_show(commit: str = "HEAD", stat: bool = False) -> dict[str, Any]:
        """
        Show git commit details.

        Args:
            commit: Commit reference (default: HEAD)
            stat: Show diffstat instead of full diff (default: False)

        Returns:
            Dictionary with git show output
        """
        # Validate commit reference to prevent option injection
        if commit.startswith("-"):
            return {
                "success": False,
                "error": f"Invalid commit reference '{commit}'. Commit refs cannot start with '-'.",
            }

        import re

        if not re.match(r"^[a-zA-Z0-9._/~^-]+$", commit):
            return {
                "success": False,
                "error": f"Invalid commit reference '{commit}'. Only alphanumerics, dots, hyphens, underscores, slashes, tildes, and carets allowed.",
            }

        cmd = ["git", "show", commit]

        if stat:
            cmd.append("--stat")

        return run_command(cmd)

    # === TESTING OPERATIONS ===

    @server.tool(description="Run all unit tests")
    async def run_unit_tests() -> dict[str, Any]:
        """
        Run all unit tests for the project.

        Returns:
            Dictionary with pytest output and parsed summary.
            If output is large, stdout is truncated but summary is always included.
        """
        result = run_command(["make", "test-unit"], timeout=300)

        # Add parsed summary if pytest output is present
        if result.get("stdout"):
            summary = parse_pytest_output(result["stdout"])
            if summary:
                result["summary"] = summary

            # Truncate large output but keep last 5000 chars (includes summary)
            if len(result["stdout"]) > 10000:
                result["stdout_truncated"] = True
                result["stdout"] = (
                    "... [output truncated] ...\n\n" + result["stdout"][-5000:]
                )

        return result

    @server.tool(description="Run integration tests")
    async def run_integration_tests() -> dict[str, Any]:
        """
        Run integration tests.

        Returns:
            Dictionary with pytest output and parsed summary.
            If output is large, stdout is truncated but summary is always included.
        """
        result = run_command(["make", "test-integration"], timeout=300)

        # Add parsed summary if pytest output is present
        if result.get("stdout"):
            summary = parse_pytest_output(result["stdout"])
            if summary:
                result["summary"] = summary

            # Truncate large output but keep last 5000 chars (includes summary)
            if len(result["stdout"]) > 10000:
                result["stdout_truncated"] = True
                result["stdout"] = (
                    "... [output truncated] ...\n\n" + result["stdout"][-5000:]
                )

        return result

    @server.tool(description="Run tests for a specific service")
    async def run_service_tests(service_name: str) -> dict[str, Any]:
        """
        Run tests for a specific service.

        Args:
            service_name: Name of the service to test (e.g., 'evaluator', 'auth')

        Returns:
            Dictionary with pytest output and parsed summary.
            If output is large, stdout is truncated but summary is always included.
        """
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", service_name):
            return {
                "success": False,
                "error": f"Invalid service name '{service_name}'. Only alphanumerics, hyphens, and underscores are allowed.",
            }

        result = run_command(["make", f"test-unit-{service_name}"], timeout=300)

        # Add parsed summary if pytest output is present
        if result.get("stdout"):
            summary = parse_pytest_output(result["stdout"])
            if summary:
                result["summary"] = summary

            # Truncate large output but keep last 5000 chars (includes summary)
            if len(result["stdout"]) > 10000:
                result["stdout_truncated"] = True
                result["stdout"] = (
                    "... [output truncated] ...\n\n" + result["stdout"][-5000:]
                )

        return result

    @server.tool(description="Run pytest on a directory or file")
    async def run_pytest(
        path: str = ".",
        verbose: bool = True,
        markers: str | None = None,
    ) -> dict[str, Any]:
        """
        Run pytest on a specific directory or file with optional filtering.

        Args:
            path: Directory or file path to test (default: "." for all tests)
                  Must be a relative path within the repository
            verbose: Show verbose output with test names (default: True)
            markers: Optional pytest marker expression (e.g., "not slow", "unit")

        Returns:
            Dictionary with pytest output and parsed summary.
            If output is large, stdout is truncated but summary is always included.

        Examples:
            - run_pytest("tools/mcp-dev-tools/tests") - Run all tests in directory
            - run_pytest("packages/nmp_common", verbose=False) - Run without verbose
        """
        import re

        # Validate path using shared validation function
        error = validate_path(path, "path")
        if error:
            return error

        # Validate markers if provided
        if markers:
            # Allow pytest marker syntax: alphanumerics (covers "and"/"or"/"not"),
            # underscores, spaces, and parentheses.
            if not re.match(r"^[a-zA-Z0-9_ ()]+$", markers):
                return {
                    "success": False,
                    "error": f"Invalid markers '{markers}'. Only alphanumerics, spaces, and logical operators (and, or, not) allowed.",
                }

        # Build pytest command
        cmd = ["uv", "run", "--frozen", "pytest", path]

        if verbose:
            cmd.append("-v")

        if markers:
            cmd.extend(["-m", markers])

        result = run_command(cmd, timeout=300)

        # Add parsed summary if pytest output is present
        if result.get("stdout"):
            summary = parse_pytest_output(result["stdout"])
            if summary:
                result["summary"] = summary

            # Truncate large output but keep last 5000 chars (includes summary)
            if len(result["stdout"]) > 10000:
                result["stdout_truncated"] = True
                result["stdout"] = (
                    "... [output truncated] ...\n\n" + result["stdout"][-5000:]
                )

        return result

    @server.tool(description="Check pre-commit hooks")
    async def run_precommit() -> dict[str, Any]:
        """
        Run pre-commit hooks on all files.

        Returns:
            Dictionary with pre-commit output
        """
        return run_command(["uv", "run", "pre-commit", "run", "-a"], timeout=300)

    # === LINTING AND TYPE CHECKING ===

    @server.tool(description="Run ruff linter")
    async def run_ruff_check(path: str | None = None) -> dict[str, Any]:
        """
        Run ruff linter.

        Args:
            path: Optional specific path to check (default: all files)

        Returns:
            Dictionary with ruff check output
        """
        cmd = ["uv", "run", "ruff", "check"]
        if path:
            # Validate path to prevent option injection and path traversal
            if path.startswith("-"):
                return {
                    "success": False,
                    "error": f"Invalid path '{path}'. Paths cannot start with '-'.",
                }

            # Normalize and validate path
            try:
                resolved_path = Path(path).resolve()
                if not resolved_path.exists():
                    return {
                        "success": False,
                        "error": f"Path '{path}' does not exist.",
                    }

                # Ensure path is under working directory (prevent escaping)
                try:
                    resolved_path.relative_to(cwd)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Path '{path}' is outside the working directory.",
                    }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Invalid path '{path}': {e}",
                }

            # Use -- separator to prevent path from being interpreted as option
            cmd.append("--")
            cmd.append(str(resolved_path))

        return run_command(cmd, timeout=120)

    @server.tool(description="Run ruff formatter")
    async def run_ruff_format(
        path: str | None = None, check_only: bool = True
    ) -> dict[str, Any]:
        """
        Run ruff formatter.

        Args:
            path: Optional specific path to format (default: all files)
            check_only: Only check formatting without modifying files (default: True)

        Returns:
            Dictionary with ruff format output
        """
        cmd = ["uv", "run", "ruff", "format"]
        if check_only:
            cmd.append("--check")

        if path:
            # Validate path to prevent option injection and path traversal
            if path.startswith("-"):
                return {
                    "success": False,
                    "error": f"Invalid path '{path}'. Paths cannot start with '-'.",
                }

            # Normalize and validate path
            try:
                resolved_path = Path(path).resolve()
                if not resolved_path.exists():
                    return {
                        "success": False,
                        "error": f"Path '{path}' does not exist.",
                    }

                # Ensure path is under working directory (prevent escaping)
                try:
                    resolved_path.relative_to(cwd)
                except ValueError:
                    return {
                        "success": False,
                        "error": f"Path '{path}' is outside the working directory.",
                    }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Invalid path '{path}': {e}",
                }

            # Use -- separator to prevent path from being interpreted as option
            cmd.append("--")
            cmd.append(str(resolved_path))

        return run_command(cmd, timeout=120)

    @server.tool(description="Run type checker")
    async def run_type_check() -> dict[str, Any]:
        """
        Run type checker (ty) on the codebase.

        Returns:
            Dictionary with type checker output
        """
        return run_command(
            ["uv", "run", "--frozen", "--extra", "cpu", "ty", "check"], timeout=300
        )

    # === PROJECT NAVIGATION ===

    @server.tool(description="List directory contents")
    async def list_directory(path: str = ".") -> dict[str, Any]:
        """
        List contents of a directory in the repository.

        Args:
            path: Directory path relative to repo root (default: ".")
                  Must be a relative path within the repository

        Returns:
            Dictionary with directory listing

        Examples:
            - list_directory("services") - List all services
            - list_directory("packages") - List all packages
            - list_directory("tools") - List tools directory
            - list_directory() - List repo root
        """
        # Validate path using shared validation function
        error = validate_path(path, "path")
        if error:
            return error

        return run_command(["ls", "-1", path])

    @server.tool(description="Find files in repository")
    async def find_files(
        pattern: str,
        path: str = ".",
        file_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Find files in the repository matching a pattern.

        Args:
            pattern: Filename pattern (e.g., "*.py", "test_*.py", "api.py")
                    Supports wildcards: * (any chars), ? (single char)
            path: Starting directory relative to repo root (default: ".")
                  Must be a relative path within the repository
            file_type: Optional filter: "f" (files only), "d" (directories only)

        Returns:
            Dictionary with list of matching file paths

        Examples:
            - find_files("*.py", "services") - All Python files in services/
            - find_files("test_*.py") - All test files in repo
            - find_files("*api*", "packages", "f") - Files with 'api' in name under packages/
        """
        import re

        # Validate pattern: allow wildcards, alphanumerics, common filename chars
        if not re.match(r"^[a-zA-Z0-9_.*?-]+$", pattern):
            return {
                "success": False,
                "error": f"Invalid pattern '{pattern}'. Only alphanumerics, underscores, hyphens, dots, and wildcards (* ?) allowed.",
            }

        # Validate path using shared validation function
        error = validate_path(path, "path")
        if error:
            return error

        # Validate file_type
        if file_type is not None and file_type not in ("f", "d"):
            return {
                "success": False,
                "error": f"Invalid file_type '{file_type}'. Must be 'f' (file) or 'd' (directory).",
            }

        # Build find command
        cmd = ["find", path, "-name", pattern]

        if file_type:
            cmd.extend(["-type", file_type])

        return run_command(cmd)

    # === MAKE TARGET OPERATIONS ===

    @server.tool(description="Run safe make target")
    async def make_target(target: str) -> dict[str, Any]:
        """
        Run a safe make target from an allowlist.

        Args:
            target: Make target name. Allowed targets:
                   Testing: test, test-unit, test-integration, test-all, test-fast,
                            test-regression, test-canary, test-coverage, test-debug,
                            test-failed, test-list, test-markers, test-clean,
                            test-policy, test-jobs-launcher, test-gpu-integration
                   SDK/CLI: refresh-openapi, stainless, update-sdk, update-cli,
                            generate-cli-reference-docs, generate-config-reference-docs
                   Vendoring: vendor, vendor-nemo-platform-ext
                   Policy: build-policy, check-policy
                   Licenses: update-licenses, check-licenses
                   Build: build-jobs-launcher
                   Lint: lint

        Returns:
            Dictionary with make command results
        """
        # Allowlist of safe make targets (read-only or safe generation)
        allowed_targets = {
            # Help
            "help",
            # Testing - unit
            "test",
            "test-unit",
            "test-unit-ci",
            "test-fast",
            "test-debug",
            "test-failed",
            "test-list",
            "test-markers",
            "test-clean",
            "test-coverage",
            "test-coverage-report",
            # Testing - integration
            "test-integration",
            "test-integration-ci",
            # Testing - GPU
            "test-gpu-integration",
            "test-gpu-integration-ci",
            # Testing - other
            "test-all",
            "test-all-script",
            "test-regression",
            "test-canary",
            # Testing - policy
            "test-policy",
            # Testing - jobs launcher
            "test-jobs-launcher",
            # SDK and OpenAPI generation
            "refresh-openapi",
            "stainless",
            "update-sdk",
            "update-cli",
            "generate-cli-reference-docs",
            "generate-config-reference-docs",
            # Vendoring
            "vendor",
            "vendor-nemo-platform-ext",
            # Policy operations
            "build-policy",
            "check-policy",
            # License management
            "update-licenses",
            "check-licenses",
            # Build
            "build-jobs-launcher",
            # Linting
            "lint",
        }

        if target not in allowed_targets:
            return {
                "success": False,
                "error": f"Target '{target}' not in allowlist. Allowed: {sorted(allowed_targets)}",
            }

        return run_command(["make", target], timeout=600)

    # === DOCS MAKE TARGET OPERATIONS ===

    @server.tool(
        description="Run a Fern docs Makefile target (docs-*) from the repo root"
    )
    async def make_docs(target: str) -> dict[str, Any]:
        """
        Run a Fern docs make target from the repo root (equivalent to `make <target>`).

        The docs site is built with Fern; targets live in the root Makefile.

        Args:
            target: Make target name. Allowed targets:
                   Setup: docs-deps, docs-login
                   Serve: docs
                   Quality: docs-check, docs-broken-links, docs-fix-links
                   Publish: docs-preview, docs-publish

        Returns:
            Dictionary with make command results
        """
        allowed_docs_targets = {
            "docs-deps",
            "docs-login",
            "docs",
            "docs-check",
            "docs-broken-links",
            "docs-fix-links",
            "docs-preview",
            "docs-publish",
        }

        if target not in allowed_docs_targets:
            return {
                "success": False,
                "error": f"Target '{target}' not in allowlist. Allowed: {sorted(allowed_docs_targets)}",
            }

        return run_command(["make", target], timeout=600)

    return server


def main() -> None:
    """Main entry point for the dev tools MCP server."""
    parser = argparse.ArgumentParser(
        description="NeMo Platform Dev Tools MCP Server - Narrowly-scoped development operations"
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        help="Working directory for commands (default: repo root)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for HTTP transport (default: 8001)",
    )

    args = parser.parse_args()

    # Create server
    logger.info("Creating NeMo Platform dev tools MCP server...")
    server = create_server(working_dir=args.working_dir)

    # Run server with specified transport
    if args.transport == "stdio":
        logger.info("Starting MCP server with stdio transport...")
        server.run(transport="stdio")
    else:
        logger.info(
            f"Starting MCP server with HTTP transport on {args.host}:{args.port}..."
        )
        server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

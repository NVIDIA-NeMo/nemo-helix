# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI commands for license scanning.

This module provides typer commands for:
- Generating license reports
- Finding missing licenses
- Discovering license overrides from PyPI
"""

import logging
import os
from pathlib import Path
from typing import Annotated

import typer
from nemo_helix_sdk_tools.license.find_missing import find_missing_licenses
from nemo_helix_sdk_tools.license.generator import LicenseGenerationError, generate_all_licenses
from nemo_helix_sdk_tools.license.license_overrides import generate_overrides
from nemo_helix_sdk_tools.license.license_utils import get_workspace_root
from nemo_helix_sdk_tools.printer import print_color
from rich.logging import RichHandler

logger = logging.getLogger(__name__)

help_text = """License Scanner

Use this tool to scan the monorepo for licenses, generate overrides, and find missing licenses.

Commands:
  generate             Generate license report for the main project
  find-missing         Find packages with UNKNOWN or NON-STANDARD licenses
  discover-overrides   Fetch license info from PyPI for missing packages
"""

app = typer.Typer(help=help_text, no_args_is_help=True)


@app.callback()
def common():
    """Callback to be used by all commands to set up common args ahead of other kube commands"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
    )


@app.command()
def generate(
    sequential: Annotated[
        bool, typer.Option("--sequential", help="Run scans sequentially instead of in parallel")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: jsonl (default), table, json, csv, markdown, text",
        ),
    ] = "jsonl",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Optional path for the formatted license report. "
                "When omitted, the path is built from LICENSE_DIR and LICENSE_NAME "
                "(default: third_party/licenses.jsonl)."
            ),
        ),
    ] = None,
):
    """
    Generate license report for the main project.

    This command:
    1. Runs osv-scanner on the uv-exported Python dependency snapshot and configured Go modules
    2. Generates JSON output with license information
    3. Formats the output in your chosen format
    4. Runs scans in parallel for faster execution

    Available formats:
    - table: Rich Unicode table (default, for terminal viewing)
    - jsonl: JSON Lines - one JSON object per line (RECOMMENDED for automation)
    - json: Compact JSON array
    - csv: CSV format (good for spreadsheets)
    - markdown: Markdown table (good for documentation)
    - text: Simple tab-separated text
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        ws_root = get_workspace_root()
        logger.info(f"Workspace root: {ws_root}")

        output_path = output.resolve() if output is not None else None

        generate_all_licenses(ws_root, parallel=not sequential, format_type=output_format, output_file=output_path)

        # Get the actual output paths from the generator
        from nemo_helix_sdk_tools.license.generator import get_projects

        projects = get_projects(ws_root, output_file=output_path)

        print_color("✓ License generation complete!")
        print_color("\nOutput files:")
        for project in projects:
            print_color(f"  • {project['output_file']}")

    except LicenseGenerationError as e:
        print_color(f"Error: {e}", "red")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        print_color(f"Unexpected error: {e}", "red")
        raise typer.Exit(1)


@app.command("find-missing")
def find_missing():
    """
    Find packages with UNKNOWN or NON-STANDARD licenses, and error out if overrides don't exist
    for certain license types

    """
    find_missing_licenses()


@app.command("discover-overrides")
def discover_overrides(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
):
    """
    Discover license information from PyPI for packages with missing licenses.

    Fetches license information from PyPI and prints suggested YAML overrides
    that can be manually added to third_party/license_overrides.yaml.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        ws_root = get_workspace_root()

        old_cwd = os.getcwd()
        os.chdir(ws_root)

        try:
            generate_overrides(ws_root)
        finally:
            os.chdir(old_cwd)

    except Exception as e:
        logger.exception("Error discovering license overrides")
        print_color(f"Error: {e}", "red")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

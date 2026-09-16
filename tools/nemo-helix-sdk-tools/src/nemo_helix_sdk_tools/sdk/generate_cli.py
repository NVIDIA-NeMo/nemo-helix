# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI command to generate NeMo Helix CLI commands."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Annotated

import typer
from nemo_helix_sdk_tools.sdk.cli_generator.generator import generate_all
from nemo_helix_sdk_tools.sdk.core.common import get_project_dir


def generate_cli(
    stainless_config: Annotated[
        Path | None,
        typer.Option(help="Path to stainless.yaml (defaults to sdk/stainless.yaml)"),
    ] = None,
    cli_config: Annotated[
        Path | None,
        typer.Option(help="Path to cli_config.yaml (defaults to cli_generator/cli_config.yaml)"),
    ] = None,
) -> None:
    """Generate CLI commands from Stainless config and SDK.

    This command generates CLI command files in packages/nemo_helix_ext/src/nemo_helix_ext/cli/commands/api/
    based on the Stainless configuration and SDK introspection.

    The generation process:
    1. Clears existing generated files
    2. Reads the Stainless config to discover resources and methods
    3. Reads the CLI config for table column definitions
    4. Introspects the SDK to extract parameter details
    5. Applies Jinja2 templates to generate CLI command files
    6. Generates __init__.py files for command registration

    Examples:
        # Generate all CLI commands
        uv run nemo-helix-sdk-tools generate-cli

        # Generate with custom config
        uv run nemo-helix-sdk-tools generate-cli --stainless-config path/to/stainless.yaml
    """

    # Determine config paths
    repo_root = get_project_dir()
    if stainless_config is None:
        stainless_config = repo_root / "sdk" / "stainless.yaml"

    if cli_config is None:
        cli_config = Path(__file__).parent / "cli_generator" / "cli_config.yaml"

    if not stainless_config.exists():
        typer.echo(f"Error: Stainless config not found at {stainless_config}", err=True)
        raise typer.Exit(code=1)

    if not cli_config.exists():
        typer.echo(f"Error: CLI config not found at {cli_config}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Generating CLI from: {stainless_config}")
    typer.echo(f"Using CLI config: {cli_config}")

    logging.getLogger("nemo_helix_sdk_tools.sdk.core.stainless").setLevel(logging.WARNING)
    try:
        generate_all(stainless_config, cli_config)
        typer.echo("\n✓ CLI generation completed successfully!")
    except Exception as e:
        typer.echo(f"\n✗ CLI generation failed: {e}", err=True)
        traceback.print_exc()
        raise typer.Exit(code=1)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo CLI - Command-line interface for NeMo Platform."""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint
from typing import TYPE_CHECKING, Annotated, cast

import typer

from nemo_platform_ext.cli.commands.manifest_registry import TOP_LEVEL_ENTRIES
from nemo_platform_ext.cli.core.help_formatter import HELP_OPTION_NAMES
from nemo_platform_ext.cli.core.lazy_load import (
    ManifestBackedNmpGroup,
    attach_lazy_entries,
)
from nemo_platform_ext.cli.core.logging import configure_logging
from nemo_platform_ext.cli.core.types import ListOutputFormat, TimestampFormat
from nemo_platform_ext.cli.manifest import (
    TopLevelEntry,
    build_top_level_entries,
)
from nemo_platform_ext.config.types import OutputFormat as ConfigOutputFormat

if TYPE_CHECKING:
    from nemo_platform_plugin.cli import NemoCLI
    from nemo_platform_plugin.function import NemoFunction
    from nemo_platform_plugin.job import NemoJob

    from nemo_platform_ext.config.models import ConfigParams

logger = logging.getLogger(__name__)

_SKIP_AUTH_CHECK_SUBCOMMANDS = frozenset(
    {"agent", "auth", "config", "setup", "quickstart", "cluster-info", "skills", "docs", "services", "plugins"}
)
# Create the main CLI app with custom help formatting
app = typer.Typer(
    name="nemo",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    cls=ManifestBackedNmpGroup,
    context_settings=dict(help_option_names=list(HELP_OPTION_NAMES)),
)


# Panels whose built-in groups a plugin ``nemo.cli`` entry point of the same
# name may replace. Setup and CLI-function commands are never shadowed.
_PLUGIN_OVERRIDABLE_PANELS = frozenset({"Core plugins", "Functional plugins"})


def _build_top_level_lazy_entries() -> tuple[TopLevelEntry, ...]:
    plugin_entry_points = _installed_plugin_command_entry_points()
    return build_top_level_entries(
        module_entries_without_plugin_overrides(TOP_LEVEL_ENTRIES, plugin_entry_points),
        plugin_entry_points,
        include_hidden=True,
    )


def module_entries_without_plugin_overrides(
    entries: tuple[TopLevelEntry, ...], plugin_entry_points: dict[str, EntryPoint]
) -> tuple[TopLevelEntry, ...]:
    """Drop built-in resource groups that an installed plugin CLI replaces by name.

    Plugin ``nemo.cli`` entry points own their command name, so a plugin that
    ships e.g. its own ``files`` group takes precedence over the built-in one.
    """
    return tuple(
        entry
        for entry in entries
        if not (entry.panel in _PLUGIN_OVERRIDABLE_PANELS and entry.name in plugin_entry_points)
    )


def _installed_plugin_command_entry_points() -> dict[str, EntryPoint]:
    """Return installed plugin CLI entry points without importing plugin code."""
    try:
        from nemo_platform_plugin.discovery import discover_entry_points
    except ImportError:
        return {}
    try:
        return discover_entry_points("nemo.cli")
    except Exception:  # noqa: BLE001
        logger.warning("Failed to discover CLI plugin entry points", exc_info=True)
        return {}


def _add_plugin_job_commands(
    plugin_app: typer.Typer,
    plugin_jobs: dict[str, type[NemoJob]],
    *,
    cli: NemoCLI | None = None,
) -> None:
    # TODO: nemo-platform-plugin is temporarily optional while it is being published
    # to the nightly PyPI feed. Once available, it should become an
    # unconditional dependency and these guards can be removed.
    try:
        from nemo_platform_plugin.commands import add_job_commands
    except ImportError:
        logger.warning(
            "nemo_platform_plugin.commands unavailable; skipping plugin job command injection", exc_info=True
        )
        return

    add_job_commands(plugin_app, plugin_jobs, cli=cli)


def _discover_plugin_job_entry_points() -> dict[str, EntryPoint] | None:
    # TODO: nemo-platform-plugin is temporarily optional while it is being published
    # to the nightly PyPI feed. Once available, it should become an
    # unconditional dependency and these guards can be removed.
    try:
        from nemo_platform_plugin.discovery import discover_entry_points
    except ImportError:
        return None

    return discover_entry_points("nemo.jobs")


def _add_plugin_function_commands(
    plugin_app: typer.Typer,
    plugin_functions: dict[str, type[NemoFunction]],
    *,
    cli: NemoCLI | None = None,
) -> None:
    # Same nemo-platform-plugin optionality guard as the jobs path. When the
    # package isn't installed we silently skip — the plugin's bare CLI
    # surface (whatever it shipped via `nemo.cli`) still loads.
    try:
        from nemo_platform_plugin.commands import add_function_commands
    except ImportError:
        logger.warning(
            "nemo_platform_plugin.commands unavailable; skipping plugin function command injection",
            exc_info=True,
        )
        return

    add_function_commands(plugin_app, plugin_functions, cli=cli)


def _discover_plugin_function_entry_points() -> dict[str, EntryPoint] | None:
    try:
        from nemo_platform_plugin.discovery import discover_entry_points
    except ImportError:
        return None

    return discover_entry_points("nemo.functions")


# Global options
@app.callback(options_metavar="[GLOBAL OPTIONS]")
def main(
    ctx: typer.Context,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version information and exit.",
            callback=_version_callback,
            is_eager=True,
            rich_help_panel="Help",
        ),
    ] = None,
    context_name: Annotated[
        str | None,
        typer.Option(
            "--context",
            "-c",
            help="The name of the context to use. Overrides the current context in the config file.",
            rich_help_panel="Global Options",
            hidden=True,
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="Base URL for the NeMo Platform API",
            rich_help_panel="Global Options",
        ),
    ] = None,
    output_format: Annotated[
        ListOutputFormat | None,
        typer.Option(
            "--output-format",
            "--output",
            "-f",
            help="Output format for how results are printed.",
            rich_help_panel="Global Options",
        ),
    ] = None,
    no_truncate: Annotated[
        bool | None,
        typer.Option(
            "--no-truncate",
            help="Don't truncate long values in table/markdown/csv output",
            rich_help_panel="Global Options",
        ),
    ] = None,
    timestamp_format: Annotated[
        TimestampFormat | None,
        typer.Option(
            help="Timestamp format for table/markdown/csv output",
            rich_help_panel="Global Options",
        ),
    ] = None,
    verbose: Annotated[
        bool | None,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose messaging. This only impacts logs that are visible, it doesn't change any data outputs.",
            rich_help_panel="Global Options",
        ),
    ] = None,
    agent_mode: Annotated[
        bool | None,
        typer.Option(
            "--agent-mode",
            "-A",
            help="Enable agent-friendly output mode with extra context for coding agents.",
            rich_help_panel="Global Options",
        ),
    ] = None,
    no_auto_refresh: Annotated[
        bool,
        typer.Option(
            "--no-auto-refresh",
            help="Disable automatic token refresh when token is about to expire.",
            rich_help_panel="Global Options",
            hidden=True,
        ),
    ] = False,
    no_telemetry: Annotated[
        bool,
        typer.Option(
            "--no-telemetry",
            help="Disable anonymous usage telemetry for this invocation.",
            rich_help_panel="Global Options",
        ),
    ] = False,
) -> None:
    """
    Command-line interface for NeMo Platform.

    :books: Documentation: https://docs.nvidia.com/nemo-platform

    [green]Getting started:[/]
    - Browse documentation with [cyan]`nemo docs --list`[/]
    - Run local platform services with [cyan]`nemo services run --help`[/]
    - Read the Kubernetes deployment guide with [cyan]`nemo docs set-up/helm/install`[/]

    [green]Examples:[/]
    nemo workspaces list --output-format markdown
    nemo workspaces get default -f json

    [green]Exit codes:[/]
    - 0: Success
    - 1: Local or unexpected error
    - 2: Command usage error
    - 3: Remote/API error
    """
    # Lazy imports for performance (avoid loading pydantic_settings for --help)
    from nemo_platform_ext.cli.core.context import CLIContext
    from nemo_platform_ext.quickstart import QuickstartConfig

    # Configure logging (always call to silence httpx in non-verbose mode)
    configure_logging(1 if verbose else 0)

    if ctx.obj is None:
        ctx.obj = CLIContext()

    # Resolve agent mode: explicit flag > env var > default False
    import os

    if agent_mode is None:
        env_val = os.environ.get("NMP_AGENT_MODE", "").lower()
        agent_mode = env_val in ("1", "true", "yes")
    ctx.obj.agent_mode = agent_mode

    # Capture command name + agent mode for the command_invoked telemetry event, wire
    # the per-invocation opt-out, and print the first-run notice. Best effort inside.
    from nemo_platform_ext.cli.telemetry import runtime as telemetry_runtime

    telemetry_runtime.on_callback(ctx, no_telemetry=no_telemetry)

    # Build ConfigParams from CLI args
    overrides: ConfigParams = {}
    if context_name is not None:
        overrides["current_context"] = context_name
    if base_url is not None:
        overrides["base_url"] = base_url
    if output_format is not None:
        overrides["output_format"] = cast(ConfigOutputFormat, output_format)
    elif agent_mode:
        overrides["output_format"] = "markdown"
    if timestamp_format is not None:
        overrides["timestamp_format"] = timestamp_format
    if no_truncate is not None:
        overrides["truncate"] = not no_truncate

    # Update CLIContext overrides
    ctx.obj.overrides.update(overrides)
    ctx.obj.verbosity = 1 if verbose else 0
    ctx.obj.quickstart_config = QuickstartConfig.load()

    # Non-quickstart contexts always require auth. Some quickstart contexts require auth (opt-in)
    context_requires_auth = ctx.obj.quickstart_config is None or ctx.obj.quickstart_config.auth_enabled
    command_requires_auth = ctx.invoked_subcommand not in _SKIP_AUTH_CHECK_SUBCOMMANDS
    auto_refresh_enabled = not no_auto_refresh

    if context_requires_auth and command_requires_auth and auto_refresh_enabled:
        from nemo_platform_ext.cli.commands.auth import AuthError, ensure_valid_token

        try:
            token_valid = ensure_valid_token(ctx.obj.get_sdk_context())
            if not token_valid:
                typer.echo(
                    "Error: Your access token has expired and could not be refreshed.\n"
                    "Hint: Run 'nemo auth login' to re-authenticate.",
                    err=True,
                )
                raise typer.Exit(code=1)

            # reset the context, so we load the refreshed auth info
            ctx.obj.reset_sdk_context()
        except typer.Exit:
            raise  # Re-raise Exit to stop execution
        except AuthError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Warning: Failed to check/refresh token: {e}", err=True)


attach_lazy_entries(main, _build_top_level_lazy_entries())


def _version_callback(value: bool) -> None:
    """Print version information and exit."""
    if value:
        from nemo_platform_ext.cli.version import client_version

        typer.echo(f"nemo version {client_version()}")
        raise typer.Exit()


def cli() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()

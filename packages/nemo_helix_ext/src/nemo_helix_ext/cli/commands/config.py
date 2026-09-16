# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for NeMo Helix configuration management."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Annotated

import typer

from nemo_helix_ext.cli.commands.config_help import CONFIG_APP_HELP
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import format_output
from nemo_helix_ext.cli.core.help_formatter import create_typer_app
from nemo_helix_ext.cli.core.types import ConfigOutputFormatOption, ListOutputFormat, TimestampFormat

OutputFormat = ListOutputFormat

app = create_typer_app(
    name="config",
    help=CONFIG_APP_HELP,
)


class ConfigError(Exception):
    """Exception for config-related errors."""


def _validate_base_url(url: str) -> str:
    """Validate that a base URL is a valid HTTP(S) URL. Returns the url unchanged."""
    from pydantic import HttpUrl

    try:
        HttpUrl(url)
    except Exception as exc:
        raise ConfigError(
            f"Invalid base URL: '{url}'\nHint: The URL must include a scheme, e.g. https://nhx.example.com"
        ) from exc
    return url


def _make_completer(collection_name: str):
    """Create an autocompletion callback for a named collection (clusters, contexts, users)."""

    def completer(incomplete: str) -> list[str]:
        try:
            from nemo_helix_ext.config.config import Config

            config = Config.load()
            config_file = config.get_config_file()
            items = getattr(config_file, collection_name, None) or []
            return [item.name for item in items if item.name.startswith(incomplete)]
        except Exception:
            return []

    return completer


_complete_context_names = _make_completer("contexts")


@app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("current-context")
@handle_errors
def current_context(ctx: typer.Context) -> None:
    """Display the effective current context name.

    For full context details, use: nemo config view
    """
    from nemo_helix_ext.config.config import Config

    cli_context: CLIContext = ctx.obj
    config = Config.load()
    config_file = config.get_config_file()
    context_name = cli_context.overrides.get("current_context") or config.current_context or config_file.current_context

    if context_name:
        typer.echo(context_name)
    else:
        raise ConfigError("No current context set")


@app.command("set")
@handle_errors
def set_config(
    ctx: typer.Context,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="NeMo Helix API base URL"),
    ] = None,
    certificate_authority: Annotated[
        str | None,
        typer.Option(
            "--certificate-authority",
            help="Path to a PEM certificate authority bundle for the cluster",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="API key for authentication", hide_input=True),
    ] = None,
    access_token: Annotated[
        str | None,
        typer.Option("--access-token", help="OAuth access token for authentication", hide_input=True),
    ] = None,
    workspace: Annotated[
        str | None,
        typer.Option("--workspace", "-w", help="Default workspace"),
    ] = None,
    output_format: Annotated[
        OutputFormat | None,
        typer.Option("--output-format", "-f", help="Default output format"),
    ] = None,
    timestamp_format: Annotated[
        TimestampFormat | None,
        typer.Option("--timestamp-format", help="Timestamp display format"),
    ] = None,
    truncate: Annotated[bool | None, typer.Option(help="Truncate long output values")] = None,
    context_name: Annotated[
        str | None,
        typer.Option(
            "--context",
            help="Context name to create or modify (user-defined label; default: current context)",
        ),
    ] = None,
    activate: Annotated[
        bool,
        typer.Option("--activate", help="Set this context as the current context", show_default=True),
    ] = False,
) -> None:
    """
    Set configuration values in the current or specified context.

    If no config file exists, creates one. Without --context, uses the effective
    current context, or 'default' when no current context is selected.
    If the saved config has no current context, the next written context becomes
    current automatically. Otherwise, creating additional contexts does not switch
    the current context; use --activate or use-context.
    At least one option must be provided.

    Examples:
      nemo config set --base-url https://api.example.com
      nemo config set --context staging --base-url https://nhx.staging.example.com
      nemo config set --context production --base-url https://nhx.example.com --activate
      nemo config set --context local-tls --base-url https://localhost:8443 --certificate-authority /path/to/ca.crt
      nemo config set --workspace my-workspace --output-format json
      nemo config set --api-key YOUR_API_KEY
      nemo config set --api-key -        # prompts for API key securely
      nemo config set --access-token YOUR_ACCESS_TOKEN
      nemo config set --access-token -   # prompts for access token securely
    """
    from nemo_helix_ext.config.config import Config
    from nemo_helix_ext.config.models import ConfigParams

    cli_context: CLIContext = ctx.obj
    # Handle --api-key - (prompt securely)
    if api_key == "-":
        from nemo_helix_ext.ui.prompts import prompt_password

        api_key = prompt_password("API Key: ")
        if api_key is None:
            raise typer.Exit(0)

    # Handle --access-token - (prompt securely)
    if access_token == "-":
        from nemo_helix_ext.ui.prompts import prompt_password

        access_token = prompt_password("Access Token: ")
        if access_token is None:
            raise typer.Exit(0)

    if api_key and access_token:
        raise ConfigError("Cannot specify both --api-key and --access-token")

    # Check if any options were provided
    has_options = any(
        [
            base_url,
            certificate_authority is not None,
            api_key,
            access_token,
            workspace,
            output_format,
            timestamp_format,
            truncate is not None,
        ]
    )

    if not has_options and not activate:
        # No options provided - show help
        typer.echo(ctx.get_help())
        raise typer.Exit(1)

    if activate and not (context_name or cli_context.overrides.get("current_context")):
        raise ConfigError("--activate requires --context to be specified")

    if base_url:
        _validate_base_url(base_url)

    # Build ConfigParams from CLI flags
    params: ConfigParams = {}
    if base_url:
        params["base_url"] = base_url
    if certificate_authority is not None:
        params["certificate_authority"] = (
            str(Path(certificate_authority).expanduser().resolve()) if certificate_authority else None
        )
    token = access_token or api_key
    if token:
        params["access_token"] = token
    if workspace is not None:
        params["workspace"] = workspace
    if output_format is not None:
        params["output_format"] = output_format
    if timestamp_format is not None:
        params["timestamp_format"] = timestamp_format
    if truncate is not None:
        params["truncate"] = truncate

    target_context_name = context_name or cli_context.overrides.get("current_context")

    if activate and target_context_name:
        params["current_context"] = target_context_name

    write_result = Config.write_with_result(
        params,
        context_name=target_context_name,
    )
    effective_name = write_result.context_name
    # is_current checks the persisted file value; --activate writes current_context into params,
    # so a newly-created context with --activate will already appear here as current.
    is_current = write_result.config.get_config_file().current_context == effective_name
    if write_result.created and is_current:
        typer.echo(f"Context '{effective_name}' created and set as current")
    elif write_result.created:
        typer.echo(f"Context '{effective_name}' created")
        context_arg = shlex.quote(effective_name)
        if effective_name.startswith("-"):
            context_arg = f"-- {context_arg}"
        typer.echo(f"To make it current, run: nemo config use-context {context_arg}")
    elif activate and has_options:
        typer.echo(f"Configuration updated for context '{effective_name}' and set as current")
    elif activate:
        typer.echo(f"Switched to context '{effective_name}'")
    else:
        typer.echo(f"Configuration updated for context '{effective_name}'")


@app.command("use-context")
@handle_errors
def use_context(
    context_name: Annotated[str, typer.Argument(help="Context name to use", autocompletion=_complete_context_names)],
) -> None:
    """Set the current context in the configuration file."""
    from nemo_helix_ext.config.config import Config

    config = Config.load()
    config.set_current_context(context_name)
    typer.echo(f"Switched to context '{context_name}'")


@app.command("delete-context")
@handle_errors
def delete_context(
    context_name: Annotated[str, typer.Argument(help="Context name to delete", autocompletion=_complete_context_names)],
    prune_orphans: Annotated[
        bool,
        typer.Option(
            "--prune-orphans",
            help="Also delete cluster and user records no longer referenced by any context",
        ),
    ] = False,
) -> None:
    """Delete a context from the configuration file.

    By default this follows kubectl's convention and deletes only the context;
    referenced cluster and user records are kept.
    """
    from nemo_helix_ext.config.config import Config

    result = Config.delete_context(context_name, prune_orphans=prune_orphans)
    typer.echo(f"Deleted context '{result.context_name}'")
    if result.current_context_cleared:
        typer.echo("Current context cleared")
    if result.pruned_clusters:
        typer.echo(f"Pruned clusters: {', '.join(result.pruned_clusters)}")
    if result.pruned_users:
        typer.echo(f"Pruned users: {', '.join(result.pruned_users)}")


@app.command("view")
@handle_errors
def view_config(
    output_format: ConfigOutputFormatOption = None,
    all_contexts: Annotated[
        bool,
        typer.Option("--all-contexts", help="Show all contexts, clusters, and users"),
    ] = False,
) -> None:
    """Display the configuration file.

    By default, this shows only the current context and its referenced cluster and user.
    Use --all-contexts to show the full configuration.
    """
    from nemo_helix_ext.config.config import Config

    config = Config.load()
    config_file = config.get_config_file()
    config_path = config.get_config_path()

    # Get the config data (secrets automatically redacted by Pydantic serializers)
    config_data = config_file.model_dump(mode="json", exclude_none=True)

    if not all_contexts:
        # Filter to only show current context and its references
        if not config_file.current_context:
            raise ConfigError("No current context set")

        # Find the current context
        current_ctx = None
        for context in config_file.contexts or []:
            if context.name == config_file.current_context:
                current_ctx = context
                break

        if current_ctx is None:
            raise ConfigError(f"Current context '{config_file.current_context}' not found")

        # Build current-context config with only referenced items
        current_context_config: dict = {"current_context": config_file.current_context}

        # Include only the current context
        current_context_config["contexts"] = [current_ctx.model_dump(mode="json", exclude_none=True)]

        # Include only the referenced cluster
        if current_ctx.cluster and config_file.clusters:
            for cluster in config_file.clusters:
                if cluster.name == current_ctx.cluster:
                    current_context_config["clusters"] = [cluster.model_dump(mode="json", exclude_none=True)]
                    break

        # Include only the referenced user
        if current_ctx.user and config_file.users:
            for user in config_file.users:
                if user.name == current_ctx.user:
                    current_context_config["users"] = [user.model_dump(mode="json", exclude_none=True)]
                    break

        config_data = current_context_config

    # Add metadata about where the config came from
    if config_path:
        typer.echo(f"# Config file: {config_path}\n", err=True)

    if not all_contexts:
        context_count = len(config_file.contexts or [])
        context_message = f"# Showing config for context: {config_file.current_context}"
        if context_count > 1:
            context_message += ", to see all use `--all-contexts`"
        typer.echo(f"{context_message}\n", err=True)

    format_output(config_data, output_format=output_format)


_CONFIG_COMMAND_ORDER = {
    "view": 0,
    "set": 1,
    "current-context": 2,
    "use-context": 3,
    "delete-context": 4,
}
app.registered_commands.sort(
    key=lambda command: _CONFIG_COMMAND_ORDER.get(command.name or "", len(_CONFIG_COMMAND_ORDER))
)

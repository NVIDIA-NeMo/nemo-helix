# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pkgutil import resolve_name
from typing import Annotated, Any, Callable, Sequence

import click
from click import Command
from typer import Argument, Context, Exit, Typer
from typer.main import get_command as typer_get_command

from nemo_helix_ext.cli.core.help_formatter import NhxGroup
from nemo_helix_ext.cli.manifest import TopLevelEntry

logger = logging.getLogger(__name__)
_UNAVAILABLE_COMMAND_CONTEXT_SETTINGS = {"allow_extra_args": True, "ignore_unknown_options": True}


def attach_lazy_entries(
    callback: Callable[..., Any],
    entries: Sequence[TopLevelEntry],
) -> None:
    """Attach metadata-only child entries to a group callback.

    The root callback carries the top-level manifest so `ManifestBackedNhxGroup`
    can render root help without materializing every child command first.
    """
    callback.__nhx_lazy_entries__ = tuple(entries)


class ManifestBackedNhxGroup(NhxGroup):
    """Group that renders lazy child metadata and loads real children on demand."""

    _nhx_emit_command_invoked = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lazy_entries: dict[str, TopLevelEntry] = {}
        # The callback is the stable place where the root app keeps its lazy
        # child manifest. Group help reads this metadata, and subcommand lookup
        # loads just the requested child command.
        callback = getattr(self, "callback", None)
        lazy_entries = getattr(callback, "__nhx_lazy_entries__", ())
        for entry in lazy_entries:
            self._lazy_entries[entry.name] = entry

    def list_commands(self, _ctx: click.Context) -> list[str]:
        names = list(self.commands)
        for name in self._lazy_entries:
            if name not in self.commands:
                names.append(name)
        return names

    def get_command(self, ctx: click.Context, cmd_name: str) -> Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command

        entry = self._lazy_entries.get(cmd_name)
        if entry is None:
            return None

        loader = build_lazy_loader(entry)
        command = self._patch_command(loader())
        command.name = cmd_name
        command.hidden = command.hidden or entry.hidden
        panel = getattr(command, "rich_help_panel", None)
        if panel is None or type(panel).__name__ == "DefaultPlaceholder":
            command.rich_help_panel = entry.panel
        self.commands[cmd_name] = command
        return command


def build_lazy_loader(entry: TopLevelEntry) -> Callable[[], click.Command]:
    if entry.source == "plugin":
        return lazy_plugin_loader(entry.name, entry.import_path)
    if entry.kind == "group":
        return lazy_group_loader(entry.import_path)
    return lazy_command_loader(entry.import_path)


def _plugin_entry_leaf_name(plugin_name: str, entry_point_name: str) -> str:
    prefix = f"{plugin_name}."
    if entry_point_name.startswith(prefix):
        return entry_point_name.removeprefix(prefix)
    return entry_point_name


def _add_unavailable_plugin_primitive(
    plugin_app: Typer,
    *,
    plugin_name: str,
    entry_point_name: str,
    entry_point_value: object,
    primitive_kind: str,
    verbs: Sequence[str],
    rich_help_panel: str,
    exc: Exception,
) -> None:
    command_name = _plugin_entry_leaf_name(plugin_name, entry_point_name)
    message = f"Plugin {primitive_kind} command {command_name!r} is unavailable due to import error: {exc}"
    logger.warning(
        "Failed to load plugin %s %r from %r (%s); registering unavailable command",
        primitive_kind,
        entry_point_name,
        "nemo.jobs" if primitive_kind == "job" else "nemo.functions",
        entry_point_value,
        exc_info=True,
    )

    unavailable_app = Typer(
        name=command_name,
        help=f"Plugin {primitive_kind} command {command_name!r} is unavailable due to import error.",
        context_settings=_UNAVAILABLE_COMMAND_CONTEXT_SETTINGS,
        no_args_is_help=False,
    )

    def _raise_unavailable() -> None:
        click.echo(f"Error: {message}", err=True)
        raise Exit(code=1)

    @unavailable_app.callback(invoke_without_command=True)
    def _root(
        ctx: Context,
        _args: Annotated[list[str] | None, Argument(hidden=True)] = None,
    ) -> None:
        if ctx.invoked_subcommand is None or _args:
            _raise_unavailable()

    def _unavailable_command(
        _ctx: Context,
        _args: Annotated[list[str] | None, Argument(hidden=True)] = None,
    ) -> None:
        _raise_unavailable()

    for verb in verbs:
        unavailable_app.command(
            name=verb,
            help="Unavailable due to import error.",
            context_settings=_UNAVAILABLE_COMMAND_CONTEXT_SETTINGS,
        )(_unavailable_command)

    plugin_app.add_typer(unavailable_app, name=command_name, rich_help_panel=rich_help_panel)


def lazy_plugin_loader(plugin_name: str, import_path: str) -> Callable[[], click.Command]:
    """Resolve a plugin CLI from its entry-point import path when needed."""

    def _load_plugin_cli() -> click.Command:
        from nemo_helix_ext.cli.app import (
            _add_plugin_function_commands,
            _add_plugin_job_commands,
            _discover_plugin_function_entry_points,
            _discover_plugin_job_entry_points,
        )

        try:
            from nemo_helix_plugin.customization_contributor import CustomizationContributorDiscoveryError

            cli_cls = resolve_name(import_path)
            cli_obj = cli_cls()
            plugin_app = cli_obj.get_cli()
        except CustomizationContributorDiscoveryError as exc:
            raise click.ClickException(str(exc)) from exc
        except Exception as exc:
            logger.warning("Failed to load CLI plugin %r from %r", plugin_name, import_path, exc_info=True)
            raise click.ClickException(f"Failed to load plugin commands for {plugin_name!r}: {exc}") from exc

        job_entry_points = _discover_plugin_job_entry_points()
        if job_entry_points is not None:
            plugin_jobs = {}
            for job_name, job_entry_point in job_entry_points.items():
                if not job_name.startswith(f"{plugin_name}."):
                    continue
                try:
                    plugin_jobs[job_name] = job_entry_point.load()
                except Exception as exc:
                    _add_unavailable_plugin_primitive(
                        plugin_app,
                        plugin_name=plugin_name,
                        entry_point_name=job_name,
                        entry_point_value=getattr(job_entry_point, "value", "<unknown>"),
                        primitive_kind="job",
                        verbs=("run", "submit", "explain"),
                        rich_help_panel="Jobs",
                        exc=exc,
                    )
                    continue
            if plugin_jobs:
                _add_plugin_job_commands(plugin_app, plugin_jobs, cli=cli_obj)

        # Functions live alongside jobs at the plugin level; mirror the
        # same per-plugin filtering so a multi-plugin install only sees
        # this plugin's functions in the resolved CLI tree.
        function_entry_points = _discover_plugin_function_entry_points()
        if function_entry_points is not None:
            plugin_functions = {}
            for fn_name, fn_entry_point in function_entry_points.items():
                if not fn_name.startswith(f"{plugin_name}."):
                    continue
                try:
                    plugin_functions[fn_name] = fn_entry_point.load()
                except Exception as exc:
                    _add_unavailable_plugin_primitive(
                        plugin_app,
                        plugin_name=plugin_name,
                        entry_point_name=fn_name,
                        entry_point_value=getattr(fn_entry_point, "value", "<unknown>"),
                        primitive_kind="function",
                        verbs=("run", "submit"),
                        rich_help_panel="Functions",
                        exc=exc,
                    )
                    continue
            if plugin_functions:
                _add_plugin_function_commands(plugin_app, plugin_functions, cli=cli_obj)

        try:
            return typer_get_command(plugin_app)
        except RuntimeError as exc:
            logger.warning("Failed to build CLI plugin %r", plugin_name, exc_info=True)
            raise click.ClickException(f"Failed to build plugin commands for {plugin_name!r}: {exc}") from exc

    return _load_plugin_cli


def lazy_group_loader(import_path: str) -> Callable[[], click.Command]:
    """Resolve a Typer app from ``module:attribute`` when needed."""

    def _load_group() -> click.Command:
        nested_name = "__lazy_group__"
        temp_app = Typer()
        temp_app.add_typer(resolve_name(import_path), name=nested_name)
        root_command = typer_get_command(temp_app)
        nested_command = root_command.get_command(click.Context(root_command), nested_name)
        if nested_command is None:
            raise click.ClickException(f"Failed to build nested CLI group for {import_path!r}")
        return nested_command

    def _lazy_loader() -> click.Command:
        try:
            return _load_group()
        except (ModuleNotFoundError, AttributeError) as exc:
            raise click.ClickException(f"Failed to load CLI group {import_path!r}: {exc}") from exc

    return _lazy_loader


def lazy_command_loader(import_path: str) -> Callable[[], click.Command]:
    """Resolve a top-level command from ``module:attribute`` when needed."""

    def _load_command() -> click.Command:
        holder_name = "__lazy_command_holder__"
        command = resolve_name(import_path)
        holder_app = Typer()
        holder_app.command()(command)
        temp_app = Typer()
        temp_app.add_typer(holder_app, name=holder_name)
        root_command = typer_get_command(temp_app)
        holder_command = root_command.get_command(click.Context(root_command), holder_name)
        if holder_command is None or not isinstance(holder_command, click.Group):
            raise click.ClickException(f"Failed to build nested CLI command holder for {import_path!r}")
        nested_commands = list(holder_command.commands.values())
        if len(nested_commands) != 1:
            raise click.ClickException(f"Failed to build nested CLI command for {import_path!r}")
        nested_command = nested_commands[0]
        if nested_command is None:
            raise click.ClickException(f"Failed to build nested CLI command for {import_path!r}")
        return nested_command

    def _lazy_loader() -> click.Command:
        try:
            return _load_command()
        except (ModuleNotFoundError, AttributeError) as exc:
            raise click.ClickException(f"Failed to load CLI command {import_path!r}: {exc}") from exc

    return _lazy_loader

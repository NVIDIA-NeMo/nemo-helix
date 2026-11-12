# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo preferences`` command group: read and edit your preference documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from nemo_helix_plugin.auth.profiles.types import PreferenceDocument, PreferenceQueryParams, PreferenceValue
from rich.console import Console
from rich.table import Table
from rich.text import Text

from nemo_helix_ext.cli.commands.profile import profiles_client
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import format_output
from nemo_helix_ext.cli.core.help_formatter import create_typer_app

# Preferences are JSON documents per namespace (studio.ui, cli, plugin.<name>), optionally per workspace.
app = create_typer_app(name="preferences", help="Read and edit your preferences.")

NamespaceArg = Annotated[str, typer.Argument(help="Preference namespace, e.g. studio.ui.")]
KeyOption = Annotated[str | None, typer.Option("--key", "-k", help="Dotted path inside the document.")]
WorkspaceOption = Annotated[
    str | None, typer.Option("--workspace", "-w", help="Workspace-scoped document; omit for global.")
]


def _query(workspace: str | None) -> PreferenceQueryParams | None:
    return {"workspace": workspace} if workspace else None


def _path(key: str) -> list[str]:
    parts = key.split(".")
    if not all(parts):
        raise typer.BadParameter(f"Invalid key path: {key!r}", param_hint="--key")
    return parts


def _nested(key: str, value: Any) -> dict[str, Any]:
    for part in reversed(_path(key)):
        value = {part: value}
    return value


def _lookup(document: dict[str, Any], key: str) -> Any:
    value: Any = document
    for part in _path(key):
        if not isinstance(value, dict) or part not in value:
            typer.echo(f"Key {key!r} is not set.", err=True)
            raise typer.Exit(1)
        value = value[part]
    return value


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _print_value(value: Any, output_format: str) -> None:
    if output_format == "yaml":
        typer.echo(yaml.safe_dump(value, sort_keys=False).rstrip())
    elif isinstance(value, str) and output_format != "json":
        typer.echo(value)
    else:
        typer.echo(json.dumps(value, indent=2))


def _print_document(document: PreferenceDocument, state: CLIContext, *, raw: bool) -> None:
    output_format = state.get_output_format()
    if output_format != "table":
        format_output(document, output_format=output_format)
        return
    table = Table(title=document.namespace, show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", overflow="fold")
    # Keys and values are user text: render them as-is, never as Rich markup.
    table.add_row("[dim]workspace[/]", Text(document.workspace) if document.workspace else "[dim]global[/]")
    table.add_row("[dim]db_version[/]", str(document.db_version))
    for key, value in (document.value if raw else document.effective).items():
        table.add_row(Text(key), Text(value if isinstance(value, str) else json.dumps(value)))
    Console().print(table)


@app.command("get")
@handle_errors
def get_preference(
    ctx: typer.Context,
    namespace: NamespaceArg,
    key: KeyOption = None,
    workspace: WorkspaceOption = None,
    raw: Annotated[bool, typer.Option("--raw", help="Show only your own values, without defaults.")] = False,
) -> None:
    """Show a preference document, with deployment defaults applied unless --raw.

    Preferences are JSON documents grouped by namespace (studio.ui, cli, plugin.<name>),
    optionally scoped to a workspace. --key addresses a dotted path inside the document.

    Examples:
      nemo preferences get studio.ui
      nemo preferences get studio.ui --key theme
      nemo preferences get studio.onboarding --workspace default --raw
    """
    state: CLIContext = ctx.obj
    document = profiles_client(state).get_my_preference(namespace=namespace, query_params=_query(workspace)).data()
    if key is None:
        _print_document(document, state, raw=raw)
        return
    _print_value(_lookup(document.value if raw else document.effective, key), state.get_output_format())


@app.command("set")
@handle_errors
def set_preference(
    ctx: typer.Context,
    namespace: NamespaceArg,
    value: Annotated[str | None, typer.Argument(help="Value for --key; parsed as JSON, else used as a string.")] = None,
    key: KeyOption = None,
    json_document: Annotated[str | None, typer.Option("--json", help="Replace the whole document.")] = None,
    file: Annotated[
        Path | None, typer.Option("--file", exists=True, dir_okay=False, help="Replace the whole document from a file.")
    ] = None,
    workspace: WorkspaceOption = None,
) -> None:
    """Set one value with --key, or replace the whole document with --json or --file.

    Examples:
      nemo preferences set studio.ui --key theme dark
      nemo preferences set cli --key table.maxWidth 120
      nemo preferences set plugin.my-plugin --json '{"enabled": true}'
    """
    replacements = [source for source in (json_document, file) if source is not None]
    if key is None and len(replacements) != 1:
        raise typer.BadParameter("Pass --key with a value, or exactly one of --json and --file.")
    if key is not None and (replacements or value is None):
        raise typer.BadParameter("--key needs a value and cannot be combined with --json or --file.")

    state: CLIContext = ctx.obj
    client = profiles_client(state)
    query = _query(workspace)
    if key is not None:
        body = PreferenceValue(_nested(key, _parse_value(value or "")))
        document = client.patch_my_preference(namespace=namespace, body=body, query_params=query).data()
    else:
        raw_document = file.read_text() if file is not None else json_document or ""
        try:
            parsed = json.loads(raw_document)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Document is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise typer.BadParameter("Document must be a JSON object.")
        document = client.replace_my_preference(
            namespace=namespace, body=PreferenceValue(parsed), query_params=query
        ).data()
    _print_document(document, state, raw=False)


@app.command("unset")
@handle_errors
def unset_preference(
    ctx: typer.Context,
    namespace: NamespaceArg,
    key: KeyOption = None,
    workspace: WorkspaceOption = None,
) -> None:
    """Remove one value with --key, or delete the whole document. Defaults then apply.

    Examples:
      nemo preferences unset studio.ui --key theme
      nemo preferences unset studio.onboarding --workspace default
    """
    state: CLIContext = ctx.obj
    client = profiles_client(state)
    query = _query(workspace)
    if key is None:
        client.delete_my_preference(namespace=namespace, query_params=query)
        typer.echo(f"Deleted preferences {namespace}.", err=True)
        return
    body = PreferenceValue(_nested(key, None))
    _print_document(
        client.patch_my_preference(namespace=namespace, body=body, query_params=query).data(), state, raw=False
    )

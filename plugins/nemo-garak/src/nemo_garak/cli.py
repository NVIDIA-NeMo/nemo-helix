# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI surface for the garak plugin.

The CLI talks to the garak plugin's own service routes (mounted at
``/apis/garak/v2/workspaces/{workspace}/...``) through the typed
:class:`~nemo_helix_plugin.garak.client.GarakClient`, which it takes from
the ``nemo`` CLI state so the global ``--base-url``, context, and auth apply.

Two sub-groups are exposed:

- ``nemo garak configs <create|list|get|update|delete>``
- ``nemo garak targets <create|list|get|update|delete>``

``--data-file PATH`` and ``--data JSON`` are mutually-exclusive ways to supply
the request body for ``create`` / ``update``. The body shape follows the
plugin's ``CreateAuditConfigRequest`` / ``CreateAuditTargetRequest`` schemas
— there is no ``data`` envelope. Unknown fields are rejected before anything
is sent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import typer
from nemo_helix_plugin.cli import NemoCLI, create_typer_app
from nemo_helix_plugin.cli_codegen import handle_code_generation
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_input import build_request_body
from nemo_helix_plugin.cli_options import (
    AllPagesOption,
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    WorkspaceOption,
)
from nemo_helix_plugin.cli_output import Column, check_output_columns_with_format, format_output
from nemo_helix_plugin.cli_pagination import PaginationType, collect_offset_pages, warn_if_more_pages
from nemo_helix_plugin.cli_state import cli_state, resolve_cli_workspace, resolve_output_format
from nemo_helix_plugin.cli_warnings import collect_warnings
from nemo_helix_plugin.garak.client import GarakClient
from nemo_helix_plugin.garak.types import (
    CreateAuditConfigRequest,
    CreateAuditTargetRequest,
    UpdateAuditConfigRequest,
    UpdateAuditTargetRequest,
)
from pydantic import BaseModel


@dataclass(frozen=True)
class _Resource:
    """One garak resource and the ``GarakClient`` methods that serve it."""

    path: str
    singular: str
    help: str
    create_request: type[BaseModel]
    update_request: type[BaseModel]
    columns: list[Column]

    def method(self, verb: str) -> str:
        noun = f"audit_{self.singular}"
        return f"{verb}_{noun}s" if verb == "list" else f"{verb}_{noun}"


_CONFIGS = _Resource(
    path="configs",
    singular="config",
    help="Manage audit configurations.",
    create_request=CreateAuditConfigRequest,
    update_request=UpdateAuditConfigRequest,
    columns=[Column("name"), Column("description"), Column("created_at")],
)
_TARGETS = _Resource(
    path="targets",
    singular="target",
    help="Manage audit targets.",
    create_request=CreateAuditTargetRequest,
    update_request=UpdateAuditTargetRequest,
    columns=[Column("name"), Column("type"), Column("model"), Column("created_at")],
)


def _load_data(data_file: Path | None, data: str | None) -> dict[str, Any]:
    if data_file is not None and data is not None:
        typer.echo("Error: pass either --data-file or --data, not both.", err=True)
        raise typer.Exit(code=2)
    if data_file is not None:
        raw = data_file.read_text(encoding="utf-8")
    elif data is not None:
        raw = data
    else:
        typer.echo("Error: one of --data-file or --data is required.", err=True)
        raise typer.Exit(code=2)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: invalid JSON — {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not isinstance(parsed, dict):
        typer.echo("Error: data must be a JSON object.", err=True)
        raise typer.Exit(code=2)
    return parsed


def _build_crud_app(resource: _Resource) -> typer.Typer:
    """Build a Typer sub-app with the standard 5 CRUD commands for one resource."""
    sub = create_typer_app(name=resource.path, help=resource.help)
    singular = resource.singular

    @sub.command("create")
    @collect_warnings
    @handle_errors
    def create(
        ctx: typer.Context,
        name: str = typer.Argument(..., help=f"{singular.capitalize()} name."),
        data_file: Path | None = typer.Option(
            None,
            "--data-file",
            help=f"JSON file with the {singular} body (without name/workspace).",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
        data: str | None = typer.Option(None, "--data", "-d", help=f"Inline JSON body for the {singular}."),
        workspace: WorkspaceOption = None,
        output_format: EntityOutputFormatOption = None,
    ) -> None:
        state = cli_state(ctx)
        resolved_output_format = resolve_output_format(ctx, output_format)
        body = build_request_body(
            resource.create_request,
            {"name": name, **_load_data(data_file, data)},
            command_name=f"garak {resource.path} create",
        )
        kwargs = {"workspace": resolve_cli_workspace(ctx, workspace), "body": body}
        method = resource.method("create")
        if handle_code_generation(GarakClient, method, kwargs, resolved_output_format, state):
            return
        response = getattr(state.typed_client(GarakClient), method)(**kwargs)
        format_output(response, output_format=resolved_output_format)

    @sub.command("list")
    @collect_warnings
    @handle_errors
    def list_cmd(
        ctx: typer.Context,
        workspace: WorkspaceOption = None,
        all_pages: AllPagesOption = False,
        output_format: ListOutputFormatOption = None,
        no_truncate: NoTruncateOption = None,
        columns: OutputColumnsOption = None,
    ) -> None:
        state = cli_state(ctx)
        resolved_output_format = resolve_output_format(ctx, output_format)
        check_output_columns_with_format(columns, resolved_output_format)
        kwargs = {"workspace": resolve_cli_workspace(ctx, workspace)}
        method = resource.method("list")
        if handle_code_generation(
            GarakClient,
            method,
            kwargs,
            resolved_output_format,
            state,
            result="all-pages" if all_pages else "list",
        ):
            return
        response = getattr(state.typed_client(GarakClient), method)(**kwargs)
        result = collect_offset_pages(response, all_pages=all_pages)
        format_output(
            result,
            is_list=True,
            output_format=resolved_output_format,
            output_columns=columns if columns and columns.strip() != "default" else resource.columns,
            no_truncate=state.get_no_truncate(no_truncate),
            timestamp_format=state.get_timestamp_format(),
        )
        if not all_pages:
            warn_if_more_pages(result, PaginationType.PAGE_NUMBER)

    @sub.command("get")
    @collect_warnings
    @handle_errors
    def get(
        ctx: typer.Context,
        name: str = typer.Argument(..., help=f"{singular.capitalize()} name."),
        workspace: WorkspaceOption = None,
        output_format: EntityOutputFormatOption = None,
    ) -> None:
        state = cli_state(ctx)
        resolved_output_format = resolve_output_format(ctx, output_format)
        kwargs = {"workspace": resolve_cli_workspace(ctx, workspace), "name": name}
        method = resource.method("get")
        if handle_code_generation(GarakClient, method, kwargs, resolved_output_format, state):
            return
        response = getattr(state.typed_client(GarakClient), method)(**kwargs)
        format_output(response, output_format=resolved_output_format)

    @sub.command("update")
    @collect_warnings
    @handle_errors
    def update(
        ctx: typer.Context,
        name: str = typer.Argument(..., help=f"{singular.capitalize()} name."),
        data_file: Path | None = typer.Option(
            None,
            "--data-file",
            help=f"JSON file with the new {singular} body.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
        data: str | None = typer.Option(None, "--data", "-d", help="Inline JSON body."),
        workspace: WorkspaceOption = None,
        output_format: EntityOutputFormatOption = None,
    ) -> None:
        state = cli_state(ctx)
        resolved_output_format = resolve_output_format(ctx, output_format)
        body = build_request_body(
            resource.update_request,
            _load_data(data_file, data),
            command_name=f"garak {resource.path} update",
        )
        kwargs = {"workspace": resolve_cli_workspace(ctx, workspace), "name": name, "body": body}
        method = resource.method("update")
        if handle_code_generation(GarakClient, method, kwargs, resolved_output_format, state):
            return
        response = getattr(state.typed_client(GarakClient), method)(**kwargs)
        format_output(response, output_format=resolved_output_format)

    @sub.command("delete")
    @collect_warnings
    @handle_errors
    def delete(
        ctx: typer.Context,
        name: str = typer.Argument(..., help=f"{singular.capitalize()} name."),
        workspace: WorkspaceOption = None,
    ) -> None:
        workspace = resolve_cli_workspace(ctx, workspace)
        getattr(cli_state(ctx).typed_client(GarakClient), resource.method("delete"))(workspace=workspace, name=name)
        typer.echo(f"{singular.capitalize()} '{name}' deleted.")

    return sub


class GarakPluginCLI(NemoCLI):
    """CLI surface for the garak plugin."""

    name: ClassVar[str] = "garak"
    description: ClassVar[str] = "Garak plugin commands."

    def get_cli(self) -> typer.Typer:
        app = create_typer_app(name=self.name, help=self.description)

        @app.command("info")
        def info() -> None:
            """Print the current plugin status."""
            typer.echo(
                json.dumps(
                    {
                        "plugin": self.name,
                        "status": "ready",
                        "service": "/apis/garak/v1/healthz",
                        "jobs": ["garak.audit"],
                        "sdk": "audit",
                    },
                    indent=2,
                )
            )

        app.add_typer(_build_crud_app(_CONFIGS))
        app.add_typer(_build_crud_app(_TARGETS))

        return app

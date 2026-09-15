# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from importlib import import_module as _importlib_import_module
from typing import Annotated, Literal

import typer

from nemo_platform_ext.cli.core.api import build_kwargs
from nemo_platform_ext.cli.core.api import merge_filter_dict as merge_filter_dict
from nemo_platform_ext.cli.core.code_generator import handle_code_generation
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import handle_errors
from nemo_platform_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_output,
    validate_stream_output_format,
)
from nemo_platform_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_platform_ext.cli.core.pagination import PaginationType, fetch_all_pages, warn_if_more_pages
from nemo_platform_ext.cli.core.stdin_utils import read_data_input_with_flags, validate_required_fields
from nemo_platform_ext.cli.core.stdin_utils import read_payload as read_payload
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

_cli_child_members = _importlib_import_module("nemo_platform_ext.cli.commands.api.workspaces.members")

app = create_typer_app(name="workspaces", help="Manage workspaces")

app.add_typer(_cli_child_members.app, name="members")


@app.command("create")
@collect_warnings
@handle_errors
def create_workspaces(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Argument(
            help="Workspace name (unique identifier). Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)"
        ),
    ] = None,
    wait_role_propagation: Annotated[
        bool | None,
        typer.Option(
            "--wait-role-propagation",
            help="If true, wait for Admin role to propagate before returning (default: true). Set to false for bulk operations.",
        ),
    ] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Optional description of the workspace")
    ] = None,
    exist_ok: Annotated[
        bool | None,
        typer.Option(
            "--exist-ok", help="Do not raise an error if the resource already exists. Returns the existing resource."
        ),
    ] = None,
    input_file: Annotated[
        str | None,
        typer.Option("--input-file", help="Path to JSON file (use '-' for stdin)", rich_help_panel="Input Options"),
    ] = None,
    input_data: Annotated[
        str | None,
        typer.Option("--input-data", help="Input data for the request (JSON or YAML)", rich_help_panel="Input Options"),
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Create a new workspace.

    The creator is automatically granted Admin role on the workspace. By default,
    this endpoint waits for the Admin role to propagate before returning. Use
    `wait_role_propagation=false` to skip waiting (useful for bulk operations).

    Example:

    ```
    POST /apis/entities/v2/workspaces
    {"name": "ml-team", "description": "Machine Learning Team workspace"}
    ```

        [bold red]Required fields:[/] name

        [green]Examples:[/]
        nemo workspaces create <name> --input-file config.json
        nemo workspaces create <name> --input-data '{"name": "value"}'
        echo '{"json": "data"}' | nemo workspaces create <name> --input-file -
        nemo workspaces create <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if name is not None:
        input_payload["name"] = name
    if wait_role_propagation is not None:
        input_payload["wait_role_propagation"] = wait_role_propagation
    if description is not None:
        input_payload["description"] = description
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["name"],
        "workspaces create",
        {
            "name": "Workspace name (unique identifier). Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["workspaces"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.workspaces.create(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("delete")
@collect_warnings
@handle_errors
def delete_workspaces(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
) -> None:
    """Delete a workspace.

    This marks the workspace for deletion and returns immediately. The workspace
    will no longer be accessible via the API. An asynchronous cleanup controller
    will handle deletion of all entities and external resources.

    Role bindings are immediately deleted to revoke access.

    Example:

    ```
    DELETE /apis/entities/v2/workspaces/ml-team
    ```"""
    state: CLIContext = ctx.obj
    client = state.get_client()

    kwargs = build_kwargs()
    client.workspaces.delete(name, **kwargs)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_workspaces(
    ctx: typer.Context,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            help='Query filter expression. Supports text and JSON syntaxes:\n- Text: name:"value" AND status>500 with operators : ~ > >= < <= IN NOT IN AND OR and negation prefix -\n- Object (JSON): {"name":{"$like":"value"}} with operators $eq, $like, $lt, $lte, $gt, $gte, $in, $nin, $and, $or, $not',
        ),
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Items per page")] = None,
    sort: Annotated[
        Literal["created_at", "-created_at", "updated_at", "-updated_at", "name", "-name"] | None,
        typer.Option("--sort", help="Sort field"),
    ] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List all workspaces with pagination.

    Workspaces marked for deletion (non-null deletion_stage) are omitted so the list
    matches GET/DELETE, which treat those workspaces as not found.

    When authentication is enabled, only workspaces the principal has access to are
    returned. Service principals and platform admins have access to all workspaces.

    Query Parameters:

    - page, page_size: Pagination
    - sort: Sort field
    - filter: Advanced filters (JSON, text, or bracket notation)

    Example:

    ```
    GET /apis/entities/v2/workspaces?sort=-created_at&page=1&page_size=10
    ```"""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)
    validate_stream_output_format(output_format, stream)

    check_output_columns_with_format(columns, output_format)

    default_columns = [
        Column("name", None),
        Column("description", None),
        Column("created_at", None),
    ]
    if columns is None or str(columns).strip() == "default":
        columns = default_columns

    kwargs = build_kwargs(
        filter=filter,
        page=page,
        page_size=page_size,
        sort=sort,
    )

    if handle_code_generation(["workspaces"], "list", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = ()
    pagination_type = PaginationType.PAGE_NUMBER
    if all_pages:
        items = fetch_all_pages(
            client.workspaces.list,
            path_args=path_args,
            body_args=kwargs,
            pagination_type=pagination_type,
        )
    else:
        items = client.workspaces.list(*path_args, **kwargs)

    format_output(
        items,
        is_list=True,
        output_format=output_format,
        output_columns=columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(items, pagination_type)


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_workspaces(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a specific workspace by ID.

    Example:

    ```
    GET /apis/entities/v2/workspaces/ml-team
    ```"""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs()
    if handle_code_generation(["workspaces"], "retrieve", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.workspaces.retrieve(name, **kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("update")
@collect_warnings
@handle_errors
def update_workspaces(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    description: Annotated[str | None, typer.Option("--description", help="Updated description")] = None,
    input_file: Annotated[
        str | None,
        typer.Option("--input-file", help="Path to JSON file (use '-' for stdin)", rich_help_panel="Input Options"),
    ] = None,
    input_data: Annotated[
        str | None,
        typer.Option("--input-data", help="Input data for the request (JSON or YAML)", rich_help_panel="Input Options"),
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Update a workspace's description.

    Example:

    ```
    PUT /apis/entities/v2/workspaces/ml-team
    {"description": "Updated description for ML Team"}
    ```

        [green]Examples:[/]
        nemo workspaces update <name> --input-file config.json
        nemo workspaces update <name> --input-data '{"field": "value"}'
        echo '{"json": "data"}' | nemo workspaces update <name> --input-file -
        nemo workspaces update <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if description is not None:
        input_payload["description"] = description

    all_kwargs = {"name": name, **input_payload}

    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["workspaces"], "update", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.workspaces.update(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

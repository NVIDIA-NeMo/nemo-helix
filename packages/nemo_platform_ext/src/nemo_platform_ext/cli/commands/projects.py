# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo projects`` command group, backed by the typed Projects client."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import typer
from nemo_platform_plugin.projects.client import ProjectsClient
from nemo_platform_plugin.projects.types import (
    CreateProjectRequest,
    ListProjectsQueryParams,
    UpdateProjectRequest,
)

from nemo_platform_ext.cli.core.api import build_kwargs
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
from nemo_platform_ext.cli.core.pagination import PaginationType, collect_offset_pages, warn_if_more_pages
from nemo_platform_ext.cli.core.stdin_utils import (
    build_request_body,
    read_data_input_with_flags,
    validate_required_fields,
)
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="projects", help="Manage projects.")

ProjectSortField = Literal["created_at", "-created_at", "updated_at", "-updated_at", "name", "-name"]

_NAME_HELP = (
    "Project name (unique within workspace). Name must start with a lowercase letter, be 2-63 characters, "
    "and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). "
    "(required)"
)
_INPUT_FILE_HELP = "Path to JSON file (use '-' for stdin)"
_INPUT_DATA_HELP = "Input data for the request (JSON or YAML)"
_INPUT_PANEL = "Input Options"


def _read_input_payload(input_file: str | None, input_data: str | None) -> dict[str, Any]:
    """Return the ``--input-file`` / ``--input-data`` payload, or an empty dict when neither is given."""
    if input_file or input_data:
        return read_data_input_with_flags(input_file=input_file, input_data=input_data)
    return {}


def _list_projects_query_params(
    *, filter: str | None, page: int | None, page_size: int | None, sort: str | None
) -> ListProjectsQueryParams | None:
    query_params: ListProjectsQueryParams = {}
    if filter is not None:
        query_params["filter"] = filter
    if page is not None:
        query_params["page"] = page
    if page_size is not None:
        query_params["page_size"] = page_size
    if sort is not None:
        query_params["sort"] = sort
    return query_params or None


@app.command("create")
@collect_warnings
@handle_errors
def create_projects(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help=_NAME_HELP)] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Optional description of the project")
    ] = None,
    exist_ok: Annotated[
        bool | None,
        typer.Option(
            "--exist-ok", help="Do not raise an error if the resource already exists. Returns the existing resource."
        ),
    ] = None,
    input_file: Annotated[
        str | None, typer.Option("--input-file", help=_INPUT_FILE_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    input_data: Annotated[
        str | None, typer.Option("--input-data", help=_INPUT_DATA_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Create a new project in the given workspace.

    Example:

    ```
    POST /apis/entities/v2/workspaces/default/projects
    {"name": "ml-project", "description": "Machine Learning project"}
    ```

        [bold red]Required fields:[/] name

        [green]Examples:[/]
        nemo projects create <name> --input-file config.json
        nemo projects create <name> --input-data '{"name": "value"}'
        echo '{"json": "data"}' | nemo projects create <name> --input-file -
        nemo projects create <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags), then apply
    # CLI flag overrides (flags take precedence).
    input_payload = _read_input_payload(input_file, input_data)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if name is not None:
        input_payload["name"] = name
    if description is not None:
        input_payload["description"] = description
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    validate_required_fields(input_payload, ["name"], "projects create", {"name": _NAME_HELP})

    resolved_workspace = input_payload.get("workspace")
    body = build_request_body(
        CreateProjectRequest, input_payload, exclude={"workspace", "exist_ok"}, command_name="projects create"
    )
    resolved_exist_ok = bool(input_payload.get("exist_ok", False))

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(workspace=resolved_workspace, body=body, exist_ok=resolved_exist_ok or None)
    if handle_code_generation(ProjectsClient, "create_project", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ProjectsClient).create_project(
        workspace=resolved_workspace, body=body, exist_ok=resolved_exist_ok
    )

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("delete")
@collect_warnings
@handle_errors
def delete_projects(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete a project.

    Example:

    ```
    DELETE /apis/entities/v2/workspaces/default/projects/ml-project
    ```"""
    state: CLIContext = ctx.obj
    state.typed_client(ProjectsClient).delete_project(name=name, workspace=workspace)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_projects(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            help='Query filter expression. Supports text and JSON syntaxes:\n- Text: name:"value" AND status>500 with operators : ~ > >= < <= IN NOT IN AND OR and negation prefix -\n- Object (JSON): {"name":{"$like":"value"}} with operators $eq, $like, $lt, $lte, $gt, $gte, $in, $nin, $and, $or, $not',
        ),
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Items per page")] = None,
    sort: Annotated[ProjectSortField | None, typer.Option("--sort", help="Sort field")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List all projects in a workspace with pagination.

    Query Parameters:

    - page, page_size: Pagination
    - sort: Sort field
    - filter: Advanced filters

    Example:

    ```
    GET /apis/entities/v2/workspaces/default/projects?sort=-created_at&page=1&page_size=10
    ```"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("name", None),
        Column("description", None),
        Column("created_at", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    query_params = _list_projects_query_params(filter=filter, page=page, page_size=page_size, sort=sort)
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)
    if handle_code_generation(ProjectsClient, "list_projects", kwargs, resolved_output_format, state, result="list"):
        return

    response = state.typed_client(ProjectsClient).list_projects(workspace=workspace, query_params=query_params)
    pagination_type = PaginationType.PAGE_NUMBER
    items = collect_offset_pages(response, all_pages=all_pages)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(items, pagination_type)


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_projects(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a specific project by its workspace and name.

    Example:

    ```
    GET /apis/entities/v2/workspaces/default/projects/ml-project
    ```"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(ProjectsClient, "get_project", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ProjectsClient).get_project(name=name, workspace=workspace)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("update")
@collect_warnings
@handle_errors
def update_projects(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    description: Annotated[str | None, typer.Option("--description", help="Updated description")] = None,
    input_file: Annotated[
        str | None, typer.Option("--input-file", help=_INPUT_FILE_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    input_data: Annotated[
        str | None, typer.Option("--input-data", help=_INPUT_DATA_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Update a project's description.

    Example:

    ```
    PUT /apis/entities/v2/workspaces/default/projects/ml-project
    {"description": "Updated description for ML project"}
    ```

        [green]Examples:[/]
        nemo projects update <name> --input-file config.json
        nemo projects update <name> --input-data '{"field": "value"}'
        echo '{"json": "data"}' | nemo projects update <name> --input-file -
        nemo projects update <name> --<option> "value"
    """
    input_payload = _read_input_payload(input_file, input_data)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if description is not None:
        input_payload["description"] = description

    resolved_workspace = input_payload.get("workspace")
    body = build_request_body(
        UpdateProjectRequest, input_payload, exclude={"workspace"}, command_name="projects update"
    )

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=resolved_workspace, body=body)
    if handle_code_generation(ProjectsClient, "update_project", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ProjectsClient).update_project(name=name, workspace=resolved_workspace, body=body)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo workspaces`` command group, backed by the typed Workspaces client."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import typer
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import (
    CreateWorkspaceMemberRequest,
    CreateWorkspaceQueryParams,
    CreateWorkspaceRequest,
    ListWorkspacesQueryParams,
    UpdateWorkspaceMemberRequest,
    UpdateWorkspaceRequest,
    WorkspaceMemberQueryParams,
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

app = create_typer_app(name="workspaces", help="Manage workspaces.")
members_app = create_typer_app(name="members", help="Manage members")
app.add_typer(members_app, name="members")

WorkspaceSortField = Literal["created_at", "-created_at", "updated_at", "-updated_at", "name", "-name"]

_WAIT_ROLE_PROPAGATION_HELP = (
    "If true, wait for roles to propagate before returning (default: true). Set to false for bulk operations."
)
_INPUT_FILE_HELP = "Path to JSON file (use '-' for stdin)"
_INPUT_DATA_HELP = "Input data for the request (JSON or YAML)"
_INPUT_PANEL = "Input Options"


def _read_input_payload(input_file: str | None, input_data: str | None) -> dict[str, Any]:
    """Return the ``--input-file`` / ``--input-data`` payload, or an empty dict when neither is given."""
    if input_file or input_data:
        return read_data_input_with_flags(input_file=input_file, input_data=input_data)
    return {}


def _list_workspaces_query_params(
    *, filter: str | None, page: int | None, page_size: int | None, sort: str | None
) -> ListWorkspacesQueryParams | None:
    query_params: ListWorkspacesQueryParams = {}
    if filter is not None:
        query_params["filter"] = filter
    if page is not None:
        query_params["page"] = page
    if page_size is not None:
        query_params["page_size"] = page_size
    if sort is not None:
        query_params["sort"] = sort
    return query_params or None


def _create_workspace_query_params(wait_role_propagation: bool | None) -> CreateWorkspaceQueryParams | None:
    if wait_role_propagation is None:
        return None
    return {"wait_role_propagation": wait_role_propagation}


def _member_query_params(wait_role_propagation: bool | None) -> WorkspaceMemberQueryParams | None:
    if wait_role_propagation is None:
        return None
    return {"wait_role_propagation": wait_role_propagation}


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
        str | None, typer.Option("--input-file", help=_INPUT_FILE_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    input_data: Annotated[
        str | None, typer.Option("--input-data", help=_INPUT_DATA_HELP, rich_help_panel=_INPUT_PANEL)
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
    # Read base input (optional if all fields provided via flags), then apply
    # CLI flag overrides (flags take precedence).
    input_payload = _read_input_payload(input_file, input_data)
    if name is not None:
        input_payload["name"] = name
    if wait_role_propagation is not None:
        input_payload["wait_role_propagation"] = wait_role_propagation
    if description is not None:
        input_payload["description"] = description
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    validate_required_fields(
        input_payload,
        ["name"],
        "workspaces create",
        {
            "name": "Workspace name (unique identifier). Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)",
        },
    )

    body = build_request_body(
        CreateWorkspaceRequest,
        input_payload,
        exclude={"wait_role_propagation", "exist_ok"},
        command_name="workspaces create",
    )
    query_params = _create_workspace_query_params(input_payload.get("wait_role_propagation"))
    resolved_exist_ok = bool(input_payload.get("exist_ok", False))

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(body=body, query_params=query_params, exist_ok=resolved_exist_ok or None)
    if handle_code_generation(WorkspacesClient, "create_workspace", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(WorkspacesClient).create_workspace(
        body=body, query_params=query_params, exist_ok=resolved_exist_ok
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
    state.typed_client(WorkspacesClient).delete_workspace(name=name)

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
    sort: Annotated[WorkspaceSortField | None, typer.Option("--sort", help="Sort field")] = None,
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

    query_params = _list_workspaces_query_params(filter=filter, page=page, page_size=page_size, sort=sort)
    kwargs = build_kwargs(query_params=query_params)
    if handle_code_generation(
        WorkspacesClient, "list_workspaces", kwargs, resolved_output_format, state, result="list"
    ):
        return

    response = state.typed_client(WorkspacesClient).list_workspaces(query_params=query_params)
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
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name)
    if handle_code_generation(WorkspacesClient, "get_workspace", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(WorkspacesClient).get_workspace(name=name)

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
def update_workspaces(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    description: Annotated[str | None, typer.Option("--description", help="Updated description")] = None,
    input_file: Annotated[
        str | None, typer.Option("--input-file", help=_INPUT_FILE_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    input_data: Annotated[
        str | None, typer.Option("--input-data", help=_INPUT_DATA_HELP, rich_help_panel=_INPUT_PANEL)
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
    input_payload = _read_input_payload(input_file, input_data)
    if description is not None:
        input_payload["description"] = description

    body = build_request_body(UpdateWorkspaceRequest, input_payload, command_name="workspaces update")

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, body=body)
    if handle_code_generation(WorkspacesClient, "update_workspace", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(WorkspacesClient).update_workspace(name=name, body=body)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@members_app.command("create")
@collect_warnings
@handle_errors
def create_members(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    principal: Annotated[
        str | None,
        typer.Option("--principal", help="The principal identifier (email, user ID, or group ID) (required)"),
    ] = None,
    wait_role_propagation: Annotated[
        bool | None, typer.Option("--wait-role-propagation", help=_WAIT_ROLE_PROPAGATION_HELP)
    ] = None,
    roles: Annotated[
        list[str] | None, typer.Option("--roles", help="List of roles to grant to the principal (can be repeated)")
    ] = None,
    input_file: Annotated[
        str | None, typer.Option("--input-file", help=_INPUT_FILE_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    input_data: Annotated[
        str | None, typer.Option("--input-data", help=_INPUT_DATA_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Add a new member to the workspace with specified roles.

    This creates role bindings for the specified principal with the given roles. By
    default, this endpoint waits for the roles to propagate before returning. Use
    `wait_role_propagation=false` to skip waiting (useful for bulk operations).

    Example:

    ```
    POST /apis/entities/v2/workspaces/ml-team/members
    {"principal": "user@example.com", "roles": ["Editor"]}
    ```

        [bold red]Required fields:[/] principal

        [green]Examples:[/]
        nemo workspaces members create --input-file config.json
        nemo workspaces members create --input-data '{"principal": "value"}'
        echo '{"json": "data"}' | nemo workspaces members create --input-file -
        nemo workspaces members create --<option> "value"
    """
    input_payload = _read_input_payload(input_file, input_data)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if principal is not None:
        input_payload["principal"] = principal
    if wait_role_propagation is not None:
        input_payload["wait_role_propagation"] = wait_role_propagation
    if roles:  # Check for non-empty list
        input_payload["roles"] = roles
    validate_required_fields(
        input_payload,
        ["principal"],
        "workspaces members create",
        {
            "principal": "The principal identifier (email, user ID, or group ID) (required)",
        },
    )

    resolved_workspace = input_payload.get("workspace")
    body = build_request_body(
        CreateWorkspaceMemberRequest,
        input_payload,
        exclude={"workspace", "wait_role_propagation"},
        command_name="workspaces members create",
    )
    query_params = _member_query_params(input_payload.get("wait_role_propagation"))

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(workspace=resolved_workspace, body=body, query_params=query_params)
    if handle_code_generation(WorkspacesClient, "create_workspace_member", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(WorkspacesClient).create_workspace_member(
        workspace=resolved_workspace, body=body, query_params=query_params
    )

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@members_app.command("delete")
@collect_warnings
@handle_errors
def delete_members(
    ctx: typer.Context,
    principal_id: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    wait_role_propagation: Annotated[
        bool | None, typer.Option("--wait-role-propagation", help=_WAIT_ROLE_PROPAGATION_HELP)
    ] = None,
) -> None:
    """Remove a member from the workspace by revoking all their roles.

    This revokes all active role bindings for the principal in the workspace. By
    default, this endpoint waits for all roles to be revoked before returning. Use
    `wait_role_propagation=false` to skip waiting (useful for bulk operations).

    Example:

    ```
    DELETE /apis/entities/v2/workspaces/ml-team/members/user@example.com
    ```"""
    state: CLIContext = ctx.obj
    state.typed_client(WorkspacesClient).delete_workspace_member(
        workspace=workspace, principal_id=principal_id, query_params=_member_query_params(wait_role_propagation)
    )

    typer.echo("✓ Deleted successfully")


@members_app.command("list")
@collect_warnings
@handle_errors
def list_members(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """List all members of a workspace with their roles.

    Returns a list of all principals with active role bindings in the workspace.

    Example:

    ```
    GET /apis/entities/v2/workspaces/ml-team/members
    ```"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("principal", None),
        Column("roles", None),
        Column("granted_by", None),
        Column("granted_at", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    kwargs = build_kwargs(workspace=workspace)
    if handle_code_generation(WorkspacesClient, "list_workspace_members", kwargs, resolved_output_format, state):
        return

    items = state.typed_client(WorkspacesClient).list_workspace_members(workspace=workspace)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )


@members_app.command("update")
@collect_warnings
@handle_errors
def update_members(
    ctx: typer.Context,
    principal_id: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    roles: Annotated[
        list[str] | None,
        typer.Option("--roles", help="Updated list of roles for the principal (can be repeated) (required)"),
    ] = None,
    wait_role_propagation: Annotated[
        bool | None, typer.Option("--wait-role-propagation", help=_WAIT_ROLE_PROPAGATION_HELP)
    ] = None,
    input_file: Annotated[
        str | None, typer.Option("--input-file", help=_INPUT_FILE_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    input_data: Annotated[
        str | None, typer.Option("--input-data", help=_INPUT_DATA_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Update the roles for a workspace member.

    This will revoke existing roles not in the new list and add new roles. By
    default, this endpoint waits for the roles to propagate before returning. Use
    `wait_role_propagation=false` to skip waiting (useful for bulk operations).

    Example:

    ```
    PUT /apis/entities/v2/workspaces/ml-team/members/user@example.com
    {"roles": ["Viewer", "Editor"]}
    ```

        [bold red]Required fields:[/] roles

        [green]Examples:[/]
        nemo workspaces members update <principal_id> --input-file config.json
        nemo workspaces members update <principal_id> --input-data '{"roles": "value"}'
        echo '{"json": "data"}' | nemo workspaces members update <principal_id> --input-file -
        nemo workspaces members update <principal_id> --<option> "value"
    """
    input_payload = _read_input_payload(input_file, input_data)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if roles:  # Check for non-empty list
        input_payload["roles"] = roles
    if wait_role_propagation is not None:
        input_payload["wait_role_propagation"] = wait_role_propagation
    validate_required_fields(
        input_payload,
        ["roles"],
        "workspaces members update",
        {
            "roles": "Updated list of roles for the principal (can be repeated) (required)",
        },
    )

    resolved_workspace = input_payload.get("workspace")
    body = build_request_body(
        UpdateWorkspaceMemberRequest,
        input_payload,
        exclude={"workspace", "wait_role_propagation"},
        command_name="workspaces members update",
    )
    query_params = _member_query_params(input_payload.get("wait_role_propagation"))

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(workspace=resolved_workspace, principal_id=principal_id, body=body, query_params=query_params)
    if handle_code_generation(WorkspacesClient, "update_workspace_member", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(WorkspacesClient).update_workspace_member(
        workspace=resolved_workspace, principal_id=principal_id, body=body, query_params=query_params
    )

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

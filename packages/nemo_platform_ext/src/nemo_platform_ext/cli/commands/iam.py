# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo iam`` command group, backed by the typed IAM client."""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer
from nemo_platform_plugin.iam.client import IAMClient
from nemo_platform_plugin.iam.types import (
    ListRoleBindingsQueryParams,
    RoleBindingInput,
    RolePropagationQueryParams,
)

from nemo_platform_ext.cli.core.api import build_kwargs, merge_filter_dict
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

app = create_typer_app(name="iam", help="IAM operations.")
role_bindings_app = create_typer_app(name="role_bindings", help="Manage role_bindings")
app.add_typer(role_bindings_app, name="role-bindings")

_WAIT_ROLE_PROPAGATION_HELP = (
    "If true, wait for role to propagate before returning (default: true). Set to false for bulk operations."
)
_PRINCIPAL_HELP = "The principal identifier (email, user ID, or group ID) (required)"
_ROLE_HELP = "The role name (e.g., 'Viewer', 'Editor', 'Admin') (required)"
_INPUT_FILE_HELP = "Path to JSON file (use '-' for stdin)"
_INPUT_DATA_HELP = "Input data for the request (JSON or YAML)"
_INPUT_PANEL = "Input Options"
_FILTER_PANEL = "Filter Options"


def _read_input_payload(input_file: str | None, input_data: str | None) -> dict[str, Any]:
    """Return the ``--input-file`` / ``--input-data`` payload, or an empty dict when neither is given."""
    if input_file or input_data:
        return read_data_input_with_flags(input_file=input_file, input_data=input_data)
    return {}


def _filter_query(value: str | dict[str, Any] | None) -> str | None:
    """Serialize a merged filter (text expression or JSON object) into the ``filter`` query value."""
    if isinstance(value, dict):
        return json.dumps(value)
    return value


def _list_role_bindings_query_params(
    *, filter_value: str | None, page: int | None, page_size: int | None, sort: str | None
) -> ListRoleBindingsQueryParams | None:
    query_params: ListRoleBindingsQueryParams = {}
    if filter_value is not None:
        query_params["filter"] = filter_value
    if page is not None:
        query_params["page"] = page
    if page_size is not None:
        query_params["page_size"] = page_size
    if sort is not None:
        query_params["sort"] = sort
    return query_params or None


def _role_propagation_query_params(wait_role_propagation: bool | None) -> RolePropagationQueryParams | None:
    if wait_role_propagation is None:
        return None
    return {"wait_role_propagation": wait_role_propagation}


@role_bindings_app.command("create")
@collect_warnings
@handle_errors
def create_role_bindings(
    ctx: typer.Context,
    principal: Annotated[str | None, typer.Option("--principal", help=_PRINCIPAL_HELP)] = None,
    role: Annotated[str | None, typer.Option("--role", help=_ROLE_HELP)] = None,
    wait_role_propagation: Annotated[
        bool | None, typer.Option("--wait-role-propagation", help=_WAIT_ROLE_PROPAGATION_HELP)
    ] = None,
    workspace: Annotated[
        str | None,
        typer.Option("--workspace", help="The workspace this binding applies to. None for platform-level roles."),
    ] = None,
    input_file: Annotated[
        str | None, typer.Option("--input-file", help=_INPUT_FILE_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    input_data: Annotated[
        str | None, typer.Option("--input-data", help=_INPUT_DATA_HELP, rich_help_panel=_INPUT_PANEL)
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Create a new role binding (Platform Admin only)

    [bold red]Required fields:[/] principal, role

    [green]Examples:[/]
    nemo iam role-bindings create --input-file config.json
    nemo iam role-bindings create --input-data '{"principal": "value", "role": "value"}'
    echo '{"json": "data"}' | nemo iam role-bindings create --input-file -
    nemo iam role-bindings create --<option> "value"
    """
    # Read base input (optional if all fields provided via flags), then apply
    # CLI flag overrides (flags take precedence).
    input_payload = _read_input_payload(input_file, input_data)
    if principal is not None:
        input_payload["principal"] = principal
    if role is not None:
        input_payload["role"] = role
    if wait_role_propagation is not None:
        input_payload["wait_role_propagation"] = wait_role_propagation
    if workspace is not None:
        input_payload["workspace"] = workspace
    validate_required_fields(
        input_payload,
        ["principal", "role"],
        "iam role-bindings create",
        {"principal": _PRINCIPAL_HELP, "role": _ROLE_HELP},
    )

    body = build_request_body(
        RoleBindingInput, input_payload, exclude={"wait_role_propagation"}, command_name="iam role-bindings create"
    )
    query_params = _role_propagation_query_params(input_payload.get("wait_role_propagation"))

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(body=body, query_params=query_params)
    if handle_code_generation(IAMClient, "create_role_binding", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IAMClient).create_role_binding(body=body, query_params=query_params)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@role_bindings_app.command("delete")
@collect_warnings
@handle_errors
def delete_role_bindings(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    wait_role_propagation: Annotated[
        bool | None, typer.Option("--wait-role-propagation", help=_WAIT_ROLE_PROPAGATION_HELP)
    ] = None,
) -> None:
    """Revoke a role binding (Platform Admin only)"""
    state: CLIContext = ctx.obj
    state.typed_client(IAMClient).revoke_role_binding(
        name=name, query_params=_role_propagation_query_params(wait_role_propagation)
    )

    typer.echo("✓ Deleted successfully")


@role_bindings_app.command("list")
@collect_warnings
@handle_errors
def list_role_bindings(
    ctx: typer.Context,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  granted_at: {gte: str, lte: str}\n  revoked_at: {gte: str, lte: str}\n\nFilter role bindings by principal, workspace, role, granted_by, is_active, granted_at, and revoked_at.",
            rich_help_panel=_FILTER_PANEL,
        ),
    ] = None,
    filter_granted_by: Annotated[str | None, typer.Option("--filter.granted-by", rich_help_panel=_FILTER_PANEL)] = None,
    filter_is_active: Annotated[bool | None, typer.Option("--filter.is-active", rich_help_panel=_FILTER_PANEL)] = None,
    filter_principal: Annotated[str | None, typer.Option("--filter.principal", rich_help_panel=_FILTER_PANEL)] = None,
    filter_role: Annotated[str | None, typer.Option("--filter.role", rich_help_panel=_FILTER_PANEL)] = None,
    filter_workspace: Annotated[str | None, typer.Option("--filter.workspace", rich_help_panel=_FILTER_PANEL)] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[
        str | None,
        typer.Option(
            "--sort", help="The field to sort by. To sort in decreasing order, use `-` in front of the field name."
        ),
    ] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List all role bindings (Platform Admin only)"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("name", None),
        Column("workspace", None),
        Column("granted_at", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    filter_value = _filter_query(
        merge_filter_dict(
            filter,
            granted_by=filter_granted_by,
            is_active=filter_is_active,
            principal=filter_principal,
            role=filter_role,
            workspace=filter_workspace,
        )
    )
    query_params = _list_role_bindings_query_params(
        filter_value=filter_value, page=page, page_size=page_size, sort=sort
    )
    kwargs = build_kwargs(query_params=query_params)
    if handle_code_generation(IAMClient, "list_role_bindings", kwargs, resolved_output_format, state, result="list"):
        return

    response = state.typed_client(IAMClient).list_role_bindings(query_params=query_params)
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


@role_bindings_app.command("get")
@collect_warnings
@handle_errors
def retrieve_role_bindings(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a specific role binding (Platform Admin only)"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name)
    if handle_code_generation(IAMClient, "get_role_binding", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(IAMClient).get_role_binding(name=name)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

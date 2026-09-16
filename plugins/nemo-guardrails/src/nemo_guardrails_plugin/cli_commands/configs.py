# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo guardrail configs`` command group, backed by the typed Guardrail client."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

import typer
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
    read_payload,
    validate_required_fields,
)
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)
from nemo_platform_plugin.guardrail.client import GuardrailClient
from nemo_platform_plugin.guardrail.types import (
    CreateGuardrailConfigRequest,
    ListGuardrailConfigsQueryParams,
    UpdateGuardrailConfigRequest,
)
from pydantic import TypeAdapter

app = create_typer_app(name="configs", help="Manage configs")

GuardrailConfigSortField = Literal["created_at", "-created_at", "updated_at", "-updated_at", "name", "-name"]


_BOOL = TypeAdapter(bool)


def _filter_query(value: str | dict[str, Any] | None) -> str | None:
    if isinstance(value, dict):
        return json.dumps(value)
    return value


def _list_configs_query_params(
    *, filter_value: str | None, page: int | None, page_size: int | None, sort: str | None
) -> ListGuardrailConfigsQueryParams | None:
    query_params: ListGuardrailConfigsQueryParams = {}
    if filter_value is not None:
        query_params["filter"] = filter_value
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
def create_configs(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="The name of the guardrail config (required)")] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    data: Annotated[str | None, typer.Option("--data", help="Guardrail configuration data (JSON string)")] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Description of the guardrail config")
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
    """Create a new guardrail config.

    [bold red]Required fields:[/] name

    [green]Examples:[/]
    nemo guardrail configs create <name> --input-file config.json
    nemo guardrail configs create <name> --input-data '{"name": "value"}'
    echo '{"json": "data"}' | nemo guardrail configs create <name> --input-file -
    nemo guardrail configs create <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if name is not None:
        input_payload["name"] = name
    if data is not None:
        input_payload["data"] = read_payload("data", data)
    if description is not None:
        input_payload["description"] = description
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["name"],
        "guardrail configs create",
        {
            "name": "The name of the guardrail config (required)",
        },
    )

    body = build_request_body(
        CreateGuardrailConfigRequest,
        input_payload,
        exclude={"workspace", "exist_ok"},
        command_name="guardrail configs create",
    )
    resolved_workspace = input_payload.get("workspace")
    # --input-data may carry exist_ok as a JSON/YAML string; parse it like the server would.
    resolved_exist_ok = _BOOL.validate_python(input_payload.get("exist_ok", False))

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(workspace=resolved_workspace, body=body, exist_ok=resolved_exist_ok or None)
    if handle_code_generation(GuardrailClient, "create_guardrail_config", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(GuardrailClient).create_guardrail_config(
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
def delete_configs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete a guardrail config."""
    state: CLIContext = ctx.obj
    state.typed_client(GuardrailClient).delete_guardrail_config(name=name, workspace=workspace)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_configs(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  updated_at: {gte: str, lte: str}\n\nFilter guardrail configs by name, description, project, created_at, and updated_at.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_description: Annotated[
        str | None, typer.Option("--filter.description", rich_help_panel="Filter Options")
    ] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_project: Annotated[str | None, typer.Option("--filter.project", rich_help_panel="Filter Options")] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[
        GuardrailConfigSortField | None,
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
    """List available guardrail configs.

    Lists guardrail configs for a specific workspace."""
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

    filter_value = _filter_query(
        merge_filter_dict(filter, description=filter_description, name=filter_name, project=filter_project)
    )
    query_params = _list_configs_query_params(filter_value=filter_value, page=page, page_size=page_size, sort=sort)
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)
    if handle_code_generation(
        GuardrailClient, "list_guardrail_configs", kwargs, resolved_output_format, state, result="list"
    ):
        return

    response = state.typed_client(GuardrailClient).list_guardrail_configs(
        workspace=workspace, query_params=query_params
    )
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
def retrieve_configs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get info about a guardrail configuration."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(GuardrailClient, "get_guardrail_config", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(GuardrailClient).get_guardrail_config(name=name, workspace=workspace)

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
def update_configs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    data: Annotated[str | None, typer.Option("--data", help="Guardrail configuration data (JSON string)")] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Description of the guardrail config")
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
    """Update model metadata.

    If the request body has an empty field, keep the old
    value.

        [green]Examples:[/]
        nemo guardrail configs update <name> --input-file config.json
        nemo guardrail configs update <name> --input-data '{"field": "value"}'
        echo '{"json": "data"}' | nemo guardrail configs update <name> --input-file -
        nemo guardrail configs update <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if data is not None:
        input_payload["data"] = read_payload("data", data)
    if description is not None:
        input_payload["description"] = description

    body = build_request_body(
        UpdateGuardrailConfigRequest, input_payload, exclude={"workspace"}, command_name="guardrail configs update"
    )
    resolved_workspace = input_payload.get("workspace")

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=resolved_workspace, body=body)
    if handle_code_generation(GuardrailClient, "update_guardrail_config", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(GuardrailClient).update_guardrail_config(
        name=name, workspace=resolved_workspace, body=body
    )

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

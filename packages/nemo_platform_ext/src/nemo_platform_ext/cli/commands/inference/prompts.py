# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo inference prompts`` command group, backed by the typed Models client."""

from __future__ import annotations

from typing import Annotated, Literal, cast

import typer
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import CreatePromptRequest, ListPromptsQueryParams, UpdatePromptRequest

from nemo_platform_ext.cli.commands.inference._common import (
    filter_query_value,
    offset_query_params,
    pop_exist_ok,
    pop_workspace,
    read_input_payload,
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
from nemo_platform_ext.cli.core.stdin_utils import build_request_body, read_payload, validate_required_fields
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="prompts", help="Manage prompts")

_INFERENCE_PARAMS_HELP = "Parameters for model inference. Extra fields can be supplied for additional options applied to the inference request directly. Fields not supported by the model may cause inference errors during evaluation. (JSON string)"


@app.command("create")
@collect_warnings
@handle_errors
def create_prompts(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="Name of the prompt. (required)")] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    inference_params: Annotated[str | None, typer.Option("--inference-params", help=_INFERENCE_PARAMS_HELP)] = None,
    input_variables: Annotated[
        list[str] | None, typer.Option("--input-variables", help="Can be repeated for multiple values")
    ] = None,
    messages: Annotated[str | None, typer.Option("--messages", help="JSON string")] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="The URN of the project associated with this prompt.")
    ] = None,
    response_format: Annotated[str | None, typer.Option("--response-format", help="JSON string")] = None,
    tags: Annotated[list[str] | None, typer.Option("--tags", help="Can be repeated for multiple values")] = None,
    tool_choice: Annotated[str | None, typer.Option("--tool-choice", help="JSON string")] = None,
    tools: Annotated[str | None, typer.Option("--tools", help="JSON string")] = None,
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
    """Create a new prompt.

    [bold red]Required fields:[/] name

    [green]Examples:[/]
    nemo inference prompts create <name> --input-file config.json
    nemo inference prompts create <name> --input-data '{"name": "value"}'
    echo '{"json": "data"}' | nemo inference prompts create <name> --input-file -
    nemo inference prompts create <name> --<option> "value"
    """
    input_payload = read_input_payload(input_file, input_data)

    if workspace is not None:
        input_payload["workspace"] = workspace
    if name is not None:
        input_payload["name"] = name
    if description is not None:
        input_payload["description"] = description
    if inference_params is not None:
        input_payload["inference_params"] = read_payload("inference_params", inference_params)
    if input_variables:  # Check for non-empty list
        input_payload["input_variables"] = input_variables
    if messages is not None:
        input_payload["messages"] = read_payload("messages", messages)
    if project is not None:
        input_payload["project"] = project
    if response_format is not None:
        input_payload["response_format"] = read_payload("response_format", response_format)
    if tags:  # Check for non-empty list
        input_payload["tags"] = tags
    if tool_choice is not None:
        input_payload["tool_choice"] = read_payload("tool_choice", tool_choice)
    if tools is not None:
        input_payload["tools"] = read_payload("tools", tools)
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    validate_required_fields(
        input_payload,
        ["name"],
        "inference prompts create",
        {
            "name": "Name of the prompt. (required)",
        },
    )

    request_workspace = pop_workspace(input_payload)
    request_exist_ok = pop_exist_ok(input_payload)
    body = build_request_body(CreatePromptRequest, input_payload, command_name="inference prompts create")
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(workspace=request_workspace, body=body, exist_ok=request_exist_ok)
    if handle_code_generation(ModelsClient, "create_prompt", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ModelsClient).create_prompt(
        workspace=request_workspace, body=body, exist_ok=bool(request_exist_ok)
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
def delete_prompts(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete a prompt by workspace and name."""
    state: CLIContext = ctx.obj
    state.typed_client(ModelsClient).delete_prompt(name=name, workspace=workspace)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_prompts(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  updated_at: {gte: str, lte: str}\n\nFilter prompts by workspace, project, name, description, created_at, and updated_at.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_description: Annotated[
        str | None, typer.Option("--filter.description", rich_help_panel="Filter Options")
    ] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_project: Annotated[str | None, typer.Option("--filter.project", rich_help_panel="Filter Options")] = None,
    filter_workspace: Annotated[
        str | None, typer.Option("--filter.workspace", rich_help_panel="Filter Options")
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[
        Literal["created_at", "-created_at", "updated_at", "-updated_at", "name", "-name"] | None,
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
    """List prompts for a specific workspace."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("name", None),
        Column("workspace", None),
        Column("created_at", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    filter_value = filter_query_value(
        merge_filter_dict(
            filter, description=filter_description, name=filter_name, project=filter_project, workspace=filter_workspace
        )
    )
    query_params = cast(
        "ListPromptsQueryParams | None",
        offset_query_params(filter_value=filter_value, page=page, page_size=page_size, sort=sort),
    )
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)
    if handle_code_generation(ModelsClient, "list_prompts", kwargs, resolved_output_format, state, result="list"):
        return

    response = state.typed_client(ModelsClient).list_prompts(workspace=workspace, query_params=query_params)
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
def retrieve_prompts(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a prompt by workspace and name."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(ModelsClient, "get_prompt", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ModelsClient).get_prompt(name=name, workspace=workspace)

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
def update_prompts(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    inference_params: Annotated[str | None, typer.Option("--inference-params", help=_INFERENCE_PARAMS_HELP)] = None,
    input_variables: Annotated[
        list[str] | None, typer.Option("--input-variables", help="Can be repeated for multiple values")
    ] = None,
    messages: Annotated[str | None, typer.Option("--messages", help="JSON string")] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="The URN of the project associated with this prompt.")
    ] = None,
    response_format: Annotated[str | None, typer.Option("--response-format", help="JSON string")] = None,
    tags: Annotated[list[str] | None, typer.Option("--tags", help="Can be repeated for multiple values")] = None,
    tool_choice: Annotated[str | None, typer.Option("--tool-choice", help="JSON string")] = None,
    tools: Annotated[str | None, typer.Option("--tools", help="JSON string")] = None,
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
    """Update an existing prompt (full replacement of mutable fields).

    [green]Examples:[/]
    nemo inference prompts update <name> --input-file config.json
    nemo inference prompts update <name> --input-data '{"field": "value"}'
    echo '{"json": "data"}' | nemo inference prompts update <name> --input-file -
    nemo inference prompts update <name> --<option> "value"
    """
    input_payload = read_input_payload(input_file, input_data)

    if workspace is not None:
        input_payload["workspace"] = workspace
    if description is not None:
        input_payload["description"] = description
    if inference_params is not None:
        input_payload["inference_params"] = read_payload("inference_params", inference_params)
    if input_variables:  # Check for non-empty list
        input_payload["input_variables"] = input_variables
    if messages is not None:
        input_payload["messages"] = read_payload("messages", messages)
    if project is not None:
        input_payload["project"] = project
    if response_format is not None:
        input_payload["response_format"] = read_payload("response_format", response_format)
    if tags:  # Check for non-empty list
        input_payload["tags"] = tags
    if tool_choice is not None:
        input_payload["tool_choice"] = read_payload("tool_choice", tool_choice)
    if tools is not None:
        input_payload["tools"] = read_payload("tools", tools)

    request_workspace = pop_workspace(input_payload)
    body = build_request_body(UpdatePromptRequest, input_payload, command_name="inference prompts update")
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=request_workspace, body=body)
    if handle_code_generation(ModelsClient, "update_prompt", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ModelsClient).update_prompt(name=name, workspace=request_workspace, body=body)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

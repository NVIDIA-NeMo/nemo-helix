# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from typing import Annotated, Literal

import typer

from nemo_helix_ext.cli.core.api import build_kwargs, merge_filter_dict
from nemo_helix_ext.cli.core.code_generator import handle_code_generation
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_output,
    validate_stream_output_format,
)
from nemo_helix_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_helix_ext.cli.core.pagination import PaginationType, fetch_all_pages, warn_if_more_pages
from nemo_helix_ext.cli.core.stdin_utils import read_data_input_with_flags, read_payload, validate_required_fields
from nemo_helix_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="prompts", help="Manage prompts")


@app.command("create")
@collect_warnings
@handle_errors
def create_prompts(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="Name of the prompt. (required)")] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    inference_params: Annotated[
        str | None,
        typer.Option(
            "--inference-params",
            help="Parameters for model inference. Extra fields can be supplied for additional options applied to the inference request directly. Fields not supported by the model may cause inference errors during evaluation. (JSON string)",
        ),
    ] = None,
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
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["name"],
        "inference prompts create",
        {
            "name": "Name of the prompt. (required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "prompts"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.prompts.create(**all_kwargs)

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
def delete_prompts(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete a prompt by workspace and name."""
    state: CLIContext = ctx.obj
    client = state.get_client()

    kwargs = build_kwargs(
        workspace=workspace,
    )
    client.inference.prompts.delete(name, **kwargs)

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
    output_format = state.get_output_format(output_format)
    validate_stream_output_format(output_format, stream)

    check_output_columns_with_format(columns, output_format)

    default_columns = [
        Column("name", None),
        Column("workspace", None),
        Column("created_at", None),
    ]
    if columns is None or str(columns).strip() == "default":
        columns = default_columns

    kwargs = build_kwargs(
        workspace=workspace,
        filter=merge_filter_dict(
            filter, description=filter_description, name=filter_name, project=filter_project, workspace=filter_workspace
        ),
        page=page,
        page_size=page_size,
        sort=sort,
    )

    if handle_code_generation(["inference", "prompts"], "list", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = ()
    pagination_type = PaginationType.PAGE_NUMBER
    if all_pages:
        items = fetch_all_pages(
            client.inference.prompts.list,
            path_args=path_args,
            body_args=kwargs,
            pagination_type=pagination_type,
        )
    else:
        items = client.inference.prompts.list(*path_args, **kwargs)

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
def retrieve_prompts(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a prompt by workspace and name."""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
    )
    if handle_code_generation(["inference", "prompts"], "retrieve", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.prompts.retrieve(name, **kwargs)

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
def update_prompts(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    inference_params: Annotated[
        str | None,
        typer.Option(
            "--inference-params",
            help="Parameters for model inference. Extra fields can be supplied for additional options applied to the inference request directly. Fields not supported by the model may cause inference errors during evaluation. (JSON string)",
        ),
    ] = None,
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
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
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

    all_kwargs = {"name": name, **input_payload}

    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "prompts"], "update", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.prompts.update(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

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

app = create_typer_app(name="adapters", help="Manage adapters")


@app.command("create")
@collect_warnings
@handle_errors
def create_adapters(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Argument(
            help="Name of the adapter. Name must be unique in the workspace. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)"
        ),
    ] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    fileset: Annotated[
        str | None,
        typer.Option(
            "--fileset",
            help="Location where adapter files are stored - expected format {workspace}/{fileset_name} (required)",
        ),
    ] = None,
    finetuning_type: Annotated[
        Literal[
            "lora_merged",
            "all_weights",
            "last_layer",
            "top_layers",
            "gradual_unfreezing",
            "bias_only",
            "attention_only",
            "lora",
            "qlora",
            "adalora",
            "dora",
            "lora_plus",
            "prompt_tuning",
            "prefix_tuning",
            "p_tuning",
            "p_tuning_v2",
            "soft_prompt",
            "ppo",
            "dpo",
            "cdpo",
            "ipo",
            "orpo",
            "kto",
            "rrhf",
            "grpo",
        ]
        | None,
        typer.Option("--finetuning-type", help="Finetuning types. (required)"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Base model entity. Use `{workspace}/{model_name}` to reference a model in any workspace, or a single `{model_name}` resolved in the path workspace. A single name (2-63 characters) or 'workspace/model*name' where each segment is a valid name (lowercase, digits, hyphens, and temporarily @ . + *; no leading/trailing or consecutive hyphens). If one slash, both sides must be non-empty. (required)",
        ),
    ] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Optional description of the adapter")
    ] = None,
    enabled: Annotated[
        bool | None,
        typer.Option("--enabled", help="Whether to make this adapter available for inference post training"),
    ] = None,
    lora_config: Annotated[
        str | None, typer.Option("--lora-config", help="Lora configuration specifics (JSON string)")
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
    """Create an adapter under a base model specified by the "model" field in the body.

    [bold red]Required fields:[/] fileset, finetuning_type, model, name

    [green]Examples:[/]
    nemo adapters create <name> --input-file config.json
    nemo adapters create <name> --input-data '{"fileset": "value", "finetuning_type": "value", "model": "value", "name": "value"}'
    echo '{"json": "data"}' | nemo adapters create <name> --input-file -
    nemo adapters create <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if fileset is not None:
        input_payload["fileset"] = fileset
    if finetuning_type is not None:
        input_payload["finetuning_type"] = finetuning_type
    if model is not None:
        input_payload["model"] = model
    if name is not None:
        input_payload["name"] = name
    if description is not None:
        input_payload["description"] = description
    if enabled is not None:
        input_payload["enabled"] = enabled
    if lora_config is not None:
        input_payload["lora_config"] = read_payload("lora_config", lora_config)
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["fileset", "finetuning_type", "model", "name"],
        "adapters create",
        {
            "fileset": "Location where adapter files are stored - expected format {workspace}/{fileset_name} (required)",
            "finetuning_type": "Finetuning types. (required)",
            "model": "Base model entity. Use `{workspace}/{model_name}` to reference a model in any workspace, or a single `{model_name}` resolved in the path workspace. A single name (2-63 characters) or 'workspace/model*name' where each segment is a valid name (lowercase, digits, hyphens, and temporarily @ . + *; no leading/trailing or consecutive hyphens). If one slash, both sides must be non-empty. (required)",
            "name": "Name of the adapter. Name must be unique in the workspace. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["adapters"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.adapters.create(**all_kwargs)

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
def delete_adapters(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete Adapter"""
    state: CLIContext = ctx.obj
    client = state.get_client()

    kwargs = build_kwargs(
        workspace=workspace,
    )
    client.adapters.delete(name, **kwargs)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_adapters(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  updated_at: {gte: str, lte: str}\n\nFilter adapters by name, model (parent model ref string, stored on the adapter), description, fileset, finetuning_type, enabled, created_at, and updated_at.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_description: Annotated[
        str | None, typer.Option("--filter.description", rich_help_panel="Filter Options")
    ] = None,
    filter_enabled: Annotated[bool | None, typer.Option("--filter.enabled", rich_help_panel="Filter Options")] = None,
    filter_fileset: Annotated[str | None, typer.Option("--filter.fileset", rich_help_panel="Filter Options")] = None,
    filter_finetuning_type: Annotated[
        str | None, typer.Option("--filter.finetuning-type", rich_help_panel="Filter Options")
    ] = None,
    filter_model: Annotated[str | None, typer.Option("--filter.model", rich_help_panel="Filter Options")] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
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
    """List Adapters"""
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
            filter,
            description=filter_description,
            enabled=filter_enabled,
            fileset=filter_fileset,
            finetuning_type=filter_finetuning_type,
            model=filter_model,
            name=filter_name,
        ),
        page=page,
        page_size=page_size,
        sort=sort,
    )

    if handle_code_generation(["adapters"], "list", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = ()
    pagination_type = PaginationType.PAGE_NUMBER
    if all_pages:
        items = fetch_all_pages(
            client.adapters.list,
            path_args=path_args,
            body_args=kwargs,
            pagination_type=pagination_type,
        )
    else:
        items = client.adapters.list(*path_args, **kwargs)

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


@app.command("patch")
@collect_warnings
@handle_errors
def patch_adapters(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Optional description of the adapter")
    ] = None,
    enabled: Annotated[
        bool | None,
        typer.Option("--enabled", help="Whether to make this adapter available for inference post training"),
    ] = None,
    fileset: Annotated[str | None, typer.Option("--fileset", help="Updated fileset for the adapter")] = None,
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
    """Update Adapter

    [green]Examples:[/]
    nemo adapters patch <name> --input-file config.json
    nemo adapters patch <name> --input-data '{"field": "value"}'
    echo '{"json": "data"}' | nemo adapters patch <name> --input-file -
    nemo adapters patch <name> --<option> "value"
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
    if enabled is not None:
        input_payload["enabled"] = enabled
    if fileset is not None:
        input_payload["fileset"] = fileset

    all_kwargs = {"name": name, **input_payload}

    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["adapters"], "patch", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.adapters.patch(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_adapters(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get Adapter"""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
    )
    if handle_code_generation(["adapters"], "retrieve", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.adapters.retrieve(name, **kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

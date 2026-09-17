# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from importlib import import_module as _importlib_import_module
from typing import Annotated, Literal

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
from nemo_platform_ext.cli.core.pagination import PaginationType, fetch_all_pages, warn_if_more_pages
from nemo_platform_ext.cli.core.stdin_utils import read_data_input_with_flags, read_payload, validate_required_fields
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

_cli_child_adapters = _importlib_import_module("nemo_platform_ext.cli.commands.api.models.adapters")

app = create_typer_app(name="models", help="Manage models")

app.add_typer(_cli_child_adapters.app, name="adapters")


@app.command("create")
@collect_warnings
@handle_errors
def create_models(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Argument(
            help="Name of the model entity. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)"
        ),
    ] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    api_endpoint: Annotated[
        str | None, typer.Option("--api-endpoint", help="Data about an inference endpoint. (JSON string)")
    ] = None,
    backend_format: Annotated[
        Literal["OPENAI_CHAT", "ANTHROPIC_MESSAGES"] | None,
        typer.Option(
            "--backend-format", help="Inference backend API wire formats understood by IGW and middleware plugins."
        ),
    ] = None,
    base_model: Annotated[
        str | None,
        typer.Option("--base-model", help="Link to another model which is used as a base for the current model"),
    ] = None,
    custom_fields: Annotated[
        str | None, typer.Option("--custom-fields", help="Custom fields for additional metadata (JSON string)")
    ] = None,
    description: Annotated[str | None, typer.Option("--description", help="Optional description of the model")] = None,
    fileset: Annotated[
        str | None,
        typer.Option(
            "--fileset",
            help="A set of checkpoint files, configs, and other auxiliary info associated with this model - expected format {workspace}/{fileset_name}",
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
        typer.Option("--finetuning-type", help="Finetuning types."),
    ] = None,
    model_providers: Annotated[
        list[str] | None,
        typer.Option(
            "--model-providers",
            help="List of ModelProvider workspace/name resource names that provide inference for this Model Entity (can be repeated)",
        ),
    ] = None,
    ownership: Annotated[
        str | None, typer.Option("--ownership", help="Ownership information for the model (JSON string)")
    ] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="The URN of the project associated with this model entity")
    ] = None,
    prompt: Annotated[
        str | None, typer.Option("--prompt", help="Configuration for prompt engineering. (JSON string)")
    ] = None,
    spec: Annotated[
        str | None, typer.Option("--spec", help="Detailed specification for a model. (JSON string)")
    ] = None,
    trust_remote_code: Annotated[
        bool | None,
        typer.Option(
            "--trust-remote-code",
            help="Whether to trust remote code for the checkpoint. Some models without support in certain libraries such as Transformers require additional custom Python code to execute. Due to security ramifications of running arbitrary code, this can only be set to true on one of the following conditions: (1) the model's fileset's source is pre-approved in the platform config, or (2) the user creating this model is an administrator.",
        ),
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
    """Create a new model entity.

    This endpoint creates a new Model Entity in the Models service database. The
    Model Entity will be registered for use within the platform.

        [bold red]Required fields:[/] name

        [green]Examples:[/]
        nemo models create <name> --input-file config.json
        nemo models create <name> --input-data '{"name": "value"}'
        echo '{"json": "data"}' | nemo models create <name> --input-file -
        nemo models create <name> --<option> "value"
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
    if api_endpoint is not None:
        input_payload["api_endpoint"] = read_payload("api_endpoint", api_endpoint)
    if backend_format is not None:
        input_payload["backend_format"] = backend_format
    if base_model is not None:
        input_payload["base_model"] = base_model
    if custom_fields is not None:
        input_payload["custom_fields"] = read_payload("custom_fields", custom_fields)
    if description is not None:
        input_payload["description"] = description
    if fileset is not None:
        input_payload["fileset"] = fileset
    if finetuning_type is not None:
        input_payload["finetuning_type"] = finetuning_type
    if model_providers:  # Check for non-empty list
        input_payload["model_providers"] = model_providers
    if ownership is not None:
        input_payload["ownership"] = read_payload("ownership", ownership)
    if project is not None:
        input_payload["project"] = project
    if prompt is not None:
        input_payload["prompt"] = read_payload("prompt", prompt)
    if spec is not None:
        input_payload["spec"] = read_payload("spec", spec)
    if trust_remote_code is not None:
        input_payload["trust_remote_code"] = trust_remote_code
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["name"],
        "models create",
        {
            "name": "Name of the model entity. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["models"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.models.create(**all_kwargs)

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
def delete_models(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete Model entity.

    Permanently deletes a model entity from the platform."""
    state: CLIContext = ctx.obj
    client = state.get_client()

    kwargs = build_kwargs(
        workspace=workspace,
    )
    client.models.delete(name, **kwargs)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_models(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  updated_at: {gte: str, lte: str}\n\nFilter models by name, project, workspace, base_model, adapters, finetuning_type, prompt, lora_enabled, description, created_at, and updated_at.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_adapters: Annotated[str | None, typer.Option("--filter.adapters", rich_help_panel="Filter Options")] = None,
    filter_base_model: Annotated[
        str | None, typer.Option("--filter.base-model", rich_help_panel="Filter Options")
    ] = None,
    filter_description: Annotated[
        str | None, typer.Option("--filter.description", rich_help_panel="Filter Options")
    ] = None,
    filter_fileset: Annotated[str | None, typer.Option("--filter.fileset", rich_help_panel="Filter Options")] = None,
    filter_finetuning_type: Annotated[
        bool | None, typer.Option("--filter.finetuning-type", rich_help_panel="Filter Options")
    ] = None,
    filter_lora_enabled: Annotated[
        bool | None, typer.Option("--filter.lora-enabled", rich_help_panel="Filter Options")
    ] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_project: Annotated[str | None, typer.Option("--filter.project", rich_help_panel="Filter Options")] = None,
    filter_prompt: Annotated[bool | None, typer.Option("--filter.prompt", rich_help_panel="Filter Options")] = None,
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
    verbose: Annotated[bool | None, typer.Option("--verbose", help="Whether to include full spec details")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List Models endpoint with filtering, pagination, and sorting.

    Supports filter parameters for various criteria (including peft, custom fields),
    pagination (page, page_size), sorting, and workspace filtering via query
    parameter."""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)
    validate_stream_output_format(output_format, stream)

    check_output_columns_with_format(columns, output_format)

    default_columns = [
        Column("name", None),
        Column("description", None),
    ]
    if columns is None or str(columns).strip() == "default":
        columns = default_columns

    kwargs = build_kwargs(
        workspace=workspace,
        filter=merge_filter_dict(
            filter,
            adapters=filter_adapters,
            base_model=filter_base_model,
            description=filter_description,
            fileset=filter_fileset,
            finetuning_type=filter_finetuning_type,
            lora_enabled=filter_lora_enabled,
            name=filter_name,
            project=filter_project,
            prompt=filter_prompt,
            workspace=filter_workspace,
        ),
        page=page,
        page_size=page_size,
        sort=sort,
        verbose=verbose,
    )

    if handle_code_generation(["models"], "list", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = ()
    pagination_type = PaginationType.PAGE_NUMBER
    if all_pages:
        items = fetch_all_pages(
            client.models.list,
            path_args=path_args,
            body_args=kwargs,
            pagination_type=pagination_type,
        )
    else:
        items = client.models.list(*path_args, **kwargs)

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
def retrieve_models(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    verbose: Annotated[bool | None, typer.Option("--verbose", help="Whether to include full spec details")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get Model by Workspace and Name.

    Returns the details of a specific model entity identified by its workspace and
    name."""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
        verbose=verbose,
    )
    if handle_code_generation(["models"], "retrieve", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.models.retrieve(name, **kwargs)

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
def update_models(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    verbose: Annotated[bool | None, typer.Option("--verbose", help="Whether to include full spec details")] = None,
    api_endpoint: Annotated[
        str | None, typer.Option("--api-endpoint", help="Data about an inference endpoint. (JSON string)")
    ] = None,
    backend_format: Annotated[
        Literal["OPENAI_CHAT", "ANTHROPIC_MESSAGES"] | None,
        typer.Option(
            "--backend-format", help="Inference backend API wire formats understood by IGW and middleware plugins."
        ),
    ] = None,
    base_model: Annotated[
        str | None,
        typer.Option("--base-model", help="Link to another model which is used as a base for the current model"),
    ] = None,
    custom_fields: Annotated[
        str | None, typer.Option("--custom-fields", help="Custom fields for additional metadata (JSON string)")
    ] = None,
    description: Annotated[str | None, typer.Option("--description", help="Optional description of the model")] = None,
    fileset: Annotated[
        str | None,
        typer.Option(
            "--fileset",
            help="A set of checkpoint files, configs, and other auxiliary info associated with this model - expected format {workspace}/{fileset_name}",
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
        typer.Option("--finetuning-type", help="Finetuning types."),
    ] = None,
    model_providers: Annotated[
        list[str] | None,
        typer.Option(
            "--model-providers",
            help="List of ModelProvider workspace/name resource names that provide inference for this Model Entity (can be repeated)",
        ),
    ] = None,
    ownership: Annotated[
        str | None, typer.Option("--ownership", help="Ownership information for the model (JSON string)")
    ] = None,
    prompt: Annotated[
        str | None, typer.Option("--prompt", help="Configuration for prompt engineering. (JSON string)")
    ] = None,
    spec: Annotated[
        str | None, typer.Option("--spec", help="Detailed specification for a model. (JSON string)")
    ] = None,
    trust_remote_code: Annotated[
        bool | None,
        typer.Option(
            "--trust-remote-code",
            help="Whether to trust remote code for the checkpoint. Some models without support in certain libraries such as Transformers require additional custom Python code to execute. Due to security ramifications of running arbitrary code, this can only be set to true on one of the following conditions: (1) the model's fileset's source is pre-approved in the platform config, or (2) the user creating this model is an administrator.",
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
    """Update Model metadata.

    Updates the metadata of an existing model entity.

    If the request body has an
    empty field, the old value is kept.

        [green]Examples:[/]
        nemo models update <name> --input-file config.json
        nemo models update <name> --input-data '{"field": "value"}'
        echo '{"json": "data"}' | nemo models update <name> --input-file -
        nemo models update <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if verbose is not None:
        input_payload["verbose"] = verbose
    if api_endpoint is not None:
        input_payload["api_endpoint"] = read_payload("api_endpoint", api_endpoint)
    if backend_format is not None:
        input_payload["backend_format"] = backend_format
    if base_model is not None:
        input_payload["base_model"] = base_model
    if custom_fields is not None:
        input_payload["custom_fields"] = read_payload("custom_fields", custom_fields)
    if description is not None:
        input_payload["description"] = description
    if fileset is not None:
        input_payload["fileset"] = fileset
    if finetuning_type is not None:
        input_payload["finetuning_type"] = finetuning_type
    if model_providers:  # Check for non-empty list
        input_payload["model_providers"] = model_providers
    if ownership is not None:
        input_payload["ownership"] = read_payload("ownership", ownership)
    if prompt is not None:
        input_payload["prompt"] = read_payload("prompt", prompt)
    if spec is not None:
        input_payload["spec"] = read_payload("spec", spec)
    if trust_remote_code is not None:
        input_payload["trust_remote_code"] = trust_remote_code

    all_kwargs = {"name": name, **input_payload}

    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["models"], "update", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.models.update(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

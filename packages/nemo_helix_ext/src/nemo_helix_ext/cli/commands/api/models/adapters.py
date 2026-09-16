# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from typing import Annotated, Literal

import typer

from nemo_helix_ext.cli.core.api import build_kwargs
from nemo_helix_ext.cli.core.code_generator import handle_code_generation
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import format_output
from nemo_helix_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_helix_ext.cli.core.stdin_utils import read_data_input_with_flags, read_payload, validate_required_fields
from nemo_helix_ext.cli.core.types import EntityOutputFormatOption

app = create_typer_app(name="adapters", help="Manage adapters")


@app.command("create")
@collect_warnings
@handle_errors
def create_adapters(
    ctx: typer.Context,
    model_name: Annotated[str, typer.Argument()],
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
    """Adds an Adapter to the Model

    [bold red]Required fields:[/] fileset, finetuning_type, name

    [green]Examples:[/]
    nemo models adapters create <model_name> <name> --input-file config.json
    nemo models adapters create <model_name> <name> --input-data '{"fileset": "value", "finetuning_type": "value", "name": "value"}'
    echo '{"json": "data"}' | nemo models adapters create <model_name> <name> --input-file -
    nemo models adapters create <model_name> <name> --<option> "value"
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
        ["fileset", "finetuning_type", "name"],
        "models adapters create",
        {
            "fileset": "Location where adapter files are stored - expected format {workspace}/{fileset_name} (required)",
            "finetuning_type": "Finetuning types. (required)",
            "name": "Name of the adapter. Name must be unique in the workspace. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)",
        },
    )

    all_kwargs = {"model_name": model_name, **input_payload}
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["models", "adapters"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.models.adapters.create(**all_kwargs)

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
    adapter: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    model_name: Annotated[str, typer.Option("--model-name")] = ...,
) -> None:
    """Delete Adapter from Model entity.

    Permanently deletes an adapter from a model entity, if it was deployed, it will
    be cleaned up automatically."""
    state: CLIContext = ctx.obj
    client = state.get_client()

    kwargs = build_kwargs(
        workspace=workspace,
        model_name=model_name,
    )
    client.models.adapters.delete(adapter, **kwargs)

    typer.echo("✓ Deleted successfully")


@app.command("update")
@collect_warnings
@handle_errors
def update_adapters(
    ctx: typer.Context,
    adapter: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    model_name: Annotated[str | None, typer.Option("--model-name", help="(required)")] = None,
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
    """Update Adapter deployment or description.

    [bold red]Required fields:[/] model_name

    [green]Examples:[/]
    nemo models adapters update <adapter> --input-file config.json
    nemo models adapters update <adapter> --input-data '{"model_name": "value"}'
    echo '{"json": "data"}' | nemo models adapters update <adapter> --input-file -
    nemo models adapters update <adapter> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if model_name is not None:
        input_payload["model_name"] = model_name
    if description is not None:
        input_payload["description"] = description
    if enabled is not None:
        input_payload["enabled"] = enabled
    if fileset is not None:
        input_payload["fileset"] = fileset
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["model_name"],
        "models adapters update",
        {
            "model_name": "(required)",
        },
    )

    all_kwargs = {"adapter": adapter, **input_payload}

    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["models", "adapters"], "update", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.models.adapters.update(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

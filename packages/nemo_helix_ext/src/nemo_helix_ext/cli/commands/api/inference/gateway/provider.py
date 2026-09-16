# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from typing import Annotated

import typer

from nemo_helix_ext.cli.core.api import build_kwargs
from nemo_helix_ext.cli.core.code_generator import handle_code_generation
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import format_output
from nemo_helix_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_helix_ext.cli.core.stdin_utils import read_data_input_with_flags, read_payload, validate_required_fields
from nemo_helix_ext.cli.core.types import EntityOutputFormatOption

app = create_typer_app(name="provider", help="Manage provider")


@app.command("delete")
@collect_warnings
@handle_errors
def delete_provider(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str, typer.Option("--name")] = ...,
) -> None:
    """Proxy requests to provider inference endpoints."""
    state: CLIContext = ctx.obj
    client = state.get_client()

    kwargs = build_kwargs(
        workspace=workspace,
        name=name,
    )
    client.inference.gateway.provider.delete(trailing_uri, **kwargs)

    typer.echo("✓ Deleted successfully")


@app.command("get")
@collect_warnings
@handle_errors
def get_provider(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str, typer.Option("--name")] = ...,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Proxy requests to provider inference endpoints."""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
        name=name,
    )
    if handle_code_generation(["inference", "gateway", "provider"], "get", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.gateway.provider.get(trailing_uri, **kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("patch")
@collect_warnings
@handle_errors
def patch_provider(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str | None, typer.Option("--name", help="(required)")] = None,
    body: Annotated[str | None, typer.Option("--body", help="JSON string")] = None,
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
    """Proxy requests to provider inference endpoints.

    [bold red]Required fields:[/] name

    [green]Examples:[/]
    nemo inference gateway provider patch <trailing_uri> --input-file config.json
    nemo inference gateway provider patch <trailing_uri> --input-data '{"name": "value"}'
    echo '{"json": "data"}' | nemo inference gateway provider patch <trailing_uri> --input-file -
    nemo inference gateway provider patch <trailing_uri> --<option> "value"
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
    if body is not None:
        input_payload["body"] = read_payload("body", body)
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["name"],
        "inference gateway provider patch",
        {
            "name": "(required)",
        },
    )

    all_kwargs = {"trailing_uri": trailing_uri, **input_payload}

    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "gateway", "provider"], "patch", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.gateway.provider.patch(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("post")
@collect_warnings
@handle_errors
def post_provider(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    name: Annotated[str | None, typer.Argument(help="(required)")] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    body: Annotated[str | None, typer.Option("--body", help="JSON string")] = None,
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
    """Proxy requests to provider inference endpoints.

    [bold red]Required fields:[/] name

    [green]Examples:[/]
    nemo inference gateway provider post <trailing_uri> <name> --input-file config.json
    nemo inference gateway provider post <trailing_uri> <name> --input-data '{"name": "value"}'
    echo '{"json": "data"}' | nemo inference gateway provider post <trailing_uri> <name> --input-file -
    nemo inference gateway provider post <trailing_uri> <name> --<option> "value"
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
    if body is not None:
        input_payload["body"] = read_payload("body", body)
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["name"],
        "inference gateway provider post",
        {
            "name": "(required)",
        },
    )

    all_kwargs = {"trailing_uri": trailing_uri, **input_payload}
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "gateway", "provider"], "post", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.gateway.provider.post(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("put")
@collect_warnings
@handle_errors
def put_provider(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    name: Annotated[str | None, typer.Argument(help="(required)")] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    body: Annotated[str | None, typer.Option("--body", help="JSON string")] = None,
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
    """Proxy requests to provider inference endpoints.

    [bold red]Required fields:[/] name

    [green]Examples:[/]
    nemo inference gateway provider put <trailing_uri> <name> --input-file config.json
    nemo inference gateway provider put <trailing_uri> <name> --input-data '{"name": "value"}'
    echo '{"json": "data"}' | nemo inference gateway provider put <trailing_uri> <name> --input-file -
    nemo inference gateway provider put <trailing_uri> <name> --<option> "value"
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
    if body is not None:
        input_payload["body"] = read_payload("body", body)
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["name"],
        "inference gateway provider put",
        {
            "name": "(required)",
        },
    )

    all_kwargs = {"trailing_uri": trailing_uri, **input_payload}
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "gateway", "provider"], "put", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.gateway.provider.put(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("ready")
@collect_warnings
@handle_errors
def ready_provider(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Check if a model provider is registered in the gateway's cache.

    This is a lightweight endpoint that only checks the gateway's internal state,
    without making any requests to the actual provider backend. Use this to verify
    the gateway is ready to route requests to a provider after deployment.

    Returns: 200 OK with provider info if the provider is registered 404 Not Found
    if the provider is not yet in the gateway's cache"""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
    )
    if handle_code_generation(["inference", "gateway", "provider"], "ready", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.gateway.provider.ready(name, **kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

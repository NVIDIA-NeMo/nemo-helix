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

app = create_typer_app(name="model", help="Manage model")


@app.command("delete")
@collect_warnings
@handle_errors
def delete_model(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str, typer.Option("--name")] = ...,
) -> None:
    """Proxy requests to model entity inference endpoints.

    All inference requests must resolve to a `VirtualModel`. The platform's provider
    reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
    served model entity (named after the entity, with `default_model_entity` set to
    the entity ref) so this is the typical case; operators can also create custom
    VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
    which no VirtualModel can be found return `404`."""
    state: CLIContext = ctx.obj
    client = state.get_client()

    kwargs = build_kwargs(
        workspace=workspace,
        name=name,
    )
    client.inference.gateway.model.delete(trailing_uri, **kwargs)

    typer.echo("✓ Deleted successfully")


@app.command("get")
@collect_warnings
@handle_errors
def get_model(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str, typer.Option("--name")] = ...,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Proxy requests to model entity inference endpoints.

    All inference requests must resolve to a `VirtualModel`. The platform's provider
    reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
    served model entity (named after the entity, with `default_model_entity` set to
    the entity ref) so this is the typical case; operators can also create custom
    VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
    which no VirtualModel can be found return `404`."""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
        name=name,
    )
    if handle_code_generation(["inference", "gateway", "model"], "get", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.gateway.model.get(trailing_uri, **kwargs)

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
def patch_model(
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
    """Proxy requests to model entity inference endpoints.

    All inference requests must resolve to a `VirtualModel`. The platform's provider
    reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
    served model entity (named after the entity, with `default_model_entity` set to
    the entity ref) so this is the typical case; operators can also create custom
    VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
    which no VirtualModel can be found return `404`.

        [bold red]Required fields:[/] name

        [green]Examples:[/]
        nemo inference gateway model patch <trailing_uri> --input-file config.json
        nemo inference gateway model patch <trailing_uri> --input-data '{"name": "value"}'
        echo '{"json": "data"}' | nemo inference gateway model patch <trailing_uri> --input-file -
        nemo inference gateway model patch <trailing_uri> --<option> "value"
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
        "inference gateway model patch",
        {
            "name": "(required)",
        },
    )

    all_kwargs = {"trailing_uri": trailing_uri, **input_payload}

    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "gateway", "model"], "patch", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.gateway.model.patch(**all_kwargs)

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
def post_model(
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
    """Proxy requests to model entity inference endpoints.

    All inference requests must resolve to a `VirtualModel`. The platform's provider
    reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
    served model entity (named after the entity, with `default_model_entity` set to
    the entity ref) so this is the typical case; operators can also create custom
    VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
    which no VirtualModel can be found return `404`.

        [bold red]Required fields:[/] name

        [green]Examples:[/]
        nemo inference gateway model post <trailing_uri> <name> --input-file config.json
        nemo inference gateway model post <trailing_uri> <name> --input-data '{"name": "value"}'
        echo '{"json": "data"}' | nemo inference gateway model post <trailing_uri> <name> --input-file -
        nemo inference gateway model post <trailing_uri> <name> --<option> "value"
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
        "inference gateway model post",
        {
            "name": "(required)",
        },
    )

    all_kwargs = {"trailing_uri": trailing_uri, **input_payload}
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "gateway", "model"], "post", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.gateway.model.post(**all_kwargs)

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
def put_model(
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
    """Proxy requests to model entity inference endpoints.

    All inference requests must resolve to a `VirtualModel`. The platform's provider
    reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
    served model entity (named after the entity, with `default_model_entity` set to
    the entity ref) so this is the typical case; operators can also create custom
    VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
    which no VirtualModel can be found return `404`.

        [bold red]Required fields:[/] name

        [green]Examples:[/]
        nemo inference gateway model put <trailing_uri> <name> --input-file config.json
        nemo inference gateway model put <trailing_uri> <name> --input-data '{"name": "value"}'
        echo '{"json": "data"}' | nemo inference gateway model put <trailing_uri> <name> --input-file -
        nemo inference gateway model put <trailing_uri> <name> --<option> "value"
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
        "inference gateway model put",
        {
            "name": "(required)",
        },
    )

    all_kwargs = {"trailing_uri": trailing_uri, **input_payload}
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "gateway", "model"], "put", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.gateway.model.put(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

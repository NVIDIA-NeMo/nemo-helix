# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo inference models``: the workspace's OpenAI-compatible model catalog, via the gateway client."""

from __future__ import annotations

from typing import Annotated

import typer
from nemo_platform_plugin.inference_gateway.client import InferenceGatewayClient

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
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="models", help="Manage models")


@app.command("get")
@collect_warnings
@handle_errors
def get_models(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Retrieve information about a specific OpenAI-compatible model.

    Workspace is
    always taken from the URL path; name may be the VirtualModel name or
    workspace/name (workspace prefix is ignored). Resolves against routable
    VirtualModels, including custom ones, so this route agrees with the list route
    and the inference proxy."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(InferenceGatewayClient, "get_openai_model", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(InferenceGatewayClient).get_openai_model(name=name, workspace=workspace)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("list")
@collect_warnings
@handle_errors
def list_models(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """This endpoint lists the routable VirtualModels in the requested workspace and
    returns them in OpenAI's list models format. Each model ID is the VirtualModel
    identifier in format workspace/name. This includes both autoprovisioned
    VirtualModels (one per served model entity) and custom VirtualModels, keeping
    the catalog in agreement with the inference proxy, which also resolves
    VirtualModels scoped to the request workspace."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("id", None),
        Column("owned_by", None),
        Column("created", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    kwargs = build_kwargs(workspace=workspace)
    if handle_code_generation(InferenceGatewayClient, "list_openai_models", kwargs, resolved_output_format, state):
        return

    items = state.typed_client(InferenceGatewayClient).list_openai_models(workspace=workspace)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )

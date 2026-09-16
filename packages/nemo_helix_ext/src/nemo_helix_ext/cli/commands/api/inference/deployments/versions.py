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
from nemo_helix_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_output,
    validate_stream_output_format,
)
from nemo_helix_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_helix_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="versions", help="Manage versions")


@app.command("delete")
@collect_warnings
@handle_errors
def delete_versions(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    deployment: Annotated[str, typer.Option("--deployment")] = ...,
) -> None:
    """Delete a specific version of a ModelDeployment.

    If the deployment is in any state other than DELETED, this will set its status
    to DELETING. The models controller will then:

    1. Delete the infrastructure (e.g., K8s NimService)
    2. Update the status to DELETED

    If the deployment is already in DELETED status, calling delete again will
    permanently remove it from the database.

    Returns:

    - 202 Accepted: Deployment version marked for deletion (status set to DELETING)
    - 204 No Content: Deployment version permanently removed from database (was
      already DELETED)
    - 404 Not Found: Deployment version doesn't exist"""
    state: CLIContext = ctx.obj
    client = state.get_client()

    kwargs = build_kwargs(
        workspace=workspace,
        deployment=deployment,
    )
    client.inference.deployments.versions.delete(name, **kwargs)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_versions(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """List all versions of a ModelDeployment."""
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
    )

    if handle_code_generation(["inference", "deployments", "versions"], "list", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = (name,)
    items = client.inference.deployments.versions.list(*path_args, **kwargs)

    format_output(
        items,
        is_list=True,
        output_format=output_format,
        output_columns=columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_versions(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    deployment: Annotated[str, typer.Option("--deployment")] = ...,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a specific version of a ModelDeployment."""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
        deployment=deployment,
    )
    if handle_code_generation(["inference", "deployments", "versions"], "retrieve", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.deployments.versions.retrieve(name, **kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

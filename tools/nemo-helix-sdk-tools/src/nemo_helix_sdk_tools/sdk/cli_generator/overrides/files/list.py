# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, cast

import typer
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import Column, check_output_columns_with_format, format_output
from nemo_helix_ext.cli.core.help_formatter import collect_warnings
from nemo_helix_ext.cli.core.types import ListOutputFormatOption, NoTruncateOption, OutputColumnsOption

app = cast(Any, None)  # override-skip: provided by generated file

DEFAULT_COLUMNS = [
    Column(field="path", header="PATH"),
    Column(field="size", header="SIZE"),
]


@app.command("list")
@collect_warnings
@handle_errors
def list_files(
    ctx: typer.Context,
    fileset: Annotated[str, typer.Argument(help="Name of the fileset to list files from")],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    remote_path: Annotated[
        str,
        typer.Option("--remote-path", help="Path within the fileset. Defaults to root."),
    ] = "",
    output_format: ListOutputFormatOption = None,
    columns: OutputColumnsOption = None,
    no_truncate: NoTruncateOption = False,
) -> None:
    """
    List files in a fileset.

    Lists all files recursively from the specified path within the fileset.

    Examples:
        # List all files in a fileset
        nemo files list my-fileset

        # List files in a subdirectory
        nemo files list my-fileset --remote-path data/
    """
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    check_output_columns_with_format(columns, output_format)

    output_columns = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = DEFAULT_COLUMNS

    client = state.get_client()
    if workspace is None:
        workspace = client._get_workspace_path_param()

    response = client.files.list(
        fileset=fileset,
        workspace=workspace,
        remote_path=remote_path,
    )

    format_output(
        response.data,
        is_list=True,
        output_format=output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
    )

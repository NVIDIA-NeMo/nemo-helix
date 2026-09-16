# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any, cast

import typer
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors

app = cast(Any, None)  # override-skip: provided by generated file


@app.command("delete")
@handle_errors
def delete_file(
    ctx: typer.Context,
    fileset: Annotated[str, typer.Argument(help="Name of the fileset containing the file")],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    remote_path: Annotated[
        str,
        typer.Option("--remote-path", help="Path of the file to delete within the fileset"),
    ] = ...,  # ty: ignore[invalid-parameter-default]
) -> None:
    """
    Delete a file from a fileset.

    Examples:
        # Delete a specific file
        nemo files delete my-fileset --remote-path data/old-file.txt
    """
    state: CLIContext = ctx.obj

    client = state.get_client()
    if workspace is None:
        workspace = client._get_workspace_path_param()

    client.files.delete(
        fileset=fileset,
        workspace=workspace,
        remote_path=remote_path,
    )
    typer.echo(f"Deleted {fileset}#{remote_path}")

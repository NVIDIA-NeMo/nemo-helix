# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated, Any, cast

import typer
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient

app = cast(Any, None)  # override-skip: provided by generated file


@app.command("upload")
@handle_errors
def upload_files(
    ctx: typer.Context,
    # Note: local_path is accessed via ctx.params.get("local_path") to preserve trailing slashes
    local_path: Annotated[Path, typer.Argument(help="Local path to upload", dir_okay=True, exists=True)],  # noqa: ARG001
    fileset: Annotated[
        str | None,
        typer.Argument(help="Name of the fileset to upload to. If not provided, a new fileset is created."),
    ] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    remote_path: Annotated[
        str,
        typer.Option("--remote-path", help="Path within the fileset. Defaults to root."),
    ] = "",
) -> None:
    """
    Upload local files to a fileset.

    Supports uploading single files or directories. For directories, contents
    are uploaded recursively.

    Examples:
        # Upload a file to the root of a fileset
        nemo files upload ./data.csv my-fileset

        # Upload a directory to a subdirectory in the fileset
        nemo files upload ./data/ my-fileset --remote-path uploads/

        # Upload without specifying a fileset (auto-creates one)
        nemo files upload ./data.csv
    """
    state: CLIContext = ctx.obj

    # Use raw path that user provides, as trailing slashes matter with fsspec
    raw_local_path: str = ctx.params.get("local_path")

    client = state.get_client()
    files = client_from_platform(client, FilesClient)
    if workspace is None:
        workspace = client._get_workspace_path_param()

    from filesets import RichProgressCallback

    with RichProgressCallback(description="Uploading") as callback:
        if fileset is not None:
            # Validate fileset exists before uploading
            files.get_fileset(name=fileset, workspace=workspace)
            client.files.upload(
                local_path=raw_local_path,
                remote_path=remote_path,
                fileset=fileset,
                workspace=workspace,
                callback=callback,
            )
        else:
            # Auto-create a new fileset
            result = client.files.upload(
                local_path=raw_local_path,
                remote_path=remote_path,
                workspace=workspace,
                callback=callback,
                fileset_auto_create=True,
            )
            fileset = result.name

    if remote_path:
        typer.echo(f"Completed upload to {fileset}#{remote_path}")
    else:
        typer.echo(f"Completed upload to {fileset}")

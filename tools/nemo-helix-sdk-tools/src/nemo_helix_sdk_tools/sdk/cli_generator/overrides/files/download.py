# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Annotated, Any, cast

import typer
from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors

app = cast(Any, None)  # override-skip: provided by generated file


@app.command("download")
@handle_errors
def download_files(
    ctx: typer.Context,
    fileset: Annotated[str, typer.Argument(help="Name of the fileset to download from")],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    remote_path: Annotated[
        str,
        typer.Option("--remote-path", help="Path within the fileset. Defaults to root."),
    ] = "",
    output: Annotated[  # noqa: ARG001
        Path,
        typer.Option("--output", "-o", help="Local path to download to."),
    ] = ...,  # ty: ignore[invalid-parameter-default]
) -> None:
    """
    Download files from a fileset to a local path.

    Supports downloading single files or directories. For directories, contents
    are downloaded recursively.

    Examples:
        # Download entire fileset to current directory
        nemo files download my-fileset -o ./

        # Download a subdirectory from the fileset
        nemo files download my-fileset --remote-path data/ -o ./downloads/
    """
    state: CLIContext = ctx.obj

    # Use raw path that user provides, as trailing slashes matter with fsspec
    raw_output_path: str = ctx.params.get("output")

    client = state.get_client()
    if workspace is None:
        workspace = client._get_workspace_path_param()

    from filesets import RichProgressCallback

    with RichProgressCallback(description="Downloading") as callback:
        client.files.download(
            remote_path=remote_path,
            local_path=raw_output_path,
            fileset=fileset,
            workspace=workspace,
            callback=callback,
        )
    typer.echo(f"Downloaded {fileset}#{remote_path or '/'} to {raw_output_path!r}")

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from importlib import import_module as _importlib_import_module
from pathlib import Path
from typing import Annotated

import typer
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.files.client import FilesClient

from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.formatters import Column, check_output_columns_with_format, format_output
from nemo_helix_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_helix_ext.cli.core.types import ListOutputFormatOption, NoTruncateOption, OutputColumnsOption

_cli_child_filesets = _importlib_import_module("nemo_helix_ext.cli.commands.api.files.filesets")
_cli_child_otlp = _importlib_import_module("nemo_helix_ext.cli.commands.api.files.otlp")

app = create_typer_app(name="files", help="Manage files")

app.add_typer(_cli_child_filesets.app, name="filesets")
app.add_typer(_cli_child_otlp.app, name="otlp")


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

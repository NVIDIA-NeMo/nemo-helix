# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo files`` command group, backed by the typed Files client.

File transfers (``upload``/``download``/``list``/``delete``) go through
:mod:`filesets.transfer`, which drives the fileset filesystem on the same
typed client; fileset CRUD and OTLP logs call the client endpoints directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import click
import typer
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.endpoints import upload_otlp_logs
from nemo_platform_plugin.files.types import (
    CreateFilesetRequest,
    ListFilesetsQueryParams,
    OtlpLogQueryRequest,
    UpdateFilesetRequest,
    UploadOtlpLogsQueryParams,
)

from nemo_platform_ext.cli.core.api import build_kwargs, merge_filter_dict
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
from nemo_platform_ext.cli.core.pagination import PaginationType, collect_offset_pages, warn_if_more_pages
from nemo_platform_ext.cli.core.stdin_utils import (
    build_request_body,
    read_data_input_with_flags,
    read_payload,
    validate_required_fields,
)
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="files", help="Manage files.")
filesets_app = create_typer_app(name="filesets", help="Manage filesets")
otlp_app = create_typer_app(name="otlp", help="OTLP operations")
logs_app = create_typer_app(name="logs", help="Manage logs")

app.add_typer(filesets_app, name="filesets")
app.add_typer(otlp_app, name="otlp")
otlp_app.add_typer(logs_app, name="logs")

FILESET_NAME_HELP = (
    "The name of the fileset. Name must start with a lowercase letter, be 2-63 characters, and use lowercase "
    "letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)"
)
FILESET_METADATA_HELP = (
    "Tagged metadata container - the key indicates the type.Example: metadata = FilesetMetadata( "
    'dataset=DatasetMetadataContent( schema={"columns": ["id", "name"]}, ) ) (JSON string)'
)

DEFAULT_COLUMNS = [
    Column(field="path", header="PATH"),
    Column(field="size", header="SIZE"),
]


def _read_input_payload(input_file: str | None, input_data: str | None) -> dict[str, Any]:
    """Return the ``--input-file``/``--input-data`` payload, or an empty dict when neither was given."""
    if input_file or input_data:
        return read_data_input_with_flags(input_file=input_file, input_data=input_data)
    return {}


def _filter_query_value(value: str | dict[str, Any] | None) -> str | None:
    """Serialize a merged filter for the ``filter`` query parameter (JSON object or text expression)."""
    if isinstance(value, dict):
        return json.dumps(value)
    return value


def _list_filesets_query_params(
    *,
    filter_value: str | None,
    page: int | None,
    page_size: int | None,
    sort: str | None,
) -> ListFilesetsQueryParams | None:
    query_params: ListFilesetsQueryParams = {}
    if filter_value is not None:
        query_params["filter"] = filter_value
    if page is not None:
        query_params["page"] = page
    if page_size is not None:
        query_params["page_size"] = page_size
    if sort is not None:
        query_params["sort"] = sort
    return query_params or None


# ---------------------------------------------------------------------------
# File transfers
# ---------------------------------------------------------------------------


@app.command("upload")
@handle_errors
def upload_files(
    ctx: typer.Context,
    # Note: local_path is accessed via ctx.params["local_path"] to preserve trailing slashes
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
    raw_local_path = str(ctx.params["local_path"])

    files = state.typed_client(FilesClient)
    workspace = files.require_workspace(workspace)

    from filesets import RichProgressCallback, transfer

    with RichProgressCallback(description="Uploading") as callback:
        if fileset is not None:
            # Validate fileset exists before uploading
            files.get_fileset(name=fileset, workspace=workspace)
            transfer.upload(
                files,
                local_path=raw_local_path,
                remote_path=remote_path,
                fileset=fileset,
                workspace=workspace,
                callback=callback,
            )
        else:
            # Auto-create a new fileset
            result = transfer.upload(
                files,
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
    raw_output_path = str(ctx.params["output"])

    files = state.typed_client(FilesClient)
    workspace = files.require_workspace(workspace)

    from filesets import RichProgressCallback, transfer

    with RichProgressCallback(description="Downloading") as callback:
        transfer.download(
            files,
            remote_path=remote_path,
            local_path=raw_output_path,
            fileset=fileset,
            workspace=workspace,
            callback=callback,
        )
    typer.echo(f"Downloaded {fileset}#{remote_path or '/'} to {raw_output_path!r}")


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
    resolved_output_format = state.get_output_format(output_format)

    check_output_columns_with_format(columns, resolved_output_format)

    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = DEFAULT_COLUMNS

    files = state.typed_client(FilesClient)
    workspace = files.require_workspace(workspace)

    from filesets import transfer
    from fsspec.core import has_magic

    if resolved_output_format == "code" and has_magic(remote_path):
        raise click.UsageError(
            "--output-format code cannot express a glob --remote-path: the pattern is matched client-side "
            "after listing. Use a path prefix, or run the command without --output-format code."
        )
    query_params = transfer.list_query_params(remote_path)
    kwargs = build_kwargs(name=fileset, workspace=workspace, query_params=query_params)
    if handle_code_generation(FilesClient, "list_files", kwargs, resolved_output_format, state):
        return

    response = transfer.list_files(files, fileset=fileset, workspace=workspace, remote_path=remote_path)

    format_output(
        response.data,
        is_list=True,
        output_format=resolved_output_format,
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

    files = state.typed_client(FilesClient)
    workspace = files.require_workspace(workspace)

    from filesets import transfer

    transfer.delete(files, fileset=fileset, workspace=workspace, remote_path=remote_path)
    typer.echo(f"Deleted {fileset}#{remote_path}")


# ---------------------------------------------------------------------------
# Filesets
# ---------------------------------------------------------------------------


@filesets_app.command("create")
@collect_warnings
@handle_errors
def create_filesets(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help=FILESET_NAME_HELP)] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    cache: Annotated[
        bool | None, typer.Option("--cache", help="Cache all files after creation. Only applies to external storage.")
    ] = None,
    custom_fields: Annotated[
        str | None, typer.Option("--custom-fields", help="Custom fields for the fileset. (JSON string)")
    ] = None,
    description: Annotated[str | None, typer.Option("--description", help="The description of the fileset.")] = None,
    metadata: Annotated[str | None, typer.Option("--metadata", help=FILESET_METADATA_HELP)] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="The name of the project associated with this fileset.")
    ] = None,
    purpose: Annotated[
        Literal["dataset", "environment", "generic", "model"] | None,
        typer.Option("--purpose", help="The purpose of the fileset."),
    ] = None,
    storage: Annotated[
        str | None,
        typer.Option(
            "--storage",
            help="The storage configuration for the fileset. If not provided, uses default storage. (JSON string)",
        ),
    ] = None,
    exist_ok: Annotated[
        bool | None,
        typer.Option(
            "--exist-ok", help="Do not raise an error if the resource already exists. Returns the existing resource."
        ),
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
    """Create a new fileset.

    If no storage configuration is provided, the default storage backend will be
    used.

        [bold red]Required fields:[/] name

        [green]Examples:[/]
        nemo files filesets create <name> --input-file config.json
        nemo files filesets create <name> --input-data '{"name": "value"}'
        echo '{"json": "data"}' | nemo files filesets create <name> --input-file -
        nemo files filesets create <name> --<option> "value"
    """
    input_payload = _read_input_payload(input_file, input_data)

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if name is not None:
        input_payload["name"] = name
    if cache is not None:
        input_payload["cache"] = cache
    if custom_fields is not None:
        input_payload["custom_fields"] = read_payload("custom_fields", custom_fields)
    if description is not None:
        input_payload["description"] = description
    if metadata is not None:
        input_payload["metadata"] = read_payload("metadata", metadata)
    if project is not None:
        input_payload["project"] = project
    if purpose is not None:
        input_payload["purpose"] = purpose
    if storage is not None:
        input_payload["storage"] = read_payload("storage", storage)
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    # Validate required fields are present after merging
    validate_required_fields(input_payload, ["name"], "files filesets create", {"name": FILESET_NAME_HELP})

    workspace = input_payload.pop("workspace", None)
    exist_ok = input_payload.pop("exist_ok", None)
    body = build_request_body(CreateFilesetRequest, input_payload, command_name="files filesets create")

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(workspace=workspace, body=body, exist_ok=exist_ok)
    if handle_code_generation(FilesClient, "create_fileset", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(FilesClient).create_fileset(workspace=workspace, body=body, exist_ok=bool(exist_ok))

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@filesets_app.command("delete")
@collect_warnings
@handle_errors
def delete_filesets(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete Fileset.

    Permanently deletes an unreferenced fileset from the platform.

    Referencing model
    or adapter entities must be relinked or deleted first. For local storage
    backends, this also deletes the underlying files."""
    state: CLIContext = ctx.obj
    state.typed_client(FilesClient).delete_fileset(name=name, workspace=workspace)

    typer.echo("✓ Deleted successfully")


@filesets_app.command("list")
@collect_warnings
@handle_errors
def list_filesets(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  updated_at: {gte: str, lte: str}\n\nFilter filesets by name, description, purpose, storage_type, created_at, and updated_at.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_description: Annotated[
        str | None, typer.Option("--filter.description", rich_help_panel="Filter Options")
    ] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_purpose: Annotated[str | None, typer.Option("--filter.purpose", rich_help_panel="Filter Options")] = None,
    filter_storage_type: Annotated[
        str | None, typer.Option("--filter.storage-type", rich_help_panel="Filter Options")
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[
        Literal["-created_at", "created_at", "-updated_at", "updated_at", "-name", "name"] | None,
        typer.Option(
            "--sort", help="The field to sort by. To sort in decreasing order, use `-` in front of the field name."
        ),
    ] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List Filesets endpoint with filtering and pagination.

    Supports filtering by name, description, purpose, storage_type, created_at, and
    updated_at via query parameters. Returns paginated results with sorting options."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("name", None),
        Column("workspace", None),
        Column("created_at", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    filter_value = _filter_query_value(
        merge_filter_dict(
            filter,
            description=filter_description,
            name=filter_name,
            purpose=filter_purpose,
            storage_type=filter_storage_type,
        )
    )
    query_params = _list_filesets_query_params(filter_value=filter_value, page=page, page_size=page_size, sort=sort)
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)
    if handle_code_generation(FilesClient, "list_filesets", kwargs, resolved_output_format, state, result="list"):
        return

    response = state.typed_client(FilesClient).list_filesets(workspace=workspace, query_params=query_params)
    pagination_type = PaginationType.PAGE_NUMBER
    items = collect_offset_pages(response, all_pages=all_pages)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(items, pagination_type)


@filesets_app.command("refresh")
@collect_warnings
@handle_errors
def refresh_filesets(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Re-resolve a fileset's tracked revision against its source.

    A fileset created from a mutable ref is pinned to an immutable id so its
    contents cannot shift under a deployment. This re-resolves that same ref and
    repoints the fileset at whatever it names now. Everything else about the storage
    config, including the repository and directory, is left alone.

    A running deployment keeps serving the revision it staged. It moves to this one
    when the runner next stages the fileset, and records the revision it staged."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(FilesClient, "refresh_fileset", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(FilesClient).refresh_fileset(name=name, workspace=workspace)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@filesets_app.command("get")
@collect_warnings
@handle_errors
def retrieve_filesets(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get Fileset by Workspace and Name.

    Returns the details of a specific fileset identified by its workspace and name."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(FilesClient, "get_fileset", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(FilesClient).get_fileset(name=name, workspace=workspace)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@filesets_app.command("update")
@collect_warnings
@handle_errors
def update_filesets(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    custom_fields: Annotated[
        str | None, typer.Option("--custom-fields", help="Custom fields for the fileset. (JSON string)")
    ] = None,
    description: Annotated[str | None, typer.Option("--description", help="The description of the fileset.")] = None,
    metadata: Annotated[str | None, typer.Option("--metadata", help=FILESET_METADATA_HELP)] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="The name of the project associated with this fileset.")
    ] = None,
    purpose: Annotated[
        Literal["dataset", "environment", "generic", "model"] | None,
        typer.Option("--purpose", help="The purpose of the fileset."),
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
    """Update Fileset Metadata.

    [green]Examples:[/]
    nemo files filesets update <name> --input-file config.json
    nemo files filesets update <name> --input-data '{"field": "value"}'
    echo '{"json": "data"}' | nemo files filesets update <name> --input-file -
    nemo files filesets update <name> --<option> "value"
    """
    input_payload = _read_input_payload(input_file, input_data)

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if custom_fields is not None:
        input_payload["custom_fields"] = read_payload("custom_fields", custom_fields)
    if description is not None:
        input_payload["description"] = description
    if metadata is not None:
        input_payload["metadata"] = read_payload("metadata", metadata)
    if project is not None:
        input_payload["project"] = project
    if purpose is not None:
        input_payload["purpose"] = purpose

    workspace = input_payload.pop("workspace", None)
    body = build_request_body(UpdateFilesetRequest, input_payload, command_name="files filesets update")

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace, body=body)
    if handle_code_generation(FilesClient, "update_fileset", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(FilesClient).update_fileset(name=name, workspace=workspace, body=body)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


# ---------------------------------------------------------------------------
# OTLP logs
# ---------------------------------------------------------------------------


@logs_app.command("create")
@collect_warnings
@handle_errors
def create_logs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    artifact_base_path: Annotated[
        str | None, typer.Option("--artifact-base-path", help="Folder inside the fileset to nest logs under")
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
    """Upload OTLP logs to a specified fileset in JSON or Protobuf format.

    Supports both application/json and application/x-protobuf content types.

        [green]Examples:[/]
        nemo files otlp logs create <name> --input-file config.json
        nemo files otlp logs create <name> --input-data '{"field": "value"}'
        echo '{"json": "data"}' | nemo files otlp logs create <name> --input-file -
        nemo files otlp logs create <name> --<option> "value"
    """
    input_payload = _read_input_payload(input_file, input_data)

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if artifact_base_path is not None:
        input_payload["artifact_base_path"] = artifact_base_path

    # Routing fields travel as path/query parameters; the rest is the OTLP export payload.
    workspace = input_payload.pop("workspace", None)
    artifact_base_path = input_payload.pop("artifact_base_path", None)
    query_params: UploadOtlpLogsQueryParams | None = None
    if artifact_base_path is not None:
        query_params = {"artifact_base_path": artifact_base_path}
    content = json.dumps(input_payload).encode()

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace, content=content, query_params=query_params)
    if handle_code_generation(FilesClient, "upload_otlp_logs", kwargs, resolved_output_format, state):
        return

    # The payload is the OTLP JSON export, so label it as such rather than as opaque bytes.
    result = state.typed_client(FilesClient).send(
        upload_otlp_logs(name=name, workspace=workspace, content=content, query_params=query_params),
        headers={"Content-Type": "application/json"},
    )

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@logs_app.command("query")
@collect_warnings
@handle_errors
def query_logs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    artifact_base_path: Annotated[
        str | None,
        typer.Option(
            "--artifact-base-path",
            help="Folder inside the fileset the logs were nested under (must match the value used on write)",
        ),
    ] = None,
    filters: Annotated[str | None, typer.Option("--filters", help="Key-value filters to apply to the query")] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Maximum number of results to return")] = None,
    page_cursor: Annotated[str | None, typer.Option("--page-cursor", help="Cursor for pagination")] = None,
    tail: Annotated[int | None, typer.Option("--tail", help="Number of newest log lines to return")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Query logs from parquet files in a fileset.

    This is an internal endpoint that runs DuckDB queries with direct storage
    access."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    body = OtlpLogQueryRequest.model_validate(
        build_kwargs(
            artifact_base_path=artifact_base_path,
            filters=read_payload("filters", filters) if filters is not None else None,
            limit=limit,
            page_cursor=page_cursor,
            tail=tail,
        )
    )
    kwargs = build_kwargs(name=name, workspace=workspace, body=body)
    if handle_code_generation(FilesClient, "query_otlp_logs", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(FilesClient).query_otlp_logs(name=name, workspace=workspace, body=body)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

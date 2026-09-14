# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from typing import Annotated, Literal

import typer

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
from nemo_platform_ext.cli.core.pagination import PaginationType, fetch_all_pages, warn_if_more_pages
from nemo_platform_ext.cli.core.stdin_utils import read_data_input_with_flags, read_payload, validate_required_fields
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="filesets", help="Manage filesets")


@app.command("create")
@collect_warnings
@handle_errors
def create_filesets(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Argument(
            help="The name of the fileset. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)"
        ),
    ] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    cache: Annotated[
        bool | None, typer.Option("--cache", help="Cache all files after creation. Only applies to external storage.")
    ] = None,
    custom_fields: Annotated[
        str | None, typer.Option("--custom-fields", help="Custom fields for the fileset. (JSON string)")
    ] = None,
    description: Annotated[str | None, typer.Option("--description", help="The description of the fileset.")] = None,
    metadata: Annotated[
        str | None,
        typer.Option(
            "--metadata",
            help='Tagged metadata container - the key indicates the type.Example: metadata = FilesetMetadata( dataset=DatasetMetadataContent( schema={"columns": ["id", "name"]}, ) ) (JSON string)',
        ),
    ] = None,
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
    validate_required_fields(
        input_payload,
        ["name"],
        "files filesets create",
        {
            "name": "The name of the fileset. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["files", "filesets"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.files.filesets.create(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("delete")
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
    client = state.get_client()

    kwargs = build_kwargs(
        workspace=workspace,
    )
    client.files.filesets.delete(name, **kwargs)

    typer.echo("✓ Deleted successfully")


@app.command("list")
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
        filter=merge_filter_dict(
            filter,
            description=filter_description,
            name=filter_name,
            purpose=filter_purpose,
            storage_type=filter_storage_type,
        ),
        page=page,
        page_size=page_size,
        sort=sort,
    )

    if handle_code_generation(["files", "filesets"], "list", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = ()
    pagination_type = PaginationType.PAGE_NUMBER
    if all_pages:
        items = fetch_all_pages(
            client.files.filesets.list,
            path_args=path_args,
            body_args=kwargs,
            pagination_type=pagination_type,
        )
    else:
        items = client.files.filesets.list(*path_args, **kwargs)

    format_output(
        items,
        is_list=True,
        output_format=output_format,
        output_columns=columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(items, pagination_type)


@app.command("refresh")
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

    Deployments stage the fileset when they are created, so existing deployments
    keep serving the revision they were staged from."""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
    )
    if handle_code_generation(["files", "filesets"], "refresh", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.files.filesets.refresh(name, **kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("get")
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
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
    )
    if handle_code_generation(["files", "filesets"], "retrieve", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.files.filesets.retrieve(name, **kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("update")
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
    metadata: Annotated[
        str | None,
        typer.Option(
            "--metadata",
            help='Tagged metadata container - the key indicates the type.Example: metadata = FilesetMetadata( dataset=DatasetMetadataContent( schema={"columns": ["id", "name"]}, ) ) (JSON string)',
        ),
    ] = None,
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
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

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

    all_kwargs = {"name": name, **input_payload}

    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["files", "filesets"], "update", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.files.filesets.update(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

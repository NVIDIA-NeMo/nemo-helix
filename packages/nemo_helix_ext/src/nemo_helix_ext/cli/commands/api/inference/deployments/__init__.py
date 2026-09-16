# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from importlib import import_module as _importlib_import_module
from typing import Annotated, Literal

import typer

from nemo_helix_ext.cli.core.api import build_kwargs, merge_filter_dict
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
from nemo_helix_ext.cli.core.pagination import PaginationType, fetch_all_pages, warn_if_more_pages
from nemo_helix_ext.cli.core.stdin_utils import read_data_input_with_flags, validate_required_fields
from nemo_helix_ext.cli.core.stdin_utils import read_payload as read_payload
from nemo_helix_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)
from nemo_helix_ext.cli.core.waiters import wait_for_inference_deployment

_cli_child_versions = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.deployments.versions")

app = create_typer_app(name="deployments", help="Manage deployments")

app.add_typer(_cli_child_versions.app, name="versions")


@app.command("create")
@collect_warnings
@handle_errors
def create_deployments(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Argument(
            help="Name of the deployment. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)"
        ),
    ] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    config: Annotated[
        str | None, typer.Option("--config", help="Reference to the ModelDeploymentConfig name (required)")
    ] = None,
    config_version: Annotated[
        int | None,
        typer.Option(
            "--config-version",
            help="Reference to a specific ModelDeploymentConfig version. If not specified, uses latest.",
        ),
    ] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="The URN of the project associated with this deployment")
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
    wait: Annotated[
        bool,
        typer.Option(
            "--wait", help="Wait for the created deployment to be up and running", rich_help_panel="Lifecycle Options"
        ),
    ] = False,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            help="Watch the created deployment until it is stable while streaming status updates",
            rich_help_panel="Lifecycle Options",
        ),
    ] = False,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout", min=1, help="Maximum time to wait or watch in seconds", rich_help_panel="Lifecycle Options"
        ),
    ] = 1200,
    poll_interval: Annotated[
        int,
        typer.Option(
            "--poll-interval", min=1, help="Seconds between status checks", rich_help_panel="Lifecycle Options"
        ),
    ] = 3,
) -> None:
    """Create a new ModelDeployment (version 1).

    [bold red]Required fields:[/] config, name

    [green]Examples:[/]
    nemo inference deployments create <name> --input-file config.json
    nemo inference deployments create <name> --input-data '{"config": "value", "name": "value"}'
    echo '{"json": "data"}' | nemo inference deployments create <name> --input-file -
    nemo inference deployments create <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if config is not None:
        input_payload["config"] = config
    if name is not None:
        input_payload["name"] = name
    if config_version is not None:
        input_payload["config_version"] = config_version
    if project is not None:
        input_payload["project"] = project
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["config", "name"],
        "inference deployments create",
        {
            "config": "Reference to the ModelDeploymentConfig name (required)",
            "name": "Name of the deployment. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if wait and watch:
        raise typer.BadParameter("Cannot combine --wait and --watch.")

    if handle_code_generation(
        ["inference", "deployments"],
        "create",
        all_kwargs,
        output_format,
        state,
        watch_config={"type": "inference_deployment", "resource_label": "deployment"} if watch else None,
        watch_options={"timeout": timeout, "poll_interval": poll_interval} if watch else None,
        wait_config={"type": "inference_deployment", "resource_label": "deployment"} if wait else None,
        wait_options={"timeout": timeout, "poll_interval": poll_interval} if wait else None,
    ):
        return

    client = state.get_client()
    result = client.inference.deployments.create(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )
    if wait or watch:
        wait_name = getattr(result, "name", None) or all_kwargs.get("name")
        if not wait_name:
            raise RuntimeError("Unable to determine created resource name for --wait/--watch")
        wait_workspace = all_kwargs.get("workspace")
        if not wait_for_inference_deployment(
            client,
            wait_name,
            workspace=wait_workspace,
            timeout=timeout,
            poll_interval=poll_interval,
        ):
            raise typer.Exit(1)
        return


@app.command("delete")
@collect_warnings
@handle_errors
def delete_deployments(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete all versions of a ModelDeployment.

    If the deployment is in any state other than DELETED, this will set its status
    to DELETING. The models controller will then:

    1. Delete the infrastructure (e.g., K8s NimService)
    2. Update the status to DELETED

    If the deployment is already in DELETED status, calling delete again will
    permanently remove it from the database.

    Returns:

    - 202 Accepted: Deployment marked for deletion (status set to DELETING)
    - 204 No Content: Deployment permanently removed from database (was already
      DELETED)
    - 404 Not Found: Deployment doesn't exist"""
    state: CLIContext = ctx.obj
    client = state.get_client()

    kwargs = build_kwargs(
        workspace=workspace,
    )
    client.inference.deployments.delete(name, **kwargs)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_deployments(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    all_versions: Annotated[
        bool | None,
        typer.Option(
            "--all-versions",
            help="If true, return all versions of each deployment. If false (default), return only the latest version.",
        ),
    ] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  updated_at: {gte: str, lte: str}\n\nFilter deployments by workspace, project, status, config, model_provider_id, name, status_message, created_at, and updated_at.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_config: Annotated[str | None, typer.Option("--filter.config", rich_help_panel="Filter Options")] = None,
    filter_model_provider_id: Annotated[
        str | None, typer.Option("--filter.model-provider-id", rich_help_panel="Filter Options")
    ] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_project: Annotated[str | None, typer.Option("--filter.project", rich_help_panel="Filter Options")] = None,
    filter_status: Annotated[str | None, typer.Option("--filter.status", rich_help_panel="Filter Options")] = None,
    filter_status_message: Annotated[
        str | None, typer.Option("--filter.status-message", rich_help_panel="Filter Options")
    ] = None,
    filter_workspace: Annotated[
        str | None, typer.Option("--filter.workspace", rich_help_panel="Filter Options")
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[
        str | None,
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
    """List ModelDeployments for a specific workspace.

    By default, returns only the latest version of each deployment."""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)
    validate_stream_output_format(output_format, stream)

    check_output_columns_with_format(columns, output_format)

    default_columns = [
        Column("name", None),
        Column("status", None),
        Column("created_at", None),
    ]
    if columns is None or str(columns).strip() == "default":
        columns = default_columns

    kwargs = build_kwargs(
        workspace=workspace,
        all_versions=all_versions,
        filter=merge_filter_dict(
            filter,
            config=filter_config,
            model_provider_id=filter_model_provider_id,
            name=filter_name,
            project=filter_project,
            status=filter_status,
            status_message=filter_status_message,
            workspace=filter_workspace,
        ),
        page=page,
        page_size=page_size,
        sort=sort,
    )

    if handle_code_generation(["inference", "deployments"], "list", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = ()
    pagination_type = PaginationType.PAGE_NUMBER
    if all_pages:
        items = fetch_all_pages(
            client.inference.deployments.list,
            path_args=path_args,
            body_args=kwargs,
            pagination_type=pagination_type,
        )
    else:
        items = client.inference.deployments.list(*path_args, **kwargs)

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


@app.command("list-models")
@collect_warnings
@handle_errors
def list_models_deployments(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """Get Latest ModelDeployment's Model Entities from Entity Store.

    This provides the
    API contract that NIMs expect from Entity Store today, for pulling LoRAs, but
    enables us to enforce AuthZ boundaries.

    TODO: Implement model entity retrieval based on deployment config."""
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

    if handle_code_generation(["inference", "deployments"], "list_models", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = (name,)
    items = client.inference.deployments.list_models(*path_args, **kwargs)

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
def retrieve_deployments(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get the latest version of a ModelDeployment."""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
    )
    if handle_code_generation(["inference", "deployments"], "retrieve", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.deployments.retrieve(name, **kwargs)

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
def update_deployments(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    config: Annotated[
        str | None, typer.Option("--config", help="Reference to the ModelDeploymentConfig name (required)")
    ] = None,
    config_version: Annotated[
        int | None,
        typer.Option(
            "--config-version",
            help="Reference to a specific ModelDeploymentConfig version. If not specified, uses latest.",
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
    """Update a ModelDeployment (creates a new immutable version).

    [bold red]Required fields:[/] config

    [green]Examples:[/]
    nemo inference deployments update <name> --input-file config.json
    nemo inference deployments update <name> --input-data '{"config": "value"}'
    echo '{"json": "data"}' | nemo inference deployments update <name> --input-file -
    nemo inference deployments update <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if config is not None:
        input_payload["config"] = config
    if config_version is not None:
        input_payload["config_version"] = config_version
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["config"],
        "inference deployments update",
        {
            "config": "Reference to the ModelDeploymentConfig name (required)",
        },
    )

    all_kwargs = {"name": name, **input_payload}

    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "deployments"], "update", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.deployments.update(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("update-status")
@collect_warnings
@handle_errors
def update_status_deployments(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    status: Annotated[
        Literal["UNKNOWN", "CREATED", "PENDING", "READY", "ERROR", "DELETING", "DELETED", "LOST"] | None,
        typer.Option("--status", help="Status enum for ModelDeployment objects. (required)"),
    ] = None,
    version: Annotated[str | None, typer.Option("--version")] = None,
    model_provider_id: Annotated[
        str | None,
        typer.Option(
            "--model-provider-id",
            help="Optional reference to the auto-created ModelProvider workspace/name (format: workspace/name)",
        ),
    ] = None,
    status_message: Annotated[str | None, typer.Option("--status-message", help="Detailed status message")] = None,
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
    """Update the status of a ModelDeployment (mutable operation).

    If version is not
    specified, updates the latest version.

        [bold red]Required fields:[/] status

        [green]Examples:[/]
        nemo inference deployments update-status <name> --input-file config.json
        nemo inference deployments update-status <name> --input-data '{"status": "value"}'
        echo '{"json": "data"}' | nemo inference deployments update-status <name> --input-file -
        nemo inference deployments update-status <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if status is not None:
        input_payload["status"] = status
    if version is not None:
        input_payload["version"] = version
    if model_provider_id is not None:
        input_payload["model_provider_id"] = model_provider_id
    if status_message is not None:
        input_payload["status_message"] = status_message
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["status"],
        "inference deployments update-status",
        {
            "status": "Status enum for ModelDeployment objects. (required)",
        },
    )

    all_kwargs = {"name": name, **input_payload}

    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "deployments"], "update_status", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.deployments.update_status(**all_kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

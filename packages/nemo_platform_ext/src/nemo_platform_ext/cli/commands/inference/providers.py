# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo inference providers`` command group, backed by the typed Models client."""

from __future__ import annotations

from typing import Annotated, Literal, cast

import typer
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import (
    CreateModelProviderRequest,
    ListProvidersQueryParams,
    UpdateModelProviderStatusRequest,
    UpsertModelProviderRequest,
)

from nemo_platform_ext.cli.commands.inference._common import (
    filter_query_value,
    offset_query_params,
    pop_exist_ok,
    pop_workspace,
    read_input_payload,
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
from nemo_platform_ext.cli.core.stdin_utils import build_request_body, read_payload, validate_required_fields
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="providers", help="Manage providers")

ProviderStatusValue = Literal["UNKNOWN", "CREATED", "PENDING", "READY", "ERROR", "DELETING", "DELETED", "LOST"]


@app.command("create")
@collect_warnings
@handle_errors
def create_providers(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Argument(
            help="Name of the model provider. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)"
        ),
    ] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    host_url: Annotated[
        str | None, typer.Option("--host-url", help="The network endpoint URL for the model provider (required)")
    ] = None,
    api_key_secret_name: Annotated[
        str | None,
        typer.Option(
            "--api-key-secret-name",
            help="Reference to an API key secret stored in the Secrets service. Create the secret first via secrets API, then pass the secret name here.",
        ),
    ] = None,
    auth_header_format: Annotated[
        str | None,
        typer.Option(
            "--auth-header-format",
            help="Jinja2 template string controlling how the API key secret is sent to the upstream. Must contain exactly one variable named `auth_secret`, which is substituted with the resolved secret value at request time. Example: `'X-Api-Key: {{ auth_secret }}'`. If not set, defaults to `'Authorization: Bearer {{ auth_secret }}'`.",
        ),
    ] = None,
    default_extra_body: Annotated[
        str | None,
        typer.Option(
            "--default-extra-body",
            help="Default body parameters for inference requests. Can be overridden by user requests. (JSON string)",
        ),
    ] = None,
    default_extra_headers: Annotated[
        str | None,
        typer.Option(
            "--default-extra-headers",
            help="Default headers for inference requests. Can be overridden by user requests. (JSON string)",
        ),
    ] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Optional description of the model provider")
    ] = None,
    enabled_models: Annotated[
        list[str] | None,
        typer.Option(
            "--enabled-models", help="Optional list of specific models to enable from this provider (can be repeated)"
        ),
    ] = None,
    model_deployment_id: Annotated[
        str | None,
        typer.Option(
            "--model-deployment-id",
            help="Optional reference to the ModelDeployment ID if this provider is being auto-created for a deployment",
        ),
    ] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="The URN of the project associated with this model provider")
    ] = None,
    required_extra_body: Annotated[
        str | None,
        typer.Option(
            "--required-extra-body",
            help="Required body parameters for inference requests. Cannot be overridden by user requests. (JSON string)",
        ),
    ] = None,
    required_extra_headers: Annotated[
        str | None,
        typer.Option(
            "--required-extra-headers",
            help="Required headers for inference requests. Cannot be overridden by user requests. (JSON string)",
        ),
    ] = None,
    status: Annotated[
        ProviderStatusValue | None,
        typer.Option("--status", help="Status enum for ModelProvider objects."),
    ] = None,
    status_message: Annotated[str | None, typer.Option("--status-message", help="Status message")] = None,
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
    """Create a new model provider.

    [bold red]Required fields:[/] host_url, name

    [green]Examples:[/]
    nemo inference providers create <name> --input-file config.json
    nemo inference providers create <name> --input-data '{"host_url": "value", "name": "value"}'
    echo '{"json": "data"}' | nemo inference providers create <name> --input-file -
    nemo inference providers create <name> --<option> "value"
    """
    input_payload = read_input_payload(input_file, input_data)

    if workspace is not None:
        input_payload["workspace"] = workspace
    if host_url is not None:
        input_payload["host_url"] = host_url
    if name is not None:
        input_payload["name"] = name
    if api_key_secret_name is not None:
        input_payload["api_key_secret_name"] = api_key_secret_name
    if auth_header_format is not None:
        input_payload["auth_header_format"] = auth_header_format
    if default_extra_body is not None:
        input_payload["default_extra_body"] = read_payload("default_extra_body", default_extra_body)
    if default_extra_headers is not None:
        input_payload["default_extra_headers"] = read_payload("default_extra_headers", default_extra_headers)
    if description is not None:
        input_payload["description"] = description
    if enabled_models:  # Check for non-empty list
        input_payload["enabled_models"] = enabled_models
    if model_deployment_id is not None:
        input_payload["model_deployment_id"] = model_deployment_id
    if project is not None:
        input_payload["project"] = project
    if required_extra_body is not None:
        input_payload["required_extra_body"] = read_payload("required_extra_body", required_extra_body)
    if required_extra_headers is not None:
        input_payload["required_extra_headers"] = read_payload("required_extra_headers", required_extra_headers)
    if status is not None:
        input_payload["status"] = status
    if status_message is not None:
        input_payload["status_message"] = status_message
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    validate_required_fields(
        input_payload,
        ["host_url", "name"],
        "inference providers create",
        {
            "host_url": "The network endpoint URL for the model provider (required)",
            "name": "Name of the model provider. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)",
        },
    )

    request_workspace = pop_workspace(input_payload)
    request_exist_ok = pop_exist_ok(input_payload)
    body = build_request_body(CreateModelProviderRequest, input_payload, command_name="inference providers create")
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(workspace=request_workspace, body=body, exist_ok=request_exist_ok)
    if handle_code_generation(ModelsClient, "create_provider", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ModelsClient).create_provider(
        workspace=request_workspace, body=body, exist_ok=bool(request_exist_ok)
    )

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("delete")
@collect_warnings
@handle_errors
def delete_providers(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete a model provider by workspace and name."""
    state: CLIContext = ctx.obj
    state.typed_client(ModelsClient).delete_provider(name=name, workspace=workspace)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_providers(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  updated_at: {gte: str, lte: str}\n\nFilter model providers by workspace, project, status, model_deployment_id, name, description, host_url, created_at, and updated_at.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_description: Annotated[
        str | None, typer.Option("--filter.description", rich_help_panel="Filter Options")
    ] = None,
    filter_host_url: Annotated[str | None, typer.Option("--filter.host-url", rich_help_panel="Filter Options")] = None,
    filter_model_deployment_id: Annotated[
        str | None, typer.Option("--filter.model-deployment-id", rich_help_panel="Filter Options")
    ] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_project: Annotated[str | None, typer.Option("--filter.project", rich_help_panel="Filter Options")] = None,
    filter_status: Annotated[str | None, typer.Option("--filter.status", rich_help_panel="Filter Options")] = None,
    filter_workspace: Annotated[
        str | None, typer.Option("--filter.workspace", rich_help_panel="Filter Options")
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    sort: Annotated[
        Literal["name", "-name", "created_at", "-created_at", "updated_at", "-updated_at", "status", "-status"] | None,
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
    """List model providers for a specific workspace."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("name", None),
        Column("description", None),
        Column("created_at", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    filter_value = filter_query_value(
        merge_filter_dict(
            filter,
            description=filter_description,
            host_url=filter_host_url,
            model_deployment_id=filter_model_deployment_id,
            name=filter_name,
            project=filter_project,
            status=filter_status,
            workspace=filter_workspace,
        )
    )
    query_params = cast(
        "ListProvidersQueryParams | None",
        offset_query_params(filter_value=filter_value, page=page, page_size=page_size, sort=sort),
    )
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)
    if handle_code_generation(ModelsClient, "list_providers", kwargs, resolved_output_format, state, result="list"):
        return

    response = state.typed_client(ModelsClient).list_providers(workspace=workspace, query_params=query_params)
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


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_providers(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a model provider by workspace and name."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(ModelsClient, "get_provider", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ModelsClient).get_provider(name=name, workspace=workspace)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("update")
@collect_warnings
@handle_errors
def update_providers(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    host_url: Annotated[
        str | None, typer.Option("--host-url", help="The network endpoint URL for the model provider (required)")
    ] = None,
    api_key_secret_name: Annotated[
        str | None,
        typer.Option(
            "--api-key-secret-name",
            help="Reference to an API key secret stored in the Secrets service. Create the secret first via secrets API, then pass the secret name here.",
        ),
    ] = None,
    auth_header_format: Annotated[
        str | None,
        typer.Option(
            "--auth-header-format",
            help="Jinja2 template string controlling how the API key secret is sent to the upstream. Must contain exactly one variable named `auth_secret`, which is substituted with the resolved secret value at request time. Example: `'X-Api-Key: {{ auth_secret }}'`. If not set, defaults to `'Authorization: Bearer {{ auth_secret }}'`.",
        ),
    ] = None,
    default_extra_body: Annotated[
        str | None,
        typer.Option(
            "--default-extra-body",
            help="Default body parameters for inference requests. Can be overridden by user requests. (JSON string)",
        ),
    ] = None,
    default_extra_headers: Annotated[
        str | None,
        typer.Option(
            "--default-extra-headers",
            help="Default headers for inference requests. Can be overridden by user requests. (JSON string)",
        ),
    ] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Optional description of the model provider")
    ] = None,
    enabled_models: Annotated[
        list[str] | None,
        typer.Option(
            "--enabled-models", help="Optional list of specific models to enable from this provider (can be repeated)"
        ),
    ] = None,
    model_deployment_id: Annotated[
        str | None,
        typer.Option(
            "--model-deployment-id",
            help="Optional reference to the ModelDeployment ID if this provider is associated with a deployment",
        ),
    ] = None,
    project: Annotated[
        str | None, typer.Option("--project", help="The URN of the project associated with this model provider")
    ] = None,
    required_extra_body: Annotated[
        str | None,
        typer.Option(
            "--required-extra-body",
            help="Required body parameters for inference requests. Cannot be overridden by user requests. (JSON string)",
        ),
    ] = None,
    required_extra_headers: Annotated[
        str | None,
        typer.Option(
            "--required-extra-headers",
            help="Required headers for inference requests. Cannot be overridden by user requests. (JSON string)",
        ),
    ] = None,
    status: Annotated[
        ProviderStatusValue | None,
        typer.Option("--status", help="Status enum for ModelProvider objects."),
    ] = None,
    status_message: Annotated[str | None, typer.Option("--status-message", help="Status message")] = None,
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
    """Create or update a model provider.

    [bold red]Required fields:[/] host_url

    [green]Examples:[/]
    nemo inference providers update <name> --input-file config.json
    nemo inference providers update <name> --input-data '{"host_url": "value"}'
    echo '{"json": "data"}' | nemo inference providers update <name> --input-file -
    nemo inference providers update <name> --<option> "value"
    """
    input_payload = read_input_payload(input_file, input_data)

    if workspace is not None:
        input_payload["workspace"] = workspace
    if host_url is not None:
        input_payload["host_url"] = host_url
    if api_key_secret_name is not None:
        input_payload["api_key_secret_name"] = api_key_secret_name
    if auth_header_format is not None:
        input_payload["auth_header_format"] = auth_header_format
    if default_extra_body is not None:
        input_payload["default_extra_body"] = read_payload("default_extra_body", default_extra_body)
    if default_extra_headers is not None:
        input_payload["default_extra_headers"] = read_payload("default_extra_headers", default_extra_headers)
    if description is not None:
        input_payload["description"] = description
    if enabled_models:  # Check for non-empty list
        input_payload["enabled_models"] = enabled_models
    if model_deployment_id is not None:
        input_payload["model_deployment_id"] = model_deployment_id
    if project is not None:
        input_payload["project"] = project
    if required_extra_body is not None:
        input_payload["required_extra_body"] = read_payload("required_extra_body", required_extra_body)
    if required_extra_headers is not None:
        input_payload["required_extra_headers"] = read_payload("required_extra_headers", required_extra_headers)
    if status is not None:
        input_payload["status"] = status
    if status_message is not None:
        input_payload["status_message"] = status_message
    validate_required_fields(
        input_payload,
        ["host_url"],
        "inference providers update",
        {
            "host_url": "The network endpoint URL for the model provider (required)",
        },
    )

    request_workspace = pop_workspace(input_payload)
    body = build_request_body(UpsertModelProviderRequest, input_payload, command_name="inference providers update")
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=request_workspace, body=body)
    if handle_code_generation(ModelsClient, "upsert_provider", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ModelsClient).upsert_provider(name=name, workspace=request_workspace, body=body)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("update-status")
@collect_warnings
@handle_errors
def update_status_providers(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    model_deployment_id: Annotated[
        str | None,
        typer.Option(
            "--model-deployment-id",
            help="Reference to the ModelDeployment ID if this provider is associated with a deployment",
        ),
    ] = None,
    served_models: Annotated[
        str | None,
        typer.Option(
            "--served-models",
            help="List of models served by this provider with routing information for IGW (JSON string)",
        ),
    ] = None,
    status: Annotated[
        ProviderStatusValue | None,
        typer.Option("--status", help="Status enum for ModelProvider objects."),
    ] = None,
    status_message: Annotated[
        str | None,
        typer.Option(
            "--status-message",
            help="Status message. If status is provided without status_message, defaults to empty string.",
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
    """Update status-related fields of a model provider.

    This endpoint supports partial updates for fields managed by Models Controller:

    - model_deployment_id
    - served_models
    - status
    - status_message

    If status is provided without status_message, status_message will be set to
    empty string.

        [green]Examples:[/]
        nemo inference providers update-status <name> --input-file config.json
        nemo inference providers update-status <name> --input-data '{"field": "value"}'
        echo '{"json": "data"}' | nemo inference providers update-status <name> --input-file -
        nemo inference providers update-status <name> --<option> "value"
    """
    input_payload = read_input_payload(input_file, input_data)

    if workspace is not None:
        input_payload["workspace"] = workspace
    if model_deployment_id is not None:
        input_payload["model_deployment_id"] = model_deployment_id
    if served_models is not None:
        input_payload["served_models"] = read_payload("served_models", served_models)
    if status is not None:
        input_payload["status"] = status
    if status_message is not None:
        input_payload["status_message"] = status_message

    request_workspace = pop_workspace(input_payload)
    body = build_request_body(
        UpdateModelProviderStatusRequest, input_payload, command_name="inference providers update-status"
    )
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=request_workspace, body=body)
    if handle_code_generation(ModelsClient, "update_provider_status", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ModelsClient).update_provider_status(name=name, workspace=request_workspace, body=body)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

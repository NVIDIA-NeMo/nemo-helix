# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo inference virtual-models`` command group, backed by the typed VirtualModels client."""

from __future__ import annotations

from typing import Annotated, cast

import typer
from nemo_platform_plugin.virtual_models.client import VirtualModelsClient
from nemo_platform_plugin.virtual_models.types import (
    CreateVirtualModelRequest,
    DeleteVirtualModelQueryParams,
    ListVirtualModelsQueryParams,
    UpdateVirtualModelRequest,
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

app = create_typer_app(name="virtual_models", help="Manage virtual_models")

_AUTOPROVISIONED_HELP = "Marks this VirtualModel as controller-managed. The Models controller will delete it once no ModelProvider serves the matching entity. Setting this manually opts the VirtualModel into that cleanup behavior."
_DEFAULT_MODEL_ENTITY_HELP = 'Model entity to route to, in "workspace/name" format. Written into request["model"] before the request middleware pipeline runs. If omitted, a request middleware plugin must handle backend routing itself. Set to null to clear an existing value.'
_MODELS_HELP = "Model entity references used by this VirtualModel. A per-entry backend_format overrides the referenced ModelEntity backend_format when IGW resolves the backend format for a request. (JSON string)"
_OVERRIDE_PROXY_HELP = 'Plugin-provided proxy implementation for IGW to use instead of its default aiohttp proxy. Format: "plugin-name.proxy-name". Leave unset to use the default IGW proxy. Set to null to clear an existing value.'
_POST_RESPONSE_MIDDLEWARE_HELP = "Ordered list of middleware plugins invoked after the response has been returned to the caller. Intended for fire-and-forget work (logging, analytics) that must not block or modify the response. (JSON string)"
_REQUEST_MIDDLEWARE_HELP = 'Ordered list of middleware plugins applied before proxying to the backend. Each entry is a MiddlewareCall with a "name" (plugin identifier) and optional "config_type" and "config_id" fields that reference a stored plugin configuration. (JSON string)'
_RESPONSE_MIDDLEWARE_HELP = "Ordered list of middleware plugins applied after the backend response is received, before returning it to the caller. (JSON string)"


@app.command("create")
@collect_warnings
@handle_errors
def create_virtual_models(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Argument(help="Name of the virtual model within the workspace. Must be unique per workspace. (required)"),
    ] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    autoprovisioned: Annotated[bool | None, typer.Option("--autoprovisioned", help=_AUTOPROVISIONED_HELP)] = None,
    default_model_entity: Annotated[
        str | None, typer.Option("--default-model-entity", help=_DEFAULT_MODEL_ENTITY_HELP)
    ] = None,
    models: Annotated[str | None, typer.Option("--models", help=_MODELS_HELP)] = None,
    override_proxy: Annotated[str | None, typer.Option("--override-proxy", help=_OVERRIDE_PROXY_HELP)] = None,
    post_response_middleware: Annotated[
        str | None, typer.Option("--post-response-middleware", help=_POST_RESPONSE_MIDDLEWARE_HELP)
    ] = None,
    request_middleware: Annotated[
        str | None, typer.Option("--request-middleware", help=_REQUEST_MIDDLEWARE_HELP)
    ] = None,
    response_middleware: Annotated[
        str | None, typer.Option("--response-middleware", help=_RESPONSE_MIDDLEWARE_HELP)
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
    """Create a new VirtualModel in the given workspace.

    A VirtualModel defines an ordered middleware pipeline that IGW executes when an
    inference request arrives with `model: "workspace/name"` matching this entity.

        [bold red]Required fields:[/] name

        [green]Examples:[/]
        nemo inference virtual-models create <name> --input-file config.json
        nemo inference virtual-models create <name> --input-data '{"name": "value"}'
        echo '{"json": "data"}' | nemo inference virtual-models create <name> --input-file -
        nemo inference virtual-models create <name> --<option> "value"
    """
    input_payload = read_input_payload(input_file, input_data)

    if workspace is not None:
        input_payload["workspace"] = workspace
    if name is not None:
        input_payload["name"] = name
    if autoprovisioned is not None:
        input_payload["autoprovisioned"] = autoprovisioned
    if default_model_entity is not None:
        input_payload["default_model_entity"] = default_model_entity
    if models is not None:
        input_payload["models"] = read_payload("models", models)
    if override_proxy is not None:
        input_payload["override_proxy"] = override_proxy
    if post_response_middleware is not None:
        input_payload["post_response_middleware"] = read_payload("post_response_middleware", post_response_middleware)
    if request_middleware is not None:
        input_payload["request_middleware"] = read_payload("request_middleware", request_middleware)
    if response_middleware is not None:
        input_payload["response_middleware"] = read_payload("response_middleware", response_middleware)
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    validate_required_fields(
        input_payload,
        ["name"],
        "inference virtual-models create",
        {
            "name": "Name of the virtual model within the workspace. Must be unique per workspace. (required)",
        },
    )

    request_workspace = pop_workspace(input_payload)
    request_exist_ok = pop_exist_ok(input_payload)
    body = build_request_body(CreateVirtualModelRequest, input_payload, command_name="inference virtual-models create")
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(workspace=request_workspace, body=body, exist_ok=request_exist_ok)
    if handle_code_generation(VirtualModelsClient, "create_virtual_model", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(VirtualModelsClient).create_virtual_model(
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
def delete_virtual_models(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    expected_db_version: Annotated[
        int | None,
        typer.Option(
            "--expected-db-version",
            help="Optional database version for optimistic locking. Delete only succeeds if the VirtualModel still has this version.",
        ),
    ] = None,
) -> None:
    """Permanently delete a VirtualModel.

    This does not affect any in-flight requests already being routed through this
    VirtualModel. IGW's model cache is refreshed on its next polling cycle."""
    state: CLIContext = ctx.obj
    query_params: DeleteVirtualModelQueryParams | None = None
    if expected_db_version is not None:
        query_params = {"expected_db_version": expected_db_version}
    state.typed_client(VirtualModelsClient).delete_virtual_model(
        name=name, workspace=workspace, query_params=query_params
    )

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_virtual_models(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    exclude_autoprovisioned: Annotated[
        bool | None,
        typer.Option(
            "--exclude-autoprovisioned",
            help="When true, controller-managed (autoprovisioned) passthrough VirtualModels are excluded from the results.",
        ),
    ] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  updated_at: {gte: str, lte: str}\n\nFilter virtual models by workspace, project, name, default_model_entity, guardrail_config, created_at, and updated_at.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_default_model_entity: Annotated[
        str | None, typer.Option("--filter.default-model-entity", rich_help_panel="Filter Options")
    ] = None,
    filter_guardrail_config: Annotated[
        str | None, typer.Option("--filter.guardrail-config", rich_help_panel="Filter Options")
    ] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_project: Annotated[str | None, typer.Option("--filter.project", rich_help_panel="Filter Options")] = None,
    filter_workspace: Annotated[
        str | None, typer.Option("--filter.workspace", rich_help_panel="Filter Options")
    ] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number (1-indexed).")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Number of results per page.")] = None,
    sort: Annotated[
        str | None, typer.Option("--sort", help="Sort field. Prefix with `-` for descending order.")
    ] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List VirtualModels for the given workspace.

    Use `workspace=-` to list across all workspaces accessible to the caller."""
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

    filter_value = filter_query_value(
        merge_filter_dict(
            filter,
            default_model_entity=filter_default_model_entity,
            guardrail_config=filter_guardrail_config,
            name=filter_name,
            project=filter_project,
            workspace=filter_workspace,
        )
    )
    query_params = cast(
        "ListVirtualModelsQueryParams | None",
        offset_query_params(
            filter_value=filter_value,
            page=page,
            page_size=page_size,
            sort=sort,
            exclude_autoprovisioned=exclude_autoprovisioned,
        ),
    )
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)
    if handle_code_generation(
        VirtualModelsClient, "list_virtual_models", kwargs, resolved_output_format, state, result="list"
    ):
        return

    response = state.typed_client(VirtualModelsClient).list_virtual_models(
        workspace=workspace, query_params=query_params
    )
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


@app.command("patch")
@collect_warnings
@handle_errors
def patch_virtual_models(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    autoprovisioned: Annotated[bool | None, typer.Option("--autoprovisioned", help=_AUTOPROVISIONED_HELP)] = None,
    default_model_entity: Annotated[
        str | None, typer.Option("--default-model-entity", help=_DEFAULT_MODEL_ENTITY_HELP)
    ] = None,
    models: Annotated[str | None, typer.Option("--models", help=_MODELS_HELP)] = None,
    override_proxy: Annotated[str | None, typer.Option("--override-proxy", help=_OVERRIDE_PROXY_HELP)] = None,
    post_response_middleware: Annotated[
        str | None, typer.Option("--post-response-middleware", help=_POST_RESPONSE_MIDDLEWARE_HELP)
    ] = None,
    request_middleware: Annotated[
        str | None, typer.Option("--request-middleware", help=_REQUEST_MIDDLEWARE_HELP)
    ] = None,
    response_middleware: Annotated[
        str | None, typer.Option("--response-middleware", help=_RESPONSE_MIDDLEWARE_HELP)
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
    """Partially update a VirtualModel.

    Only fields present in the request body are modified. Fields absent from the
    request body retain their current values.

        [green]Examples:[/]
        nemo inference virtual-models patch <name> --input-file config.json
        nemo inference virtual-models patch <name> --input-data '{"field": "value"}'
        echo '{"json": "data"}' | nemo inference virtual-models patch <name> --input-file -
        nemo inference virtual-models patch <name> --<option> "value"
    """
    input_payload = read_input_payload(input_file, input_data)

    if workspace is not None:
        input_payload["workspace"] = workspace
    if autoprovisioned is not None:
        input_payload["autoprovisioned"] = autoprovisioned
    if default_model_entity is not None:
        input_payload["default_model_entity"] = default_model_entity
    if models is not None:
        input_payload["models"] = read_payload("models", models)
    if override_proxy is not None:
        input_payload["override_proxy"] = override_proxy
    if post_response_middleware is not None:
        input_payload["post_response_middleware"] = read_payload("post_response_middleware", post_response_middleware)
    if request_middleware is not None:
        input_payload["request_middleware"] = read_payload("request_middleware", request_middleware)
    if response_middleware is not None:
        input_payload["response_middleware"] = read_payload("response_middleware", response_middleware)

    request_workspace = pop_workspace(input_payload)
    body = build_request_body(UpdateVirtualModelRequest, input_payload, command_name="inference virtual-models patch")
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=request_workspace, body=body)
    if handle_code_generation(VirtualModelsClient, "update_virtual_model", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(VirtualModelsClient).update_virtual_model(
        name=name, workspace=request_workspace, body=body
    )

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_virtual_models(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a VirtualModel by workspace and name."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(VirtualModelsClient, "get_virtual_model", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(VirtualModelsClient).get_virtual_model(name=name, workspace=workspace)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

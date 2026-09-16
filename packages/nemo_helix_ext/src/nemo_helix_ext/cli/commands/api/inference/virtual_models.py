# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from typing import Annotated

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
from nemo_helix_ext.cli.core.stdin_utils import read_data_input_with_flags, read_payload, validate_required_fields
from nemo_helix_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="virtual_models", help="Manage virtual_models")


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
    autoprovisioned: Annotated[
        bool | None,
        typer.Option(
            "--autoprovisioned",
            help="Marks this VirtualModel as controller-managed. The Models controller will delete it once no ModelProvider serves the matching entity. Setting this manually opts the VirtualModel into that cleanup behavior.",
        ),
    ] = None,
    default_model_entity: Annotated[
        str | None,
        typer.Option(
            "--default-model-entity",
            help='Model entity to route to, in "workspace/name" format. Written into request["model"] before the request middleware pipeline runs. If omitted, a request middleware plugin must handle backend routing itself. Set to null to clear an existing value.',
        ),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            help="Model entity references used by this VirtualModel. A per-entry backend_format overrides the referenced ModelEntity backend_format when IGW resolves the backend format for a request. (JSON string)",
        ),
    ] = None,
    override_proxy: Annotated[
        str | None,
        typer.Option(
            "--override-proxy",
            help='Plugin-provided proxy implementation for IGW to use instead of its default aiohttp proxy. Format: "plugin-name.proxy-name". Leave unset to use the default IGW proxy. Set to null to clear an existing value.',
        ),
    ] = None,
    post_response_middleware: Annotated[
        str | None,
        typer.Option(
            "--post-response-middleware",
            help="Ordered list of middleware plugins invoked after the response has been returned to the caller. Intended for fire-and-forget work (logging, analytics) that must not block or modify the response. (JSON string)",
        ),
    ] = None,
    request_middleware: Annotated[
        str | None,
        typer.Option(
            "--request-middleware",
            help='Ordered list of middleware plugins applied before proxying to the backend. Each entry is a MiddlewareCall with a "name" (plugin identifier) and optional "config_type" and "config_id" fields that reference a stored plugin configuration. (JSON string)',
        ),
    ] = None,
    response_middleware: Annotated[
        str | None,
        typer.Option(
            "--response-middleware",
            help="Ordered list of middleware plugins applied after the backend response is received, before returning it to the caller. (JSON string)",
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
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["name"],
        "inference virtual-models create",
        {
            "name": "Name of the virtual model within the workspace. Must be unique per workspace. (required)",
        },
    )

    all_kwargs = input_payload
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "virtual_models"], "create", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.virtual_models.create(**all_kwargs)

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
    client = state.get_client()

    kwargs = build_kwargs(
        workspace=workspace,
        expected_db_version=expected_db_version,
    )
    client.inference.virtual_models.delete(name, **kwargs)

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
        exclude_autoprovisioned=exclude_autoprovisioned,
        filter=merge_filter_dict(
            filter,
            default_model_entity=filter_default_model_entity,
            guardrail_config=filter_guardrail_config,
            name=filter_name,
            project=filter_project,
            workspace=filter_workspace,
        ),
        page=page,
        page_size=page_size,
        sort=sort,
    )

    if handle_code_generation(["inference", "virtual_models"], "list", kwargs, output_format, state):
        return

    client = state.get_client()
    path_args = ()
    pagination_type = PaginationType.PAGE_NUMBER
    if all_pages:
        items = fetch_all_pages(
            client.inference.virtual_models.list,
            path_args=path_args,
            body_args=kwargs,
            pagination_type=pagination_type,
        )
    else:
        items = client.inference.virtual_models.list(*path_args, **kwargs)

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


@app.command("patch")
@collect_warnings
@handle_errors
def patch_virtual_models(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    autoprovisioned: Annotated[
        bool | None,
        typer.Option(
            "--autoprovisioned",
            help="Marks this VirtualModel as controller-managed. The Models controller will delete it once no ModelProvider serves the matching entity. Setting this manually opts the VirtualModel into that cleanup behavior.",
        ),
    ] = None,
    default_model_entity: Annotated[
        str | None,
        typer.Option(
            "--default-model-entity",
            help='Model entity to route to, in "workspace/name" format. Written into request["model"] before the request middleware pipeline runs. If omitted, a request middleware plugin must handle backend routing itself. Set to null to clear an existing value.',
        ),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            help="Model entity references used by this VirtualModel. A per-entry backend_format overrides the referenced ModelEntity backend_format when IGW resolves the backend format for a request. (JSON string)",
        ),
    ] = None,
    override_proxy: Annotated[
        str | None,
        typer.Option(
            "--override-proxy",
            help='Plugin-provided proxy implementation for IGW to use instead of its default aiohttp proxy. Format: "plugin-name.proxy-name". Leave unset to use the default IGW proxy. Set to null to clear an existing value.',
        ),
    ] = None,
    post_response_middleware: Annotated[
        str | None,
        typer.Option(
            "--post-response-middleware",
            help="Ordered list of middleware plugins invoked after the response has been returned to the caller. Intended for fire-and-forget work (logging, analytics) that must not block or modify the response. (JSON string)",
        ),
    ] = None,
    request_middleware: Annotated[
        str | None,
        typer.Option(
            "--request-middleware",
            help='Ordered list of middleware plugins applied before proxying to the backend. Each entry is a MiddlewareCall with a "name" (plugin identifier) and optional "config_type" and "config_id" fields that reference a stored plugin configuration. (JSON string)',
        ),
    ] = None,
    response_middleware: Annotated[
        str | None,
        typer.Option(
            "--response-middleware",
            help="Ordered list of middleware plugins applied after the backend response is received, before returning it to the caller. (JSON string)",
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
    """Partially update a VirtualModel.

    Only fields present in the request body are modified. Fields absent from the
    request body retain their current values.

        [green]Examples:[/]
        nemo inference virtual-models patch <name> --input-file config.json
        nemo inference virtual-models patch <name> --input-data '{"field": "value"}'
        echo '{"json": "data"}' | nemo inference virtual-models patch <name> --input-file -
        nemo inference virtual-models patch <name> --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
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

    all_kwargs = {"name": name, **input_payload}

    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    if handle_code_generation(["inference", "virtual_models"], "patch", all_kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.virtual_models.patch(**all_kwargs)

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
def retrieve_virtual_models(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a VirtualModel by workspace and name."""
    state: CLIContext = ctx.obj
    output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(
        workspace=workspace,
    )
    if handle_code_generation(["inference", "virtual_models"], "retrieve", kwargs, output_format, state):
        return

    client = state.get_client()
    result = client.inference.virtual_models.retrieve(name, **kwargs)

    format_output(
        result,
        is_list=False,
        output_format=output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

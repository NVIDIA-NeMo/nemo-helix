# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo inference deployment-configs`` command group, backed by the typed Models client."""

from __future__ import annotations

from typing import Annotated, Literal, cast

import typer
from nemo_platform_plugin.models.client import ModelsClient
from nemo_platform_plugin.models.types import (
    CreateModelDeploymentConfigRequest,
    ListDeploymentConfigsQueryParams,
    UpdateModelDeploymentConfigRequest,
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

app = create_typer_app(name="deployment_configs", help="Manage deployment_configs")
versions_app = create_typer_app(name="versions", help="Manage versions")
app.add_typer(versions_app, name="versions")

EngineValue = Literal["nim", "vllm", "generic"]

_ENGINE_HELP = "Inference engine selecting the compiler path for a deployment. The engine determines what command, image, and env a deployment compiles to. The fields a compiler consumes are not engine-specific; engines take the same inputs (model_spec + executor_config) and differ in what they do with them. (required)"
_EXECUTOR_CONFIG_HELP = "Compute + container settings shared by the docker and k8s executors. Both the docker and k8s executors run containers and share this shape. A future non-container executor (e.g. subprocess) would warrant turning `executor_config` into a discriminated union. (JSON string) (required)"
_MODEL_SPEC_HELP = "What model to serve and how -- independent of the executor it runs on. Executor-invariant facts about the model. The compiler resolves the weight source per engine; serving fields override the model entity spec when set. (JSON string) (required)"


@app.command("create")
@collect_warnings
@handle_errors
def create_deployment_configs(
    ctx: typer.Context,
    name: Annotated[
        str | None,
        typer.Argument(
            help="Name of the deployment configuration. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)"
        ),
    ] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    engine: Annotated[EngineValue | None, typer.Option("--engine", help=_ENGINE_HELP)] = None,
    executor_config: Annotated[str | None, typer.Option("--executor-config", help=_EXECUTOR_CONFIG_HELP)] = None,
    model_spec: Annotated[str | None, typer.Option("--model-spec", help=_MODEL_SPEC_HELP)] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Optional description of the deployment configuration")
    ] = None,
    model_entity_id: Annotated[
        str | None,
        typer.Option("--model-entity-id", help="Optional reference to the base model entity ID for this deployment"),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option("--project", help="The URN of the project associated with this deployment configuration"),
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
    """Create a new ModelDeploymentConfig (version 1).

    [bold red]Required fields:[/] engine, executor_config, model_spec, name

    [green]Examples:[/]
    nemo inference deployment-configs create <name> --input-file config.json
    nemo inference deployment-configs create <name> --input-data '{"engine": "value", "executor_config": {}, "model_spec": {}, "name": "value"}'
    echo '{"json": "data"}' | nemo inference deployment-configs create <name> --input-file -
    nemo inference deployment-configs create <name> --<option> "value"
    """
    input_payload = read_input_payload(input_file, input_data)

    if workspace is not None:
        input_payload["workspace"] = workspace
    if engine is not None:
        input_payload["engine"] = engine
    if executor_config is not None:
        input_payload["executor_config"] = read_payload("executor_config", executor_config)
    if model_spec is not None:
        input_payload["model_spec"] = read_payload("model_spec", model_spec)
    if name is not None:
        input_payload["name"] = name
    if description is not None:
        input_payload["description"] = description
    if model_entity_id is not None:
        input_payload["model_entity_id"] = model_entity_id
    if project is not None:
        input_payload["project"] = project
    if exist_ok is not None:
        input_payload["exist_ok"] = exist_ok
    validate_required_fields(
        input_payload,
        ["engine", "executor_config", "model_spec", "name"],
        "inference deployment-configs create",
        {
            "engine": _ENGINE_HELP,
            "executor_config": _EXECUTOR_CONFIG_HELP,
            "model_spec": _MODEL_SPEC_HELP,
            "name": "Name of the deployment configuration. Name must start with a lowercase letter, be 2-63 characters, and use lowercase letters, digits, hyphens, and dots (no consecutive hyphens, cannot end with a hyphen). (required)",
        },
    )

    request_workspace = pop_workspace(input_payload)
    request_exist_ok = pop_exist_ok(input_payload)
    body = build_request_body(
        CreateModelDeploymentConfigRequest, input_payload, command_name="inference deployment-configs create"
    )
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(workspace=request_workspace, body=body, exist_ok=request_exist_ok)
    if handle_code_generation(ModelsClient, "create_deployment_config", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ModelsClient).create_deployment_config(
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
def delete_deployment_configs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
) -> None:
    """Delete all versions of a ModelDeploymentConfig.

    This operation will fail with 409 Conflict if any ModelDeployments currently
    reference this config and are not in DELETED status. Delete or wait for
    dependent deployments to reach DELETED status before deleting the config."""
    state: CLIContext = ctx.obj
    state.typed_client(ModelsClient).delete_deployment_config(name=name, workspace=workspace)

    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_deployment_configs(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    filter: Annotated[
        str | None,
        typer.Option(
            "--filter",
            metavar="FILTER_JSON",
            help="Use --filter with JSON for complex/nested queries, or --filter.FIELD options for simple fields. Both can be combined, with field options taking precedence.\nJSON-only fields:\n  created_at: {gte: str, lte: str}\n  updated_at: {gte: str, lte: str}\n\nFilter deployment configs by workspace, project, model_entity_id, name, description, created_at, and updated_at.",
            rich_help_panel="Filter Options",
        ),
    ] = None,
    filter_description: Annotated[
        str | None, typer.Option("--filter.description", rich_help_panel="Filter Options")
    ] = None,
    filter_model_entity_id: Annotated[
        str | None, typer.Option("--filter.model-entity-id", rich_help_panel="Filter Options")
    ] = None,
    filter_name: Annotated[str | None, typer.Option("--filter.name", rich_help_panel="Filter Options")] = None,
    filter_project: Annotated[str | None, typer.Option("--filter.project", rich_help_panel="Filter Options")] = None,
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
    """List ModelDeploymentConfigs for a specific workspace.

    Returns only the latest
    version of each config."""
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
            model_entity_id=filter_model_entity_id,
            name=filter_name,
            project=filter_project,
            workspace=filter_workspace,
        )
    )
    query_params = cast(
        "ListDeploymentConfigsQueryParams | None",
        offset_query_params(filter_value=filter_value, page=page, page_size=page_size, sort=sort),
    )
    kwargs = build_kwargs(workspace=workspace, query_params=query_params)
    if handle_code_generation(
        ModelsClient, "list_deployment_configs", kwargs, resolved_output_format, state, result="list"
    ):
        return

    response = state.typed_client(ModelsClient).list_deployment_configs(workspace=workspace, query_params=query_params)
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
def retrieve_deployment_configs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get the latest version of a ModelDeploymentConfig."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(ModelsClient, "get_deployment_config", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ModelsClient).get_deployment_config(name=name, workspace=workspace)

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
def update_deployment_configs(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    engine: Annotated[EngineValue | None, typer.Option("--engine", help=_ENGINE_HELP)] = None,
    executor_config: Annotated[str | None, typer.Option("--executor-config", help=_EXECUTOR_CONFIG_HELP)] = None,
    model_spec: Annotated[str | None, typer.Option("--model-spec", help=_MODEL_SPEC_HELP)] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Optional description of the deployment configuration")
    ] = None,
    model_entity_id: Annotated[
        str | None,
        typer.Option("--model-entity-id", help="Optional reference to the base model entity ID for this deployment"),
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
    """Update a ModelDeploymentConfig (creates a new immutable version).

    [bold red]Required fields:[/] engine, executor_config, model_spec

    [green]Examples:[/]
    nemo inference deployment-configs update <name> --input-file config.json
    nemo inference deployment-configs update <name> --input-data '{"engine": "value", "executor_config": {}, "model_spec": {}}'
    echo '{"json": "data"}' | nemo inference deployment-configs update <name> --input-file -
    nemo inference deployment-configs update <name> --<option> "value"
    """
    input_payload = read_input_payload(input_file, input_data)

    if workspace is not None:
        input_payload["workspace"] = workspace
    if engine is not None:
        input_payload["engine"] = engine
    if executor_config is not None:
        input_payload["executor_config"] = read_payload("executor_config", executor_config)
    if model_spec is not None:
        input_payload["model_spec"] = read_payload("model_spec", model_spec)
    if description is not None:
        input_payload["description"] = description
    if model_entity_id is not None:
        input_payload["model_entity_id"] = model_entity_id
    validate_required_fields(
        input_payload,
        ["engine", "executor_config", "model_spec"],
        "inference deployment-configs update",
        {
            "engine": _ENGINE_HELP,
            "executor_config": _EXECUTOR_CONFIG_HELP,
            "model_spec": _MODEL_SPEC_HELP,
        },
    )

    request_workspace = pop_workspace(input_payload)
    body = build_request_body(
        UpdateModelDeploymentConfigRequest, input_payload, command_name="inference deployment-configs update"
    )
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=request_workspace, body=body)
    if handle_code_generation(ModelsClient, "update_deployment_config", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ModelsClient).update_deployment_config(
        name=name, workspace=request_workspace, body=body
    )

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@versions_app.command("delete")
@collect_warnings
@handle_errors
def delete_versions(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    *,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    config: Annotated[str, typer.Option("--config")],
) -> None:
    """Delete a specific version of a ModelDeploymentConfig.

    This operation will fail with 409 Conflict if any ModelDeployments currently
    reference this specific version and are not in DELETED status. Delete or wait
    for dependent deployments to reach DELETED status before deleting the config
    version."""
    state: CLIContext = ctx.obj
    state.typed_client(ModelsClient).delete_deployment_config_version(config=config, name=name, workspace=workspace)

    typer.echo("✓ Deleted successfully")


@versions_app.command("list")
@collect_warnings
@handle_errors
def list_versions(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """List all versions of a ModelDeploymentConfig."""
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

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(ModelsClient, "list_deployment_config_versions", kwargs, resolved_output_format, state):
        return

    items = state.typed_client(ModelsClient).list_deployment_config_versions(name=name, workspace=workspace)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )


@versions_app.command("get")
@collect_warnings
@handle_errors
def retrieve_versions(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    *,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    config: Annotated[str, typer.Option("--config")],
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Get a specific version of a ModelDeploymentConfig."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(config=config, name=name, workspace=workspace)
    if handle_code_generation(ModelsClient, "get_deployment_config_version", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(ModelsClient).get_deployment_config_version(
        config=config, name=name, workspace=workspace
    )

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

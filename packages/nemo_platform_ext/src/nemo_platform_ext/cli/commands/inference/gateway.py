# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo inference gateway`` command group, backed by the typed Inference Gateway client.

The ``provider`` and ``model`` sub-groups proxy raw HTTP verbs to a provider or
model-entity route (``.../-/{trailing_uri}``); ``openai v1 models`` exposes the
workspace's OpenAI-compatible model catalog.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable

import typer
from nemo_platform_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_platform_plugin.inference_gateway.types import JsonBody

from nemo_platform_ext.cli.commands.inference._common import pop_workspace, read_input_payload
from nemo_platform_ext.cli.core.api import build_kwargs
from nemo_platform_ext.cli.core.code_generator import handle_code_generation
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import UnknownInputFieldsError, handle_errors
from nemo_platform_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_output,
    validate_stream_output_format,
)
from nemo_platform_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_platform_ext.cli.core.stdin_utils import read_payload, validate_required_fields
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormat,
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="gateway", help="Gateway operations")
model_app = create_typer_app(name="model", help="Manage model")
openai_app = create_typer_app(name="openai", help="OpenAI operations")
openai_v1_app = create_typer_app(name="v1", help="V1 operations")
openai_v1_models_app = create_typer_app(name="models", help="Manage models")
provider_app = create_typer_app(name="provider", help="Manage provider")

app.add_typer(model_app, name="model")
app.add_typer(openai_app, name="openai")
app.add_typer(provider_app, name="provider")
openai_app.add_typer(openai_v1_app, name="v1")
openai_v1_app.add_typer(openai_v1_models_app, name="models")


def _proxy_body_request(
    ctx: typer.Context,
    *,
    method: str,
    command_name: str,
    trailing_uri: str,
    name: str | None,
    workspace: str | None,
    body: str | None,
    input_file: str | None,
    input_data: str | None,
    output_format: EntityOutputFormat | None,
) -> None:
    """Shared implementation of the ``patch`` / ``post`` / ``put`` proxy commands."""
    input_payload = read_input_payload(input_file, input_data)

    if workspace is not None:
        input_payload["workspace"] = workspace
    if name is not None:
        input_payload["name"] = name
    if body is not None:
        input_payload["body"] = read_payload("body", body)
    validate_required_fields(
        input_payload,
        ["name"],
        command_name,
        {
            "name": "(required)",
        },
    )

    request_workspace = pop_workspace(input_payload)
    request_name = str(input_payload["name"])
    # The proxy request has exactly one payload field; anything else in the
    # input is a typo or a misplaced body key, and the generated command
    # rejected it loudly instead of posting an empty body.
    unknown = sorted(set(input_payload) - {"name", "body"})
    if unknown:
        raise UnknownInputFieldsError(unknown, command_name, ["name", "body", "workspace"])
    request_body = JsonBody(input_payload.get("body") or {})
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=request_name, trailing_uri=trailing_uri, workspace=request_workspace, body=request_body)
    if handle_code_generation(InferenceGatewayClient, method, kwargs, resolved_output_format, state):
        return

    send: Callable[..., Any] = getattr(state.typed_client(InferenceGatewayClient), method)
    result = send(name=request_name, trailing_uri=trailing_uri, workspace=request_workspace, body=request_body)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


def _proxy_get(
    ctx: typer.Context,
    *,
    method: str,
    trailing_uri: str,
    name: str,
    workspace: str | None,
    output_format: EntityOutputFormat | None,
) -> None:
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, trailing_uri=trailing_uri, workspace=workspace)
    if handle_code_generation(InferenceGatewayClient, method, kwargs, resolved_output_format, state):
        return

    send: Callable[..., Any] = getattr(state.typed_client(InferenceGatewayClient), method)
    result = send(name=name, trailing_uri=trailing_uri, workspace=workspace)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


def _proxy_delete(ctx: typer.Context, *, method: str, trailing_uri: str, name: str, workspace: str | None) -> None:
    state: CLIContext = ctx.obj
    send: Callable[..., Any] = getattr(state.typed_client(InferenceGatewayClient), method)
    send(name=name, trailing_uri=trailing_uri, workspace=workspace)

    typer.echo("✓ Deleted successfully")


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------


@provider_app.command("delete")
@collect_warnings
@handle_errors
def delete_provider(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    *,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str, typer.Option("--name")],
) -> None:
    """Proxy requests to provider inference endpoints."""
    _proxy_delete(ctx, method="provider_delete", trailing_uri=trailing_uri, name=name, workspace=workspace)


@provider_app.command("get")
@collect_warnings
@handle_errors
def get_provider(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    *,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str, typer.Option("--name")],
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Proxy requests to provider inference endpoints."""
    _proxy_get(
        ctx,
        method="provider_get",
        trailing_uri=trailing_uri,
        name=name,
        workspace=workspace,
        output_format=output_format,
    )


@provider_app.command("patch")
@collect_warnings
@handle_errors
def patch_provider(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str | None, typer.Option("--name", help="(required)")] = None,
    body: Annotated[str | None, typer.Option("--body", help="JSON string")] = None,
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
    """Proxy requests to provider inference endpoints.

    [bold red]Required fields:[/] name

    [green]Examples:[/]
    nemo inference gateway provider patch <trailing_uri> --input-file config.json
    nemo inference gateway provider patch <trailing_uri> --input-data '{"name": "value"}'
    echo '{"json": "data"}' | nemo inference gateway provider patch <trailing_uri> --input-file -
    nemo inference gateway provider patch <trailing_uri> --<option> "value"
    """
    _proxy_body_request(
        ctx,
        method="provider_patch",
        command_name="inference gateway provider patch",
        trailing_uri=trailing_uri,
        name=name,
        workspace=workspace,
        body=body,
        input_file=input_file,
        input_data=input_data,
        output_format=output_format,
    )


@provider_app.command("post")
@collect_warnings
@handle_errors
def post_provider(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    name: Annotated[str | None, typer.Argument(help="(required)")] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    body: Annotated[str | None, typer.Option("--body", help="JSON string")] = None,
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
    """Proxy requests to provider inference endpoints.

    [bold red]Required fields:[/] name

    [green]Examples:[/]
    nemo inference gateway provider post <trailing_uri> <name> --input-file config.json
    nemo inference gateway provider post <trailing_uri> <name> --input-data '{"name": "value"}'
    echo '{"json": "data"}' | nemo inference gateway provider post <trailing_uri> <name> --input-file -
    nemo inference gateway provider post <trailing_uri> <name> --<option> "value"
    """
    _proxy_body_request(
        ctx,
        method="provider_post",
        command_name="inference gateway provider post",
        trailing_uri=trailing_uri,
        name=name,
        workspace=workspace,
        body=body,
        input_file=input_file,
        input_data=input_data,
        output_format=output_format,
    )


@provider_app.command("put")
@collect_warnings
@handle_errors
def put_provider(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    name: Annotated[str | None, typer.Argument(help="(required)")] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    body: Annotated[str | None, typer.Option("--body", help="JSON string")] = None,
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
    """Proxy requests to provider inference endpoints.

    [bold red]Required fields:[/] name

    [green]Examples:[/]
    nemo inference gateway provider put <trailing_uri> <name> --input-file config.json
    nemo inference gateway provider put <trailing_uri> <name> --input-data '{"name": "value"}'
    echo '{"json": "data"}' | nemo inference gateway provider put <trailing_uri> <name> --input-file -
    nemo inference gateway provider put <trailing_uri> <name> --<option> "value"
    """
    _proxy_body_request(
        ctx,
        method="provider_put",
        command_name="inference gateway provider put",
        trailing_uri=trailing_uri,
        name=name,
        workspace=workspace,
        body=body,
        input_file=input_file,
        input_data=input_data,
        output_format=output_format,
    )


@provider_app.command("ready")
@collect_warnings
@handle_errors
def ready_provider(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Check if a model provider is registered in the gateway's cache.

    This is a lightweight endpoint that only checks the gateway's internal state,
    without making any requests to the actual provider backend. Use this to verify
    the gateway is ready to route requests to a provider after deployment.

    Returns: 200 OK with provider info if the provider is registered 404 Not Found
    if the provider is not yet in the gateway's cache"""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(InferenceGatewayClient, "provider_ready", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(InferenceGatewayClient).provider_ready(name=name, workspace=workspace)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


@model_app.command("delete")
@collect_warnings
@handle_errors
def delete_model(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    *,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str, typer.Option("--name")],
) -> None:
    """Proxy requests to model entity inference endpoints.

    All inference requests must resolve to a `VirtualModel`. The platform's provider
    reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
    served model entity (named after the entity, with `default_model_entity` set to
    the entity ref) so this is the typical case; operators can also create custom
    VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
    which no VirtualModel can be found return `404`."""
    _proxy_delete(ctx, method="model_delete", trailing_uri=trailing_uri, name=name, workspace=workspace)


@model_app.command("get")
@collect_warnings
@handle_errors
def get_model(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    *,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str, typer.Option("--name")],
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Proxy requests to model entity inference endpoints.

    All inference requests must resolve to a `VirtualModel`. The platform's provider
    reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
    served model entity (named after the entity, with `default_model_entity` set to
    the entity ref) so this is the typical case; operators can also create custom
    VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
    which no VirtualModel can be found return `404`."""
    _proxy_get(
        ctx,
        method="model_get",
        trailing_uri=trailing_uri,
        name=name,
        workspace=workspace,
        output_format=output_format,
    )


@model_app.command("patch")
@collect_warnings
@handle_errors
def patch_model(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    name: Annotated[str | None, typer.Option("--name", help="(required)")] = None,
    body: Annotated[str | None, typer.Option("--body", help="JSON string")] = None,
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
    """Proxy requests to model entity inference endpoints.

    All inference requests must resolve to a `VirtualModel`. The platform's provider
    reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
    served model entity (named after the entity, with `default_model_entity` set to
    the entity ref) so this is the typical case; operators can also create custom
    VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
    which no VirtualModel can be found return `404`.

        [bold red]Required fields:[/] name

        [green]Examples:[/]
        nemo inference gateway model patch <trailing_uri> --input-file config.json
        nemo inference gateway model patch <trailing_uri> --input-data '{"name": "value"}'
        echo '{"json": "data"}' | nemo inference gateway model patch <trailing_uri> --input-file -
        nemo inference gateway model patch <trailing_uri> --<option> "value"
    """
    _proxy_body_request(
        ctx,
        method="model_patch",
        command_name="inference gateway model patch",
        trailing_uri=trailing_uri,
        name=name,
        workspace=workspace,
        body=body,
        input_file=input_file,
        input_data=input_data,
        output_format=output_format,
    )


@model_app.command("post")
@collect_warnings
@handle_errors
def post_model(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    name: Annotated[str | None, typer.Argument(help="(required)")] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    body: Annotated[str | None, typer.Option("--body", help="JSON string")] = None,
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
    """Proxy requests to model entity inference endpoints.

    All inference requests must resolve to a `VirtualModel`. The platform's provider
    reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
    served model entity (named after the entity, with `default_model_entity` set to
    the entity ref) so this is the typical case; operators can also create custom
    VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
    which no VirtualModel can be found return `404`.

        [bold red]Required fields:[/] name

        [green]Examples:[/]
        nemo inference gateway model post <trailing_uri> <name> --input-file config.json
        nemo inference gateway model post <trailing_uri> <name> --input-data '{"name": "value"}'
        echo '{"json": "data"}' | nemo inference gateway model post <trailing_uri> <name> --input-file -
        nemo inference gateway model post <trailing_uri> <name> --<option> "value"
    """
    _proxy_body_request(
        ctx,
        method="model_post",
        command_name="inference gateway model post",
        trailing_uri=trailing_uri,
        name=name,
        workspace=workspace,
        body=body,
        input_file=input_file,
        input_data=input_data,
        output_format=output_format,
    )


@model_app.command("put")
@collect_warnings
@handle_errors
def put_model(
    ctx: typer.Context,
    trailing_uri: Annotated[str, typer.Argument()],
    name: Annotated[str | None, typer.Argument(help="(required)")] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    body: Annotated[str | None, typer.Option("--body", help="JSON string")] = None,
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
    """Proxy requests to model entity inference endpoints.

    All inference requests must resolve to a `VirtualModel`. The platform's provider
    reconciler auto-creates an implicit `autoprovisioned` VirtualModel for every
    served model entity (named after the entity, with `default_model_entity` set to
    the entity ref) so this is the typical case; operators can also create custom
    VirtualModels for routing, plugin chains, LoRA escape-hatches, etc. Requests for
    which no VirtualModel can be found return `404`.

        [bold red]Required fields:[/] name

        [green]Examples:[/]
        nemo inference gateway model put <trailing_uri> <name> --input-file config.json
        nemo inference gateway model put <trailing_uri> <name> --input-data '{"name": "value"}'
        echo '{"json": "data"}' | nemo inference gateway model put <trailing_uri> <name> --input-file -
        nemo inference gateway model put <trailing_uri> <name> --<option> "value"
    """
    _proxy_body_request(
        ctx,
        method="model_put",
        command_name="inference gateway model put",
        trailing_uri=trailing_uri,
        name=name,
        workspace=workspace,
        body=body,
        input_file=input_file,
        input_data=input_data,
        output_format=output_format,
    )


# ---------------------------------------------------------------------------
# openai v1 models
# ---------------------------------------------------------------------------


@openai_v1_models_app.command("get")
@collect_warnings
@handle_errors
def get_openai_models(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Retrieve information about a specific OpenAI-compatible model.

    Workspace is
    always taken from the URL path; name may be the VirtualModel name or
    workspace/name (workspace prefix is ignored). Resolves against routable
    VirtualModels, including custom ones, so this route agrees with the list route
    and the inference proxy."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    kwargs = build_kwargs(name=name, workspace=workspace)
    if handle_code_generation(InferenceGatewayClient, "get_openai_model", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(InferenceGatewayClient).get_openai_model(name=name, workspace=workspace)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@openai_v1_models_app.command("list")
@collect_warnings
@handle_errors
def list_openai_models(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
) -> None:
    """This endpoint lists the routable VirtualModels in the requested workspace and
    returns them in OpenAI's list models format. Each model ID is the VirtualModel
    identifier in format workspace/name. This includes both autoprovisioned
    VirtualModels (one per served model entity) and custom VirtualModels, keeping
    the catalog in agreement with the inference proxy, which also resolves
    VirtualModels scoped to the request workspace."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    validate_stream_output_format(resolved_output_format, stream)

    check_output_columns_with_format(columns, resolved_output_format)

    default_columns = [
        Column("id", None),
        Column("owned_by", None),
        Column("created", None),
    ]
    output_columns: str | list[Column] | None = columns
    if columns is None or str(columns).strip() == "default":
        output_columns = default_columns

    kwargs = build_kwargs(workspace=workspace)
    if handle_code_generation(InferenceGatewayClient, "list_openai_models", kwargs, resolved_output_format, state):
        return

    items = state.typed_client(InferenceGatewayClient).list_openai_models(workspace=workspace)

    format_output(
        items,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )

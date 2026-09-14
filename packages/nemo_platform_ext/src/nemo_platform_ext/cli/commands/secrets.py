# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated

import typer
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.response import NemoPaginatedResponse
from nemo_platform_plugin.client.types import OffsetPagination, OffsetPaginationMetadata
from nemo_platform_plugin.secrets.client import SecretsClient
from nemo_platform_plugin.secrets.types import (
    ListSecretsQueryParams,
    PlatformSecretCreateRequest,
    PlatformSecretResponse,
    PlatformSecretUpdateRequest,
)
from pydantic import SecretStr

from nemo_platform_ext.cli.core.code_generator import format_code_output
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import handle_errors
from nemo_platform_ext.cli.core.formatters import (
    Column,
    check_output_columns_with_format,
    format_csv,
    format_json,
    format_markdown_table,
    format_table,
    format_yaml,
    iter_json_lines,
    validate_stream_output_format,
)
from nemo_platform_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_platform_ext.cli.core.pagination import PaginationType, warn_if_more_pages
from nemo_platform_ext.cli.core.stdin_utils import resolve_secret_value, validate_required_fields
from nemo_platform_ext.cli.core.types import (
    EntityOutputFormatOption,
    ListOutputFormatOption,
    NoTruncateOption,
    OutputColumnsOption,
    StreamOutputOption,
)

app = create_typer_app(name="secrets", help="Manage secrets.")
admin_app = create_typer_app(name="admin", help="Manage admin")
app.add_typer(admin_app, name="admin")

_CodeArg = str | int | bool | None
_SECRET_VALUE_PLACEHOLDER = "<secret-value>"


@dataclass(frozen=True)
class _PaginationMetadata:
    page: int
    page_size: int
    current_page_size: int
    total_pages: int
    total_results: int

    @classmethod
    def from_offset_metadata(cls, metadata: OffsetPaginationMetadata) -> _PaginationMetadata:
        return cls(
            page=metadata["page"],
            page_size=metadata["page_size"],
            current_page_size=metadata["current_page_size"],
            total_pages=metadata["total_pages"],
            total_results=metadata["total_results"],
        )

    @classmethod
    def for_all_items(cls, items: list[PlatformSecretResponse], page_size: int | None) -> _PaginationMetadata:
        return cls(
            page=1,
            page_size=page_size or len(items),
            current_page_size=len(items),
            total_pages=1,
            total_results=len(items),
        )

    def model_dump(self, mode: str = "json") -> dict[str, int]:
        del mode
        return {
            "page": self.page,
            "page_size": self.page_size,
            "current_page_size": self.current_page_size,
            "total_pages": self.total_pages,
            "total_results": self.total_results,
        }


@dataclass(frozen=True)
class _SecretsPageResponse:
    data: list[PlatformSecretResponse]
    pagination: _PaginationMetadata
    sort: None = None

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        return {
            "data": [item.model_dump(mode=mode) for item in self.data],
            "sort": self.sort,
            "pagination": self.pagination.model_dump(mode=mode),
        }


def _secrets_client_from_state(state: CLIContext) -> SecretsClient:
    sdk = state.get_client()
    return client_from_platform(sdk, SecretsClient)


def _list_query_params(*, page: int | None, page_size: int | None) -> ListSecretsQueryParams | None:
    query_params: ListSecretsQueryParams = {}
    if page is not None:
        query_params["page"] = page
    if page_size is not None:
        query_params["page_size"] = page_size
    return query_params or None


def _update_body(*, description: str | None, secret_data: str | None) -> PlatformSecretUpdateRequest:
    if description is not None and secret_data is not None:
        return PlatformSecretUpdateRequest(description=description, value=SecretStr(secret_data))
    if description is not None:
        return PlatformSecretUpdateRequest(description=description)
    if secret_data is not None:
        return PlatformSecretUpdateRequest(value=SecretStr(secret_data))
    return PlatformSecretUpdateRequest()


def _resolve_direct_value_alias(*, value: str | None, data: str | None, command_name: str) -> str | None:
    if value is not None and data is not None:
        raise typer.BadParameter(
            "Pass either --value or legacy --data, not both.",
            param_hint=f"{command_name} --value",
        )
    return value if value is not None else data


def _source_secret_page(
    response: NemoPaginatedResponse[PlatformSecretResponse, OffsetPagination],
    *,
    all_pages: bool,
    page_size: int | None,
) -> _SecretsPageResponse:
    if all_pages:
        items = list(response.items())
        return _SecretsPageResponse(data=items, pagination=_PaginationMetadata.for_all_items(items, page_size))

    page = response.page()
    return _SecretsPageResponse(
        data=list(page.items),
        pagination=_PaginationMetadata.from_offset_metadata(page.metadata),
    )


def _format_python_literal(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def _handle_code_generation(
    method: str, args: dict[str, _CodeArg], output_format: str | None, state: CLIContext
) -> bool:
    if output_format != "code":
        return False

    generated_code = _generate_secrets_python_code(
        method=method,
        args=args,
        base_url=state.get_base_url("http://localhost:8080"),
    )
    typer.echo(format_code_output(generated_code, language="python"))
    return True


def _emit_secrets_output(
    data: object,
    *,
    is_list: bool = False,
    output_format: str | None = None,
    output_columns: str | list[Column] | None = None,
    no_truncate: bool | None = None,
    timestamp_format: str | None = None,
    stream: bool = False,
) -> None:
    from nemo_platform_ext.cli.core.table_config import resolve_and_validate_columns, validate_output_columns

    timestamp_format = timestamp_format or "iso"

    if stream:
        validate_stream_output_format(output_format, stream)
        for line in iter_json_lines(data, is_list=is_list):
            typer.echo(line)
        return

    if not is_list and output_format in {"table", "markdown", "csv"}:
        output_format = "json"

    truncate = not no_truncate
    if is_list and output_format in {"table", "markdown", "csv"}:
        if isinstance(output_columns, str):
            output_columns = validate_output_columns(output_columns)
        if output_columns is None:
            output_columns = "all"
        resolved_columns = resolve_and_validate_columns(output_columns, data)

        if output_format == "table":
            output = format_table(data, columns=resolved_columns, truncate=truncate, timestamp_format=timestamp_format)
            typer.echo(output)
            return
        if output_format == "markdown":
            output = format_markdown_table(
                data,
                columns=resolved_columns,
                truncate=truncate,
                timestamp_format=timestamp_format,
            )
            typer.echo(output)
            return
        output = format_csv(data, columns=resolved_columns, truncate=truncate, timestamp_format=timestamp_format)
        typer.echo(output, nl=False)
        return

    if output_format == "yaml":
        output = format_yaml(data, syntax_highlight=True, background=False)
    elif output_format == "raw":
        output = format_json(data, indent=None, syntax_highlight=False, background=False)
    else:
        output = format_json(data, indent=2, syntax_highlight=True, background=False)
    typer.echo(output)


def _generate_secrets_python_code(*, method: str, args: dict[str, _CodeArg], base_url: str | None) -> str:
    lines = [
        "from nemo_platform import NeMoPlatform",
        "from nemo_platform_plugin.client.adapter import client_from_platform",
        "from nemo_platform_plugin.secrets.client import SecretsClient",
        (
            "from nemo_platform_plugin.secrets.types import "
            "ListSecretsQueryParams, PlatformSecretCreateRequest, PlatformSecretUpdateRequest"
        ),
        "from pydantic import SecretStr",
        "",
        f"platform_client = NeMoPlatform(base_url={_format_python_literal(base_url)})"
        if base_url
        else "platform_client = NeMoPlatform()",
        "secrets = client_from_platform(platform_client, SecretsClient)",
        f"args = {_format_python_literal(args)}",
        "",
    ]
    lines.extend(_render_secrets_call(method, args))
    return "\n".join(lines)


def _render_secrets_call(method: str, args: dict[str, _CodeArg]) -> list[str]:
    if method == "access":
        return [
            "response = secrets.access_secret(",
            '    name=str(args["name"]),',
            '    workspace=args.get("workspace"),',
            ")",
            "print(response.data())",
        ]
    if method == "create":
        return [
            "response = secrets.create_secret(",
            '    workspace=args.get("workspace"),',
            "    body=PlatformSecretCreateRequest(",
            '        name=str(args["name"]),',
            '        value=SecretStr(str(args["value"])),',
            '        description=args.get("description"),',
            "    ),",
            ")",
            "print(response.data())",
        ]
    if method == "delete":
        return [
            "secrets.delete_secret(",
            '    name=str(args["name"]),',
            '    workspace=args.get("workspace"),',
            ").data()",
        ]
    if method == "list":
        lines = [
            "query_params: ListSecretsQueryParams = {}",
            'page = args.get("page")',
            "if page is not None:",
            '    query_params["page"] = int(page)',
            'page_size = args.get("page_size")',
            "if page_size is not None:",
            '    query_params["page_size"] = int(page_size)',
            "response = secrets.list_secrets(",
            '    workspace=args.get("workspace"),',
            "    query_params=query_params or None,",
            ")",
        ]
        if args.get("all_pages"):
            lines.extend(
                [
                    "for secret in response.items():",
                    "    print(secret)",
                ]
            )
        else:
            lines.extend(
                [
                    "page_result = response.page()",
                    "for secret in page_result.items:",
                    "    print(secret)",
                ]
            )
        return lines
    if method == "get":
        return [
            "response = secrets.get_secret(",
            '    name=str(args["name"]),',
            '    workspace=args.get("workspace"),',
            ")",
            "print(response.data())",
        ]
    if method == "update":
        return [
            'secret_value = args.get("value")',
            'description = args.get("description")',
            "if description is not None and secret_value is not None:",
            "    body = PlatformSecretUpdateRequest(description=str(description), value=SecretStr(str(secret_value)))",
            "elif description is not None:",
            "    body = PlatformSecretUpdateRequest(description=str(description))",
            "elif secret_value is not None:",
            "    body = PlatformSecretUpdateRequest(value=SecretStr(str(secret_value)))",
            "else:",
            "    body = PlatformSecretUpdateRequest()",
            "response = secrets.update_secret(",
            '    name=str(args["name"]),',
            '    workspace=args.get("workspace"),',
            "    body=body,",
            ")",
            "print(response.data())",
        ]
    if method == "rotate_encryption_keys":
        return [
            "response = secrets.rotate_encryption_keys()",
            "print(response.data())",
        ]
    raise ValueError(f"Unsupported secrets method for code generation: {method}")


@app.command("access")
@collect_warnings
@handle_errors
def access_secrets(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Access the value of a secret."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    args: dict[str, _CodeArg] = {"name": name, "workspace": workspace}
    if _handle_code_generation("access", args, resolved_output_format, state):
        return

    result = _secrets_client_from_state(state).access_secret(name=name, workspace=workspace).data()
    _emit_secrets_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("create")
@collect_warnings
@handle_errors
def create_secrets(
    ctx: typer.Context,
    name: Annotated[str | None, typer.Argument(help="The name of the secret to create")] = None,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    from_file: Annotated[
        str | None,
        typer.Option("--from-file", help="Path to file containing the secret value. Use '-' to read from stdin."),
    ] = None,
    value: Annotated[
        str | None,
        typer.Option("--value", help="Secret value directly. Use --from-file for large or sensitive input."),
    ] = None,
    data: Annotated[
        str | None,
        typer.Option("--data", hidden=True),
    ] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="An optional description of the secret")
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Create a new secret.

    [green]Examples:[/]
    [dim]# Pass secret value directly[/]
    nemo secrets create my-secret --value "abc123"
    [dim]# Read secret from a file[/]
    nemo secrets create my-secret --from-file ./secret.txt --description "API key for X"
    [dim]# Read secret from stdin[/]
    cat secret.txt | nemo secrets create my-secret --from-file -
    [dim]# Read secret from environment variable[/]
    echo "$API_KEY" | nemo secrets create my-secret --from-file -
    """
    input_payload: dict[str, _CodeArg] = {}
    if name is not None:
        input_payload["name"] = name
    if workspace is not None:
        input_payload["workspace"] = workspace
    if description is not None:
        input_payload["description"] = description

    validate_required_fields(
        input_payload,
        ["name"],
        "secrets create",
        {
            "name": "The name of the secret to create",
        },
    )
    assert name is not None

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    code_args: dict[str, _CodeArg] = dict(input_payload)
    code_args["value"] = _SECRET_VALUE_PLACEHOLDER
    if _handle_code_generation("create", code_args, resolved_output_format, state):
        return

    direct_value = _resolve_direct_value_alias(value=value, data=data, command_name="secrets create")
    secret_data = resolve_secret_value(from_file, direct_value, required=True, command_name="secrets create")
    assert secret_data is not None

    result = (
        _secrets_client_from_state(state)
        .create_secret(
            workspace=workspace,
            body=PlatformSecretCreateRequest(
                name=name,
                value=SecretStr(secret_data),
                description=description,
            ),
        )
        .data()
    )
    _emit_secrets_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("delete")
@collect_warnings
@handle_errors
def delete_secrets(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Delete a secret."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    args: dict[str, _CodeArg] = {"name": name, "workspace": workspace}
    if _handle_code_generation("delete", args, resolved_output_format, state):
        return

    _secrets_client_from_state(state).delete_secret(name=name, workspace=workspace).data()
    typer.echo("✓ Deleted successfully")


@app.command("list")
@collect_warnings
@handle_errors
def list_secrets(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    page: Annotated[int | None, typer.Option("--page", help="Page number.")] = None,
    page_size: Annotated[int | None, typer.Option("--page-size", help="Page size.")] = None,
    output_format: ListOutputFormatOption = None,
    no_truncate: NoTruncateOption = None,
    columns: OutputColumnsOption = None,
    stream: StreamOutputOption = False,
    all_pages: Annotated[bool, typer.Option("--all-pages", help="Fetch all pages")] = False,
) -> None:
    """List available secrets"""
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
    if output_columns is None or str(output_columns).strip() == "default":
        output_columns = default_columns

    args: dict[str, _CodeArg] = {"workspace": workspace, "page": page, "page_size": page_size, "all_pages": all_pages}
    if _handle_code_generation("list", args, resolved_output_format, state):
        return

    response = _secrets_client_from_state(state).list_secrets(
        workspace=workspace,
        query_params=_list_query_params(page=page, page_size=page_size),
    )
    result = _source_secret_page(response, all_pages=all_pages, page_size=page_size)
    _emit_secrets_output(
        result,
        is_list=True,
        output_format=resolved_output_format,
        output_columns=output_columns,
        no_truncate=state.get_no_truncate(no_truncate),
        timestamp_format=state.get_timestamp_format(),
        stream=stream,
    )
    if not all_pages:
        warn_if_more_pages(result, PaginationType.PAGE_NUMBER)


@app.command("get")
@collect_warnings
@handle_errors
def retrieve_secrets(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Retrieve a secret by its name."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    args: dict[str, _CodeArg] = {"name": name, "workspace": workspace}
    if _handle_code_generation("get", args, resolved_output_format, state):
        return

    result = _secrets_client_from_state(state).get_secret(name=name, workspace=workspace).data()
    _emit_secrets_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@app.command("update")
@collect_warnings
@handle_errors
def update_secrets(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument()],
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    from_file: Annotated[
        str | None,
        typer.Option("--from-file", help="Path to file containing the secret value. Use '-' to read from stdin."),
    ] = None,
    value: Annotated[
        str | None,
        typer.Option("--value", help="Secret value directly. Use --from-file for large or sensitive input."),
    ] = None,
    data: Annotated[
        str | None,
        typer.Option("--data", hidden=True),
    ] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="An optional description of the secret")
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Update a secret's metadata and/or value.

    [green]Examples:[/]
    [dim]# Update secret value directly[/]
    nemo secrets update my-secret --value "new-value"
    [dim]# Read secret from a file[/]
    nemo secrets update my-secret --from-file ./secret.txt --description "Updated!"
    [dim]# Read secret from stdin[/]
    cat secret.txt | nemo secrets update my-secret --from-file -
    [dim]# Read secret from environment variable[/]
    echo "$API_KEY" | nemo secrets update my-secret --from-file -
    """
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    code_args: dict[str, _CodeArg] = {"name": name, "workspace": workspace, "description": description}
    if from_file is not None or value is not None or data is not None:
        code_args["value"] = _SECRET_VALUE_PLACEHOLDER
    if _handle_code_generation("update", code_args, resolved_output_format, state):
        return

    direct_value = _resolve_direct_value_alias(value=value, data=data, command_name="secrets update")
    secret_data = resolve_secret_value(from_file, direct_value, required=False)

    result = (
        _secrets_client_from_state(state)
        .update_secret(
            name=name,
            workspace=workspace,
            body=_update_body(description=description, secret_data=secret_data),
        )
        .data()
    )
    _emit_secrets_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )


@admin_app.command("rotate-encryption-keys")
@collect_warnings
@handle_errors
def rotate_encryption_keys(
    ctx: typer.Context,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Rotate encryption keys for all platform secrets."""
    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)
    if _handle_code_generation("rotate_encryption_keys", {}, resolved_output_format, state):
        return

    result = _secrets_client_from_state(state).rotate_encryption_keys().data()
    _emit_secrets_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
import sys
import traceback
import typing

import click
import httpx
import typer
from pydantic import ValidationError

REMOTE_ERROR_EXIT_CODE = 3


class MissingRequiredFieldsError(Exception):
    """Raised when required fields are missing from CLI input."""

    def __init__(
        self,
        missing_fields: list[str],
        command_name: str,
        field_help: dict[str, str] | None = None,
    ):
        self.missing_fields = missing_fields
        self.command_name = command_name
        self.field_help = field_help or {}

        missing_str = ", ".join(f"--{f.replace('_', '-')}" for f in missing_fields)
        super().__init__(f"Missing required fields: {missing_str}")


class UnknownInputFieldsError(Exception):
    """Raised when ``--input-data`` / ``--input-file`` carries keys the request does not define."""

    def __init__(self, unknown_fields: list[str], command_name: str, known_fields: list[str]):
        self.unknown_fields = unknown_fields
        self.command_name = command_name
        self.known_fields = known_fields
        super().__init__(f"Unknown fields: {', '.join(unknown_fields)}")


class InvalidSearchPatternError(Exception):
    """Raised when --filter is given JSON that fails to parse or an otherwise unusable value."""

    def __init__(self, value: str, parse_error: str | None = None):
        self.value = value
        self.parse_error = parse_error
        if parse_error:
            super().__init__(f"Invalid filter JSON: {parse_error}. Input: {value!r}")
        else:
            super().__init__(f"Invalid filter value: {value!r}")


def _build_list_cmd(ctx: click.Context | None, prog: str) -> str | None:
    """Return the list command path if the parent group has a list subcommand, else None."""
    parent_ctx = ctx.parent if ctx else None
    if parent_ctx is None or "list" not in getattr(parent_ctx.command, "commands", {}):
        return None
    parts = []
    while parent_ctx and parent_ctx.parent:
        if parent_ctx.info_name:
            parts.insert(0, parent_ctx.info_name)
        parent_ctx = parent_ctx.parent
    return " ".join([prog, *parts, "list"])


_MISSING_WORKSPACE_MARKERS = (
    "Missing workspace argument",
    "Missing path parameter 'workspace'",
    "workspace must be provided",
)


def _is_missing_workspace_error(error: ValueError) -> bool:
    """Return whether *error* is the typed client reporting an unresolved workspace."""
    message = str(error)
    return any(marker in message for marker in _MISSING_WORKSPACE_MARKERS)


def _format_api_error(error: object) -> str:
    """Extract a clean error message from an API error."""
    if hasattr(error, "body") and error.body is not None:
        if isinstance(error.body, dict):
            body = typing.cast(dict[str, object], error.body)
            detail = body.get("detail")
            if detail:
                return str(detail)
            message = body.get("message")
            if message:
                return str(message)
    detail = getattr(error, "detail", None)
    if isinstance(detail, str) and detail:
        return detail
    message = getattr(error, "message", None)
    if isinstance(message, str) and message:
        return message
    return str(error)


def _format_api_request(error: object) -> str | None:
    request = getattr(error, "request", None)
    if request is None:
        response = getattr(error, "http_response", None)
        request = getattr(response, "request", None)
    method = getattr(request, "method", None)
    url = getattr(request, "url", None)
    if not isinstance(method, str) or not isinstance(url, (str, httpx.URL)):
        return None
    return f"{method} {url}"


def _format_api_target(error: object) -> str | None:
    request = getattr(error, "request", None)
    if request is None:
        response = getattr(error, "http_response", None)
        request = getattr(response, "request", None)
    url = getattr(request, "url", None)
    if isinstance(url, str):
        try:
            url = httpx.URL(url)
        except httpx.InvalidURL:
            return None
    if not isinstance(url, httpx.URL):
        return None
    path = url.path
    parts = [part for part in path.split("/") if part]
    if not parts:
        return None
    if len(parts) >= 3 and parts[0] == "apis":
        return f"{parts[1]} API route {path}"
    return f"route {path}"


def _print_api_request_context(console, error: object) -> None:
    request = _format_api_request(error)
    if request:
        console.print(f"[bold]Request:[/] {request}")
    target = _format_api_target(error)
    if target:
        console.print(f"[bold]Target:[/] {target}")


def _format_not_found_hint(ctx: click.Context | None, prog: str) -> str:
    list_cmd = _build_list_cmd(ctx, prog)
    if list_cmd:
        hint = f"Check the resource name/ID with [cyan]{list_cmd}[/]."
    else:
        hint = "Check the resource name/ID."
    hint += (
        " A 404 can mean the resource does not exist, the workspace/base URL is wrong, "
        "or the endpoint is not deployed on this cluster."
    )
    hint += f" Verify base-url/workspace with [cyan]{prog} config view[/]."
    return hint


def handle_errors(func):
    """Decorator to handle errors in CLI commands."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            ctx = None
            if args and hasattr(args[0], "parent") and hasattr(args[0], "info_name"):
                ctx = args[0]
            elif "ctx" in kwargs:
                ctx = kwargs["ctx"]
            handle_exception(e, ctx)

    return wrapper


def handle_exception(error: Exception, ctx: click.Context | None = None) -> None:
    """
    Handle all CLI errors and exit with appropriate error code.

    Exit codes: 0 = Success, 1 = Local/unexpected error, 2 = Usage error, 3 = Remote/API error
    """

    from rich.console import Console

    console = Console(stderr=True)

    import click.exceptions
    from nemo_platform_plugin.client import errors as plugin_errors

    prog = "nemo"

    if logging.getLogger("nemo_platform_ext.cli").isEnabledFor(logging.DEBUG):
        console.print(f"\n{'=' * 80}")
        console.print("DEBUG: Full traceback:")
        console.print("=" * 80)
        traceback.print_exc(file=sys.stderr)
        console.print("=" * 80 + "\n")

    if isinstance(error, click.UsageError):
        error_ctx = ctx or error.ctx

        # Handle NoArgsIsHelpError specially - display help to stderr and exit with code 2 since no command was provided
        # Note: the `NoArgsIsHelpError` was introduced in click 8.2.0, so we need to guard against it (see https://github.com/pallets/click/pull/1489)
        #   In earlier versions, this case wasn't an exception, so it's handled outside of this error_handler.
        if hasattr(click.exceptions, "NoArgsIsHelpError") and isinstance(error, click.exceptions.NoArgsIsHelpError):
            if error_ctx is not None:
                # Print help to stderr using click.echo which respects TTY/color settings
                click.echo(error_ctx.get_help(), err=True)
            raise SystemExit(2)

        if error_ctx is not None:
            parts: list[str] = []
            current_ctx: click.Context | None = error_ctx
            while current_ctx is not None:
                if current_ctx.info_name and current_ctx.parent is not None:
                    parts.insert(0, current_ctx.info_name)
                current_ctx = current_ctx.parent
            cmd_path = " ".join(parts)

            if cmd_path:
                console.print(f"[bold bright_green]Usage:[/] {prog} [GLOBAL OPTIONS] {cmd_path} [OPTIONS]")
                console.print(f"Try [cyan]{prog} {cmd_path} --help[/] for help.")
            else:
                console.print(f"[bold bright_green]Usage:[/] {prog} [GLOBAL OPTIONS] [OPTIONS]")
                console.print(f"Try [cyan]{prog} --help[/] for help.")
            console.print()

        if isinstance(error, click.MissingParameter) and error.param is not None and error_ctx is not None:
            param_help = getattr(error.param, "help", None)
            param_name = error.param.name.upper() if error.param.name else "VALUE"
            console.print("[bold red]Error:[/] Missing required argument:")
            if param_help:
                console.print(f"  [yellow]<{param_name}>[/]  {param_help}")
            else:
                console.print(f"  [yellow]<{param_name}>[/]")
        else:
            error_msg = str(error.format_message())
            console.print(f"[bold red]Error:[/] {error_msg}")
            if isinstance(error, click.NoSuchOption) and error.option_name == "--name" and error_ctx is not None:
                console.print(
                    f"[yellow]Hint:[/] [cyan]--name[/] is now a positional argument. Use: [cyan]{prog} {cmd_path} <name>[/]"
                )

        raise SystemExit(2)

    if isinstance(error, click.ClickException):
        console.print(f"[bold red]Error:[/] {error.message}")
        raise typer.Exit(code=error.exit_code)

    if isinstance(error, typer.Exit):
        # Re-raise typer.Exit with its original exit code (don't treat Exit(0) as error)
        raise error
    if isinstance(error, plugin_errors.AuthenticationError):
        console.print(f"[bold red]Authentication error:[/] ({error.status_code}) {_format_api_error(error)}")
        console.print(
            "[yellow]Hint:[/] Run [cyan]'nemo auth login'[/] or set the token manually "
            "with [cyan]nemo config set --access-token <token>[/], or use [cyan]NMP_ACCESS_TOKEN[/]."
        )
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.PermissionDeniedError):
        console.print(f"[bold red]Permission denied:[/] ({error.status_code}) {_format_api_error(error)}")
        console.print(
            "[yellow]Hint:[/] Your current credentials do not have access to perform this operation. "
            "Contact your administrator to request access."
        )
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.NotFoundError):
        console.print(f"[bold red]Not found:[/] ({error.status_code}) {_format_api_error(error)}")
        _print_api_request_context(console, error)
        console.print(f"[yellow]Hint:[/] {_format_not_found_hint(ctx, prog)}")
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.BadRequestError):
        console.print(f"[bold red]Bad request:[/] ({error.status_code}) {_format_api_error(error)}")
        console.print("[yellow]Hint:[/] Check your input values. Run with [cyan]--help[/] to see required options.")
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.ConflictError):
        console.print(f"[bold red]Conflict:[/] ({error.status_code}) {_format_api_error(error)}")
        console.print(
            "[yellow]Hint:[/] This can mean a resource with that name already exists (try a different name or delete the existing one), "
            "or a concurrent update conflicted (retry the operation)."
        )
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.UnprocessableEntityError):
        console.print(f"[bold red]Invalid input:[/] ({error.status_code}) {_format_api_error(error)}")
        console.print("[yellow]Hint:[/] Check your input values. Run with [cyan]--help[/] to see required options.")
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.RateLimitError):
        console.print(f"[bold red]Rate limit exceeded:[/] ({error.status_code}) {_format_api_error(error)}")
        console.print("[yellow]Hint:[/] Too many requests. Wait a moment and try again.")
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.InternalServerError):
        formatted = _format_api_error(error)
        console.print(f"[bold red]Server error:[/] ({error.status_code}) {formatted}")
        list_cmd = _build_list_cmd(ctx, prog) if ("404" in formatted or "not found" in formatted.lower()) else None
        if list_cmd:
            console.print(f"[yellow]Hint:[/] {_format_not_found_hint(ctx, prog)}")
        else:
            console.print("[yellow]Hint:[/] This is a server-side issue. Try again later or contact support.")
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.NemoTransportError) and isinstance(error.error, httpx.TimeoutException):
        console.print(f"[bold red]Timeout error:[/] {_format_api_error(error)}")
        _print_api_request_context(console, error)
        console.print("[yellow]Hint:[/] The request timed out. The server may be busy - try again later.")
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.NemoTransportError):
        console.print(f"[bold red]Connection error:[/] {_format_api_error(error)}")
        _print_api_request_context(console, error)
        console.print(
            "[yellow]Hint:[/] Check your network connection and verify that [cyan]base-url[/] you configured is correct."
        )
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.NemoHTTPError):
        console.print(f"[bold red]API error:[/] ({error.status_code}) {_format_api_error(error)}")
        _print_api_request_context(console, error)
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.NemoResponseValidationError):
        console.print(f"[bold red]API response error:[/] {_format_api_error(error)}")
        _print_api_request_context(console, error)
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, plugin_errors.NemoClientError):
        console.print(f"[bold red]API error:[/] {_format_api_error(error)}")
        _print_api_request_context(console, error)
        raise typer.Exit(code=REMOTE_ERROR_EXIT_CODE)
    elif isinstance(error, ValidationError):
        console.print("[bold red]Invalid input:[/]")
        for detail in error.errors():
            location = ".".join(str(part) for part in detail.get("loc", ())) or "input"
            console.print(f"  [yellow]{location}[/]  {detail.get('msg', 'invalid value')}")
        console.print("[yellow]Hint:[/] Check your input values. Run with [cyan]--help[/] to see required options.")
        raise typer.Exit(code=2)
    elif isinstance(error, ValueError) and _is_missing_workspace_error(error):
        console.print("[bold red]Missing workspace:[/] No workspace configured for this command.")
        console.print(
            f"[yellow]Hint:[/] Run [cyan]{prog} config set --workspace <name>[/] or use the [cyan]--workspace[/] option."
        )
        raise typer.Exit(code=2)
    elif isinstance(error, MissingRequiredFieldsError):
        # Show usage and help hint like missing argument errors
        console.print(f"[bold bright_green]Usage:[/] {prog} [GLOBAL OPTIONS] {error.command_name} [OPTIONS]")
        console.print(f"Try [cyan]{prog} {error.command_name} --help[/] for help.")
        console.print()

        # Format each missing field on its own line (like help output)
        console.print("[bold red]Error:[/] Missing required options:")
        for field in error.missing_fields:
            opt_name = f"--{field.replace('_', '-')}"
            metavar = field.upper().replace("_", "-")
            help_text = error.field_help.get(field, "")
            if help_text:
                console.print(f"  [cyan]{opt_name}[/] [yellow]<{metavar}>[/]  {help_text}")
            else:
                console.print(f"  [cyan]{opt_name}[/] [yellow]<{metavar}>[/]")
        console.print()
        console.print("[yellow]Hint:[/] Provide via CLI flags or [cyan]--input-file[/]/[cyan]--input-data[/].")
        raise typer.Exit(code=2)
    elif isinstance(error, UnknownInputFieldsError):
        console.print(f"[bold bright_green]Usage:[/] {prog} [GLOBAL OPTIONS] {error.command_name} [OPTIONS]")
        console.print(f"Try [cyan]{prog} {error.command_name} --help[/] for help.")
        console.print()
        console.print(f"[bold red]Error:[/] Unknown input fields: {', '.join(error.unknown_fields)}")
        console.print(f"[yellow]Hint:[/] Accepted fields: {', '.join(error.known_fields)}.")
        raise typer.Exit(code=2)
    elif isinstance(error, InvalidSearchPatternError):
        if error.parse_error:
            console.print(f"[bold red]Error:[/] Invalid filter JSON: {error.parse_error}")
            console.print(f"Your input: {error.value!r}")
        else:
            console.print(f"[bold red]Error:[/] Invalid filter value {error.value!r}.")
        console.print()
        console.print("Provide [cyan]--filter[/] as a text expression or JSON, or use [cyan]--filter.FIELD[/] options.")
        console.print()
        console.print("Examples:")
        console.print("  [cyan]--filter 'name~\"nemotron\"'[/]   (text: substring match)")
        console.print("  [cyan]--filter 'status:\"active\" AND amount>500'[/]   (text: combined)")
        console.print('  [cyan]--filter \'{"name": {"$like": "nemotron"}}\'[/]   (JSON)')
        console.print("  [cyan]--filter.name='nemotron'[/]   (per-field option)")
        console.print()
        console.print("[yellow]Hint:[/] Use [cyan]--help[/] to see available filter fields.")
        raise typer.Exit(code=2)
    else:
        console.print(f"[bold red]Unexpected error:[/] {error}")
        console.print("[yellow]Hint:[/] Run with [cyan]--verbose[/] for more details.")
        raise typer.Exit(code=1)

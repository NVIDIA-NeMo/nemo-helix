# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
from rich.console import Console
from rich.markup import escape


def format_http_status_error(exc: httpx.HTTPStatusError, *, action: str | None = None) -> str:
    """Format an HTTP status failure with request context for CLI output."""
    details = _status_error_details(exc, action=action)

    lines = [f"Error: {details.action}: {details.message}"]
    if details.request:
        lines.append(f"Request: {details.request}")
    if details.show_target and details.target:
        lines.append(f"Target: {details.target}")
    if details.hint:
        lines.append(f"Hint: {details.hint}")
    return "\n".join(lines)


def format_http_request_error(exc: httpx.RequestError, *, action: str | None = None) -> str:
    """Format a transport/request failure with request context for CLI output."""
    details = _request_error_details(exc, action=action)

    lines = [f"Error: {details.action}: {details.message}"]
    if details.request:
        lines.append(f"Request: {details.request}")
    if details.target:
        lines.append(f"Target: {details.target}")
    lines.append("Hint: Check your network connection and verify the base URL/workspace with `nemo config view`.")
    return "\n".join(lines)


def print_http_status_error(
    exc: httpx.HTTPStatusError,
    *,
    action: str | None = None,
    console: Console | None = None,
) -> None:
    """Print an HTTP status failure using the CLI's existing Rich error styling."""
    details = _status_error_details(exc, action=action)
    console = console or Console(stderr=True, soft_wrap=True)

    console.print(f"[bold red]Error:[/] {escape(details.action)}: {escape(details.message)}")
    if details.request:
        console.print(f"[bold]Request:[/] {escape(details.request)}")
    if details.show_target and details.target:
        console.print(f"[bold]Target:[/] {escape(details.target)}")
    if details.rich_hint:
        console.print(f"[yellow]Hint:[/] {details.rich_hint}")


def print_http_request_error(
    exc: httpx.RequestError,
    *,
    action: str | None = None,
    console: Console | None = None,
) -> None:
    """Print a transport/request failure using the CLI's existing Rich error styling."""
    details = _request_error_details(exc, action=action)
    console = console or Console(stderr=True, soft_wrap=True)

    console.print(f"[bold red]Error:[/] {escape(details.action)}: {escape(details.message)}")
    if details.request:
        console.print(f"[bold]Request:[/] {escape(details.request)}")
    if details.target:
        console.print(f"[bold]Target:[/] {escape(details.target)}")
    console.print(
        "[yellow]Hint:[/] Check your network connection and verify the base URL/workspace "
        "with [cyan]nemo config view[/]."
    )


@dataclass(frozen=True)
class _StatusErrorDetails:
    action: str
    status: int
    message: str
    request: str | None
    target: str | None
    show_target: bool
    hint: str | None
    rich_hint: str | None


@dataclass(frozen=True)
class _RequestErrorDetails:
    action: str
    message: str
    request: str | None
    target: str | None


def _status_error_details(exc: httpx.HTTPStatusError, *, action: str | None) -> _StatusErrorDetails:
    response = exc.response
    status = response.status_code
    status_line = f"HTTP {status}"
    if response.reason_phrase:
        status_line = f"{status_line} {response.reason_phrase}"

    request = _response_request(response)
    request_line, target = _request_context(request)
    response_message = _response_message(response)
    use_status_line = response_message is None or (status == 404 and _is_generic_not_found(response_message))
    is_generic_404 = status == 404 and use_status_line
    hint, rich_hint = _status_hint(status=status, is_generic_404=is_generic_404)
    message = status_line if use_status_line else f"{response_message} ({status_line})"
    return _StatusErrorDetails(
        action=f"{action} failed" if action else "HTTP request failed",
        status=status,
        message=message,
        request=request_line,
        target=target,
        show_target=is_generic_404,
        hint=hint,
        rich_hint=rich_hint,
    )


def _request_error_details(exc: httpx.RequestError, *, action: str | None) -> _RequestErrorDetails:
    request = _exception_request(exc)
    request_line, target = _request_context(request)
    message = str(exc) or exc.__class__.__name__
    return _RequestErrorDetails(
        action=f"{action} failed" if action else "HTTP request failed",
        message=message,
        request=request_line,
        target=target,
    )


def _response_request(response: httpx.Response) -> httpx.Request | None:
    try:
        return response.request
    except RuntimeError:
        return None


def _exception_request(exc: httpx.RequestError) -> httpx.Request | None:
    try:
        return exc.request
    except RuntimeError:
        return None


def _request_context(request: httpx.Request | None) -> tuple[str | None, str | None]:
    if request is None:
        return None, None
    return f"{request.method} {request.url}", _target_from_path(request.url.path)


def _response_text(response: httpx.Response) -> str:
    try:
        return response.text
    except httpx.ResponseNotRead:
        try:
            response.read()
            return response.text
        except httpx.HTTPError:
            return ""


def _response_message(response: httpx.Response) -> str | None:
    text = _response_text(response)
    if not text:
        return None
    try:
        body: Any = json.loads(text)
    except ValueError:
        return _clean_response_text(text)

    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        if detail:
            return json.dumps(detail, separators=(",", ":"))
        message = body.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        if message:
            return json.dumps(message, separators=(",", ":"))
    return _clean_response_text(text)


def _clean_response_text(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("<"):
        return None
    return stripped


def _is_generic_not_found(message: str) -> bool:
    normalized = " ".join(message.casefold().split())
    return normalized in {"not found", "404 not found"}


def _status_hint(*, status: int, is_generic_404: bool) -> tuple[str | None, str | None]:
    if status != 404:
        return None, None
    if is_generic_404:
        hint = (
            "This route may not be deployed on the selected cluster. "
            "Verify the base URL/workspace with `nemo config view`."
        )
        rich_hint = (
            "This route may not be deployed on the selected cluster. "
            "Verify the base URL/workspace with [cyan]nemo config view[/]."
        )
        return hint, rich_hint
    hint = "Check the resource name and workspace. Verify the base URL with `nemo config view`."
    rich_hint = "Check the resource name and workspace. Verify the base URL with [cyan]nemo config view[/]."
    return hint, rich_hint


def _target_from_path(path: str) -> str | None:
    parts = [part for part in path.split("/") if part]
    if not parts:
        return None
    if len(parts) >= 3 and parts[0] == "apis":
        return f"{parts[1]} API route {path}"
    return f"route {path}"

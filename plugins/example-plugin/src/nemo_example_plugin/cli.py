# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example plugin CLI commands — registered under ``nemo.cli``.

Thin wrapper that exposes the core business logic and middleware config CRUD
via the CLI.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx
import typer
from nemo_example_plugin.core import say_hello
from nemo_helix_plugin.cli import NemoCLI
from nemo_helix_plugin.cli_errors import print_http_request_error, print_http_status_error


def _request_json(method: str, url: str, *, json_body: dict | None = None) -> Any:
    try:
        response = httpx.request(method, url, json=json_body, timeout=30)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print_http_status_error(exc, action=f"{method} example API")
        raise typer.Exit(code=1) from exc
    except httpx.RequestError as exc:
        print_http_request_error(exc, action=f"{method} example API")
        raise typer.Exit(code=1) from exc

    if response.status_code == 204 or not response.content:
        return None
    return response.json()


class ExampleCLI(NemoCLI):
    """Exposes plugin commands as ``nemo example ...``."""

    name = "example"
    description = "Example plugin commands."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(help="Example plugin commands.")

        # ── hello ─────────────────────────────────────────────────────

        @app.command()
        def hello(name: str = typer.Option(default="world", help="Name to greet.")) -> None:
            """Greet a name."""
            typer.echo(say_hello(name))

        # ── middleware-configs subgroup ────────────────────────────────

        mw = typer.Typer(help="Manage ExampleMiddlewareConfig entities.")
        app.add_typer(mw, name="middleware-configs")

        @mw.command("create")
        def create_middleware_config(
            workspace: str = typer.Option(..., help="Workspace name."),
            name: str = typer.Option(..., help="Config name (unique within workspace)."),
            blocked_keywords: Optional[str] = typer.Option(None, help="Comma-separated list of keywords to block."),
            block_message: Optional[str] = typer.Option(
                None, help="Refusal message returned when a request is blocked."
            ),
            base_url: str = typer.Option("http://localhost:8000", envvar="NHX_BASE_URL"),
        ) -> None:
            """Create a new middleware config entity."""

            body: dict = {"name": name}
            if blocked_keywords:
                body["blocked_keywords"] = [k.strip() for k in blocked_keywords.split(",")]
            if block_message:
                body["block_message"] = block_message

            url = f"{base_url.rstrip('/')}/apis/example/v2/workspaces/{workspace}/middleware-configs"
            response = _request_json("POST", url, json_body=body)
            typer.echo(json.dumps(response, indent=2))

        @mw.command("list")
        def list_middleware_configs(
            workspace: str = typer.Option(..., help="Workspace name."),
            base_url: str = typer.Option("http://localhost:8000", envvar="NHX_BASE_URL"),
        ) -> None:
            """List middleware configs in a workspace."""

            url = f"{base_url.rstrip('/')}/apis/example/v2/workspaces/{workspace}/middleware-configs"
            response = _request_json("GET", url)
            typer.echo(json.dumps(response, indent=2))

        @mw.command("get")
        def get_middleware_config(
            workspace: str = typer.Option(..., help="Workspace name."),
            name: str = typer.Option(..., help="Config name."),
            base_url: str = typer.Option("http://localhost:8000", envvar="NHX_BASE_URL"),
        ) -> None:
            """Get a single middleware config by name."""

            url = f"{base_url.rstrip('/')}/apis/example/v2/workspaces/{workspace}/middleware-configs/{name}"
            response = _request_json("GET", url)
            typer.echo(json.dumps(response, indent=2))

        @mw.command("update")
        def update_middleware_config(
            workspace: str = typer.Option(..., help="Workspace name."),
            name: str = typer.Option(..., help="Config name."),
            blocked_keywords: Optional[str] = typer.Option(
                None, help="Comma-separated keywords (replaces existing list)."
            ),
            block_message: Optional[str] = typer.Option(None, help="New refusal message."),
            base_url: str = typer.Option("http://localhost:8000", envvar="NHX_BASE_URL"),
        ) -> None:
            """Partially update a middleware config (omitted fields unchanged)."""

            body: dict = {}
            if blocked_keywords is not None:
                body["blocked_keywords"] = [k.strip() for k in blocked_keywords.split(",")]
            if block_message is not None:
                body["block_message"] = block_message

            url = f"{base_url.rstrip('/')}/apis/example/v2/workspaces/{workspace}/middleware-configs/{name}"
            response = _request_json("PATCH", url, json_body=body)
            typer.echo(json.dumps(response, indent=2))

        @mw.command("delete")
        def delete_middleware_config(
            workspace: str = typer.Option(..., help="Workspace name."),
            name: str = typer.Option(..., help="Config name."),
            base_url: str = typer.Option("http://localhost:8000", envvar="NHX_BASE_URL"),
            yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
        ) -> None:
            """Delete a middleware config."""

            if not yes:
                typer.confirm(f"Delete middleware config '{workspace}/{name}'?", abort=True)

            url = f"{base_url.rstrip('/')}/apis/example/v2/workspaces/{workspace}/middleware-configs/{name}"
            _request_json("DELETE", url)
            typer.echo(f"Deleted '{workspace}/{name}'.")

        return app

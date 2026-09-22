# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI surface for the auditor plugin.

The CLI talks to the auditor plugin's own service routes (mounted at
``/apis/auditor/v2/workspaces/{workspace}/...``) rather than the generic
entity store. The plugin's POST/PUT handlers pydantic-validate the request
body before persistence, so invalid payloads come back as a structured 422
without ever reaching the entity store.

Two sub-groups are exposed:

- ``nemo auditor configs <create|list|get|update|delete>``
- ``nemo auditor targets <create|list|get|update|delete>``

``--data-file PATH`` and ``--data JSON`` are mutually-exclusive ways to supply
the request body for ``create`` / ``update``. The body shape follows the
plugin's ``CreateAuditConfigRequest`` / ``CreateAuditTargetRequest`` schemas
— there is no ``data`` envelope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import httpx
import typer
from nemo_platform_plugin.cli import NemoCLI
from nemo_platform_plugin.cli_errors import print_http_request_error, print_http_status_error
from nemo_platform_plugin.cli_state import resolve_workspace

_DEFAULT_BASE_URL = "http://localhost:8080"
_WORKSPACE_HELP = "Target workspace. Defaults to the active CLI context's workspace."


def _plugin_path(workspace: str, resource: str, name: str | None = None) -> str:
    base = f"/apis/auditor/v2/workspaces/{workspace}/{resource}"
    return f"{base}/{name}" if name else base


def _load_data(data_file: Path | None, data: str | None) -> dict[str, Any]:
    if data_file is not None and data is not None:
        typer.echo("Error: pass either --data-file or --data, not both.", err=True)
        raise typer.Exit(code=2)
    if data_file is not None:
        raw = data_file.read_text(encoding="utf-8")
    elif data is not None:
        raw = data
    else:
        typer.echo("Error: one of --data-file or --data is required.", err=True)
        raise typer.Exit(code=2)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: invalid JSON — {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not isinstance(parsed, dict):
        typer.echo("Error: data must be a JSON object.", err=True)
        raise typer.Exit(code=2)
    return parsed


def _api_request(method: str, base_url: str, path: str, *, json_body: dict | None = None) -> Any:
    url = base_url.rstrip("/") + path
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.request(method, url, json=json_body)
            resp.raise_for_status()
            if resp.status_code == 204 or not resp.content:
                return None
            return resp.json()
    except httpx.HTTPStatusError as exc:
        print_http_status_error(exc, action=f"{method} auditor API")
        raise typer.Exit(code=1) from exc
    except httpx.RequestError as exc:
        print_http_request_error(exc, action=f"{method} auditor API")
        raise typer.Exit(code=1) from exc


def _build_crud_app(resource: str, singular: str, help_text: str) -> typer.Typer:
    """Build a Typer sub-app with the standard 5 CRUD commands for one resource path.

    ``resource`` is the URL path segment under ``/apis/auditor/v2/workspaces/{ws}/``
    (``"configs"`` or ``"targets"``). ``singular`` is the user-facing noun used
    in argument help text and confirmation messages.
    """
    sub = typer.Typer(name=resource, help=help_text, no_args_is_help=True)

    @sub.command("create")
    def create(
        name: str = typer.Argument(..., help=f"{singular.capitalize()} name."),
        data_file: Path | None = typer.Option(
            None,
            "--data-file",
            "-f",
            help=f"JSON file with the {singular} body (without name/workspace).",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
        data: str | None = typer.Option(None, "--data", "-d", help=f"Inline JSON body for the {singular}."),
        workspace: str | None = typer.Option(
            None,
            "--workspace",
            "-w",
            help=_WORKSPACE_HELP,
            show_default="active context workspace",
        ),
        base_url: str = typer.Option(_DEFAULT_BASE_URL, "--base-url", envvar="NMP_BASE_URL"),
    ) -> None:
        workspace = resolve_workspace(workspace)
        body = _load_data(data_file, data)
        payload = {"name": name, **body}
        resp = _api_request("POST", base_url, _plugin_path(workspace, resource), json_body=payload)
        typer.echo(json.dumps(resp, indent=2))

    @sub.command("list")
    def list_cmd(
        workspace: str | None = typer.Option(
            None,
            "--workspace",
            "-w",
            help=_WORKSPACE_HELP,
            show_default="active context workspace",
        ),
        base_url: str = typer.Option(_DEFAULT_BASE_URL, "--base-url", envvar="NMP_BASE_URL"),
    ) -> None:
        workspace = resolve_workspace(workspace)
        resp = _api_request("GET", base_url, _plugin_path(workspace, resource))
        typer.echo(json.dumps(resp, indent=2))

    @sub.command("get")
    def get(
        name: str = typer.Argument(..., help=f"{singular.capitalize()} name."),
        workspace: str | None = typer.Option(
            None,
            "--workspace",
            "-w",
            help=_WORKSPACE_HELP,
            show_default="active context workspace",
        ),
        base_url: str = typer.Option(_DEFAULT_BASE_URL, "--base-url", envvar="NMP_BASE_URL"),
    ) -> None:
        workspace = resolve_workspace(workspace)
        resp = _api_request("GET", base_url, _plugin_path(workspace, resource, name))
        typer.echo(json.dumps(resp, indent=2))

    @sub.command("update")
    def update(
        name: str = typer.Argument(..., help=f"{singular.capitalize()} name."),
        data_file: Path | None = typer.Option(
            None,
            "--data-file",
            "-f",
            help=f"JSON file with the new {singular} body.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
        data: str | None = typer.Option(None, "--data", "-d", help="Inline JSON body."),
        workspace: str | None = typer.Option(
            None,
            "--workspace",
            "-w",
            help=_WORKSPACE_HELP,
            show_default="active context workspace",
        ),
        base_url: str = typer.Option(_DEFAULT_BASE_URL, "--base-url", envvar="NMP_BASE_URL"),
    ) -> None:
        workspace = resolve_workspace(workspace)
        body = _load_data(data_file, data)
        resp = _api_request("PUT", base_url, _plugin_path(workspace, resource, name), json_body=body)
        typer.echo(json.dumps(resp, indent=2))

    @sub.command("delete")
    def delete(
        name: str = typer.Argument(..., help=f"{singular.capitalize()} name."),
        workspace: str | None = typer.Option(
            None,
            "--workspace",
            "-w",
            help=_WORKSPACE_HELP,
            show_default="active context workspace",
        ),
        base_url: str = typer.Option(_DEFAULT_BASE_URL, "--base-url", envvar="NMP_BASE_URL"),
    ) -> None:
        workspace = resolve_workspace(workspace)
        _api_request("DELETE", base_url, _plugin_path(workspace, resource, name))
        typer.echo(f"{singular.capitalize()} '{name}' deleted.")

    return sub


class AuditorPluginCLI(NemoCLI):
    """CLI surface for the auditor plugin."""

    name: ClassVar[str] = "auditor"
    description: ClassVar[str] = "Auditor plugin commands."

    def get_cli(self) -> typer.Typer:
        app = typer.Typer(
            name=self.name,
            help=self.description,
            no_args_is_help=True,
        )

        @app.command("info")
        def info() -> None:
            """Print the current plugin status."""
            typer.echo(
                json.dumps(
                    {
                        "plugin": self.name,
                        "status": "ready",
                        "service": "/apis/auditor/v1/healthz",
                        "jobs": ["auditor.audit"],
                        "sdk": "audit",
                    },
                    indent=2,
                )
            )

        app.add_typer(_build_crud_app("configs", "config", "Manage audit configurations."))
        app.add_typer(_build_crud_app("targets", "target", "Manage audit targets."))

        return app

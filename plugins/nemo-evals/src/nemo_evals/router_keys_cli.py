# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo evals router-keys``: mint the closed credential a router calls models back through the gateway with."""

from __future__ import annotations

import json
from collections.abc import Callable

import typer
from nemo_helix_plugin.client.client import NemoClient

DAY_SEC = 24 * 60 * 60


def add_router_keys_commands(app: typer.Typer, client_factory: Callable[[], NemoClient]) -> None:
    """Register ``router-keys create`` under ``app``."""
    router_keys = typer.Typer(
        name="router-keys",
        help="Closed credentials for routers, such as Switchyard, that call models back through the gateway.",
        no_args_is_help=True,
    )
    app.add_typer(router_keys)

    @router_keys.command("create")
    def create(
        secret: str = typer.Option(
            ..., "--secret", help="Platform secret to store the key in; give it to the router's ModelProvider."
        ),
        models: list[str] = typer.Option(
            ..., "--model", help="A closed-book VirtualModel the router may call, workspace/name. Repeat for each."
        ),
        expires_in_days: int = typer.Option(30, "--expires-in-days", min=1, help="The key's lifetime in days."),
        workspace: str | None = typer.Option(
            None, "--workspace", help="Workspace for the secret; default: the configured workspace."
        ),
    ) -> None:
        """Mint a router key that reaches only the given closed-book models, and store it in a platform secret."""
        client = client_factory()
        workspace = client.resolve_workspace(workspace)
        url = f"{client.base_url.rstrip('/')}/apis/evals/v2/workspaces/{workspace}/router-keys"
        body = {"secret": secret, "models": models, "expires_in_seconds": expires_in_days * DAY_SEC}
        response = client.http_client.post(url, json=body, headers=client.request_headers(None, url=url))
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text
            typer.echo(f"Error: creating the router key failed ({response.status_code}): {detail}", err=True)
            raise typer.Exit(code=1)
        typer.echo(json.dumps(response.json(), indent=2))

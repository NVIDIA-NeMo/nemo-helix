# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file is auto-generated
from __future__ import annotations

from importlib import import_module as _importlib_import_module
from typing import Annotated

import typer

from nemo_helix_ext.cli.core.context import CLIContext
from nemo_helix_ext.cli.core.errors import handle_errors
from nemo_helix_ext.cli.core.help_formatter import create_typer_app

_cli_child_deployment_configs = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.deployment_configs")
_cli_child_deployments = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.deployments")
_cli_child_gateway = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.gateway")
_cli_child_models = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.models")
_cli_child_prompts = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.prompts")
_cli_child_providers = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.providers")
_cli_child_virtual_models = _importlib_import_module("nemo_helix_ext.cli.commands.api.inference.virtual_models")

app = create_typer_app(name="inference", help="Manage inference")

app.add_typer(_cli_child_deployment_configs.app, name="deployment-configs")
app.add_typer(_cli_child_deployments.app, name="deployments")
app.add_typer(_cli_child_gateway.app, name="gateway")
app.add_typer(_cli_child_models.app, name="models")
app.add_typer(_cli_child_prompts.app, name="prompts")
app.add_typer(_cli_child_providers.app, name="providers")
app.add_typer(_cli_child_virtual_models.app, name="virtual-models")


@app.command("get-url")
@handle_errors
def get_url(
    ctx: typer.Context,
    workspace: Annotated[
        str | None,
        typer.Option("--workspace", help="Workspace to scope the URL to. Defaults to the CLI context workspace."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Print the provider proxy route for this provider name."),
    ] = None,
    virtual_model: Annotated[
        str | None,
        typer.Option("--virtual-model", help="Print the model entity proxy route for this virtual model name."),
    ] = None,
) -> None:
    """Print the OpenAI-compatible base URL for the inference gateway.

    [green]Examples:[/]
    [dim]# Workspace-scoped OpenAI base URL (use as OpenAI client's base_url)[/]
    nemo inference get-url
    [dim]# Provider proxy route (append your own trailing path, e.g. /v1/chat/completions)[/]
    nemo inference get-url --provider llama-3-2-1b-deployment
    [dim]# Model-entity proxy route[/]
    nemo inference get-url --virtual-model meta-llama-3-2-1b-instruct
    """
    if provider is not None and virtual_model is not None:
        raise typer.BadParameter("--provider and --virtual-model are mutually exclusive")

    state: CLIContext = ctx.obj
    client = state.get_client()
    ws = workspace if workspace is not None else client._get_workspace_path_param()

    if provider is not None:
        provider_obj = client.inference.providers.retrieve(provider, workspace=ws)
        url = client.models.get_provider_route_openai_url(provider_obj).removesuffix("/v1")
    elif virtual_model is not None:
        entity = client.models.retrieve(virtual_model, workspace=ws)
        url = client.models.get_model_entity_route_openai_url(entity).removesuffix("/v1")
    else:
        url = client.models.get_openai_route_base_url(workspace=ws)

    typer.echo(url)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo inference`` command group, backed by the typed Models, VirtualModels and Inference Gateway clients."""

from __future__ import annotations

from typing import Annotated

import typer
from nemo_platform_plugin.models.client import ModelsClient

from nemo_platform_ext.cli.commands.inference import (
    deployment_configs,
    deployments,
    gateway,
    models,
    prompts,
    providers,
    virtual_models,
)
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import handle_errors
from nemo_platform_ext.cli.core.help_formatter import create_typer_app

app = create_typer_app(name="inference", help="Inference operations.")

app.add_typer(deployment_configs.app, name="deployment-configs")
app.add_typer(deployments.app, name="deployments")
app.add_typer(gateway.app, name="gateway")
app.add_typer(models.app, name="models")
app.add_typer(prompts.app, name="prompts")
app.add_typer(providers.app, name="providers")
app.add_typer(virtual_models.app, name="virtual-models")


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
    models_client = state.typed_client(ModelsClient)
    ws = models_client.require_workspace(workspace)

    if provider is not None:
        provider_obj = models_client.get_provider(name=provider, workspace=ws).data()
        url = models_client.get_provider_route_openai_url(provider_obj).removesuffix("/v1")
    elif virtual_model is not None:
        entity = models_client.get_model(name=virtual_model, workspace=ws).data()
        url = models_client.get_model_entity_route_openai_url(entity).removesuffix("/v1")
    else:
        url = models_client.get_openai_route_base_url(workspace=ws)

    typer.echo(url)

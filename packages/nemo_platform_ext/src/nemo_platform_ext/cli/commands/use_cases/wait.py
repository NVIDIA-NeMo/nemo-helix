# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI commands for waiting on deployment and provider status."""

from __future__ import annotations

import sys
from typing import Annotated, Literal

import typer

from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import handle_errors
from nemo_platform_ext.cli.core.help_formatter import create_typer_app
from nemo_platform_ext.cli.core.waiters import wait_for_gateway, wait_for_inference_deployment

app = create_typer_app(name="wait", help="Wait for resources to reach a desired status.")

inference_app = create_typer_app(name="inference", help="Wait for inference resources")
app.add_typer(inference_app, name="inference")


@inference_app.command("deployment")
@handle_errors
def wait_deployment(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Name of the deployment to wait for")],
    workspace: Annotated[str | None, typer.Option("--workspace", help="Workspace name")] = None,
    status: Annotated[
        Literal["READY", "DELETED", "PENDING", "ERROR"],
        typer.Option(
            "--status",
            "-s",
            help="Desired status to wait for",
        ),
    ] = "READY",
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            "-t",
            min=1,
            help="Maximum time to wait in seconds",
        ),
    ] = 1200,
    check_gateway: Annotated[
        bool,
        typer.Option(
            "--check-gateway/--no-check-gateway",
            help="When waiting for READY, also verify gateway can route to the provider",
        ),
    ] = True,
    poll_interval: Annotated[
        int,
        typer.Option(
            "--poll-interval",
            min=1,
            help="Seconds between status checks",
        ),
    ] = 3,
) -> None:
    """
    Wait for a deployment to reach a desired status.

    Polls the deployment status until it reaches the desired state or times out.
    For READY status, optionally verifies the gateway can route to the provider.
    For DELETED status, waits for the resource to be fully garbage collected.

    Exit codes:
      0: Desired status reached
      1: Timeout or error

    [green]Examples:[/]
      nemo wait inference deployment my-deployment --status READY
      nemo wait inference deployment my-deployment --status READY --timeout 600 --no-check-gateway
      nemo wait inference deployment my-deployment --status DELETED --timeout 90
    """
    state: CLIContext = ctx.obj
    client = state.get_client()
    success = wait_for_inference_deployment(
        client,
        name,
        workspace=workspace,
        status=status,
        timeout=timeout,
        poll_interval=poll_interval,
        check_gateway=check_gateway,
    )
    sys.exit(0 if success else 1)


@inference_app.command("provider")
@handle_errors
def wait_provider(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Name of the provider to wait for")],
    workspace: Annotated[str | None, typer.Option("--workspace", help="Workspace name")] = None,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            "-t",
            min=1,
            help="Maximum time to wait in seconds",
        ),
    ] = 60,
    poll_interval: Annotated[
        int,
        typer.Option(
            "--poll-interval",
            min=1,
            help="Seconds between status checks",
        ),
    ] = 1,
) -> None:
    """
    Wait for the inference gateway to be ready to route to a provider.

    Polls the gateway's ready endpoint until it can route requests to the
    specified provider. This is useful after creating a deployment to ensure
    the gateway has refreshed its cache.

    Exit codes:
      0: Gateway is ready
      1: Timeout

    [green]Examples:[/]
      nemo wait inference provider my-deployment
      nemo wait inference provider my-deployment --timeout 120
    """
    state: CLIContext = ctx.obj
    client = state.get_client()

    workspace = client.require_workspace(workspace)

    success = wait_for_gateway(client, name, workspace, timeout, poll_interval)
    sys.exit(0 if success else 1)

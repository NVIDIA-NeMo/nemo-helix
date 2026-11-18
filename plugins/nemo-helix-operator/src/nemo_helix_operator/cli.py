# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI commands: ``nemo helix-operator upload | build | status``."""

import json
import time
from pathlib import Path

import httpx
import typer
from nemo_helix_operator.config import HelixOperatorConfig
from nemo_helix_operator.schemas import BuildSource, CreateBuildRequest, HelixBuild
from nemo_helix_operator.source import pack
from nemo_helix_plugin.cli import NemoCLI, create_typer_app
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_options import WorkspaceOption
from nemo_helix_plugin.cli_state import cli_state, resolve_base_url, resolve_cli_workspace
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest

FilesetOption = typer.Option("helix-src", "--fileset", help="Fileset holding the uploaded checkout.")
_WAIT_SECONDS = 20 * 60


def _call(ctx: typer.Context, method: str, workspace: str | None, path: str, body: object = None) -> HelixBuild:
    base_url = HelixOperatorConfig.get().operator_url or resolve_base_url()
    url = f"{base_url.rstrip('/')}/apis/helix-operator/v2/workspaces/{resolve_cli_workspace(ctx, workspace)}{path}"
    # ponytail: plain httpx with the CLI client's default headers; a bearer-token provider isn't forwarded,
    # which is fine only while platform auth is off.
    response = httpx.request(method, url, json=body, headers=cli_state(ctx).get_client().default_headers, timeout=60)
    if response.is_error:
        typer.echo(f"HTTP {response.status_code}: {response.text}", err=True)
        raise typer.Exit(1)
    return HelixBuild.model_validate_json(response.content)


class HelixOperatorCLI(NemoCLI):
    name = "helix-operator"
    description = "Build nhx-api images from a local NeMo Helix checkout."

    def get_cli(self) -> typer.Typer:
        app = create_typer_app(help=self.description)

        @app.command()
        @handle_errors
        def upload(
            ctx: typer.Context,
            directory: Path = typer.Argument(Path("."), help="The Helix checkout."),
            fileset: str = FilesetOption,
            base_image: str | None = typer.Option(None, "--base-image", help="The generated Dockerfile's FROM."),
            workspace: WorkspaceOption = None,
        ) -> None:
            """Pack the checkout's git-visible files and upload them to a fileset."""
            workspace = resolve_cli_workspace(ctx, workspace)
            archive, metadata = pack(directory, base_image or HelixOperatorConfig.get().base_image)
            files = cli_state(ctx).typed_client(FilesClient)
            files.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=fileset), exist_ok=True)
            files.upload_file(workspace=workspace, name=fileset, path="source.tar.gz", content=archive)
            files.upload_file(
                workspace=workspace, name=fileset, path="source.json", content=json.dumps(metadata).encode()
            )
            typer.echo(json.dumps({"fileset": fileset, "bytes": len(archive), **metadata}))

        @app.command()
        @handle_errors
        def build(
            ctx: typer.Context,
            fileset: str = FilesetOption,
            archive: str = typer.Option("source.tar.gz", "--archive", help="The archive in the fileset."),
            wait: bool = typer.Option(False, "--wait", help="Poll until the build is ready or failed (20 minutes)."),
            workspace: WorkspaceOption = None,
        ) -> None:
            """Build an uploaded checkout."""
            body = CreateBuildRequest(source=BuildSource(fileset=fileset, archive=archive)).model_dump()
            result = _call(ctx, "POST", workspace, "/builds", body)
            deadline = time.monotonic() + _WAIT_SECONDS
            while wait and result.status not in ("ready", "failed") and time.monotonic() < deadline:
                time.sleep(10)
                result = _call(ctx, "GET", workspace, f"/builds/{result.name}")
            typer.echo(result.model_dump_json(indent=2))
            if wait and result.status != "ready":
                raise typer.Exit(1)

        @app.command()
        @handle_errors
        def status(
            ctx: typer.Context,
            name: str = typer.Argument(..., help="The build name `build` printed."),
            workspace: WorkspaceOption = None,
        ) -> None:
            """Read a build."""
            typer.echo(_call(ctx, "GET", workspace, f"/builds/{name}").model_dump_json(indent=2))

        return app

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI commands: ``nemo helix-operator upload | build | status``."""

import json
import time
from pathlib import Path
from typing import TypeVar

import httpx
import typer
from nemo_helix_operator.config import HelixOperatorConfig
from nemo_helix_operator.deploy import CreateDeploymentRequest, HelixDeployment
from nemo_helix_operator.schemas import BuildSource, CreateBuildRequest, HelixBuild
from nemo_helix_operator.source import pack
from nemo_helix_operator.testrun import CreateTestRunRequest, HelixTestRun
from nemo_helix_plugin.cli import NemoCLI, create_typer_app
from nemo_helix_plugin.cli_error_handling import handle_errors
from nemo_helix_plugin.cli_options import WorkspaceOption
from nemo_helix_plugin.cli_state import cli_state, resolve_base_url, resolve_cli_workspace
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from pydantic import BaseModel

FilesetOption = typer.Option("helix-src", "--fileset", help="Fileset holding the uploaded checkout.")
_WAIT_SECONDS = 20 * 60
T = TypeVar("T", bound=BaseModel)


def _call(ctx: typer.Context, model: type[T], method: str, workspace: str | None, path: str, body: object = None) -> T:
    base_url = HelixOperatorConfig.get().operator_url or resolve_base_url()
    url = f"{base_url.rstrip('/')}/apis/helix-operator/v2/workspaces/{resolve_cli_workspace(ctx, workspace)}{path}"
    # ponytail: plain httpx with the CLI client's default headers; a bearer-token provider isn't forwarded,
    # which is fine only while platform auth is off.
    response = httpx.request(method, url, json=body, headers=cli_state(ctx).get_client().default_headers, timeout=60)
    if response.is_error:
        typer.echo(f"HTTP {response.status_code}: {response.text}", err=True)
        raise typer.Exit(1)
    return model.model_validate_json(response.content)


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
            result = _call(ctx, HelixBuild, "POST", workspace, "/builds", body)
            deadline = time.monotonic() + _WAIT_SECONDS
            while wait and result.status not in ("ready", "failed") and time.monotonic() < deadline:
                time.sleep(10)
                result = _call(ctx, HelixBuild, "GET", workspace, f"/builds/{result.name}")
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
            typer.echo(_call(ctx, HelixBuild, "GET", workspace, f"/builds/{name}").model_dump_json(indent=2))

        @app.command()
        @handle_errors
        def deploy(
            ctx: typer.Context,
            api_image: str = typer.Option(..., "--api-image", help="`<repository>:<tag>` for the api deployment."),
            core_image: str | None = typer.Option(None, "--core-image", help="For core; defaults to --api-image."),
            rollback_on_failure: bool = typer.Option(
                False, "--rollback-on-failure", help="Roll the release back if the upgrade fails."
            ),
            fileset: str = FilesetOption,
            archive: str = typer.Option("source.tar.gz", "--archive", help="The archive whose Helm chart is used."),
            wait: bool = typer.Option(False, "--wait", help="Poll until the upgrade finishes (20 minutes)."),
            workspace: WorkspaceOption = None,
        ) -> None:
            """Upgrade the Helix release with these images and the uploaded checkout's chart."""
            body = CreateDeploymentRequest(
                source=BuildSource(fileset=fileset, archive=archive),
                api_image=api_image,
                core_image=core_image,
                rollback_on_failure=rollback_on_failure,
            ).model_dump()
            result = _call(ctx, HelixDeployment, "POST", workspace, "/deployments", body)
            deadline = time.monotonic() + _WAIT_SECONDS
            while wait and result.status.startswith("pending") and time.monotonic() < deadline:
                time.sleep(10)
                result = _call(ctx, HelixDeployment, "GET", workspace, f"/deployments/{result.revision}")
            typer.echo(result.model_dump_json(indent=2))
            if wait and result.status != "deployed":
                raise typer.Exit(1)

        @app.command()
        @handle_errors
        def deployment(
            ctx: typer.Context,
            revision: int = typer.Argument(..., help="The release revision `deploy` printed."),
            workspace: WorkspaceOption = None,
        ) -> None:
            """Read a revision of the Helix release."""
            result = _call(ctx, HelixDeployment, "GET", workspace, f"/deployments/{revision}")
            typer.echo(result.model_dump_json(indent=2))

        @app.command()
        @handle_errors
        def test(
            ctx: typer.Context,
            fileset: str = FilesetOption,
            archive: str = typer.Option("source.tar.gz", "--archive", help="The archive whose e2e tests run."),
            wait: bool = typer.Option(False, "--wait", help="Poll until the run finishes (20 minutes)."),
            workspace: WorkspaceOption = None,
        ) -> None:
            """Run the uploaded checkout's e2e hello-world test against the Helix API, in-cluster."""
            body = CreateTestRunRequest(source=BuildSource(fileset=fileset, archive=archive)).model_dump()
            result = _call(ctx, HelixTestRun, "POST", workspace, "/test-runs", body)
            deadline = time.monotonic() + _WAIT_SECONDS
            while wait and result.status == "running" and time.monotonic() < deadline:
                time.sleep(10)
                result = _call(ctx, HelixTestRun, "GET", workspace, f"/test-runs/{result.id}")
            typer.echo(result.model_dump_json(indent=2))
            if wait and result.status != "passed":
                raise typer.Exit(1)

        @app.command("test-run")
        @handle_errors
        def test_run(
            ctx: typer.Context,
            run_id: str = typer.Argument(..., help="The run id `test` printed."),
            workspace: WorkspaceOption = None,
        ) -> None:
            """Read a test run."""
            typer.echo(_call(ctx, HelixTestRun, "GET", workspace, f"/test-runs/{run_id}").model_dump_json(indent=2))

        return app

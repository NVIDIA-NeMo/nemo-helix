# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create, trigger, watch and download named workflow demos."""

import argparse
import sys
from enum import Enum
from pathlib import Path
from subprocess import run
from uuid import uuid4

from filesets import transfer
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoClientError, NemoHTTPError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest, HelixJobResponse, JobStatusDetailsUpdate
from nemo_helix_plugin.jobs.workflows import WorkflowInvalidation
from nemo_helix_plugin.secrets.client import SecretsClient
from pydantic import BaseModel, ConfigDict

from demo.workflow_specs import layered_workflow, montage_workflow

ROOT = Path(__file__).resolve().parents[1]


class Workflow(str, Enum):
    LAYERS = "layers"
    MONTAGE = "montage"


class DemoSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow: Workflow
    filesets: dict[str, str]

    @property
    def output_roles(self) -> tuple[str, ...]:
        return ("output",) if self.workflow == Workflow.LAYERS else ("edges", "montage", "captions")


def roles_for(workflow: Workflow) -> tuple[str, ...]:
    return ("input", "config", "output") if workflow == Workflow.LAYERS else ("input", "edges", "montage", "captions")


def job_request(
    workflow: Workflow, name: str, bindings: dict[str, str], *, profile: str, image: str, secret: str | None
) -> CreateHelixJobRequest:
    if workflow == Workflow.LAYERS:
        spec = layered_workflow(bindings, image=image, profile=profile)
    else:
        if not secret:
            raise ValueError("The montage workflow requires --secret naming an existing NVIDIA API key Secret")
        spec = montage_workflow(
            bindings["input"],
            bindings["edges"],
            bindings["montage"],
            bindings["captions"],
            profile=profile,
            image=image,
            secret=secret,
        )
    metadata = DemoSpec(workflow=workflow, filesets=bindings)
    return CreateHelixJobRequest(
        name=name, source="workflow-demo", spec={"demo": metadata.model_dump(mode="json")}, platform_spec=spec
    )


def create_run(
    client: NemoClient, workflow: Workflow, name: str, *, workspace: str, profile: str, image: str, secret: str | None
) -> HelixJobResponse:
    roles = roles_for(workflow)
    job_request(
        workflow,
        name,
        {role: f"fileset-placeholder-{role}" for role in roles},
        profile=profile,
        image=image,
        secret=secret,
    )
    requests = {role: CreateFilesetRequest(name=f"{name}-{role}") for role in roles}
    files_client = FilesClient.from_client(client)
    jobs_client = JobsClient.from_client(client)
    try:
        jobs_client.get_job(workspace=workspace, name=name).data()
    except NemoHTTPError as exc:
        if exc.status_code != 404:
            raise
    else:
        raise ValueError(f"Job {name!r} already exists; choose a new name")
    for request in requests.values():
        try:
            files_client.get_fileset(workspace=workspace, name=request.name).data()
        except NemoHTTPError as exc:
            if exc.status_code != 404:
                raise
        else:
            raise ValueError(f"Fileset {request.name!r} already exists; choose a new run name")
    if workflow == Workflow.MONTAGE and secret is not None:
        SecretsClient.from_client(client).get_secret(workspace=workspace, name=secret).data()
    bindings: dict[str, str] = {}
    for role, request in requests.items():
        fileset = files_client.create_fileset(workspace=workspace, body=request).data()
        bindings[role] = fileset.id
        if role == "input":
            filename, content = (
                ("message.txt", b"Hello from the continuous DAG demo!\n")
                if workflow == Workflow.LAYERS
                else ("sutro-tower.jpg", (ROOT / "demo/opencv/fixtures/sutro-tower.jpg").read_bytes())
            )
        elif role == "config":
            filename, content = "settings.txt", b"partitions=10; batches=6\n"
        else:
            continue
        files_client.upload_file(workspace=workspace, name=fileset.name, path=filename, content=content).data()
        files_client.commit_fileset(workspace=workspace, name=fileset.name).data()
    request = job_request(workflow, name, bindings, profile=profile, image=image, secret=secret)
    return jobs_client.create_job(workspace=workspace, body=request).data()


def trigger_run(
    client: NemoClient,
    name: str,
    *,
    workspace: str,
    role: str,
    input_path: Path | None,
    steps: list[str],
    request_id: str,
) -> str:
    jobs_client = JobsClient.from_client(client)
    attempt = jobs_client.get_job_attempt(workspace=workspace, job=name, attempt="current").data()
    if attempt.status.is_terminal():
        raise ValueError("This Attempt is terminal; use nemo jobs rerun to create a new Attempt")
    metadata = DemoSpec.model_validate(attempt.spec.get("demo"))
    if steps:
        if input_path is not None or role != "input":
            raise ValueError("Use --step separately from --input and --fileset")
        signal = WorkflowInvalidation(request_id=request_id, steps=steps)
        unknown = set(steps) - {step.name for step in attempt.platform_spec.steps}
        if unknown:
            raise ValueError("Unknown nodes: " + ", ".join(sorted(unknown)))
        jobs_client.update_job_attempt_status_details(
            workspace=workspace,
            job=name,
            attempt=attempt.id,
            body=JobStatusDetailsUpdate(root={"workflow_invalidation": signal.model_dump()}),
        ).data()
        return f"Invalidated {', '.join(steps)} in Attempt {attempt.id}; request={request_id}"
    allowed = ("input", "config") if metadata.workflow == Workflow.LAYERS else ("input",)
    if role not in allowed:
        raise ValueError("Trigger Fileset must be one of: " + ", ".join(allowed))
    if input_path is not None and not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")
    files_client = FilesClient.from_client(client)
    fileset = files_client.get_fileset_by_id(workspace=workspace, fileset_id=metadata.filesets[role]).data()
    if input_path is not None:
        if metadata.workflow == Workflow.MONTAGE:
            if input_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Montage inputs must be JPG or PNG images")
            filename = input_path.name
        else:
            filename = "message.txt" if role == "input" else "settings.txt"
        files_client.upload_file(
            workspace=workspace, name=fileset.name, path=filename, content=input_path.read_bytes()
        ).data()
    committed = files_client.commit_fileset(workspace=workspace, name=fileset.name).data()
    return f"Committed {fileset.name}; revision={committed.commit_revision}; Attempt={attempt.id}"


def download_run(client: NemoClient, name: str, *, workspace: str, output: Path) -> None:
    attempt = JobsClient.from_client(client).get_job_attempt(workspace=workspace, job=name, attempt="current").data()
    metadata = DemoSpec.model_validate(attempt.spec.get("demo"))
    files_client = FilesClient.from_client(client)
    for role in metadata.output_roles:
        fileset = files_client.get_fileset_by_id(workspace=workspace, fileset_id=metadata.filesets[role]).data()
        destination = output / role
        destination.mkdir(parents=True, exist_ok=True)
        transfer.download(files_client, workspace=workspace, fileset=fileset.name, local_path=str(destination) + "/")
        print(destination)


def watch_run(name: str, *, workspace: str, no_tui: bool) -> int:
    return run(
        [
            sys.executable,
            "-m",
            "nemo_helix_ext.cli.app",
            "jobs",
            "watch",
            name,
            "--workspace",
            workspace,
            "--no-tui" if no_tui else "--tui",
        ],
        check=False,
    ).returncode


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List available workflow demos")
    create = commands.add_parser("create", help="Create a named continuous workflow and its private Filesets")
    create.add_argument("workflow", choices=[workflow.value for workflow in Workflow])
    create.add_argument("name")
    create.add_argument("--profile", default="gpu")
    create.add_argument("--image")
    create.add_argument("--secret", help="Existing NVIDIA API key Secret; required for montage")
    create.add_argument("--watch", action="store_true")
    trigger = commands.add_parser("trigger", help="Commit input or invalidate selected nodes")
    trigger.add_argument("name")
    trigger.add_argument("--fileset", choices=["input", "config"], default="input")
    selection = trigger.add_mutually_exclusive_group()
    selection.add_argument("--input", type=Path)
    selection.add_argument("--step", action="append", default=[])
    trigger.add_argument("--request-id", default=None)
    watch = commands.add_parser("watch", help="Open the existing DAG watcher")
    watch.add_argument("name")
    watch.add_argument("--no-tui", action="store_true", help="Unified line-oriented log follow")
    download = commands.add_parser("download", help="Download current published outputs")
    download.add_argument("name")
    download.add_argument("--output", "-o", type=Path)
    cancel = commands.add_parser("cancel", help="Cancel the current Attempt")
    cancel.add_argument("name")
    for command in (create, trigger, watch, download, cancel):
        command.add_argument("--workspace", default="default")
    return result


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    if args.command == "list":
        print("layers   19 nodes: prepare -> 10 workers -> 5 reducers -> 2 validators -> summarize")
        print("montage   3 nodes: detect-edges -> build-montage -> describe-images (NVIDIA Secret required)")
        return
    if args.command == "watch":
        raise SystemExit(watch_run(args.name, workspace=args.workspace, no_tui=args.no_tui))
    if args.command == "create" and not (ROOT / "services/core/jobs/jobs-launcher/jobs-launcher").is_file():
        cli.error("Build the Linux jobs-launcher first; see demo/README.md")
    try:
        with NemoClient.from_config() as client:
            if args.command == "create":
                workflow = Workflow(args.workflow)
                image = args.image or (
                    "docker.io/my-registry/nhx-api:local" if workflow == Workflow.LAYERS else "helix-opencv-demo:local"
                )
                job = create_run(
                    client,
                    workflow,
                    args.name,
                    workspace=args.workspace,
                    profile=args.profile,
                    image=image,
                    secret=args.secret,
                )
                print(f"Created {job.name}; Attempt={job.attempt_id}", flush=True)
                print(f"uv run python -m demo.workflows watch {job.name} --workspace {args.workspace}", flush=True)
            elif args.command == "trigger":
                print(
                    trigger_run(
                        client,
                        args.name,
                        workspace=args.workspace,
                        role=args.fileset,
                        input_path=args.input,
                        steps=args.step,
                        request_id=args.request_id or str(uuid4()),
                    )
                )
            elif args.command == "download":
                download_run(
                    client, args.name, workspace=args.workspace, output=args.output or ROOT / "demo/runs" / args.name
                )
            elif args.command == "cancel":
                JobsClient.from_client(client).cancel_job(workspace=args.workspace, name=args.name).data()
                print(f"Cancellation requested for {args.name}")
    except (NemoClientError, ValueError) as exc:
        cli.exit(1, f"{exc}\n")
    if args.command == "create" and args.watch:
        raise SystemExit(watch_run(args.name, workspace=args.workspace, no_tui=False))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create the continuous workflow demo from its saved job specification."""

import argparse
from pathlib import Path

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.providers import CPUExecutionProvider
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.jobs.workflows import WorkflowFilesetCommit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="continuous-demo")
    parser.add_argument("--workspace", default="default")
    parser.add_argument("--image", default="docker.io/my-registry/nhx-api:local")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    job = CreateHelixJobRequest.model_validate_json(Path(__file__).with_name("job.json").read_text())
    job.name = args.name
    for step in job.platform_spec.steps:
        if not isinstance(step.executor, CPUExecutionProvider):
            parser.error("The demo specification must use CPU container steps")
        step.executor.container.image = args.image
    if args.validate_only:
        print(f"Valid continuous workflow: {len(job.platform_spec.steps)} nodes, 10 parallel workers")
        return
    with NemoClient.from_config() as client:
        files_client = FilesClient.from_client(client)
        fileset = files_client.create_fileset(
            workspace=args.workspace,
            body=CreateFilesetRequest(name=f"{args.name}-input", description="Commit to trigger another demo pass"),
        ).data()
        job.platform_spec.steps[0].trigger = [WorkflowFilesetCommit(fileset_commit=fileset.id)]
        created = JobsClient.from_client(client).create_job(workspace=args.workspace, body=job).data()
    print(f"Created {args.workspace}/{created.name}")
    print(f"uv run nemo jobs watch {created.name} --workspace {args.workspace} --tui")
    print(f"uv run nemo files filesets commit {fileset.name} --workspace {args.workspace}")


if __name__ == "__main__":
    main()

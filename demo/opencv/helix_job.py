# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit the images FileSet to local Helix Jobs and download its artifacts."""

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, FilesetOutput
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.spec import (
    HelixJobEnvironmentVariable,
    HelixJobSecretEnvironmentVariableRef,
    HelixJobSpec,
    HelixJobStepSpec,
)
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.jobs.workflows import WorkflowInput, WorkflowSpec


def create_output(files: FilesClient, name: str) -> FilesetOutput:
    files.create_fileset(body=CreateFilesetRequest(name=name))
    return files.commit_fileset(name=name).data()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--profile", default="opencv")
    parser.add_argument("--secret", required=True, help="Existing NVIDIA API key Secret in the default workspace")
    args = parser.parse_args()
    run_id = uuid4().hex[:10]
    prefix = f"opencv-{run_id}"
    demo = Path(__file__).resolve().parent
    out = demo / "runs" / prefix
    out.mkdir(parents=True)
    with NemoClient(base_url=args.base_url, workspace="default") as client:
        files = FilesClient.from_client(client)
        jobs = JobsClient.from_client(client)
        images = files.get_fileset(name="images").data()
        entries = files.list_files(name=images.name).data().data
        photos = [entry.path for entry in entries if Path(entry.path).suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not photos or not images.commit_revision:
            parser.error("Add photos to the images FileSet and commit it before submitting the Job.")
        print(f"Input FileSet: {images.name} @ {images.commit_revision}", flush=True)
        edges = create_output(files, f"{prefix}-edges")
        montage = create_output(files, f"{prefix}-montage")
        captions = create_output(files, f"{prefix}-captions")

        executor = CPUExecutionProvider(
            provider="cpu",
            profile=args.profile,
            container=ContainerSpec(image="helix-opencv-demo:local", entrypoint=["python"], command=["/app/demo.py"]),
        )
        steps = [
            HelixJobStepSpec(
                name="detect-edges",
                depends_on=[],
                executor=executor,
                inputs=[WorkflowInput(fileset_id=images.id)],
                config={
                    "transform": "detect-edges",
                    "inputs": {"images": images.name},
                    "output_role": "edges",
                    "output_name": edges.name,
                },
            ),
            HelixJobStepSpec(
                name="build-montage",
                executor=executor,
                depends_on=["detect-edges"],
                inputs=[WorkflowInput(fileset_id=images.id), WorkflowInput(fileset_id=edges.id)],
                config={
                    "transform": "build-montage",
                    "inputs": {"images": images.name, "edges": edges.name},
                    "output_role": "montage",
                    "output_name": montage.name,
                },
            ),
            HelixJobStepSpec(
                name="describe-images",
                executor=executor,
                depends_on=["build-montage"],
                inputs=[
                    WorkflowInput(fileset_id=images.id),
                    WorkflowInput(fileset_id=edges.id),
                    WorkflowInput(fileset_id=montage.id),
                ],
                config={
                    "transform": "describe-images",
                    "inputs": {"images": images.name, "edges": edges.name, "montage": montage.name},
                    "output_role": "captions",
                    "output_name": captions.name,
                },
                environment=[
                    HelixJobEnvironmentVariable(
                        name="NVIDIA_API_KEY", from_secret=HelixJobSecretEnvironmentVariableRef(name=args.secret)
                    )
                ],
            ),
        ]
        job = jobs.create_job(
            body=CreateHelixJobRequest(
                name=prefix,
                source="opencv-demo",
                spec={},
                platform_spec=HelixJobSpec(steps=steps, workflow=WorkflowSpec(mode="run_to_completion")),
            )
        ).data()
        print(f"Job {job.name} ({job.id}), attempt {job.attempt_id}", flush=True)
        deadline = time.monotonic() + 300
        while True:
            status = jobs.get_job_status(name=job.id, attempt_id=job.attempt_id).data()
            if status.status.is_terminal() or time.monotonic() >= deadline:
                break
            time.sleep(2)
        (out / "status.json").write_text(status.model_dump_json(indent=2))
        if (
            status.status != HelixJobStatus.COMPLETED
            or len(status.steps) != len(steps)
            or any(s.status != HelixJobStatus.COMPLETED for s in status.steps)
        ):
            raise RuntimeError(f"Job status {status.status.value}. Inspect {out / 'status.json'}")
        outputs = {
            "edges": [Path(photo).stem + ".png" for photo in photos],
            "montage": ["contact-sheet.png"],
            "captions": ["descriptions.json"],
        }
        for role, original in (("edges", edges), ("montage", montage), ("captions", captions)):
            fs = files.get_fileset(name=original.name).data()
            if fs.commit_revision == original.commit_revision:
                raise RuntimeError(f"{fs.name} was never published by its producer")
            (out / role).mkdir()
            for filename in outputs[role]:
                path = out / role / filename
                path.write_bytes(files.download_file(name=fs.name, path=filename).read())
                print(f"{fs.name} @ {fs.commit_revision}: {path}")
        records = json.loads((out / "captions/descriptions.json").read_text())
        expected = {("images", photo) for photo in photos}
        expected.update(("edges", filename) for filename in outputs["edges"])
        expected.add(("montage", "contact-sheet.png"))
        if {(r["source"], r["image"]) for r in records if r["caption"].strip()} != expected:
            raise RuntimeError("Expected a caption for each image, edge map, and montage")


if __name__ == "__main__":
    main()

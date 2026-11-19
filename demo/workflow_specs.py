# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow definitions used by the demo CLI."""

from pathlib import Path

from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import (
    HelixJobEnvironmentVariable,
    HelixJobSecretEnvironmentVariableRef,
    HelixJobSpec,
    HelixJobStepSpec,
)
from nemo_helix_plugin.jobs.transfers import FilesetPull, FilesetPush, StepFilesets
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.jobs.workflows import WorkflowFilesetCommit, WorkflowSpec, WorkflowStepSuccess


def layered_workflow(bindings: dict[str, str], *, image: str, profile: str) -> HelixJobSpec:
    path = Path(__file__).with_name("continuous-workflow") / "job.json"
    job = CreateHelixJobRequest.model_validate_json(path.read_text())
    for step in job.platform_spec.steps:
        if not isinstance(step.executor, CPUExecutionProvider):
            raise ValueError("The layers demo requires CPU container steps")
        step.executor.container.image = image
        step.executor.profile = profile
        for condition in [*(step.depends_on or []), *(step.trigger or [])]:
            if isinstance(condition, WorkflowFilesetCommit):
                role = condition.fileset_commit.removeprefix("FILESET_").removesuffix("_ID").lower()
                condition.fileset_commit = bindings[role]
        for transfer in [*step.filesets.pull, *step.filesets.push]:
            role = transfer.fileset_id.removeprefix("FILESET_").removesuffix("_ID").lower()
            transfer.fileset_id = bindings[role]
    return HelixJobSpec.model_validate(job.platform_spec.model_dump())


def montage_workflow(
    images: str,
    edges: str,
    montage: str,
    captions: str,
    *,
    profile: str,
    image: str,
    secret: str,
) -> HelixJobSpec:
    """Wire Fileset IDs into the transforms, transfer bindings, and DAG events."""

    def step(
        name: str, reads: dict[str, str], output_role: str, output_id: str, parents: list[str]
    ) -> HelixJobStepSpec:
        directories = {role: f"/tmp/opencv/{role}" for role in [*reads, output_role]}
        command = ["/app/demo.py", name]
        for role, directory in directories.items():
            command.extend([f"--{role}", directory])
        return HelixJobStepSpec(
            name=name,
            depends_on=[WorkflowStepSuccess(step_success=parent) for parent in parents],
            trigger=(
                [WorkflowStepSuccess(step_success=parent) for parent in parents]
                if parents
                else [WorkflowFilesetCommit(fileset_commit=images)]
            ),
            executor=CPUExecutionProvider(
                provider="cpu",
                profile=profile,
                container=ContainerSpec(image=image, entrypoint=["python", "-u"], command=command),
            ),
            config={"transform": name},
            filesets=StepFilesets(
                pull=[
                    FilesetPull(fileset_id=fileset_id, directory=directories[role])
                    for role, fileset_id in reads.items()
                ],
                push=[FilesetPush(fileset_id=output_id, directory=directories[output_role], when="success")],
            ),
        )

    steps = [
        step("detect-edges", {"images": images}, "edges", edges, []),
        step("build-montage", {"images": images, "edges": edges}, "montage", montage, ["detect-edges"]),
        step(
            "describe-images",
            {"images": images, "edges": edges, "montage": montage},
            "captions",
            captions,
            ["build-montage"],
        ),
    ]
    steps[-1].environment = [
        HelixJobEnvironmentVariable(
            name="NVIDIA_API_KEY", from_secret=HelixJobSecretEnvironmentVariableRef(name=secret)
        )
    ]
    return HelixJobSpec(steps=steps, workflow=WorkflowSpec(mode="continuous", input_change_policy="after_drain"))

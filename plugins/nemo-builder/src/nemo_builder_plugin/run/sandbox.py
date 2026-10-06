# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What every sandbox provider shares: one sandbox per build context, the same mounts, and the same kaniko run.

A provider is how ``supervise`` gets a sandbox: a plain Kubernetes pod, or OpenSandbox. Either way the sandbox runs
the caller's Dockerfiles, so it holds no ServiceAccount token, no secret, and no credential in its environment.
"""

from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from nemo_builder_plugin.steps import SandboxGroup, SandboxImage, WorkLayout

#: The root of the job's `WorkLayout` in the sandbox. A `RUN` that writes here writes to the work
#: volume, so it is a name no Dockerfile would plausibly use.
SANDBOX_ROOT = PurePosixPath("/nhx-work")

KANIKO_EXECUTOR = "/kaniko/executor"

#: How long one sandbox may build its images, together.
BUILD_TIMEOUT_SECONDS = 60 * 60

#: When a sandbox is ended for it, so it ends even if this step is killed before it can delete it.
SANDBOX_DEADLINE_SECONDS = BUILD_TIMEOUT_SECONDS + 5 * 60

#: What a NetworkPolicy selects a sandbox on.
SANDBOX_LABEL = "nhx.nvidia.com/sandbox"

#: Which job a sandbox builds for, so a later attempt of the job can find and delete what an earlier one left.
JOB_LABEL = "nhx.nvidia.com/build-job"


def job_key(workspace: str, job_id: str) -> str:
    """A label value naming one job, unique across workspaces: job names are unique only within one."""
    return hashlib.sha256(f"{workspace}/{job_id}".encode()).hexdigest()[:20]


def sandbox_labels(workspace: str, job_id: str) -> dict[str, str]:
    return {
        SANDBOX_LABEL: "true",
        JOB_LABEL: job_key(workspace, job_id),
        "app.kubernetes.io/managed-by": "nemo-builder",
    }


@dataclass(frozen=True, slots=True)
class Mount:
    """A subPath of the work volume, mounted into the sandbox."""

    sub_path: str
    mount_path: str
    read_only: bool


def mounts(group: SandboxGroup, job_sub_path: str) -> list[Mount]:
    """The group's own context, read-only, and an output directory for each of its images.

    Never the whole output directory, where a later group's ``RUN`` could rewrite an earlier group's layout.
    """
    volume, view = WorkLayout(PurePosixPath(job_sub_path)), WorkLayout(SANDBOX_ROOT)
    return [
        Mount(str(volume.context(group.source)), str(view.context(group.source)), read_only=True),
        *(
            Mount(str(volume.output(image.image)), str(view.output(image.image)), read_only=False)
            for image in group.images
        ),
    ]


def clear_output(image: SandboxImage) -> str:
    """Shell that empties an image's output, or a failed build would leave an earlier attempt's layout for `push`.

    The output is a mount point, so only its contents can go.
    """
    quoted = shlex.quote(str(WorkLayout(SANDBOX_ROOT).output(image.image)))
    return f"rm -rf {quoted}/* {quoted}/.[!.]* {quoted}/..?*"


def kaniko_command(group: SandboxGroup, image: SandboxImage) -> str:
    """One image's build, writing an OCI layout to its output. ``--cleanup`` readies the sandbox for the next one."""
    view = WorkLayout(SANDBOX_ROOT)
    args = [
        KANIKO_EXECUTOR,
        f"--context=dir://{view.context(group.source)}",
        f"--dockerfile={image.dockerfile}",
        f"--custom-platform={image.platform}",
        "--no-push",
        "--no-push-cache",
        f"--oci-layout-path={view.output(image.image)}",
        "--cleanup",
        "--verbosity=info",
    ]
    return " ".join(shlex.quote(a) for a in args)


class SandboxProvider(Protocol):
    def sweep(self) -> None:
        """Delete the sandboxes an earlier attempt of this job left. Raises if they can't be deleted."""
        ...

    def build(self, index: int, group: SandboxGroup) -> dict[str, int]:
        """Build ``group``'s images in one sandbox, deleting it afterwards; each image's kaniko exit code.

        An image without a code didn't build. Raises if the sandbox can't be created or run.
        """
        ...

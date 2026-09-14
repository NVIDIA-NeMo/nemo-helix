# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a Gym environment FileSet into an evaluator job's persistent storage."""

from __future__ import annotations

import shutil
from pathlib import Path

from filesets import FilesetFileSystem, FilesetPathError, parse_fileset_ref
from nemo_evaluator.filesets import FilesetRef
from nemo_evaluator.jobs.utils import as_nemo_client
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.job import NemoJob
from nemo_platform_plugin.job_context import JobContext
from nemo_platform_plugin.sdk import NeMoPlatform
from pydantic import BaseModel, ConfigDict

#: Read-only tree the Gym host mounts at ``/job/environment``.
ENVIRONMENT_STORAGE_DIR = "environment"
#: Writable tree the Gym host mounts at ``/job/work``. Created empty so the mount exists before eval.
WORKSPACE_STORAGE_DIR = "workspace"
#: Scratch dir for the FileSet download. FileSet writes are not atomic, so we land here first
#: and rename to ``environment/`` only after the download completes.
ENVIRONMENT_STAGING_DIR = ".environment-staging"


class EnvironmentStageSpec(BaseModel):
    """Input for the evaluator-owned environment staging task."""

    model_config = ConfigDict(extra="forbid")

    environment: FilesetRef


def _remove_path(path: Path) -> None:
    """Delete a file, symlink, or directory so staging can replace it atomically."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _download_fileset_contents(*, sdk: NemoClient, workspace: str, fileset: str, destination: Path) -> None:
    """Download a FileSet root's contents directly into ``destination``."""
    files_client = FilesClient.from_client(sdk)
    fs = FilesetFileSystem(client=files_client)
    fs.get(f"{workspace}/{fileset}/", str(destination), recursive=True)


class EnvironmentStageJob(NemoJob):
    """Download a complete environment FileSet before the Gym evaluation step."""

    name = "stage-environment"
    description = "Stage a Gym environment FileSet into persistent job storage."
    container = "nmp-cpu-tasks"
    spec_schema = EnvironmentStageSpec

    def run(self, config: dict, *, ctx: JobContext, sdk: NemoClient | NeMoPlatform) -> dict:
        """Download the FileSet into ``persistent/environment``, replacing any previous tree."""
        client = as_nemo_client(sdk)
        spec = EnvironmentStageSpec.model_validate(config)
        try:
            workspace, fileset, file_path = parse_fileset_ref(
                spec.environment.root,
                workspace_fallback=ctx.workspace,
            )
        except FilesetPathError as exc:
            raise ValueError(f"invalid Gym environment FileSet reference: {spec.environment.root!r}") from exc
        if file_path:
            raise ValueError("Gym environment FileSet references must not include a file fragment")

        destination = ctx.storage.persistent / ENVIRONMENT_STORAGE_DIR
        staging = ctx.storage.persistent / ENVIRONMENT_STAGING_DIR
        workspace_dir = ctx.storage.persistent / WORKSPACE_STORAGE_DIR
        _remove_path(staging)
        staging.mkdir(parents=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        try:
            _download_fileset_contents(
                sdk=client,
                destination=staging,
                fileset=fileset,
                workspace=workspace,
            )
            _remove_path(destination)
            staging.rename(destination)
        except Exception:
            _remove_path(staging)
            raise

        return {
            "status": "completed",
            "environment": f"{workspace}/{fileset}",
            "path": str(destination),
        }

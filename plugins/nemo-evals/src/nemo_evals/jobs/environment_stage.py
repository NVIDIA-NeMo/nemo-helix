# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a Gym environment FileSet into an evaluator job's persistent storage."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Self

from filesets import FilesetFileSystem, FilesetPathError, parse_fileset_ref
from nemo_evals.filesets import FilesetRef
from nemo_evals.jobs.gym_registered_agent_package import (
    GymRegisteredAgentPackageSpec,
    write_registered_agent_package,
)
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.job import NemoJob
from nemo_helix_plugin.job_context import JobContext
from pydantic import BaseModel, ConfigDict, model_validator

#: Read-only tree the Gym host mounts at ``/job/environment``.
ENVIRONMENT_STORAGE_DIR = "environment"
#: Writable tree the Gym host mounts at ``/job/work``. Created empty so the mount exists before eval.
WORKSPACE_STORAGE_DIR = "workspace"
#: Scratch dir for the FileSet download. FileSet writes are not atomic, so we land here first
#: and rename to ``environment/`` only after the download completes.
ENVIRONMENT_STAGING_DIR = ".environment-staging"
#: Scratch dir for a registered agent's Ethos files before they are copied into the package.
AGENT_FILES_STAGING_DIR = ".agent-files-staging"


class EnvironmentStageSpec(BaseModel):
    """Input for the evaluator-owned environment staging task.

    ``environment`` is staged as the environment tree. A Gym registered agent adds its package on top:
    ``agent_files`` (the agent's Ethos FileSet) are copied into it and ``gym_registered_agent``
    describes the component, instance config and wheelhouse to write.
    """

    model_config = ConfigDict(extra="forbid")

    environment: FilesetRef | None = None
    agent_files: FilesetRef | None = None
    gym_registered_agent: GymRegisteredAgentPackageSpec | None = None

    @model_validator(mode="after")
    def _stage_something(self) -> Self:
        if self.environment is None and self.gym_registered_agent is None:
            raise ValueError("nothing to stage: set `environment` or `gym_registered_agent`")
        if self.agent_files is not None and self.gym_registered_agent is None:
            raise ValueError(
                "`agent_files` are staged into a registered agent's Gym package; set `gym_registered_agent`"
            )
        return self


def _remove_path(path: Path) -> None:
    """Delete a file, symlink, or directory so staging can replace it atomically."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _download_fileset_contents(*, client: NemoClient, workspace: str, fileset: str, destination: Path) -> None:
    """Download a FileSet root's contents directly into ``destination``."""
    files_client = FilesClient.from_client(client)
    fs = FilesetFileSystem(client=files_client)
    fs.get(f"{workspace}/{fileset}/", str(destination), recursive=True)


class EnvironmentStageJob(NemoJob):
    """Download a complete environment FileSet before the Gym evaluation step."""

    name = "stage-environment"
    description = "Stage a Gym environment FileSet into persistent job storage."
    container = "nhx-tasks"
    spec_schema = EnvironmentStageSpec

    def run(
        self,
        config: dict,
        *,
        ctx: JobContext,
        client: NemoClient,
    ) -> dict:
        """Download the FileSet into ``persistent/environment``, replacing any previous tree."""
        spec = EnvironmentStageSpec.model_validate(config)
        environment = _whole_fileset(spec.environment, ctx.workspace, "environment") if spec.environment else None
        agent_files = _whole_fileset(spec.agent_files, ctx.workspace, "agent files") if spec.agent_files else None

        destination = ctx.storage.persistent / ENVIRONMENT_STORAGE_DIR
        staging = ctx.storage.persistent / ENVIRONMENT_STAGING_DIR
        workspace_dir = ctx.storage.persistent / WORKSPACE_STORAGE_DIR
        _remove_path(staging)
        staging.mkdir(parents=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        files_dir = ctx.storage.persistent / AGENT_FILES_STAGING_DIR if agent_files is not None else None
        try:
            if environment is not None:
                _download_fileset_contents(
                    client=client, destination=staging, fileset=environment[1], workspace=environment[0]
                )
            if spec.gym_registered_agent is not None:
                if files_dir is not None and agent_files is not None:
                    _remove_path(files_dir)
                    files_dir.mkdir(parents=True)
                    _download_fileset_contents(
                        client=client, destination=files_dir, fileset=agent_files[1], workspace=agent_files[0]
                    )
                write_registered_agent_package(staging, spec.gym_registered_agent, agent_files=files_dir)
            _remove_path(destination)
            staging.rename(destination)
        except Exception:
            _remove_path(staging)
            raise
        finally:
            if files_dir is not None:
                _remove_path(files_dir)

        return {
            "status": "completed",
            "environment": f"{environment[0]}/{environment[1]}" if environment else None,
            "agent_files": f"{agent_files[0]}/{agent_files[1]}" if agent_files else None,
            "path": str(destination),
        }


def _whole_fileset(ref: FilesetRef, workspace_fallback: str, what: str) -> tuple[str, str]:
    try:
        workspace, fileset, file_path = parse_fileset_ref(ref.root, workspace_fallback=workspace_fallback)
    except FilesetPathError as exc:
        raise ValueError(f"invalid {what} FileSet reference: {ref.root!r}") from exc
    if file_path:
        raise ValueError(f"{what} FileSet references must not include a file fragment")
    return workspace, fileset

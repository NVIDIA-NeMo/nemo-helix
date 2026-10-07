# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot a registered agent's Ethos files for one evaluation job.

``resolved_config`` is captured at submit, but the files it refers to (skills, prompts, local adapters)
live in the agent's ``<agent>-ethos`` FileSet, which ``nemo agents create`` clears and rewrites every time
the agent is re-registered. A job queued across a re-registration would otherwise stage new files under
old config. So submission copies the FileSet into a job-owned snapshot, the staging step downloads that
snapshot, and the evaluation step deletes it once the run has succeeded.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

from filesets import AsyncFilesetFileSystem
from nemo_agents_plugin.entities import ethos_fileset_name
from nemo_evaluator.filesets import FilesetRef, _download_fileset_ref
from nemo_evaluator.jobs.utils import run_with_isolated_async_client
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest

logger = logging.getLogger(__name__)

#: Prefix of the job-owned snapshot FileSet; the suffix makes each submission's copy unique.
SNAPSHOT_FILESET_PREFIX = "agent-files-"


def snapshot_fileset_name() -> str:
    return f"{SNAPSHOT_FILESET_PREFIX}{uuid.uuid4().hex[:12]}"


async def snapshot_registered_agent_files(files: AsyncFilesClient, *, workspace: str, agent_name: str) -> FilesetRef:
    """Copy the agent's Ethos FileSet into a fresh job-owned FileSet and return that FileSet's ref.

    The caller has already checked the Ethos FileSet exists. A failure after the snapshot FileSet was
    created deletes it again so a failed submission does not leave a half-copied FileSet behind.
    """
    source = f"{workspace}/{ethos_fileset_name(agent_name)}"
    snapshot = snapshot_fileset_name()
    description = f"Ethos files of agent {workspace}/{agent_name} as evaluated; snapshot of {source}."
    await files.create_fileset(
        workspace=workspace, body=CreateFilesetRequest(name=snapshot, description=description[:255])
    )
    try:
        with tempfile.TemporaryDirectory(prefix=".agent-files-snapshot-") as directory:
            downloaded = await _download_fileset_ref(
                FilesetRef(root=source), directory, fs=AsyncFilesetFileSystem(client=files)
            )
            for path in sorted(p for p in Path(downloaded).rglob("*") if p.is_file()):
                await files.upload_file(
                    workspace=workspace,
                    name=snapshot,
                    path=path.relative_to(downloaded).as_posix(),
                    content=path.read_bytes(),
                )
    except Exception:
        await discard_registered_agent_files(files, FilesetRef(root=f"{workspace}/{snapshot}"))
        raise
    return FilesetRef(root=f"{workspace}/{snapshot}")


async def discard_registered_agent_files(files: AsyncFilesClient, ref: FilesetRef) -> None:
    """Delete a snapshot FileSet, logging rather than raising: the snapshot is never the job's output."""
    workspace, _, name = ref.root.partition("/")
    try:
        await files.delete_fileset(workspace=workspace, name=name)
    except Exception:
        logger.warning("Failed to delete agent files snapshot %s; it can be removed by hand", ref.root, exc_info=True)


def discard_registered_agent_files_sync(client: NemoClient | AsyncNemoClient, ref: FilesetRef) -> None:
    """The evaluation step's view of :func:`discard_registered_agent_files`, for either client color."""
    if isinstance(client, AsyncNemoClient):
        run_with_isolated_async_client(
            client, lambda async_client: discard_registered_agent_files(AsyncFilesClient.from_client(async_client), ref)
        )
        return
    workspace, _, name = ref.root.partition("/")
    try:
        FilesClient.from_client(client).delete_fileset(workspace=workspace, name=name)
    except Exception:
        logger.warning("Failed to delete agent files snapshot %s; it can be removed by hand", ref.root, exc_info=True)

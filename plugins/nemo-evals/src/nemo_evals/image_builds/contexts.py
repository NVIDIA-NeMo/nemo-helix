# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Putting a build context evals generated into Files, where Builder reads it.

A task's own context is already in Files, inside the task archive. A context evals writes itself, such as an agent
layer's, is packed into a tar archive and uploaded to a fileset.
"""

from __future__ import annotations

import asyncio
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from nemo_builder_plugin.schema import FileSetSource
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest


async def upload_context(
    files_client: AsyncFilesClient,
    workspace: str,
    fileset: str,
    directory: Path,
    *,
    archive: str,
    root: str,
    create_fileset: bool = True,
) -> FileSetSource:
    """Upload ``directory`` as ``archive``, a gzip-compressed tar archive, in ``fileset``; returns Builder's source.

    The archive holds the directory as ``root``, which is the source's context path. The fileset is created unless
    it is there already. A second archive for a fileset an earlier call made passes ``create_fileset=False``, so
    Files isn't asked for the fileset again.

    ``fileset`` is the fileset's name in ``workspace``. ``archive`` is a path in the fileset, so it may name a folder
    there. The source is made before anything else: one Builder would refuse is refused here before Files is asked
    for anything and before anything is written.

    The archive is packed in a temporary directory of its own, removed before the upload; nothing is written beside
    ``directory``. It is read whole into memory to upload it. Nothing deletes the fileset afterwards.

    Raises:
        ValidationError: Builder's source refuses the workspace, ``fileset``, ``archive`` or ``root``.
    """
    source = FileSetSource(fileset=f"{workspace}/{fileset}", archive=archive, context_path=root)
    if create_fileset:
        await files_client.create_fileset(workspace=workspace, body=CreateFilesetRequest(name=fileset), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nemo-build-context-") as scratch:
        # Under the archive's own file name: gzip records the name of the file it writes.
        packed = Path(scratch) / PurePosixPath(archive).name
        await asyncio.to_thread(_pack, directory, packed, root)
        content = await asyncio.to_thread(packed.read_bytes)
    await files_client.upload_file(workspace=workspace, name=fileset, path=archive, content=content)
    return source


def _pack(directory: Path, archive: Path, arcname: str) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(directory, arcname=arcname)

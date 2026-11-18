# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared online upgrade from a legacy Files reference to an allocation ID."""

from filesets import parse_fileset_ref
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient


def migrate_schema_fileset_id(
    files_client: FilesClient, *, ref: str, fileset_id: str | None, workspace: str | None = None
) -> str | None:
    """Keep a recorded ID; otherwise resolve the legacy reference once.

    A workspace supplies the namespace for an unqualified legacy reference.
    Only a missing fileset returns None; permission and transport failures propagate.
    """
    if fileset_id is not None:
        return fileset_id
    workspace, name, _ = parse_fileset_ref(ref, workspace_fallback=workspace)
    try:
        return files_client.get_fileset(workspace=workspace, name=name).data().id
    except NotFoundError:
        return None


async def migrate_schema_fileset_id_async(
    files_client: AsyncFilesClient, *, ref: str, fileset_id: str | None, workspace: str | None = None
) -> str | None:
    """Apply the same legacy-reference upgrade with an async Files client."""
    if fileset_id is not None:
        return fileset_id
    workspace, name, _ = parse_fileset_ref(ref, workspace_fallback=workspace)
    try:
        return (await files_client.get_fileset(workspace=workspace, name=name)).data().id
    except NotFoundError:
        return None

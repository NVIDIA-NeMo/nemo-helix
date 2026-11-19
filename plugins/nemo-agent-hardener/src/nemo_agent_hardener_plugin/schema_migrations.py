# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online upgrades applied before mutating saved Hardener records."""

from nemo_agent_hardener_plugin.entities import AgentHardenerManifest
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.schema_migrations import migrate_schema_fileset_id


def migrate_schema_manifest(manifest: AgentHardenerManifest, *, files_client: FilesClient) -> AgentHardenerManifest:
    """Return the manifest with its associated fileset ID; the caller owns the write."""
    if not manifest.agent_fileset and manifest.agent_fileset_id is None:
        return manifest
    fileset_id = migrate_schema_fileset_id(
        files_client, ref=manifest.agent_fileset, fileset_id=manifest.agent_fileset_id, workspace=manifest.workspace
    )
    return manifest.model_copy(update={"agent_fileset_id": fileset_id})

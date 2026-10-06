# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-release compatibility shims for the pre-rename plugin name.

This module is the only place in the plugin that may name the old plugin. It is
deleted in the release after the rename; see ``docs/garak-plugin/migration-guide.mdx``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entities.types import EntityCreateInput, EntityUpdate

logger = logging.getLogger(__name__)

LEGACY_ENTITY_TYPES = {
    "auditor_audit_config": "garak_plugin_scan_config",
    "auditor_audit_target": "garak_plugin_scan_target",
}
LEGACY_JOB_SOURCE = "auditor"
LEGACY_SCOPE_AREAS = ("auditor",)
LEGACY_PLUGIN_NAMES = ("auditor",)
JOB_SOURCE = "garak-plugin"
JOB_ENTITY_TYPE = "platform_job"
ALL_WORKSPACES = "-"
LEGACY_LOCAL_GARAK_PYTHON = "~/.auditor/.venv/bin/python"
LEGACY_GARAK_PYTHON_ENVVAR = "NEMO_AUDITOR_GARAK_PYTHON"
_PAGE_SIZE = 100


def getenv_with_legacy(name: str, legacy_name: str) -> str | None:
    """Return ``$name``, falling back to the deprecated ``$legacy_name`` with a warning."""
    value = os.environ.get(name)
    if value is not None:
        return value
    legacy_value = os.environ.get(legacy_name)
    if legacy_value is not None:
        logger.warning("Environment variable %s is deprecated and will be removed; use %s instead.", legacy_name, name)
    return legacy_value


def legacy_local_garak_python() -> str | None:
    """Return the pre-rename default garak interpreter path if it exists on disk."""
    path = Path(LEGACY_LOCAL_GARAK_PYTHON).expanduser()
    return str(path) if path.exists() else None


async def migrate_legacy_entities(sdk: AsyncHelixClient) -> int:
    """Re-home scan configs and targets stored under the pre-rename entity types.

    Idempotent: each entity is created under the new type (an existing one counts
    as already migrated) and only then deleted from the old type, so a partial run
    resumes where it stopped. Returns the number of entities moved.
    """
    client = client_from_platform(sdk, AsyncEntitiesClient)
    moved = 0
    for old_type, new_type in LEGACY_ENTITY_TYPES.items():
        while True:
            response = await client.list_entities(
                workspace=ALL_WORKSPACES,
                entity_type=old_type,
                query_params={"page": 1, "page_size": _PAGE_SIZE},
            )
            items = response.page().items
            if not items:
                break
            for entity in items:
                try:
                    await client.create_entity(
                        workspace=entity.workspace,
                        entity_type=new_type,
                        body=EntityCreateInput(name=entity.name, project=entity.project, data=entity.data),
                    )
                except ConflictError:
                    logger.debug("%s/%s already exists as %s", entity.workspace, entity.name, new_type)
                await client.delete_entity_by_name(workspace=entity.workspace, entity_type=old_type, name=entity.name)
                moved += 1
    if moved:
        logger.warning("Migrated %d stored scan config/target entities to the garak-plugin entity types.", moved)
    return moved


async def migrate_legacy_job_sources(sdk: AsyncHelixClient) -> int:
    """Point jobs submitted before the rename at the new service name.

    The plugin's job listing filters on the ``source`` field, so old jobs would
    otherwise disappear from it. Only ``source`` is rewritten; rows that changed
    concurrently are skipped (optimistic locking) and picked up on the next start.
    """
    client = client_from_platform(sdk, AsyncEntitiesClient)
    moved = 0
    skipped: set[tuple[str, str]] = set()
    while True:
        response = await client.list_entities(
            workspace=ALL_WORKSPACES,
            entity_type=JOB_ENTITY_TYPE,
            query_params={
                "page": 1,
                "page_size": _PAGE_SIZE,
                "filter": '{"data.source": "%s"}' % LEGACY_JOB_SOURCE,
            },
        )
        pending = [e for e in response.page().items if (e.workspace, e.name) not in skipped]
        if not pending:
            break
        for entity in pending:
            try:
                await client.update_entity_by_name(
                    workspace=entity.workspace,
                    entity_type=JOB_ENTITY_TYPE,
                    name=entity.name,
                    body=EntityUpdate(
                        data={**entity.data, "source": JOB_SOURCE}, expected_db_version=entity.db_version
                    ),
                )
                moved += 1
            except ConflictError:
                skipped.add((entity.workspace, entity.name))
    if moved:
        logger.warning("Re-labelled %d existing scan jobs with the garak-plugin service name.", moved)
    return moved

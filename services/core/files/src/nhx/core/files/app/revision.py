# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist explicit Files-service commit markers using the entity version fence."""

from random import uniform
from uuid import uuid4

from anyio import sleep
from fastapi import HTTPException
from nemo_helix_plugin.client.errors import NemoClientError
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError, EntityStoreError
from nhx.core.files.entities import Fileset


async def commit_fileset_revision(fileset: Fileset, entity_store: EntityClient) -> Fileset:
    """Assign a fresh commit marker, retrying explicit entity version conflicts."""
    revision = str(uuid4())
    current = fileset
    for attempt in range(3):
        try:
            if attempt:
                await sleep(uniform(0.01, 0.05) * 2 ** (attempt - 1))
                current = await entity_store.get_by_id(Fileset, fileset.id)
                if current.workspace != fileset.workspace:
                    raise EntityNotFoundError("Fileset is no longer in the original workspace")
            proposed = current.model_copy(update={"commit_revision": revision})
            return await entity_store.update(proposed)
        except EntityNotFoundError as exc:
            raise HTTPException(404, "Fileset no longer exists.") from exc
        except EntityConflictError as exc:
            if attempt == 2:
                raise HTTPException(503, "Fileset commit could not be persisted after concurrent updates.") from exc
        except (EntityStoreError, NemoClientError) as exc:
            raise HTTPException(503, "Fileset commit could not be confirmed.") from exc
    raise AssertionError("Commit update attempts exhausted without a result")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist Files-service mutation receipts using the existing entity version fence."""

from collections.abc import Callable
from random import uniform
from uuid import uuid4

from anyio import sleep
from fastapi import HTTPException
from nemo_helix_plugin.client.errors import NemoClientError
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError, EntityStoreError
from nhx.core.files.entities import Fileset


def _unchanged(fileset: Fileset) -> Fileset:
    return fileset


async def advance_fileset_revision(
    fileset: Fileset,
    entity_store: EntityClient,
    transform: Callable[[Fileset], Fileset] = _unchanged,
) -> Fileset:
    """Return this mutation's committed entity, retrying only explicit version conflicts.

    The caller performs any storage effect once before calling. A transform reapplies
    only the intended metadata changes to the latest entity on conflict.
    """
    revision = str(uuid4())
    current = fileset
    for attempt in range(3):
        try:
            if attempt:
                await sleep(uniform(0.01, 0.05) * 2 ** (attempt - 1))
                current = await entity_store.get_by_id(Fileset, fileset.id)
                if current.workspace != fileset.workspace:
                    raise EntityNotFoundError("Fileset is no longer in the original workspace")
            proposed = transform(current).model_copy(update={"revision": revision})
            return await entity_store.update(proposed)
        except EntityNotFoundError as exc:
            raise HTTPException(404, "Fileset no longer exists.") from exc
        except EntityConflictError as exc:
            if attempt == 2:
                raise HTTPException(503, "Fileset revision could not be persisted after concurrent updates.") from exc
        except (EntityStoreError, NemoClientError) as exc:
            raise HTTPException(
                503, "Fileset revision could not be persisted; storage may already have changed."
            ) from exc
    raise AssertionError("Revision update attempts exhausted without a result")

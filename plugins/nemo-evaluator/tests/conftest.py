# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test doubles for the evaluator plugin."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from nemo_platform_plugin.entities import EntityBase, EntityClient, ListResponse, PaginationInfo
from nemo_platform_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_platform_plugin.filter_ops import LogicalOperation


def matches_filter(entity, operation) -> bool:
    """Evaluate the AND-of-equality filters services emit.

    Filters are *evaluated*, not ignored: a fake that returned every row would make a revision
    lookup that forgot its ``parent`` predicate look correct, and cross-record confusion is exactly
    the bug that predicate prevents.
    """
    if operation is None:
        return True
    if isinstance(operation, LogicalOperation):
        return all(matches_filter(entity, child) for child in operation.operations)
    field = operation.field
    actual = entity.parent if field == "parent" else getattr(entity, field.removeprefix("data."), None)
    return actual == operation.value


class FakeEntityStore(EntityClient):
    """In-memory entity store standing in for ``NemoEntitiesClient``.

    Two behaviors are reproduced deliberately, because service logic depends on them:

    - **Parent-scoped uniqueness.** Records key on ``(entity_type, workspace, name, parent)``, which
      is what lets revision children of different tasks both be named ``rev.1`` while making a
      duplicate ordinal under one parent conflict.
    - **Optimistic locking.** ``update`` rejects a write whose ``db_version`` is stale. A fake that
      accepted every update would make correct retry logic untestable and broken retry logic look
      fine.
    - **Copy-on-read and copy-on-write.** Every record crosses the boundary as a deep copy, because
      the real client serializes over HTTP and cannot share objects with its caller. A fake that
      handed back the stored instance would let a service mutate the store just by touching an
      entity it read — so staging changes in memory and writing them later would be
      indistinguishable from writing them immediately, and a missing write would still pass.

    Signatures follow the *concrete* client (positional ``name``, optional ``parent``) rather than
    the narrower shared protocols, since that is what services actually call.
    """

    def __init__(self) -> None:
        # The job transformer now accepts only the concrete typed-client boundary. This fake
        # implements that boundary directly while keeping all storage in memory.
        self.entities: dict[tuple[str, str, str, str | None], EntityBase] = {}
        #: Monotonic tick for creation timestamps. Wall-clock ``now()`` can repeat within a test,
        #: which would make ``-created_at`` ordering non-deterministic — the real store's inserts
        #: are genuinely ordered, so the fake must be too.
        self._tick = 0

    def _now(self) -> datetime:
        self._tick += 1
        return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=self._tick)

    def _key(self, entity_type, name: str, workspace: str, parent: str | None):
        return (entity_type.__entity_type__, workspace, name, parent)

    async def create(self, entity):
        key = self._key(type(entity), entity.name, entity.workspace, entity._parent)
        if key in self.entities:
            raise NemoEntityConflictError(f"{key} exists")
        now = self._now()
        entity._id = f"{entity.__entity_type__}-{entity.name}"
        entity._created_at = now
        entity._updated_at = now
        entity._db_version = 0
        self.entities[key] = entity.model_copy(deep=True)
        return entity.model_copy(deep=True)

    async def get(self, entity_type, name: str, *, workspace: str, parent: str | None = None):
        key = self._key(entity_type, name, workspace, parent)
        if key not in self.entities:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")
        return self.entities[key].model_copy(deep=True)

    async def update(self, entity, *, original_name: str | None = None):
        key = self._key(type(entity), original_name or entity.name, entity.workspace, entity._parent)
        stored = self.entities.get(key)
        if stored is not None and entity._db_version != stored._db_version:
            raise NemoEntityConflictError(f"stale update for {entity.name}")
        entity._db_version = (stored._db_version if stored is not None else 0) + 1
        entity._updated_at = self._now()
        self.entities[key] = entity.model_copy(deep=True)
        return entity.model_copy(deep=True)

    async def delete(
        self,
        entity_type,
        name: str,
        *,
        workspace: str,
        parent: str | None = None,
        expected_db_version: int | None = None,
    ) -> None:
        # ``parent`` is part of the key, not decoration: revision children of different heads share
        # the name ``rev.1``, so ignoring it here would delete whichever one was inserted first.
        key = self._key(entity_type, name, workspace, parent)
        if key not in self.entities:
            raise NemoEntityNotFoundError(f"{workspace}/{name} not found")
        del self.entities[key]

    async def list(
        self,
        entity_type,
        *,
        workspace: str,
        filter_operation=None,
        sort: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ListResponse:
        items = [
            entity
            for (entity_type_name, record_workspace, _, _), entity in self.entities.items()
            if entity_type_name == entity_type.__entity_type__ and record_workspace == workspace
        ]
        items = [entity for entity in items if matches_filter(entity, filter_operation)]
        if sort and items:
            # Honour ``sort`` rather than ignoring it: services rely on server-side ordering, and a
            # fake that returned insertion order would make a wrong ``sort`` argument invisible.
            # Nothing to order when the result is empty, and probing the field on a bare ``object``
            # would reject a legitimate sort over an empty page.
            field = sort.lstrip("-")
            if not hasattr(items[0], field):
                raise NotImplementedError(f"fake cannot sort on {field!r}")
            items = sorted(items, key=lambda entity: getattr(entity, field), reverse=sort.startswith("-"))
        # Totals describe the whole result set, not the page — that distinction is the only way a
        # caller can tell a truncated history from a complete one, which is what ``list_revisions``
        # documents.
        total_results = len(items)
        start = (page - 1) * page_size
        items = [entity.model_copy(deep=True) for entity in items[start : start + page_size]]
        return ListResponse(
            data=items,
            pagination=PaginationInfo(
                page=page,
                page_size=page_size,
                current_page_size=len(items),
                total_pages=max(1, math.ceil(total_results / page_size)),
                total_results=total_results,
            ),
        )


@pytest.fixture
def entity_store() -> FakeEntityStore:
    """A fresh in-memory entity store.

    Exposed as a fixture rather than an importable name because ``from conftest import ...`` is
    ambiguous once more than one test root is collected in the same run — the wrong ``conftest``
    module wins and the import fails.
    """
    return FakeEntityStore()

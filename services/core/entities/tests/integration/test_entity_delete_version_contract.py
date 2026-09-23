# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client/server contract for optimistic-locking deletes.

``expected_db_version`` is built as a query param by ``EntityClient`` and read back
by the delete route's ``Query(...)`` declaration. Nothing else pins those two names
to each other:

- the client-side unit tests assert the key the client builds, against a mock
- the repository tests exercise the version check with no HTTP involved
- the endpoint tests assert query params reach ``PreparedRequest``, using a literal

So a rename on either side leaves every one of those green while silently turning a
guarded delete into an unconditional one, which is the exact TOCTOU hole the guard
was added to close.

These tests run a real ``EntityClient`` against a live entities app, so the query
string is genuinely built, sent, parsed and honoured.
"""

import pytest
from nhx.common.entities import EntityBase
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.testing import create_test_client


class VersionedWidget(EntityBase):
    colour: str


@pytest.fixture
def entity_client():
    with create_test_client(client_type=EntityClient) as client:
        yield client


async def _make_widget(entity_client: EntityClient, name: str) -> VersionedWidget:
    return await entity_client.create(VersionedWidget(name=name, workspace="default", colour="red"))


@pytest.mark.integration
@pytest.mark.asyncio
class TestDeleteVersionContract:
    async def test_stale_version_is_rejected_and_entity_survives(self, entity_client: EntityClient):
        """The guard has to actually reach the server, not just be built client-side."""
        created = await _make_widget(entity_client, "stale-guard")

        with pytest.raises(EntityConflictError):
            await entity_client.delete(
                VersionedWidget,
                "stale-guard",
                workspace="default",
                expected_db_version=created.db_version + 99,
            )

        # A guard that silently went missing would have deleted this.
        survivor = await entity_client.get(VersionedWidget, "stale-guard", workspace="default")
        assert survivor.colour == "red"

    async def test_matching_version_deletes(self, entity_client: EntityClient):
        created = await _make_widget(entity_client, "matching-guard")

        await entity_client.delete(
            VersionedWidget,
            "matching-guard",
            workspace="default",
            expected_db_version=created.db_version,
        )

        with pytest.raises(EntityNotFoundError):
            await entity_client.get(VersionedWidget, "matching-guard", workspace="default")

    async def test_version_bumped_by_update_invalidates_the_original_guard(self, entity_client: EntityClient):
        """The TOCTOU case: read a version, someone else writes, the delete must fail."""
        created = await _make_widget(entity_client, "raced-guard")
        stale_version = created.db_version

        created.colour = "blue"
        await entity_client.update(created)

        with pytest.raises(EntityConflictError):
            await entity_client.delete(
                VersionedWidget, "raced-guard", workspace="default", expected_db_version=stale_version
            )

    async def test_delete_without_guard_stays_unconditional(self, entity_client: EntityClient):
        """Omitting the guard must not start sending one, or every caller becomes racy."""
        created = await _make_widget(entity_client, "no-guard")
        created.colour = "blue"
        await entity_client.update(created)

        await entity_client.delete(VersionedWidget, "no-guard", workspace="default")

        with pytest.raises(EntityNotFoundError):
            await entity_client.get(VersionedWidget, "no-guard", workspace="default")

    async def test_delete_by_id_sends_the_version_it_resolved(self, entity_client: EntityClient):
        """delete_by_id always guards, using the version from its own lookup."""
        created = await _make_widget(entity_client, "by-id-guard")
        assert created.id is not None

        await entity_client.delete_by_id(VersionedWidget, created.id)

        with pytest.raises(EntityNotFoundError):
            await entity_client.get(VersionedWidget, "by-id-guard", workspace="default")

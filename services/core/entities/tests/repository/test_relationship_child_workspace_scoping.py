# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relationship filter EXISTS only counts child entities in allowed workspaces.

Mirrors the API: ``relationship_child_workspaces`` matches
``get_accessible_workspaces`` (None = do not filter children by workspace).
"""

import pytest
from nhx.core.entities.app.repository import SQLAlchemyEntityRepository
from nhx.core.entities.utils.filter import RelationshipFilterOperation, _parse_json_filter
from nhx.core.entities.utils.relationships import Relationship

pytestmark = pytest.mark.asyncio


def _adapters_exists_filter() -> RelationshipFilterOperation:
    """Filter for model has at least one adapter child (EXISTS on `adapters` by `parent`)."""
    rel = Relationship(kind="one_to_many", target_entity_type="adapter", via="parent")
    return RelationshipFilterOperation(
        relationship_name="adapters",
        relationship=rel,
        condition=None,
        exists=True,
    )


class TestRelationshipChildWorkspaceScoping:
    async def test_child_in_inaccessible_workspace_excluded_when_scoping_enabled(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """With relationship_child_workspaces=parent ws only, adapter in other ws is ignored."""
        m = await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="model",
            name="m1",
            data={},
        )
        await entity_repo.create_entity(
            workspace="workspace-2",
            entity_type="adapter",
            name="a1",
            parent=m.id,
            data={"finetuning_type": "LoRA"},
        )
        f = _adapters_exists_filter()
        # Child lives in workspace-2; only workspace-1 is "visible" for child rows
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="model",
            page=1,
            page_size=20,
            filter_op=f,
            relationship_child_workspaces={"workspace-1"},
        )
        assert total == 0
        assert entities == []

    async def test_child_visible_when_its_workspace_is_allowed(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """With both parent and child workspaces in the set, relationship EXISTS matches."""
        m = await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="model",
            name="m1",
            data={},
        )
        await entity_repo.create_entity(
            workspace="workspace-2",
            entity_type="adapter",
            name="a1",
            parent=m.id,
            data={"finetuning_type": "LoRA"},
        )
        f = _adapters_exists_filter()
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="model",
            page=1,
            page_size=20,
            filter_op=f,
            relationship_child_workspaces={"workspace-1", "workspace-2"},
        )
        assert total == 1
        assert [e.name for e in entities] == ["m1"]

    async def test_no_scoping_allows_child_in_different_workspace(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """With relationship_child_workspaces=None, EXISTS is unchanged (cross-ws child)."""
        m = await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="model",
            name="m1",
            data={},
        )
        await entity_repo.create_entity(
            workspace="workspace-2",
            entity_type="adapter",
            name="a1",
            parent=m.id,
            data={},
        )
        f = _adapters_exists_filter()
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="model",
            page=1,
            page_size=20,
            filter_op=f,
            relationship_child_workspaces=None,
        )
        assert total == 1
        assert [e.name for e in entities] == ["m1"]

    async def test_parsed_json_model_filter_respects_scoping(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        """Same as first case but using parser output (integration with RelationshipFilterOperation)."""
        m = await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="model",
            name="m1",
            data={},
        )
        await entity_repo.create_entity(
            workspace="workspace-2",
            entity_type="adapter",
            name="a1",
            parent=m.id,
            data={"finetuning_type": "LoRA"},
        )
        f = _parse_json_filter('{"adapters": {"$exists": true}}', entity_type="model")
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="model",
            page=1,
            page_size=20,
            filter_op=f,
            relationship_child_workspaces={"workspace-1"},
        )
        assert total == 0
        assert entities == []

    async def test_empty_allowed_child_workspaces_makes_exists_false(
        self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces
    ):
        m = await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="model",
            name="m1",
            data={},
        )
        await entity_repo.create_entity(
            workspace="workspace-2",
            entity_type="adapter",
            name="a1",
            parent=m.id,
            data={"finetuning_type": "LoRA"},
        )
        f = _adapters_exists_filter()
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="model",
            page=1,
            page_size=20,
            filter_op=f,
            relationship_child_workspaces=set(),
        )
        assert total == 0
        assert entities == []

    async def test_combined_user_filter_and_scoping(self, entity_repo: SQLAlchemyEntityRepository, setup_workspaces):
        """Relationship child scoping is independent of other AND'd predicates (name match)."""
        m = await entity_repo.create_entity(
            workspace="workspace-1",
            entity_type="model",
            name="match-me",
            data={},
        )
        await entity_repo.create_entity(
            workspace="workspace-2",
            entity_type="adapter",
            name="a1",
            parent=m.id,
            data={},
        )
        f = _parse_json_filter(
            '{"$and": [{"name": {"$like": "match%"}}, {"adapters": {"$exists": true}}]}',
            entity_type="model",
        )
        entities, total = await entity_repo.list_entities(
            workspace="workspace-1",
            entity_type="model",
            page=1,
            page_size=20,
            filter_op=f,
            relationship_child_workspaces={"workspace-1"},
        )
        assert total == 0
        assert entities == []

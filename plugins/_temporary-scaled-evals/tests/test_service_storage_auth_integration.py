# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scaled Evals storage initializer with real Entities persistence and authorization."""

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.core.entities.service import EntitiesService
from nhx.testing import ClientContext, create_test_client, short_unique_name

pytest.importorskip("scaled_evals")

from scaled_evals.api import entity_store as entity_store_module
from scaled_evals.api.config_profile_store import ConfigProfileStore
from scaled_evals.api.settings import settings
from scaled_evals.api.task_store import TaskStore

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def entity_store() -> None:
    """Use the real storage functions instead of the parent suite's fake store."""


@pytest.mark.parametrize("auth_enabled", [True, False])
@pytest.mark.parametrize("preexisting", [True, False])
def test_private_storage_and_execution_inputs_survive_creation_and_restart(
    monkeypatch: pytest.MonkeyPatch, auth_enabled: bool, preexisting: bool
) -> None:
    with create_test_client(
        EntitiesService, auth_enabled=auth_enabled, workspaces=[], projects=[], client_type=ClientContext
    ) as context:
        client = context.client.with_headers({"X-NHX-Principal-Id": "service:scaled-evals"})

        def service_client(*, as_service: str, internal: bool) -> NemoClient:
            assert as_service == "scaled-evals" and internal
            return client

        monkeypatch.setattr(entity_store_module, "get_nemo_client", service_client)
        monkeypatch.setattr(entity_store_module, "_client", None)
        name = short_unique_name("scaled-private")
        workspaces = WorkspacesClient.from_client(client)
        if preexisting:
            workspaces.create_workspace(body=CreateWorkspaceRequest(name=name)).data()
        monkeypatch.setattr(settings, "entity_store_private_workspace", name)
        entity_store_module.private_workspace.cache_clear()
        try:
            assert entity_store_module.private_workspace() == name
            members = workspaces.list_workspace_members(workspace=name).data().data
            assert set(next(member.roles for member in members if member.principal == "service:scaled-evals")) == {
                "Admin",
                "ServiceEntityStore",
            }
            tasks = TaskStore()
            assert tasks.count(owner_id="ordinary-user") == 0
            tasks.create_with_initial_revision(
                "private-task",
                name="Private task",
                slug="private-task",
                description=None,
                visibility="private",
                object_key="private-task/revision-1.tar",
                owner_id="ordinary-user",
            )
            profiles = ConfigProfileStore()
            profile = profiles.create(
                "private-agent",
                name="Private agent",
                type="agent",
                config={"prompt": "private execution input"},
                owner_id="ordinary-user",
            )
            assert tasks.count(owner_id="ordinary-user") == 1
            assert tasks.count(owner_id="other-user") == 0
            assert len(tasks.list(limit=10, cursor=None, order="asc", owner_id="ordinary-user")) == 1
            assert tasks.update("private-task", description="Updated task") is not None
            entity_store_module.private_workspace.cache_clear()
            assert entity_store_module.private_workspace() == name
            members = workspaces.list_workspace_members(workspace=name).data().data
            assert len(next(member.roles for member in members if member.principal == "service:scaled-evals")) == 2
            assert TaskStore().count(owner_id="ordinary-user") == 1
            revision = TaskStore().revision("private-task", 1)
            assert revision is not None
            assert revision["tarball_object_key"] == "private-task/revision-1.tar"
            assert ConfigProfileStore().get_typed("private-agent", ("agent",)) == profile
            assert tasks.soft_delete("private-task")
            assert tasks.count(owner_id="ordinary-user") == 0
            assert profiles.delete("private-agent", owner_id="ordinary-user")
        finally:
            entity_store_module.private_workspace.cache_clear()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Creator-originated provider writes cannot bypass workspace deletion."""

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.models.client import ModelsClient
from nemo_helix_plugin.models.types import CreateModelProviderRequest, UpsertModelProviderRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.common.auth.models import Principal
from nhx.core.models.service import ModelsService
from nhx.testing import TEST_ADMIN_EMAIL, as_user, create_test_client, short_unique_name

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("auth_enabled", [False, True])
@pytest.mark.parametrize("delegated", [False, True])
def test_provider_writes_check_creator_workspace_before_direct_persistence(auth_enabled, delegated):
    with create_test_client(ModelsService, auth_enabled=auth_enabled, client_type=NemoClient) as base:
        workspace = short_unique_name("provider-lifecycle")
        admin = WorkspacesClient.from_client(as_user(base, TEST_ADMIN_EMAIL))
        admin.create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        # Keep route authority after deletion revokes workspace memberships.
        # Even a platform admin must respect the user-facing lifecycle guard.
        caller = base.with_headers(
            Principal(
                id="service:integration-test" if delegated else TEST_ADMIN_EMAIL,
                on_behalf_of=TEST_ADMIN_EMAIL if delegated else None,
            ).get_headers()
        )
        admin.delete_workspace(name=workspace).data()
        models = ModelsClient.from_client(caller)
        with pytest.raises(NotFoundError):
            models.create_provider(
                workspace=workspace, body=CreateModelProviderRequest(name="new", host_url="https://provider.invalid")
            )
        with pytest.raises(NotFoundError):
            models.upsert_provider(
                workspace=workspace,
                name="upserted",
                body=UpsertModelProviderRequest(host_url="https://provider.invalid"),
            )

        direct = as_user(base, "service:models")
        created = (
            ModelsClient.from_client(direct)
            .create_provider(
                workspace=workspace,
                body=CreateModelProviderRequest(name="cleanup", host_url="https://provider.invalid"),
            )
            .data()
        )
        assert created.name == "cleanup"
        entities = EntitiesClient.from_client(direct)
        names = {
            entity.name for entity in entities.list_entities(workspace=workspace, entity_type="model_provider").items()
        }
        assert names == {"cleanup"}

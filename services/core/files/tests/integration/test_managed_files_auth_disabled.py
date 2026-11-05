# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth-disabled Files still protects producer metadata and managed contents."""

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest, UpdateFilesetRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest
from nhx.core.files.service import FilesService
from nhx.testing import TEST_ADMIN_EMAIL, as_user, create_test_client

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("auth_enabled", [False, True])
def test_resource_rules_preserve_managed_files(auth_enabled: bool) -> None:
    with create_test_client(FilesService, auth_enabled=auth_enabled, client_type=NemoClient) as base:
        if auth_enabled:
            WorkspacesClient.from_client(as_user(base, TEST_ADMIN_EMAIL)).create_workspace_member(
                workspace="default",
                body=CreateWorkspaceMemberRequest(principal="alice", roles=["Editor"]),
                query_params={"wait_role_propagation": True},
            ).data()
        user = FilesClient.from_client(as_user(base, "alice"))
        producer = FilesClient.from_client(as_user(base, "service:jobs"))
        forged = user.create_fileset(
            workspace="default",
            body=CreateFilesetRequest(name="forged", custom_fields={"service_source": "jobs"}),
        ).data()
        assert "service_source" not in forged.custom_fields

        producer.create_fileset(
            workspace="default",
            body=CreateFilesetRequest(name="managed", custom_fields={"service_source": "jobs"}),
        ).data()
        updated = user.update_fileset(
            workspace="default",
            name="managed",
            body=UpdateFilesetRequest(custom_fields={"service_source": "replacement", "label": "kept"}),
        ).data()
        assert updated.custom_fields == {"service_source": "jobs", "label": "kept"}
        with pytest.raises(PermissionDeniedError):
            user.upload_file(workspace="default", name="managed", path="denied.txt", content=b"denied").data()
        assert user.list_files(workspace="default", name="managed").data().data == []
        producer.upload_file(workspace="default", name="managed", path="allowed.txt", content=b"allowed").data()
        assert [item.path for item in user.list_files(workspace="default", name="managed").data().data] == [
            "allowed.txt"
        ]

        user.upload_file(workspace="default", name="forged", path="ordinary.txt", content=b"ordinary").data()

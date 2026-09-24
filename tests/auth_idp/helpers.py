# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed-client helpers shared by the auth-idp contract tests."""

from collections.abc import Sequence

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, WorkspaceMember


def grant_workspace_role(
    client: NemoClient,
    *,
    workspace: str,
    principal: str,
    roles: Sequence[str],
    wait_role_propagation: bool = True,
) -> WorkspaceMember:
    """Grant workspace roles to a principal and wait for the binding to propagate."""
    return (
        WorkspacesClient.from_client(client)
        .create_workspace_member(
            workspace=workspace,
            body=CreateWorkspaceMemberRequest(principal=principal, roles=list(roles)),
            query_params={"wait_role_propagation": wait_role_propagation},
        )
        .data()
    )

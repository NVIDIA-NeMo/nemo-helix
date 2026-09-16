# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix import APIStatusError
from nhx.testing import grant_workspace_role

from tests.auth_idp.common import require_capability

pytestmark = [
    pytest.mark.auth_idp,
    pytest.mark.auth_idp_runtime,
    pytest.mark.e2e,
]


def test_provider_workload_identity_is_denied_before_binding(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "workspace_rbac")

    with pytest.raises(APIStatusError) as exc_info:
        auth_idp_runtime.workload_provider_sdk().workspaces.retrieve(auth_idp_workspace)

    assert exc_info.value.status_code == 403


def test_provider_workload_identity_is_allowed_after_binding(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "workspace_rbac")

    e2e_setup_sdk = auth_idp_runtime.e2e_setup_sdk()
    for principal in auth_idp_runtime.workload_role_principals():
        grant_workspace_role(e2e_setup_sdk, workspace=auth_idp_workspace, principal=principal, roles=["Viewer"])

    retrieved = auth_idp_runtime.workload_provider_sdk().workspaces.retrieve(auth_idp_workspace)
    assert retrieved.name == auth_idp_workspace


def test_provider_workload_identity_returns_to_denied_after_revoke(
    auth_idp_case,
    auth_idp_runtime,
    auth_idp_workspace,
):
    require_capability(auth_idp_case, "workspace_rbac")

    e2e_setup_sdk = auth_idp_runtime.e2e_setup_sdk()
    for principal in auth_idp_runtime.workload_role_principals():
        grant_workspace_role(e2e_setup_sdk, workspace=auth_idp_workspace, principal=principal, roles=["Viewer"])
        e2e_setup_sdk.workspaces.members.delete(
            principal,
            workspace=auth_idp_workspace,
            wait_role_propagation=True,
        )

    with pytest.raises(APIStatusError) as exc_info:
        auth_idp_runtime.workload_provider_sdk().workspaces.retrieve(auth_idp_workspace)

    assert exc_info.value.status_code == 403

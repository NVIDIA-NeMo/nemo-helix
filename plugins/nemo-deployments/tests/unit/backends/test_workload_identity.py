# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_deployments_plugin.backends.k8s.workload_identity import revoke_workload_delegations
from nemo_deployments_plugin.backends.labels import deployment_key
from nemo_deployments_plugin.backends.workload_identity import workload_identity_activation_error
from nemo_deployments_plugin.entities import Container, DeploymentConfig, WorkloadIdentitySpec
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import WorkloadDelegationLookupScope


def _config() -> DeploymentConfig:
    return DeploymentConfig(
        name="cfg",
        workspace="default",
        containers=[Container(name="main", image="nginx:latest")],
        workloadIdentity=WorkloadIdentitySpec(enabled=True),
    )


def _auth_context() -> AuthContext:
    return AuthContext(principal_id="user:alice", principal_groups=["research"])


def test_workload_identity_activation_error_when_exchange_disabled() -> None:
    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=False,
    ):
        error = workload_identity_activation_error(config=_config(), auth_context=_auth_context())

    assert error == "workload_identity requires auth.oidc.workload_token_exchange_enabled to be enabled"


def test_workload_identity_activation_error_when_auth_context_missing() -> None:
    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        error = workload_identity_activation_error(config=_config(), auth_context=None)

    assert error == "workload_identity requires deployment auth_context for on-behalf-of delegation"


@pytest.mark.asyncio
async def test_revoke_workload_delegations_uses_broad_scope_when_disabled() -> None:
    store = MagicMock()
    store.revoke_by_workload = AsyncMock()

    await revoke_workload_delegations(
        store,
        config=DeploymentConfig(
            name="cfg",
            workspace="default",
            containers=[Container(name="main", image="nginx:latest")],
        ),
        workspace="default",
        deployment_name="task",
    )

    store.revoke_by_workload.assert_awaited_once_with(
        WorkloadDelegationLookupScope(
            workload_workspace="default",
            workload_kind=None,
            workload_instance_id=deployment_key("default", "task"),
        )
    )


@pytest.mark.asyncio
async def test_revoke_workload_delegations_uses_broad_scope_after_workload_kind_change() -> None:
    store = MagicMock()
    store.revoke_by_workload = AsyncMock()
    config = _config().model_copy(
        update={
            "workload_identity": WorkloadIdentitySpec(
                enabled=True,
                workloadKind="new_kind",
                workloadId="task",
            )
        }
    )

    await revoke_workload_delegations(store, config=config, workspace="default", deployment_name="task")

    store.revoke_by_workload.assert_awaited_once_with(
        WorkloadDelegationLookupScope(
            workload_workspace="default",
            workload_kind=None,
            workload_instance_id=deployment_key("default", "task"),
        )
    )

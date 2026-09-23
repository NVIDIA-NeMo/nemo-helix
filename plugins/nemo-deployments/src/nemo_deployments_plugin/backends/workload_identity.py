# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral deployment workload identity helpers."""

from __future__ import annotations

from typing import overload

from nemo_deployments_plugin.backends.labels import deployment_key
from nemo_deployments_plugin.entities import DeploymentConfig
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import WorkloadDelegationLookupScope, WorkloadDelegationScope
from nemo_helix_plugin.auth.workload_identity import (
    WorkloadIdentityConfigError,
    is_workload_identity_token_exchange_enabled,
)

DEFAULT_WORKLOAD_KIND = "deployment"


def workload_identity_requested(config: DeploymentConfig) -> bool:
    spec = config.workload_identity
    return spec is not None and spec.enabled


def workload_identity_activation_error(
    *,
    config: DeploymentConfig,
    auth_context: AuthContext | None,
) -> str | None:
    """Return a user-facing error if workload identity cannot be activated."""
    if not workload_identity_requested(config):
        return None
    try:
        token_exchange_enabled = is_workload_identity_token_exchange_enabled()
    except WorkloadIdentityConfigError as exc:
        return str(exc)
    if not token_exchange_enabled:
        return "workload_identity requires auth.oidc.workload_token_exchange_enabled to be enabled"
    if auth_context is None:
        return "workload_identity requires deployment auth_context for on-behalf-of delegation"
    return None


def workload_identity_reconcile_allowed(config: DeploymentConfig, auth_context: AuthContext | None) -> bool:
    """Return whether workload identity reconciliation should run for a status read."""
    return (
        workload_identity_requested(config)
        and workload_identity_activation_error(
            config=config,
            auth_context=auth_context,
        )
        is None
    )


def workload_kind(config: DeploymentConfig) -> str:
    spec = config.workload_identity
    if spec is not None and spec.workload_kind:
        return spec.workload_kind
    return DEFAULT_WORKLOAD_KIND


@overload
def workload_delegation_scope(
    *,
    workspace: str,
    deployment_name: str,
    config: DeploymentConfig,
) -> WorkloadDelegationScope: ...


@overload
def workload_delegation_scope(
    *,
    workspace: str,
    deployment_name: str,
    config: None,
) -> WorkloadDelegationLookupScope: ...


def workload_delegation_scope(
    *,
    workspace: str,
    deployment_name: str,
    config: DeploymentConfig | None,
) -> WorkloadDelegationLookupScope:
    """Return the typed delegation scope for one physical deployment instance."""
    if config is None:
        return WorkloadDelegationLookupScope(
            workload_workspace=workspace,
            workload_kind=None,
            workload_instance_id=deployment_key(workspace, deployment_name),
        )
    spec = config.workload_identity
    return WorkloadDelegationScope(
        workload_workspace=workspace,
        workload_kind=workload_kind(config),
        workload_instance_id=deployment_key(workspace, deployment_name),
        workload_claim_id=spec.workload_id if spec is not None else None,
    )

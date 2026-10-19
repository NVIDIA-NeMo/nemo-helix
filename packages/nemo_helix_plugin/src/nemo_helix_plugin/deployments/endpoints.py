# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the deployments-plugin API.

Single source of truth for the HTTP contract (paths carry the
``/apis/deployments`` gateway prefix) — full CRUD for Volumes, Deployments, and
DeploymentConfigs. All ``DELETE`` routes return ``204``. Volume and Deployment
deletes are soft: they set the entity's status to ``DELETING`` and the
reconciler then tears down the backing resource and removes the entity, so the
caller polls ``get_*`` for teardown. DeploymentConfig delete is immediate (a
hard entity delete, guarded by a referential 409 if a deployment still uses it).
All ``delete_*`` methods return ``None``.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import delete, get, post
from nemo_helix_plugin.client.types import Paginated, PreparedRequest
from nemo_helix_plugin.deployments.types import (
    CreateDeploymentConfigRequest,
    CreateDeploymentRequest,
    CreateVolumeRequest,
    Deployment,
    DeploymentConfig,
    ListDeploymentConfigsQueryParams,
    ListDeploymentsQueryParams,
    ListVolumesQueryParams,
    Volume,
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------


@get("/apis/deployments/v2/workspaces/{workspace}/volumes/{name}")
@abstractmethod
def get_volume(*, workspace: str | None = None, name: str) -> Volume: ...


@get("/apis/deployments/v2/workspaces/{workspace}/volumes")
@abstractmethod
def list_volumes(
    *, workspace: str | None = None, query_params: ListVolumesQueryParams | None = None
) -> Paginated[Volume]: ...


def _get_volume_on_conflict(body: CreateVolumeRequest, workspace: str | None) -> PreparedRequest[Volume]:
    """Build the retrieve request replayed when ``create_volume(exist_ok=True)`` 409s."""
    return get_volume(name=body.name, workspace=workspace)


@post("/apis/deployments/v2/workspaces/{workspace}/volumes", get_on_conflict=_get_volume_on_conflict)
@abstractmethod
def create_volume(*, workspace: str | None = None, body: CreateVolumeRequest, exist_ok: bool = False) -> Volume: ...


@delete("/apis/deployments/v2/workspaces/{workspace}/volumes/{name}")
@abstractmethod
def delete_volume(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Deployment configs
# ---------------------------------------------------------------------------


@get("/apis/deployments/v2/workspaces/{workspace}/deployment-configs/{name}")
@abstractmethod
def get_deployment_config(*, workspace: str | None = None, name: str) -> DeploymentConfig: ...


@get("/apis/deployments/v2/workspaces/{workspace}/deployment-configs")
@abstractmethod
def list_deployment_configs(
    *, workspace: str | None = None, query_params: ListDeploymentConfigsQueryParams | None = None
) -> Paginated[DeploymentConfig]: ...


def _get_deployment_config_on_conflict(
    body: CreateDeploymentConfigRequest, workspace: str | None
) -> PreparedRequest[DeploymentConfig]:
    """Build the retrieve request replayed when ``create_deployment_config(exist_ok=True)`` 409s."""
    return get_deployment_config(name=body.name, workspace=workspace)


@post(
    "/apis/deployments/v2/workspaces/{workspace}/deployment-configs",
    get_on_conflict=_get_deployment_config_on_conflict,
)
@abstractmethod
def create_deployment_config(
    *, workspace: str | None = None, body: CreateDeploymentConfigRequest, exist_ok: bool = False
) -> DeploymentConfig: ...


@delete("/apis/deployments/v2/workspaces/{workspace}/deployment-configs/{name}")
@abstractmethod
def delete_deployment_config(*, workspace: str | None = None, name: str) -> None: ...


# ---------------------------------------------------------------------------
# Deployments
# ---------------------------------------------------------------------------


@get("/apis/deployments/v2/workspaces/{workspace}/deployments/{name}")
@abstractmethod
def get_deployment(*, workspace: str | None = None, name: str) -> Deployment: ...


@get("/apis/deployments/v2/workspaces/{workspace}/deployments")
@abstractmethod
def list_deployments(
    *, workspace: str | None = None, query_params: ListDeploymentsQueryParams | None = None
) -> Paginated[Deployment]: ...


def _get_deployment_on_conflict(body: CreateDeploymentRequest, workspace: str | None) -> PreparedRequest[Deployment]:
    """Build the retrieve request replayed when ``create_deployment(exist_ok=True)`` 409s."""
    return get_deployment(name=body.name, workspace=workspace)


@post("/apis/deployments/v2/workspaces/{workspace}/deployments", get_on_conflict=_get_deployment_on_conflict)
@abstractmethod
def create_deployment(
    *, workspace: str | None = None, body: CreateDeploymentRequest, exist_ok: bool = False
) -> Deployment: ...


@delete("/apis/deployments/v2/workspaces/{workspace}/deployments/{name}")
@abstractmethod
def delete_deployment(*, workspace: str | None = None, name: str) -> None: ...

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert compiled plugin entities into deployments-plugin API request DTOs.

The compiler still builds rich ``nemo_deployments_plugin`` entity objects
(Volume / DeploymentConfig). The models controller talks to the plugin over
HTTP, whose create routes take the request-DTO subset
(``nemo_helix_plugin.deployments.types``). These converters bridge the two:
dump the entity (snake_case), drop server-owned fields (status, workspace, ids),
and validate into the request DTO. Nested shapes map by field name.
"""

from __future__ import annotations

from typing import Any

from nemo_deployments_plugin.entities import DeploymentConfig, Volume
from nemo_helix_plugin.deployments.types import CreateDeploymentConfigRequest, CreateVolumeRequest

# Server-owned / transport fields an entity carries that a create request must not.
# ``workspace`` is a route path param, not a body field; the rest are set by the store.
_SERVER_FIELDS = frozenset(
    {
        "workspace",
        "status",
        "status_message",
        "error_details",
        "id",
        "entity_id",
        "db_version",
        "created_at",
        "updated_at",
        "created_by",
        "updated_by",
        "parent",
        "project",
    }
)


def _request_data(entity: Any) -> dict[str, Any]:
    data = entity.model_dump(by_alias=False, exclude_none=True)
    for field in _SERVER_FIELDS:
        data.pop(field, None)
    return data


def volume_create_request(volume: Volume) -> CreateVolumeRequest:
    return CreateVolumeRequest.model_validate(_request_data(volume))


def deployment_config_create_request(config: DeploymentConfig) -> CreateDeploymentConfigRequest:
    return CreateDeploymentConfigRequest.model_validate(_request_data(config))

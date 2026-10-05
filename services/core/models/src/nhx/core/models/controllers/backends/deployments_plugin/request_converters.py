# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert compiled plugin entities into deployments-plugin API request DTOs.

The compiler still builds rich ``nemo_deployments_plugin`` entity objects
(Volume / DeploymentConfig). The models controller talks to the plugin over
HTTP, whose create routes take the request-DTO subset
(``nemo_helix_plugin.deployments.types``). These converters bridge the two:
dump the entity (snake_case) and keep only the fields the request DTO declares
(deriving the field set from the DTO, not a hand-maintained drop-list). Nested
shapes map by field name.
"""

from __future__ import annotations

from typing import Any

from nemo_deployments_plugin.entities import DeploymentConfig, Volume
from nemo_helix_plugin.deployments.types import CreateDeploymentConfigRequest, CreateVolumeRequest
from pydantic import BaseModel


def _request_data(entity: Any, request_type: type[BaseModel]) -> dict[str, Any]:
    # Derive the field set from the request DTO itself rather than maintaining a
    # drop-list of server-owned fields: dump the entity (snake_case) and keep only
    # the fields the create request actually declares. Anything the entity carries
    # that the DTO doesn't (status, ids, timestamps, workspace, future entity-level
    # additions) is dropped automatically, so an entity gaining a field can never
    # leak into a request. (The DTOs use ``extra="ignore"``, so unknown keys would
    # be dropped at validation anyway; filtering here makes the boundary explicit.)
    data = entity.model_dump(by_alias=False, exclude_none=True)
    return {key: value for key, value in data.items() if key in request_type.model_fields}


def volume_create_request(volume: Volume) -> CreateVolumeRequest:
    return CreateVolumeRequest.model_validate(_request_data(volume, CreateVolumeRequest))


def deployment_config_create_request(config: DeploymentConfig) -> CreateDeploymentConfigRequest:
    return CreateDeploymentConfigRequest.model_validate(_request_data(config, CreateDeploymentConfigRequest))

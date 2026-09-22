# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route exposing the default Anonymizer entity labels for the detection stage."""

from __future__ import annotations

from anonymizer import DEFAULT_ENTITY_LABELS
from fastapi import APIRouter, status
from nemo_helix_plugin.authz import AuthzScope, CallerKind, PermissionSet, path_rule, perm
from pydantic import BaseModel

scope = AuthzScope("anonymizer")


class EntityLabelPerms(PermissionSet, namespace="anonymizer.entity-labels"):
    """Permissions for the default entity-label listing."""

    LIST = perm("List the default Anonymizer entity labels")


class EntityLabelsResponse(BaseModel):
    """The default GLiNER entity labels detected when none are supplied."""

    data: list[str]


router = APIRouter()


@router.get(
    "/entity-labels",
    summary="List Default Entity Labels",
    response_description="The default entity labels used by the detection stage",
    status_code=status.HTTP_200_OK,
    response_model=EntityLabelsResponse,
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[EntityLabelPerms.LIST])
async def list_entity_labels(workspace: str) -> EntityLabelsResponse:
    """Return the default entity labels detected when a config omits ``entity_labels``."""
    return EntityLabelsResponse(data=list(DEFAULT_ENTITY_LABELS))

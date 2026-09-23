# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.entities.client import EntityClient
from nhx.common.service.dependencies import get_entity_client
from nhx.core.secrets.api.v2.admin.routines import rotate_encryption_keys
from nhx.core.secrets.api.v2.admin.schemas import HelixSecretAdminRotationResponse
from nhx.core.secrets.app.encryptor import get_current_encryptor

router = APIRouter()

logger = logging.getLogger(__name__)

# Only the platform admin role includes `secrets.rotate` (see static-authz.yaml).
_SYSTEM_WORKSPACE = "system"
_ROTATION_PERMISSIONS = ("secrets.rotate",)


async def require_rotate_encryption_keys_caller(auth_client: AuthClient) -> None:
    """Require the platform admin capability: ``secrets.rotate`` in the ``system`` workspace."""
    if not auth_client.auth_enabled:
        return
    if await auth_client.has_permissions(_SYSTEM_WORKSPACE, list(_ROTATION_PERMISSIONS)):
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


## Admin endpoints for Platform secrets ##


@router.post("/v2/rotate-encryption-keys", status_code=status.HTTP_202_ACCEPTED)
async def admin_rotate_encryption_keys(
    entity_client: EntityClient = Depends(get_entity_client),
    auth_client: AuthClient = Depends(get_auth_client),
) -> HelixSecretAdminRotationResponse:
    """Rotate encryption keys for all platform secrets."""
    await require_rotate_encryption_keys_caller(auth_client)
    try:
        current_encryptor = get_current_encryptor()
        await rotate_encryption_keys(current_encryptor, entity_client)
    except Exception as e:
        logger.exception("Failed to rotate encryption keys")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error rotating encryption keys"
        ) from e
    return HelixSecretAdminRotationResponse(rotated_secrets=0, success=True)

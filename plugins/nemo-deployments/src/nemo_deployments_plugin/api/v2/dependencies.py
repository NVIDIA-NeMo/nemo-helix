# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies for the deployments plugin API."""

from __future__ import annotations

from fastapi import HTTPException, Request
from nemo_helix_plugin.auth import is_service_principal_id
from nemo_helix_plugin.entity_client import get_entity_client

__all__ = ["get_entity_client", "require_service_principal"]

_PRINCIPAL_ID_HEADER = "X-NHX-Principal-Id"


def require_service_principal(request: Request) -> None:
    """Restrict controller-only status writes to service principals."""
    principal_id = request.headers.get(_PRINCIPAL_ID_HEADER, "")
    if not is_service_principal_id(principal_id):
        raise HTTPException(
            status_code=403,
            detail="Status updates require a service principal.",
        )

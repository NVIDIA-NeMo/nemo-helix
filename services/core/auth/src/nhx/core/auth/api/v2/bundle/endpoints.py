# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OPA bundle endpoint for Auth Service."""

import io
import logging

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse
from nhx.common.entities import EntityClient
from nhx.common.service.dependencies import get_entity_client
from nhx.core.auth.app.bundle import get_bundle_cache_seconds, get_opa_bundle_with_etag

router = APIRouter(tags=["Bundle"])
logger = logging.getLogger(__name__)


@router.get(
    "/v2/iam/opa-bundle.tar.gz",
    response_class=Response,
    include_in_schema=False,
    responses={
        200: {
            "content": {"application/gzip": {}},
            "description": "OPA bundle containing authorization policy and static data",
        },
        304: {
            "description": "Not Modified - bundle has not changed since last request",
        },
    },
    summary="Download OPA bundle",
    description="Get the OPA bundle containing the authorization policy and static configuration data.",
)
async def get_opa_bundle(
    request: Request,
    entities_client: EntityClient = Depends(get_entity_client),
):
    """Generate and return the OPA bundle with E-Tag support.

    This endpoint generates an OPA bundle containing:
    - All authorization policy files from app/policies/ (auto-discovered)
    - Static authorization data (roles, endpoint permissions)
    - Dynamic authorization data from database (workspaces, principals, role bindings)

    The bundle is returned as a tar.gz file that can be loaded directly by OPA.

    E-Tag Support:
    - Returns E-Tag header with bundle hash
    - Supports If-None-Match header for conditional requests
    - Returns 304 Not Modified if bundle hasn't changed
    - Bundle generation is debounced to avoid excessive regeneration
    """
    # Get the bundle and its E-Tag
    bundle_bytes, etag = await get_opa_bundle_with_etag(entities_client=entities_client)
    cache_seconds = get_bundle_cache_seconds()

    # Check if client has a cached version
    if_none_match = request.headers.get("If-None-Match")
    if if_none_match and if_none_match.strip('"') == etag:
        # Client has the latest version, return 304 Not Modified
        return Response(
            status_code=304,
            headers={
                "ETag": f'"{etag}"',
                "Cache-Control": f"max-age={cache_seconds}, must-revalidate",
            },
        )

    # Return the bundle with E-Tag header
    return StreamingResponse(
        io.BytesIO(bundle_bytes),
        media_type="application/gzip",
        headers={
            "Content-Disposition": "attachment; filename=opa-bundle.tar.gz",
            "ETag": f'"{etag}"',
            "Cache-Control": f"max-age={cache_seconds}, must-revalidate",
        },
    )

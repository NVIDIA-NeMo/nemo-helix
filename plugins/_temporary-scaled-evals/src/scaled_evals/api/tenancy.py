# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException
from nemo_helix_plugin.dependencies import RequestAuthorizer, get_request_authorizer

# Any route stamped ``@admin_only`` works; the platform decides whether the
# caller holds ``scaled-evals.admin`` for it.
_ADMIN_PROBE = ("GET", "/apis/scaled-evals/v1/admin/capacity")

logger = logging.getLogger(__name__)


async def caller_is_admin(authorize: Annotated[RequestAuthorizer, Depends(get_request_authorizer)]) -> bool:
    """Return whether the platform grants the caller ``scaled-evals.admin``.

    Admin-only routes are enforced by the platform before the handler runs.
    This answers the same question for handlers that widen what an admin sees.
    An unavailable admin probe falls back to normal owner/shared access.
    Authentication and unexpected errors still propagate. Platform auth disabled
    authorizes everything, so local callers are admins.
    """
    try:
        await authorize(*_ADMIN_PROBE)
    except (HTTPException, httpx.HTTPStatusError) as exc:
        status_code = exc.status_code if isinstance(exc, HTTPException) else exc.response.status_code
        if status_code == 403:
            return False
        if not 500 <= status_code < 600:
            raise
        logger.warning("Admin authorization probe returned HTTP %s; continuing without admin privileges", status_code)
        return False
    except httpx.TransportError as exc:
        logger.warning("Admin authorization probe failed (%s); continuing without admin privileges", type(exc).__name__)
        return False
    return True


IsAdmin = Annotated[bool, Depends(caller_is_admin)]

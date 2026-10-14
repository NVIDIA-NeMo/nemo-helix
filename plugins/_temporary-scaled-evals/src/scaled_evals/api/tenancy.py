# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException
from nemo_helix_plugin.dependencies import RequestAuthorizer, get_request_authorizer

# Any route stamped ``@admin_only`` works; the platform decides whether the
# caller holds ``scaled-evals.admin`` for it.
_ADMIN_PROBE = ("GET", "/apis/scaled-evals/v1/admin/capacity")


async def caller_is_admin(authorize: Annotated[RequestAuthorizer, Depends(get_request_authorizer)]) -> bool:
    """Return whether the platform grants the caller ``scaled-evals.admin``.

    Admin-only routes are enforced by the platform before the handler runs.
    This answers the same question for handlers that widen what an admin sees.
    Platform auth disabled authorizes everything, so local callers are admins.
    """
    try:
        await authorize(*_ADMIN_PROBE)
    except HTTPException as exc:
        if exc.status_code == 403:
            return False
        raise
    return True


IsAdmin = Annotated[bool, Depends(caller_is_admin)]

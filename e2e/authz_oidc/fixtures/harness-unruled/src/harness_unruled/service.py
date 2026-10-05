# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partially-invalid fixture plugin for the authz OIDC E2E harness.

One route carries a valid ``@path_rule``; one deliberately does not. Under the
default ``on_invalid_plugin=deny_route`` fail mode (decision D4) the unruled
route must be explicitly denied for *every* caller — human with permissions,
service principal (``ServiceSystem`` wildcard), and HelixAdmin — while the
ruled sibling keeps working. Under ``on_invalid_plugin=quarantine`` the whole
``/apis/harness-unruled`` namespace is fenced, ruled route included.
"""

from __future__ import annotations

from typing import ClassVar

from fastapi import APIRouter
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.service import NemoService, RouterSpec

router = APIRouter()


@router.get("/ruled")
@path_rule(callers=[CallerKind.PRINCIPAL])
async def ruled() -> dict[str, str]:
    """Properly annotated control route."""
    return {"route": "ruled", "status": "ok"}


@router.get("/unruled")
async def unruled() -> dict[str, str]:
    """Deliberately missing @path_rule — must be denied for everyone (fail-closed)."""
    return {"route": "unruled", "status": "you should never see this"}


class HarnessUnruledService(NemoService):
    """Service with a deliberate authoring error on exactly one route."""

    name: ClassVar[str] = "harness-unruled"

    def get_routers(self) -> list[RouterSpec]:
        return [RouterSpec(router=router, tag="Authz E2E Unruled Fixture")]

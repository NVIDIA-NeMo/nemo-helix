# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clean fixture plugin for the authz OIDC E2E harness.

No shipped plugin declares a ``SERVICE_PRINCIPAL``-only route, so the
caller-kind service-only deny is not observable on the
stock surface. This plugin provides:

- ``GET /apis/harness-fixture/probe/service-only`` — ``callers=[SERVICE_PRINCIPAL]``:
  humans (including PlatformAdmin, unless the exemption knob is set) must be
  denied; service principals allowed.
- ``GET /apis/harness-fixture/probe/open`` — ``callers=[PRINCIPAL]``, no
  permissions: control proving the plugin is mounted and a plain
  authenticated user reaches it (so the service-only 403 is meaningful).
"""

from __future__ import annotations

from typing import ClassVar

from fastapi import APIRouter
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.service import NemoService, RouterSpec

router = APIRouter()


@router.get("/probe/service-only")
@path_rule(callers=[CallerKind.SERVICE_PRINCIPAL])
async def probe_service_only() -> dict[str, str]:
    """Reachable only by service principals (callers=[SERVICE_PRINCIPAL])."""
    return {"probe": "service-only", "status": "ok"}


@router.get("/probe/open")
@path_rule(callers=[CallerKind.PRINCIPAL])
async def probe_open() -> dict[str, str]:
    """Reachable by any authenticated principal (mounted-and-working control)."""
    return {"probe": "open", "status": "ok"}


class HarnessFixtureService(NemoService):
    """Minimal clean service: one service-only route, one open control route."""

    name: ClassVar[str] = "harness-fixture"

    def get_routers(self) -> list[RouterSpec]:
        return [RouterSpec(router=router, tag="Authz E2E Fixture")]

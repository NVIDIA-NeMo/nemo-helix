# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List the post-run actions installed on the platform.

Mounted at ``/apis/agents/v2/post-run-actions``. Not workspace-scoped: what is
installed is a property of the platform, not of any workspace or Agent entity,
so any authenticated caller may read it.
"""

from __future__ import annotations

from fastapi import APIRouter
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.jobs.post_run_actions import list_post_run_actions as _list_installed_post_run_actions
from nemo_helix_plugin.agents.types import PostRunActionInfo
from nemo_helix_plugin.authz import CallerKind, path_rule

router = APIRouter()


@router.get("/post-run-actions", response_model=list[PostRunActionInfo], tags=["Agent Post-Run Actions"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL], permissions=[])
def list_post_run_actions() -> list[PostRunActionInfo]:
    """List every installed post-run action an execute-agent job can set as ``post_run_action.kind``."""
    # Sync on purpose: scanning entry points and importing plugin modules blocks,
    # so FastAPI runs this in its threadpool rather than on the event loop.
    return _list_installed_post_run_actions()

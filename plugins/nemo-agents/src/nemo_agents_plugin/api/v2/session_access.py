# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership-aware access helpers for persisted agent sessions."""

from __future__ import annotations

import logging

from fastapi import HTTPException
from nemo_agents_plugin.entities import AgentSession
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError

logger = logging.getLogger(__name__)


async def get_owned_session(
    entity_client: NemoEntitiesClient,
    *,
    workspace: str,
    name: str,
    effective_principal_id: str,
) -> AgentSession:
    """Resolve a session owned by the request's effective principal.

    Missing sessions and sessions owned by another principal intentionally have
    the same response so callers cannot use this endpoint to discover another
    principal's sessions.
    """
    try:
        session = await entity_client.get(AgentSession, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise session_not_found(workspace, name) from exc
    except Exception as exc:
        logger.exception("Failed to get session '%s'", name)
        raise HTTPException(status_code=500, detail="Failed to get session.") from exc

    if not _is_owned_session(session, workspace=workspace, effective_principal_id=effective_principal_id):
        raise session_not_found(workspace, name)

    return session


async def get_owned_session_by_id(
    entity_client: NemoEntitiesClient,
    *,
    workspace: str,
    session_id: str,
    effective_principal_id: str,
) -> AgentSession:
    """Resolve a session ID owned by the request's effective principal."""
    filter_obj = {"id": session_id}
    if effective_principal_id:
        filter_obj["created_by"] = effective_principal_id
    try:
        session = await entity_client.find_one(
            AgentSession,
            workspace=workspace,
            filter_obj=filter_obj,
        )
    except NemoEntityNotFoundError as exc:
        raise session_id_not_found(workspace, session_id) from exc
    except Exception as exc:
        logger.exception("Failed to get session ID '%s'", session_id)
        raise HTTPException(status_code=500, detail="Failed to get session.") from exc

    if session.id != session_id or not _is_owned_session(
        session,
        workspace=workspace,
        effective_principal_id=effective_principal_id,
    ):
        raise session_id_not_found(workspace, session_id)

    return session


def _is_owned_session(
    session: AgentSession,
    *,
    workspace: str,
    effective_principal_id: str,
) -> bool:
    """Return whether a session belongs to the request workspace and principal."""
    if session.workspace != workspace:
        return False
    # No-auth mode has no request principal. Entity writes still use the
    # platform service client and therefore carry a service ``created_by``;
    # workspace scoping is the applicable boundary in that mode.
    return not effective_principal_id or session.created_by == effective_principal_id


def session_not_found(workspace: str, name: str) -> HTTPException:
    """Return the non-disclosing response for unavailable sessions."""
    return HTTPException(
        status_code=404,
        detail=f"Session '{name}' not found in workspace '{workspace}'.",
    )


def session_id_not_found(workspace: str, session_id: str) -> HTTPException:
    """Return the non-disclosing response for an unavailable session ID."""
    return HTTPException(
        status_code=404,
        detail=f"Session ID '{session_id}' not found in workspace '{workspace}'.",
    )

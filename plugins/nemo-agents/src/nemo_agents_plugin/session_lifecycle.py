# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared persisted-session lifecycle helpers."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from urllib.parse import quote

import httpx
from nemo_agents_plugin.deployment_routing import get_deployment_endpoint, get_deployment_transport
from nemo_agents_plugin.entities import AgentDeployment, AgentSession
from nemo_platform_plugin.entity_client import NemoEntitiesClient

logger = logging.getLogger(__name__)

_FABRIC_CLEANUP_TIMEOUT_SECONDS = 5.0


def session_expiration_is_due(session: AgentSession, *, at: datetime) -> bool:
    """Return whether an active session has reached its persisted idle deadline."""
    expires_at = session.expires_at
    if expires_at is None:
        return False
    if expires_at.tzinfo is None or expires_at.utcoffset() is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    else:
        expires_at = expires_at.astimezone(UTC)
    return expires_at <= at.astimezone(UTC)


def _log_cleanup_failure(
    session: AgentSession,
    reason: str,
    *,
    exc_info: bool = False,
) -> None:
    """Record a non-blocking runtime-cleanup failure with entity context."""
    logger.warning(
        "Fabric runtime cleanup failed for session ID '%s' in workspace '%s' (deployment ID '%s'): %s.",
        session.id,
        session.workspace,
        session.deployment_id,
        reason,
        exc_info=exc_info,
    )


async def cleanup_fabric_runtime(
    entity_client: NemoEntitiesClient,
    session: AgentSession,
) -> None:
    """Best-effort removal of a session's process-local Fabric runtime.

    Cleanup never changes the persisted terminal state and never raises. Failures
    are observable through a contextual warning; Fabric idle eviction or a runtime
    restart provides eventual cleanup when the immediate request cannot complete.
    """
    try:
        deployment = await entity_client.find_one(
            AgentDeployment,
            workspace=session.workspace,
            filter_obj={"id": session.deployment_id},
        )
    except Exception as exc:
        _log_cleanup_failure(session, f"could not resolve deployment: {exc}", exc_info=True)
        return

    if deployment.id != session.deployment_id or deployment.workspace != session.workspace:
        _log_cleanup_failure(
            session,
            "resolved deployment did not match the session binding",
        )
        return

    endpoint = get_deployment_endpoint(deployment)
    if endpoint is None:
        _log_cleanup_failure(session, "deployment has no routable endpoint")
        return

    transport = get_deployment_transport(deployment)
    cleanup_url, routing_headers = transport.target(f"{endpoint.rstrip('/')}/v1/sessions/{quote(session.id, safe='')}")
    try:
        async with httpx.AsyncClient(
            timeout=_FABRIC_CLEANUP_TIMEOUT_SECONDS,
            follow_redirects=False,
            **transport.client_kwargs(),
        ) as client:
            response = await client.delete(cleanup_url, headers=routing_headers)
    except Exception as exc:
        _log_cleanup_failure(session, f"request failed: {exc}", exc_info=True)
        return

    # A missing runtime is already clean: the session may never have been invoked or may
    # have expired from the deployment's process-local registry.
    if response.status_code == 404 or 200 <= response.status_code < 300:
        return
    _log_cleanup_failure(session, f"request returned HTTP {response.status_code}")

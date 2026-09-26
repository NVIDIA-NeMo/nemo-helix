# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kern plugin service — registered under ``nemo.services``.

Exposes read-only views of the kern agent-fleet data sources:

- ``GET /v1/workspaces/{workspace}/flags``          — kern sleep flags (attention queue)
- ``POST /v1/workspaces/{workspace}/flags/{ulid}/ack`` — acknowledge a flag
- ``GET /v1/workspaces/{workspace}/failures``       — kmem failure-kind memory entries
- ``GET /v1/workspaces/{workspace}/decisions``      — gate decisions from SQLite
- ``GET /v1/workspaces/{workspace}/memory``         — all kmem memory entries (optional q)
- ``GET /v1/workspaces/{workspace}/memory/{ulid}``  — single kmem entry with full body
- ``GET /v1/workspaces/{workspace}/sessions``       — live session state

Full route: ``/apis/kern/v1/workspaces/{workspace}/{resource}``
(platform mounts all routes under ``/apis/<service-name>``,
then the RouterSpec prefix ``/v1/workspaces/{workspace}`` is appended).

The ``workspace`` path parameter is accepted but not used for filtering —
kern data is host-global (user max's machine).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entity_client import NemoEntitiesClient, get_entity_client
from nemo_helix_plugin.service import NemoService, RouterSpec
from pydantic import BaseModel

from nemo_kern._readers import (
    read_decisions,
    read_failures,
    read_flags,
    read_memory,
    read_memory_entry,
    write_ack,
)
from nemo_kern.authz import scope
from nemo_kern.config import KernConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models (CONTRACT shapes)
# ---------------------------------------------------------------------------


class FlagItem(BaseModel):
    ulid: str
    severity: str
    audience: str
    reason: str
    session_id: str | None
    created: str
    verdict: str | None
    acked: bool


class FailureItem(BaseModel):
    ulid: str
    kind: str
    category: str
    created: str
    summary: str


class DecisionItem(BaseModel):
    ts: str | None
    rule: str | None
    decision: str | None
    action: str | None
    stage: str | None
    latency_ms: float | None
    reason: str | None
    session_id: str | None


class MemoryItem(BaseModel):
    ulid: str
    kind: str
    tags: list[str]
    scope: str
    created: str
    title: str


class MemoryDetail(MemoryItem):
    body: str


class SessionItem(BaseModel):
    agent: str
    model: str | None
    status: str
    ctx_fill: float | None
    cost: float
    hold: str | None
    last_seen: str | None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class KernService(NemoService):
    """Kern plugin service.

    Registered under the ``nemo.services`` entry-point group.
    The platform mounts all routes under ``/apis/kern``.
    """

    name: ClassVar[str] = "kern"
    dependencies: ClassVar[list[str]] = []

    def get_routers(self) -> list[RouterSpec]:
        return [
            RouterSpec(
                _build_kern_router(),
                tag="Kern",
                description="Kern agent-fleet substrate: flags, failures, decisions, memory, sessions.",
                prefix="/v1/workspaces/{workspace}",
            )
        ]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def _build_kern_router() -> APIRouter:
    router = APIRouter()
    # NOTE: workspace path param is received but not used — kern data is host-global.
    # The parameter must appear in at least one route path so FastAPI resolves it.

    # ------------------------------------------------------------------
    # GET /flags
    # ------------------------------------------------------------------

    @router.get("/flags", response_model=list[FlagItem])
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
    async def list_flags(
        workspace: str,
        limit: int = Query(default=50, ge=1, le=500),
        severity: str | None = Query(default=None, description="Filter by severity: low|medium|high"),
    ) -> list[dict[str, Any]]:
        """Return kern sleep flags (attention queue), newest first."""
        cfg = KernConfig.get()
        return read_flags(
            flags_dir=cfg.flags_dir,
            proposals_file=cfg.proposals_file,
            ack_file=cfg.ack_file,
            limit=limit,
            severity=severity,
        )

    # ------------------------------------------------------------------
    # POST /flags/{ulid}/ack
    # ------------------------------------------------------------------

    @router.post("/flags/{ulid}/ack", response_model=dict)
    @scope.write
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
    async def ack_flag(workspace: str, ulid: str) -> dict[str, Any]:
        """Acknowledge a kern sleep flag.

        Records the ack in the local JSON ledger (not in the kern-sleep store).
        Returns an empty dict on success.
        """
        cfg = KernConfig.get()
        # Validate that the flag exists in the flags directory
        flag_file = cfg.flags_dir / f"{ulid}.json"
        if not flag_file.exists():
            raise HTTPException(status_code=404, detail=f"Flag '{ulid}' not found")
        try:
            write_ack(cfg.ack_file, ulid, verdict="acked")
        except OSError as exc:
            logger.exception("kern: failed to write ack for %s", ulid)
            raise HTTPException(status_code=500, detail="Failed to persist ack") from exc
        return {}

    # ------------------------------------------------------------------
    # GET /failures
    # ------------------------------------------------------------------

    @router.get("/failures", response_model=list[FailureItem])
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
    async def list_failures(
        workspace: str,
        limit: int = Query(default=50, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        """Return kmem memory entries where kind=failure, newest first."""
        cfg = KernConfig.get()
        return read_failures(memory_dir=cfg.memory_dir, limit=limit)

    # ------------------------------------------------------------------
    # GET /decisions
    # ------------------------------------------------------------------

    @router.get("/decisions", response_model=list[DecisionItem])
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
    async def list_decisions(
        workspace: str,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        """Return gate decisions from the kern SQLite database, newest first."""
        cfg = KernConfig.get()
        return read_decisions(db_path=cfg.decisions_db, limit=limit)

    # ------------------------------------------------------------------
    # GET /memory
    # ------------------------------------------------------------------

    @router.get("/memory", response_model=list[MemoryItem])
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
    async def list_memory(
        workspace: str,
        limit: int = Query(default=50, ge=1, le=500),
        q: str | None = Query(default=None, description="Substring search in title+body"),
    ) -> list[dict[str, Any]]:
        """Return kmem memory entries, newest first.

        The optional ``q`` parameter filters results to entries whose title or
        body text contains the given substring (case-insensitive).
        """
        cfg = KernConfig.get()
        return read_memory(memory_dir=cfg.memory_dir, limit=limit, q=q)

    # ------------------------------------------------------------------
    # GET /memory/{ulid}
    # ------------------------------------------------------------------

    @router.get("/memory/{ulid}", response_model=MemoryDetail)
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
    async def get_memory_entry(workspace: str, ulid: str) -> dict[str, Any]:
        """Return a single kmem memory entry with its full body text."""
        cfg = KernConfig.get()
        entry = read_memory_entry(memory_dir=cfg.memory_dir, ulid=ulid)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Memory entry '{ulid}' not found")
        return entry

    # ------------------------------------------------------------------
    # GET /sessions
    # ------------------------------------------------------------------

    @router.get("/sessions", response_model=list[SessionItem])
    @scope.read
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[])
    async def list_sessions(
        workspace: str,
        entity_client: NemoEntitiesClient = Depends(get_entity_client),
    ) -> list[dict[str, Any]]:
        """Return live (or recently active) kern agent sessions.

        Data is written by :class:`~nemo_kern.controller.KernSessionController`
        which tails the kern status log and upserts KernSession entities.
        This endpoint reads those entities back in a flat shape.
        """
        from nemo_kern.config import KernConfig
        from nemo_kern.entities import KernSession

        ws = KernConfig.get().workspace
        try:
            result = await entity_client.list(KernSession, workspace=ws)
            sessions = result.data
        except Exception:
            logger.warning("kern: failed to list KernSession entities", exc_info=True)
            return []

        from datetime import datetime, timezone

        return [
            {
                "agent": s.agent,
                "model": s.model,
                "status": s.status,
                "ctx_fill": s.ctx_fill,
                "cost": s.cost,
                "hold": s.hold,
                "last_seen": (
                    datetime.fromtimestamp(s.last_seen, timezone.utc).isoformat()
                    if s.last_seen
                    else None
                ),
            }
            for s in sessions
        ]

    return router

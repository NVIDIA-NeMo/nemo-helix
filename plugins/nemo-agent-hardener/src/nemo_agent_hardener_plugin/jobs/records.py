# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read/write the war-game's entity-store records.

AgentHardenerRun rows (create/pre-create/update + the data payload) and the manifest-entity reads the
run depends on (configured rounds, cached benign suite, persisting a reviewed suite). All best-effort:
recording never fails the war-game itself.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_agent_hardener_plugin.entities import (
    AGENT_HARDENER_MANIFEST_TYPE,
    AGENT_HARDENER_RUN_TYPE,
    AgentHardenerManifest,
    AgentHardenerRun,
)
from nemo_agent_hardener_plugin.jobs import benign_suite
from nemo_agent_hardener_plugin.jobs.errors import RunFailure
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.entities.client import EntitiesClient
from nemo_platform_plugin.entities.types import EntityCreateInput, EntityUpdate
from nemo_platform_plugin.entity_client import NemoEntitiesClient
from nemo_platform_plugin.job_context import JobContext

logger = logging.getLogger(__name__)


def _run_data(
    agent: str,
    port: int,
    manifest: str,
    status: str,
    returncode: int,
    job_id: str = "",
    hitlog_fileset: str = "",
    manifest_id: str = "",
    source_run: str = "",
    failure: RunFailure | None = None,
    events_fileset: str = "",
) -> dict[str, Any]:
    """The AgentHardenerRun data payload (whole record, since updates replace it).

    When *failure* is given (a failed run), its classified category/message/remediation are recorded and
    folded into the summary so the cause is visible even where only the summary is shown.
    """
    if status == "running":
        summary = f"running against {agent or 'agent'}"
    elif failure is not None:
        summary = f"failed ({failure.category}) against {agent or 'agent'}: {failure.message}"
    else:
        summary = f"{status} (exit {returncode}) against {agent or 'agent'}"
    return {
        "agent": agent,
        "job_id": job_id,
        "port": port,
        "manifest": manifest,
        "manifest_id": manifest_id,
        "status": status,
        "returncode": returncode,
        "summary": summary,
        "hitlog_fileset": hitlog_fileset,
        "events_fileset": events_fileset,
        "source_run": source_run,
        "error_category": failure.category if failure else "",
        "error_message": failure.message if failure else "",
        "error_remediation": failure.remediation if failure else "",
    }


def _create_run(sdk: Any, *, workspace: str, data: dict[str, Any]) -> str | None:
    """Persist a new AgentHardenerRun record; never fail the run on error. Returns its name."""
    if sdk is None or not hasattr(sdk, "entities"):
        return None
    try:
        entity = (
            client_from_platform(sdk, EntitiesClient)
            .create_entity(entity_type=AGENT_HARDENER_RUN_TYPE, workspace=workspace, body=EntityCreateInput(data=data))
            .data()
        )
        return getattr(entity, "name", None)
    except Exception:  # recording is best-effort, not part of the war-game
        logger.warning("failed to persist AgentHardenerRun record", exc_info=True)
        return None


async def _precreate_run(
    entity_client: NemoEntitiesClient, *, workspace: str, manifest_id: str, job_id: str, source_run: str = ""
) -> str | None:
    """Create the run record at submit time so Studio can open its live view immediately.

    Reads the agent/port straight off the manifest entity (no sandbox/materialization) and records a
    ``running`` row linked to the job. The worker reuses this record instead of creating its own. Best-effort:
    on any failure we return ``None`` and the worker falls back to creating the record when it starts.
    """
    try:
        manifest = await entity_client.get(AgentHardenerManifest, name=manifest_id, workspace=workspace)
        # Project-source manifests have no agent ref; label the run by the manifest name instead.
        label = manifest.agent or manifest.name
        run = AgentHardenerRun(
            workspace=workspace,
            agent=manifest.agent,
            port=manifest.port,
            job_id=job_id,
            manifest_id=manifest_id,
            status="running",
            returncode=-1,
            summary=f"running against {label}",
            source_run=source_run,
        )
        return (await entity_client.create(run)).name
    except Exception:  # pre-creation is an optimization; never block job submission on it
        logger.warning("failed to pre-create AgentHardenerRun for job %s", job_id, exc_info=True)
        return None


def _run_facts(sdk: Any, *, workspace: str, name: str) -> tuple[str, int]:
    """The ``(agent, port)`` already recorded on run *name*, or ``("", 0)`` (best-effort).

    Updates replace the whole record (see :func:`_run_data`), so a failure finalizing a pre-created
    row must carry these forward or the run is left showing no agent.
    """
    if sdk is None or not hasattr(sdk, "entities"):
        return "", 0
    try:
        record = (
            client_from_platform(sdk, EntitiesClient)
            .get_entity_by_name(name=name, entity_type=AGENT_HARDENER_RUN_TYPE, workspace=workspace)
            .data()
        )
        data = getattr(record, "data", {}) or {}
        port = data.get("port")
        return str(data.get("agent") or ""), int(port) if isinstance(port, int) else 0
    except Exception:  # reading back is best-effort; worst case the failure record loses the agent label
        logger.warning("failed to read back AgentHardenerRun %s", name, exc_info=True)
        return "", 0


def _update_run(sdk: Any, *, workspace: str, name: str, data: dict[str, Any]) -> None:
    """Overwrite an existing AgentHardenerRun record (e.g. running -> completed); best-effort."""
    if sdk is None or not hasattr(sdk, "entities"):
        return
    try:
        client_from_platform(sdk, EntitiesClient).update_entity_by_name(
            name=name, entity_type=AGENT_HARDENER_RUN_TYPE, workspace=workspace, body=EntityUpdate(data=data)
        ).data()
    except Exception:  # recording is best-effort, not part of the war-game
        logger.warning("failed to update AgentHardenerRun record", exc_info=True)


def _manifest_rounds(sdk: Any, manifest_id: str, ctx: JobContext) -> int:
    """The manifest's configured number of hardening rounds (>=1); 1 if unset/unavailable (best-effort)."""
    if sdk is None or not hasattr(sdk, "entities"):
        return 1
    try:
        record = (
            client_from_platform(sdk, EntitiesClient)
            .get_entity_by_name(name=manifest_id, entity_type=AGENT_HARDENER_MANIFEST_TYPE, workspace=ctx.workspace)
            .data()
        )
        rounds = (getattr(record, "data", {}) or {}).get("rounds")
        return rounds if isinstance(rounds, int) and rounds >= 1 else 1
    except Exception:  # reading config is best-effort; default to a single round
        logger.warning("failed to read rounds for manifest %s", manifest_id, exc_info=True)
        return 1


def _manifest_models(sdk: Any, manifest_id: str, ctx: JobContext) -> dict[str, Any]:
    """The manifest's stored default model selection (attack/analysis/agent), or ``{}`` (best-effort)."""
    if sdk is None or not hasattr(sdk, "entities"):
        return {}
    try:
        record = (
            client_from_platform(sdk, EntitiesClient)
            .get_entity_by_name(name=manifest_id, entity_type=AGENT_HARDENER_MANIFEST_TYPE, workspace=ctx.workspace)
            .data()
        )
        models = (getattr(record, "data", {}) or {}).get("models")
        return models if isinstance(models, dict) else {}
    except Exception:  # reading config is best-effort; fall back to agent-hardener's built-in model defaults
        logger.warning("failed to read models for manifest %s", manifest_id, exc_info=True)
        return {}


def _cached_benign_suite(sdk: Any, manifest_id: str, ctx: JobContext) -> list[dict[str, str]]:
    """The manifest's cached benign-suite rows, or ``[]`` when none/unavailable (best-effort)."""
    if sdk is None or not hasattr(sdk, "entities"):
        return []
    try:
        record = (
            client_from_platform(sdk, EntitiesClient)
            .get_entity_by_name(name=manifest_id, entity_type=AGENT_HARDENER_MANIFEST_TYPE, workspace=ctx.workspace)
            .data()
        )
        suite = (getattr(record, "data", {}) or {}).get("benign_suite") or []
        return [row for row in suite if isinstance(row, dict)]
    except Exception:  # reading the cache is best-effort; a miss just re-generates
        logger.warning("failed to read cached benign suite for manifest %s", manifest_id, exc_info=True)
        return []


def read_and_persist_suite(
    sdk: Any,
    ctx: JobContext,
    manifest_id: str | None,
    csv_path: Any,
    *,
    interview: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Parse the synthesized ``requests.csv`` and cache it on the manifest; return the suite rows.

    The shared line both the CLI ``synth-benign`` and Studio's serve-driven HITL converge on: read the
    suite agent-hardener wrote, then (when a ``manifest_id`` is known) persist it on the manifest entity.
    Persistence is best-effort — an empty suite or missing manifest is simply not cached.
    """
    suite = benign_suite.read_suite(csv_path)
    if manifest_id and suite:
        _persist_benign_suite(sdk, workspace=ctx.workspace, manifest_id=manifest_id, suite=suite, interview=interview)
    return suite


def _persist_benign_suite(
    sdk: Any,
    *,
    workspace: str,
    manifest_id: str,
    suite: list[dict[str, str]],
    interview: list[dict[str, Any]] | None = None,
) -> None:
    """Cache the reviewed benign suite (and the interview Q&A behind it) on the manifest; best-effort."""
    if sdk is None or not hasattr(sdk, "entities") or not suite:
        return
    try:
        record = (
            client_from_platform(sdk, EntitiesClient)
            .get_entity_by_name(name=manifest_id, entity_type=AGENT_HARDENER_MANIFEST_TYPE, workspace=workspace)
            .data()
        )
        data = dict(getattr(record, "data", {}) or {})
        data["benign_suite"] = suite
        if interview:
            data["benign_interview"] = interview
        client_from_platform(sdk, EntitiesClient).update_entity_by_name(
            name=manifest_id,
            entity_type=AGENT_HARDENER_MANIFEST_TYPE,
            workspace=workspace,
            body=EntityUpdate(data=data),
        )
    except Exception:  # caching is best-effort, not part of the war-game
        logger.warning("failed to cache benign suite on manifest %s", manifest_id, exc_info=True)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only routes over the ``AgentHardenerRun`` entity.

Mounted by the plugin service at ``/apis/agent-hardener/v2/workspaces/{workspace}``. War-game
runs are created by the job (``client.entities.create``), so the plugin only exposes reads:
list the agent's runs (Studio's Hardening tab) and fetch one. The entity is the same shape
on the wire as at rest, so it is returned directly.
"""

from __future__ import annotations

import logging
import tomllib
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from nemo_agent_hardener_plugin._perms import AgentHardenerRunPerms
from nemo_agent_hardener_plugin.agent_resolver import parse_agent_ref, strip_gateway_url
from nemo_agent_hardener_plugin.api.v2._filters import make_filter_dep
from nemo_agent_hardener_plugin.api.v2.manifests import refresh_manifest
from nemo_agent_hardener_plugin.api.v2.schemas import (
    ApplyMitigationRequest,
    ApplyMitigationResponse,
    ComposeDefenseRequest,
    ComposeDefenseResponse,
    RunFilter,
)
from nemo_agent_hardener_plugin.authz import scope
from nemo_agent_hardener_plugin.entities import AgentHardenerManifest, AgentHardenerRun
from nemo_agent_hardener_plugin.jobs.defenses import compose_defense
from nemo_agents_plugin.entities import Agent
from nemo_platform_plugin.authz import CallerKind, path_rule
from nemo_platform_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityNotFoundError,
    get_entity_client,
)
from nemo_platform_plugin.jobs.openapi_utils import generate_openapi_extra_params
from nemo_platform_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

#: The plugin kind agent-hardener registers inside the victim. Duplicated rather than imported: agent-hardener
#: is deliberately not a dependency of this plugin (its garak closure conflicts with the platform's).
_PLUGIN_KIND = "agent_hardener.pre_tool_verifier"

router = APIRouter()

_run_filter_dep = make_filter_dep(RunFilter)


@router.get(
    "/runs",
    tags=["Agent Hardener Runs"],
    openapi_extra=generate_openapi_extra_params(filter_schema=RunFilter),
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerRunPerms.LIST])
async def list_runs(
    workspace: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="-created_at"),
    filter: RunFilter = Depends(_run_filter_dep),
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> dict:
    """List war-game runs in the workspace, with pagination and an ``agent``/``status`` filter."""
    filter_dict = filter if isinstance(filter, dict) else filter.model_dump(exclude_none=True)
    try:
        result = await entity_client.list(
            AgentHardenerRun,
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_obj=filter_dict or None,
        )
    except Exception as exc:
        logger.exception("Failed to list agent-hardener runs in workspace '%s'", sanitize_for_log(workspace))
        raise HTTPException(status_code=500, detail="Failed to list agent-hardener runs.") from exc
    return {
        "data": [run.model_dump(mode="json") for run in result.data],
        "pagination": result.pagination.model_dump() if result.pagination else None,
        "sort": sort,
        "filter": filter or None,
    }


@router.get("/runs/{name}", response_model=AgentHardenerRun, tags=["Agent Hardener Runs"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerRunPerms.READ])
async def get_run(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentHardenerRun:
    """Get a single war-game run by name."""
    try:
        return await entity_client.get(AgentHardenerRun, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"AgentHardenerRun '{name}' not found in workspace '{workspace}'.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to get agent-hardener run '%s'", sanitize_for_log(name))
        raise HTTPException(status_code=500, detail="Failed to get agent-hardener run.") from exc


@router.post(
    "/runs/{name}/apply-mitigation",
    response_model=ApplyMitigationResponse,
    tags=["Agent Hardener Runs"],
)
@scope.write
# NOTE: this writes another plugin's entity (`Agent.config`) while holding only
# `agent-hardener.runs.apply`. Requiring `agents.agents.create` alongside is not possible — the platform
# fail-closes on permission ids outside a service's own namespace — so `agent-hardener.runs.apply` is
# effectively an agent-write grant. Treat it as such when assigning it.
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerRunPerms.APPLY])
async def apply_mitigation(
    workspace: str,
    name: str,
    body: ApplyMitigationRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> ApplyMitigationResponse:
    """Adopt a run's hardened guardrails onto the run's target agent config (no redeploy).

    This is the *only* place ``relay.components[]`` is produced. The guardrail runs from a plugins.toml
    inside the victim; the agent registry stores agent config, so adoption re-homes the same component
    onto the entity. Near-identity, not a translation: the ``config`` object is the one Relay loaded.

    Reverses the Inference-Gateway injection so the stored config stays deployment-neutral. The user
    must redeploy the agent for the guardrails to take effect.
    """
    try:
        guardrails = tomllib.loads(body.guardrails_toml)
    except tomllib.TOMLDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"guardrails_toml is not valid TOML: {exc}") from exc
    components = _relay_components(guardrails)
    if not components:
        raise HTTPException(status_code=422, detail="guardrails_toml declares no Agent Hardener guardrail component.")

    try:
        run = await entity_client.get(AgentHardenerRun, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"AgentHardenerRun '{name}' not found in workspace '{workspace}'."
        ) from exc

    if not run.agent:
        raise HTTPException(status_code=409, detail=f"Run '{name}' has no target agent to update.")
    # A project run carries its *manifest* name in `agent`, so the guard above passes and the lookup
    # below would target whatever agent happens to share that name.
    await _reject_project_source(entity_client, workspace, run.manifest_id, name)
    agent_ws, agent_name = parse_agent_ref(run.agent, workspace)

    try:
        agent = await entity_client.get(Agent, name=agent_name, workspace=agent_ws)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Agent '{agent_name}' not found in workspace '{agent_ws}'."
        ) from exc

    agent.config = _with_relay_components(strip_gateway_url(dict(agent.config)), components)
    try:
        await entity_client.update(agent)
    except Exception as exc:
        logger.exception("Failed to apply mitigation to agent '%s'", sanitize_for_log(agent_name))
        raise HTTPException(status_code=500, detail="Failed to update the agent config.") from exc

    # Manifests are frozen targets, so the agent edit we just made would not reach the next run.
    # Refresh the manifest this run came from, keeping "harden -> apply -> re-run to confirm" intact.
    refreshed = await _refresh_source_manifest(entity_client, workspace, run.manifest_id)

    detail = f"Updated '{agent_name}' with the hardened guardrails. Redeploy the agent to activate them."
    if run.manifest_id and not refreshed:
        detail += (
            f" Manifest '{run.manifest_id}' could not be refreshed automatically — run "
            f"`nemo agent-hardener refresh --manifest-id {run.manifest_id}` before re-running, or it will "
            "war-game the agent as it was before this change."
        )
    return ApplyMitigationResponse(applied=True, agent=agent_name, detail=detail)


async def _reject_project_source(
    entity_client: NemoEntitiesClient, workspace: str, manifest_id: str, run_name: str
) -> None:
    """Refuse adoption for a bring-your-own manifest, which has no agent entity to adopt onto."""
    if not manifest_id:
        return
    try:
        manifest = await entity_client.get(AgentHardenerManifest, name=manifest_id, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run '{run_name}' references manifest '{manifest_id}', which no longer exists, so its "
                "target cannot be confirmed. Re-run against an existing manifest before applying."
            ),
        ) from exc
    if manifest.source_type == "project":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run '{run_name}' targets the bring-your-own manifest '{manifest_id}', which has no "
                "registered agent to update. Apply the hardened guardrails to your own image instead."
            ),
        )


def _relay_components(guardrails: dict[str, Any]) -> list[dict[str, Any]]:
    """The Agent Hardener components declared in a plugins.toml, in the shape ``relay.components[]`` takes.

    A near-identity: the war-game delivers guardrails as top-level ``[[components]]`` entries, which
    is already ``{kind, enabled, config}``. Re-emitted rather than passed through so a hand-edited
    file cannot carry an unrelated component kind onto the agent entity.
    """
    return [
        {"kind": _PLUGIN_KIND, "enabled": True, "config": entry["config"]}
        for entry in guardrails.get("components", [])
        if isinstance(entry, dict)
        and entry.get("kind") == _PLUGIN_KIND
        and isinstance(entry.get("config"), dict)
        and entry["config"].get("guardrails")
    ]


def _with_relay_components(config: dict[str, Any], components: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach *components* to a ``nemo-agents-spec-v1`` config.

    Carried under ``telemetry`` because that is the section the spec already forwards to Relay. The
    translator does not read ``relay_components`` yet — see the follow-up in
    ``nemo_agents_plugin.fabric.translator._apply_telemetry`` — so today this records the adopted
    guardrail on the entity rather than activating it on the next deploy. Storing it in the shape the
    passthrough will take means adoption starts working when that lands, with no second migration.
    """
    telemetry = dict(config.get("telemetry") or {})
    telemetry["relay_components"] = components
    return {**config, "telemetry": telemetry}


async def _refresh_source_manifest(entity_client: NemoEntitiesClient, workspace: str, manifest_id: str) -> bool:
    """Re-freeze the manifest a run came from; return whether it happened.

    Best-effort on purpose: the mitigation is already applied to the agent, so a refresh failure must
    not fail the request — but the caller tells the operator, because a silently stale manifest would
    make the next run measure the unhardened agent and look like the fix did nothing.
    """
    if not manifest_id:
        return False
    try:
        await refresh_manifest(workspace=workspace, name=manifest_id, entity_client=entity_client)
        return True
    except Exception:
        logger.warning(
            "could not refresh manifest '%s' after apply-mitigation", sanitize_for_log(manifest_id), exc_info=True
        )
        return False


@router.post(
    "/runs/{name}/compose-defense",
    response_model=ComposeDefenseResponse,
    tags=["Agent Hardener Runs"],
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerRunPerms.COMPOSE])
async def compose_defense_route(
    workspace: str,
    name: str,
    body: ComposeDefenseRequest,
) -> ComposeDefenseResponse:
    """Compose a chosen subset of a run's recommended defenses into deployable workflow + policy YAML.

    Keeps only the selected guardrails in the workflow and picks the hardened-vs-baseline policy. Powers
    the harden flow's live preview and feeds the composed YAMLs to a sanity-check (validate-only) run.
    """
    try:
        guardrails_toml, policy_yaml = compose_defense(body.mitigations, body.selected_defense_ids)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to compose the selected defenses: {exc}") from exc
    return ComposeDefenseResponse(guardrails_toml=guardrails_toml, policy_yaml=policy_yaml)


@router.delete("/runs/{name}", status_code=204, tags=["Agent Hardener Runs"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentHardenerRunPerms.DELETE])
async def delete_run(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> None:
    """Delete a war-game run record. The underlying platform job is cancelled/deleted separately."""
    try:
        await entity_client.delete(AgentHardenerRun, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"AgentHardenerRun '{name}' not found in workspace '{workspace}'."
        ) from exc
    except Exception as exc:
        logger.exception("Failed to delete agent-hardener run '%s'", sanitize_for_log(name))
        raise HTTPException(status_code=500, detail="Failed to delete agent-hardener run.") from exc

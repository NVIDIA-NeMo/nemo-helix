# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Network access of an agent's OpenShell sandboxes: what was requested, and what is remembered.

A sandbox that is denied a connection files a request on the OpenShell gateway. These
routes list the requests of every sandbox running the agent (its deployments and its
execute jobs), decide them, and keep approved rules on the agent so later sandboxes
start with them.

Prototype scope:

* The gateway is reached through the deployments plugin's ``openshell`` executor, so one
  must be configured, and job sandboxes are only found when the Jobs service uses the
  same gateway and gateway workspace.
* A request is only visible while its sandbox exists. Once a finished job's sandbox is
  removed, its requests are gone and can no longer be remembered.
* Remembering a rule needs only the permission to create agents; there is no separate
  permission for approving network access.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import yaml
from fastapi import APIRouter, Depends, HTTPException
from nemo_agents_plugin.api.v2._perms import AgentPerms
from nemo_agents_plugin.api.v2.dependencies import get_entity_client
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.entities import Agent
from nemo_agents_plugin.runner.registry import get_runner_registry
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError
from nemo_helix_plugin.openshell.network_policy import agent_from_labels, rule_to_dict
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Set by the deployments runner and the Jobs service on the sandboxes they create.
_DEPLOYMENT_LABEL = "nemo.agents/deployment"
_JOB_LABEL = "nhx.nvidia.com/job_id"


class NetworkRequest(BaseModel):
    """One network connection a sandbox asked for."""

    id: str
    status: str = Field(description="pending, approved or rejected.")
    rule_name: str
    endpoints: list[str] = Field(description="Each destination as host:port.")
    binary: str = Field(description="The executable that opened the connection.")
    rationale: str = ""
    security_notes: str = ""
    rejection_reason: str = ""
    hit_count: int = 0
    first_seen: str | None = None
    last_seen: str | None = None
    decided: str | None = None
    rule: dict[str, Any] = Field(description="The rule that approving adds, in the policy YAML shape.")
    remembered: bool = Field(description="Whether the agent already carries this rule.")


class AgentSandbox(BaseModel):
    """A sandbox running the agent, with its network requests."""

    sandbox: str
    kind: Literal["deployment", "job", "sandbox"]
    name: str = Field(description="The deployment or job the sandbox belongs to.")
    phase: str
    requests: list[NetworkRequest] = Field(default_factory=list)
    error: str = Field(default="", description="Why the requests could not be read, if they could not.")


class AgentNetworkPolicy(BaseModel):
    rules: dict[str, dict[str, Any]] = Field(description="Rules the agent carries into every new sandbox.")
    agent_yaml: str = Field(description="The same rules as an agent.yaml fragment.")
    sandboxes: list[AgentSandbox] = Field(default_factory=list)
    gateway_error: str = Field(default="", description="Why sandboxes could not be listed, if they could not.")


class ApproveRequest(BaseModel):
    remember: bool = Field(
        default=False, description="Also keep the rule on the agent, so new sandboxes start with it."
    )


class RejectRequest(BaseModel):
    reason: str = ""
    forget: bool = Field(default=False, description="Also drop the same rule from the agent, if it carries it.")


def agent_yaml_fragment(rules: dict[str, dict[str, Any]]) -> str:
    """The rules as they would be written in agent.yaml."""
    if not rules:
        return ""
    return yaml.safe_dump({"helix_registration": {"network_policies": rules}}, sort_keys=False)


async def _get_agent(entity_client: NemoEntitiesClient, workspace: str, name: str) -> Agent:
    try:
        return await entity_client.get(Agent, name=name, workspace=workspace)
    except NemoEntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found in workspace '{workspace}'.") from exc


async def _gateway() -> Any:
    """The deployments plugin's OpenShell backend, which holds the gateway connection."""
    runner = get_runner_registry().backend_for("openshell")
    resolve = getattr(runner, "openshell_backend", None)
    if resolve is None:
        raise ValueError("This platform has no OpenShell executor for agents.")
    return await resolve()


def _detail(exc: Exception) -> str:
    """A gRPC error's own message, or the exception's."""
    details = getattr(exc, "details", None)
    return (details() if callable(details) else None) or str(exc) or type(exc).__name__


def _timestamp(value: Any) -> str | None:
    return value.ToJsonString() if value.seconds else None


def _same_access(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two rules allow the same thing. Their names do not matter."""
    return {k: v for k, v in a.items() if k != "name"} == {k: v for k, v in b.items() if k != "name"}


def _remembered_as(rules: dict[str, dict[str, Any]], rule: dict[str, Any]) -> str | None:
    """The name the agent carries ``rule`` under, or None if it does not carry it."""
    return next((name for name, existing in rules.items() if _same_access(existing, rule)), None)


def _remember(rules: dict[str, dict[str, Any]], name: str, rule: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``rules`` with ``rule`` added, under a free name if ``name`` is taken by a different rule.

    The gateway names a request after its destination, so two sandboxes can file the same
    name for different executables.
    """
    if _remembered_as(rules, rule) is not None:
        return rules
    free, suffix = name, 2
    while free in rules:
        free, suffix = f"{name}_{suffix}", suffix + 1
    return {**rules, free: {**rule, "name": free}}


def _request(chunk: Any, rules: dict[str, dict[str, Any]]) -> NetworkRequest:
    rule = rule_to_dict(chunk.proposed_rule)
    return NetworkRequest(
        id=chunk.id,
        status=chunk.status,
        rule_name=chunk.rule_name,
        endpoints=[f"{endpoint.host}:{endpoint.port}" for endpoint in chunk.proposed_rule.endpoints],
        binary=chunk.binary,
        rationale=chunk.rationale,
        security_notes=chunk.security_notes,
        rejection_reason=chunk.rejection_reason,
        hit_count=chunk.hit_count,
        first_seen=_timestamp(chunk.first_seen_time),
        last_seen=_timestamp(chunk.last_seen_time),
        decided=_timestamp(chunk.decided_time),
        rule=rule,
        remembered=_remembered_as(rules, rule) is not None,
    )


def _owned_by(sandbox: Any, workspace: str, agent: str) -> bool:
    return agent_from_labels(sandbox.metadata.labels) == (workspace, agent)


async def _sandboxes(gateway: Any, agent: Agent) -> list[AgentSandbox]:
    from openshell._proto import openshell_pb2 as pb  # ty: ignore[unresolved-import]

    found: list[AgentSandbox] = []
    for sandbox in await gateway.list_sandboxes():
        if not _owned_by(sandbox, agent.workspace, agent.name):
            continue
        labels = sandbox.metadata.labels
        if _DEPLOYMENT_LABEL in labels:
            kind, name = "deployment", labels[_DEPLOYMENT_LABEL]
        elif _JOB_LABEL in labels:
            kind, name = "job", labels[_JOB_LABEL]
        else:
            kind, name = "sandbox", sandbox.metadata.name
        entry = AgentSandbox(
            sandbox=sandbox.metadata.name, kind=kind, name=name, phase=_phase(pb, sandbox.status.phase)
        )
        try:
            chunks = await gateway.get_draft_chunks(sandbox.metadata.name)
        except Exception as exc:
            entry.error = _detail(exc)
        else:
            entry.requests = [_request(chunk, agent.network_policies) for chunk in chunks]
        found.append(entry)
    return found


def _phase(pb: Any, phase: int) -> str:
    return pb.SandboxPhase.Name(phase).removeprefix("SANDBOX_PHASE_").lower()


async def _view(agent: Agent) -> AgentNetworkPolicy:
    view = AgentNetworkPolicy(rules=agent.network_policies, agent_yaml=agent_yaml_fragment(agent.network_policies))
    try:
        view.sandboxes = await _sandboxes(await _gateway(), agent)
    except Exception as exc:
        # The remembered rules are still worth showing when the gateway is not.
        logger.warning("Could not list OpenShell sandboxes for agent %s/%s: %s", agent.workspace, agent.name, exc)
        view.gateway_error = _detail(exc)
    return view


async def _find_chunk(agent: Agent, sandbox_name: str, request_id: str) -> tuple[Any, Any]:
    """The gateway and the chunk, after checking the sandbox runs this agent."""
    try:
        gateway = await _gateway()
        sandboxes = await gateway.list_sandboxes()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenShell gateway: {_detail(exc)}") from exc
    if not any(s.metadata.name == sandbox_name and _owned_by(s, agent.workspace, agent.name) for s in sandboxes):
        raise HTTPException(status_code=404, detail=f"Agent '{agent.name}' has no sandbox '{sandbox_name}'.")
    try:
        chunks = await gateway.get_draft_chunks(sandbox_name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenShell gateway: {_detail(exc)}") from exc
    for chunk in chunks:
        if chunk.id == request_id:
            return gateway, chunk
    raise HTTPException(status_code=404, detail=f"Sandbox '{sandbox_name}' has no request '{request_id}'.")


@router.get("/agents/{name}/network-policy", response_model=AgentNetworkPolicy, tags=["Agents"])
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentPerms.READ])
async def get_network_policy(
    workspace: str,
    name: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentNetworkPolicy:
    """The agent's remembered network rules and the requests of its running sandboxes."""
    return await _view(await _get_agent(entity_client, workspace, name))


@router.post(
    "/agents/{name}/network-policy/sandboxes/{sandbox}/requests/{request_id}/approve",
    response_model=AgentNetworkPolicy,
    tags=["Agents"],
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentPerms.CREATE])
async def approve_network_request(
    workspace: str,
    name: str,
    sandbox: str,
    request_id: str,
    body: ApproveRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentNetworkPolicy:
    """Let the sandbox make the connection, and optionally remember the rule on the agent.

    A request that was already approved can still be remembered.
    """
    agent = await _get_agent(entity_client, workspace, name)
    gateway, chunk = await _find_chunk(agent, sandbox, request_id)
    if chunk.status == "pending":
        try:
            await gateway.approve_draft_chunk(sandbox, chunk)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"OpenShell gateway: {_detail(exc)}") from exc
    elif chunk.status != "approved":
        raise HTTPException(status_code=409, detail=f"Request is {chunk.status}; it cannot be approved.")
    if body.remember:
        agent.network_policies = _remember(agent.network_policies, chunk.rule_name, rule_to_dict(chunk.proposed_rule))
        agent = await entity_client.update(agent)
    return await _view(agent)


@router.post(
    "/agents/{name}/network-policy/sandboxes/{sandbox}/requests/{request_id}/reject",
    response_model=AgentNetworkPolicy,
    tags=["Agents"],
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentPerms.CREATE])
async def reject_network_request(
    workspace: str,
    name: str,
    sandbox: str,
    request_id: str,
    body: RejectRequest,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentNetworkPolicy:
    """Keep the connection blocked, and optionally drop the same rule from the agent."""
    agent = await _get_agent(entity_client, workspace, name)
    gateway, chunk = await _find_chunk(agent, sandbox, request_id)
    if chunk.status != "pending":
        raise HTTPException(status_code=409, detail=f"Request is already {chunk.status}.")
    try:
        await gateway.reject_draft_chunk(sandbox, chunk.id, body.reason)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenShell gateway: {_detail(exc)}") from exc
    remembered_as = _remembered_as(agent.network_policies, rule_to_dict(chunk.proposed_rule))
    if body.forget and remembered_as is not None:
        agent.network_policies = {k: v for k, v in agent.network_policies.items() if k != remembered_as}
        agent = await entity_client.update(agent)
    return await _view(agent)


@router.delete("/agents/{name}/network-policy/rules/{rule}", response_model=AgentNetworkPolicy, tags=["Agents"])
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[AgentPerms.CREATE])
async def forget_network_rule(
    workspace: str,
    name: str,
    rule: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> AgentNetworkPolicy:
    """Drop a remembered rule. Sandboxes that are already running keep it."""
    agent = await _get_agent(entity_client, workspace, name)
    if rule not in agent.network_policies:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' has no network rule '{rule}'.")
    agent.network_policies = {k: v for k, v in agent.network_policies.items() if k != rule}
    return await _view(await entity_client.update(agent))

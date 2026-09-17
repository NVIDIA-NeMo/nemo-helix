# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent Hardener plugin API request/query schemas.

The persisted entities are the same shape on the wire as at rest, so the read routes return them
directly. This module holds the list-endpoint query filters (extending ``NemoFilter``,
``extra="forbid"`` so a misspelled key 422s) plus the ``POST /manifests`` init request body.
"""

from __future__ import annotations

from typing import Any, Literal

from nemo_agent_hardener_plugin.entities import ManifestSource
from nemo_agent_hardener_plugin.model_config import WarGameModels
from nemo_platform_plugin.schema import NemoFilter
from pydantic import BaseModel, Field


class RunFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/runs``."""

    agent: str | None = Field(
        default=None,
        description="Filter to runs targeting this agent reference (workspace/name).",
    )
    manifest_id: str | None = Field(
        default=None,
        description="Filter to runs launched from this manifest (scopes 'replay last run').",
    )
    status: str | None = Field(
        default=None,
        description="Filter to runs with this status ('running', 'completed', or 'failed').",
    )


class ManifestFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/manifests``."""

    agent: str | None = Field(default=None, description="Filter to manifests for this agent reference.")
    source_type: str | None = Field(default=None, description="Filter by source ('agent' or 'project').")


class ManifestInit(BaseModel):
    """Body for ``POST /v2/workspaces/{workspace}/manifests`` — scaffold a named manifest.

    ``agent`` resolves a deployed agent; ``project`` builds the manifest from an uploaded NAT project
    (``project_fileset`` + the confirmed detection answers) by shelling ``agent-hardener init --yes``.
    """

    name: str = Field(description="User-defined manifest id (unique within the workspace).")
    source_type: ManifestSource = Field(default="agent", description="Scaffold source ('agent' or 'project').")
    agent: str | None = Field(default=None, description="Agent reference (required when source_type='agent').")
    # Project source (source_type='project') — the confirmed answers from the inspect step.
    project_fileset: str | None = Field(default=None, description="Fileset ref of the uploaded NAT project bundle.")
    manifest_yaml: str | None = Field(
        default=None,
        description="Pre-built agent-hardener manifest (project source). The CLI runs agent-hardener's own "
        "interactive `init` at the operator's terminal and sends the result; omit it and the server "
        "builds one with `init --yes`, which is what Studio does since it has no TTY.",
    )
    workflow: str | None = Field(
        default=None, description="Chosen workflow path within the project (project-relative)."
    )
    launch_mode: str | None = Field(
        default=None,
        description="Victim launch mode: 'workflow' (a generic image built from the project) or 'byo' "
        "(built from the project's own Dockerfile). 'byo' needs either a `dockerfile` here or a "
        "`manifest_yaml` that already carries one. Derived from the manifest when omitted.",
    )
    dockerfile: str | None = Field(
        default=None,
        description="Project-relative Dockerfile to build the victim image from, instead of a generic one — "
        "for agents needing system packages or a custom base image. Requires `binaries`, and a `workflow` "
        "(given or detected): the image is how the environment is built, the workflow is what gets served "
        "and hardened. The image must carry a 'sandbox' user/group, iproute2, and `nat` on the default PATH.",
    )
    binaries: list[str] | None = Field(
        default=None,
        description="In-container glob patterns scoping which processes may egress, e.g. '/app/.venv/bin/**'. "
        "Required with `dockerfile`: a BYO image's layout is unknown, so the sandbox policy cannot infer it.",
    )
    port: int | None = Field(default=None, description="Victim port (defaults to 8000).")
    secrets: list[str] | None = Field(default=None, description="Secret names the victim requires.")
    secrets_file: str | None = Field(default=None, description="Dotenv path within the project holding the secrets.")
    egress: list[str] | None = Field(
        default=None,
        description="Allow-listed egress host[:port] entries the victim may reach (external hosts the agent "
        "calls, e.g. inference-api.nvidia.com); baked into the manifest by `init --egress`.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Non-secret environment variables for the victim (agent-hardener's `agent.env`). Stored in "
        "plaintext on the manifest — credentials belong in `secrets`, which names them and "
        "resolves the values from the Secrets store at run time.",
    )
    backends: list[str] | None = Field(
        default=None,
        description="Route-only host backends the agent's tools call, each 'NAME:PORT[,PORT2]' (e.g. "
        "'finance:8086'). Rewrites the agent's localhost:PORT to host.docker.internal:PORT and opens the "
        "sandbox->host route; passed to `init --backend`.",
    )
    models: WarGameModels | None = Field(
        default=None,
        description="Stored default model selection (attack/analysis/agent groups); omit to use agent-hardener's "
        "built-in defaults.",
    )


class ValidateModelRequest(BaseModel):
    """Body for ``POST /v2/workspaces/{workspace}/model-config/validate`` — probe a model choice."""

    model: str | None = Field(default=None, description="Model name to verify against the endpoint's model list.")
    base_url: str = Field(description="OpenAI-compatible endpoint to probe (`GET {base_url}/models`).")
    api_key_secret: str | None = Field(
        default=None, description="Secret name holding the provider key; omitted probes without auth."
    )


class ValidateModelResponse(BaseModel):
    """Verdict for a model choice; ``available`` lists what the credentials can reach so the UI offers real options."""

    ok: bool = Field(description="True when the endpoint is reachable, authorized, and serves the model.")
    reason: str = Field(default="", description="'' | 'auth' | 'unreachable' | 'unknown_model'.")
    available: list[str] = Field(default_factory=list, description="Model ids the credentials can reach.")
    detail: str = Field(default="", description="Human-readable diagnostic (status code / transport error).")


class InspectProjectRequest(BaseModel):
    """Body for ``POST /v2/workspaces/{workspace}/manifests/inspect`` — detect an uploaded project."""

    project_fileset: str = Field(description="Fileset ref of the uploaded NAT project bundle to inspect.")


class InspectProjectResponse(BaseModel):
    """Detection facts + defaults for the upload wizard (the parsed ``agent-hardener inspect --json`` output)."""

    project_dir: str = Field(default="", description="Detected installable project root (relative to the bundle).")
    workflows: list[str] = Field(default_factory=list, description="Discovered workflow paths (project-relative).")
    dockerfiles: list[str] = Field(default_factory=list, description="Discovered Dockerfile paths (project-relative).")
    suggested_launch_mode: str = Field(default="workflow", description="'workflow' or 'byo'.")
    default_agent_name: str = Field(default="", description="Suggested agent name.")
    default_port: int = Field(default=8000, description="Suggested victim port.")
    secrets_file: str = Field(default="", description="Detected dotenv path (project-relative), or empty.")
    secret_names: list[str] = Field(default_factory=list, description="Secret names found in the dotenv file.")
    egress: list[str] = Field(default_factory=list, description="External hosts the agent reaches (allow-list).")
    backend_ports: list[int] = Field(
        default_factory=list,
        description="Local host-backend ports detected in the workflow (localhost:PORT the tools call).",
    )


class InspectAgentRequest(BaseModel):
    """Body for ``POST /v2/workspaces/{workspace}/manifests/inspect-agent`` — a deployed agent ref."""

    agent: str = Field(description="Deployed agent reference (``workspace/name`` or ``name``).")


class InspectAgentResponse(BaseModel):
    """Auto-derived defaults for the deployed-agent create form (port + secret names, editable)."""

    agent: str = Field(description="Resolved ``workspace/name`` of the agent.")
    port: int = Field(description="Victim port derived from the running deployment (else the default).")
    secrets: list[str] = Field(default_factory=list, description="Secret names derived from the agent config.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal notes (e.g. no running deployment).")


class ManifestUpdate(BaseModel):
    """Body for ``PATCH /v2/workspaces/{workspace}/manifests/{name}`` — edit an existing manifest.

    Only editable fields; omitted fields are left unchanged. The agent source is immutable (delete +
    recreate to retarget).
    """

    benign_suite: list[dict[str, str]] | None = Field(
        default=None, description="Replace the cached benign suite (tool,payload,label,rationale,persona rows)."
    )
    port: int | None = Field(default=None, description="Victim port the war-game will target.")
    egress: list[str] | None = Field(
        default=None, description="Allow-listed egress host[:port] entries the victim may reach."
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Non-secret environment variables for the victim (agent-hardener's `agent.env`). Stored in "
        "plaintext on the manifest — credentials belong in `secrets`, which names them and "
        "resolves the values from the Secrets store at run time.",
    )
    defenders: list[str] | None = Field(
        default=None,
        description="Enabled defender keys ('guardrails','openshell'); empty means agent-hardener defaults.",
    )
    attack_intensity: Literal["light", "standard", "thorough"] | None = Field(
        default=None, description="Attacker (garak) effort preset."
    )
    rounds: int | None = Field(
        default=None, ge=1, description="Number of iterative hardening rounds (agent-hardener `run --rounds`)."
    )
    models: WarGameModels | None = Field(
        default=None, description="Replace the stored default model selection (attack/analysis/agent groups)."
    )


class ApplyMitigationRequest(BaseModel):
    """Body for ``POST /v2/workspaces/{workspace}/runs/{name}/apply-mitigation`` — adopt the hardened workflow.

    The client passes the hardened workflow YAML from the run's mitigations artifact. The endpoint reverses
    the Inference-Gateway injection and writes it onto the run's target agent config (no redeploy).
    """

    workflow_yaml: str = Field(description="Hardened NAT workflow YAML (the mitigations 'after' document).")


class ApplyMitigationResponse(BaseModel):
    """Result of applying a hardened workflow to an agent."""

    applied: bool = Field(description="True when the agent config was updated.")
    agent: str = Field(description="Name of the agent whose config was updated.")
    detail: str = Field(description="Human-readable note (e.g. a reminder to redeploy).")


class ComposeDefenseRequest(BaseModel):
    """Body for ``POST /v2/workspaces/{workspace}/runs/{name}/compose-defense`` — build a chosen defense subset.

    The client passes the run's ``mitigations`` artifact (which it already fetched for the recommendations
    view) plus the ids of the defenses to keep. The endpoint composes the workflow with only the selected
    guardrails and picks the hardened-vs-baseline policy, for live preview and to feed a sanity-check run.
    """

    mitigations: dict[str, Any] = Field(description="The run's mitigations artifact (its 'defenses'/workflow/policy).")
    selected_defense_ids: list[str] = Field(
        default_factory=list, description="Ids of the defenses to keep (guardrail ids and/or 'openshell_policy')."
    )


class ComposeDefenseResponse(BaseModel):
    """The composed workflow + policy for the selected defenses."""

    workflow_yaml: str | None = Field(default=None, description="Workflow with only the selected guardrails, or null.")
    policy_yaml: str | None = Field(
        default=None, description="Hardened policy if selected, else the baseline, or null."
    )

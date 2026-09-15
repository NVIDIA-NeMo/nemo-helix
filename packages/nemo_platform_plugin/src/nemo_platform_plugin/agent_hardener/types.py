# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/response DTOs for the Agent Hardener service HTTP contract."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, NotRequired, TypedDict

from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from nemo_platform_plugin.jobs.types import validate_output_location
from nemo_platform_plugin.schema import DatetimeFilter, Filter, StringFilter
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

AGENT_HARDENER_RUN_TYPE = "agent_hardener_run"
AGENT_HARDENER_MANIFEST_TYPE = "agent_hardener_manifest"

JsonMap = dict[str, JsonValue]
StringMap = dict[str, str]
BenignSuiteRow = dict[str, str]
RunStatus = Literal["running", "completed", "failed"]
ManifestSource = Literal["agent", "project"]
AttackIntensity = Literal["light", "standard", "thorough"]

ATTACK_DEFAULT_MODEL = "aws/anthropic/claude-opus-4-5"
ATTACK_DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1/"
ANALYSIS_DEFAULT_MODEL = "nvidia/nvidia/Nemotron-3-Nano-30B-A3B"
ANALYSIS_DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"


class JobsSortField(str, Enum):
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"


class ModelChoice(BaseModel):
    """One group's model selection. Every field is optional; ``None`` → the group's built-in default."""

    model: str | None = Field(default=None, description="Model name/URN; null uses the group default.")
    base_url: str | None = Field(default=None, description="Custom OpenAI-compatible endpoint; null uses the default.")
    api_key_secret: str | None = Field(
        default=None,
        description="Name of a NeMo Secret holding the provider API key for a custom endpoint; null uses the "
        "platform's provisioned agent-hardener inference key.",
    )


class WarGameModels(BaseModel):
    """The three model groups for a war-game. An unset group uses agent-hardener's built-in default."""

    attack: ModelChoice | None = Field(default=None, description="garak red-team + detector model.")
    analysis: ModelChoice | None = Field(
        default=None, description="Defenders + benign validator (synth suite-generation + judge) model."
    )
    safety: ModelChoice | None = Field(
        default=None,
        description="Guardrail middleware LLM (agent-hardener's `safety_llm`); unset copies the victim's own LLM. "
        "Only `model` applies — agent-hardener pins this LLM's endpoint and key when it writes the guardrail.",
    )


class ModelGroupDefault(BaseModel):
    """The default model + endpoint the UI shows for one group."""

    model: str
    base_url: str


class ModelConfigDefaults(BaseModel):
    """Defaults surfaced to the UI so pickers pre-fill without hardcoding agent-hardener's literals."""

    attack: ModelGroupDefault
    analysis: ModelGroupDefault


def model_config_defaults() -> ModelConfigDefaults:
    """Return the built-in per-group model defaults (the values shown pre-filled in the UI)."""
    return ModelConfigDefaults(
        attack=ModelGroupDefault(model=ATTACK_DEFAULT_MODEL, base_url=ATTACK_DEFAULT_BASE_URL),
        analysis=ModelGroupDefault(model=ANALYSIS_DEFAULT_MODEL, base_url=ANALYSIS_DEFAULT_BASE_URL),
    )


class AgentHardenerRun(BaseModel):
    """A record of one Agent Hardener war-game run."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="", description="Entity name within the workspace.")
    workspace: str = Field(default="", description="Workspace identifier.")
    project: str | None = Field(default=None, description="Project associated with this entity.")
    agent: str = Field(default="", description="Targeted agent reference (workspace/name).")
    job_id: str = Field(default="", description="Platform job that drove this run (for live status/HITL).")
    port: int = Field(default=0, description="Victim port the war-game attacked.")
    manifest: str = Field(default="", description="Path to the agent-hardener.yaml manifest used.")
    manifest_id: str = Field(default="", description="Manifest this run belongs to.")
    status: RunStatus = Field(default="failed", description="Final run status.")
    returncode: int = Field(default=-1, description="Exit code from `agent-hardener run`.")
    summary: str = Field(default="", description="Short human-readable outcome summary.")
    error_category: str = Field(default="", description="Classified failure category when status is 'failed'.")
    error_message: str = Field(default="", description="Operator-facing failure message when the run failed.")
    error_remediation: str = Field(default="", description="Suggested next step to resolve the failure.")
    hitlog_fileset: str = Field(default="", description="Fileset ref of the garak hitlog this run produced.")
    events_fileset: str = Field(default="", description="Fileset ref of the run's durable events history.")
    source_run: str = Field(default="", description="Source harden run for a validate-only sanity-check run.")
    id: str | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None
    entity_id: str | None = None
    parent: str | None = None
    db_version: int | None = None


class AgentHardenerManifest(BaseModel):
    """A named, reusable war-game target scaffolded via ``init``."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="", description="Entity name within the workspace.")
    workspace: str = Field(default="", description="Workspace identifier.")
    project: str | None = Field(default=None, description="Project associated with this entity.")
    agent: str = Field(default="", description="Deployed agent reference (workspace/name) this manifest targets.")
    source_type: ManifestSource = Field(default="agent", description="How the manifest was built.")
    project_fileset: str = Field(default="", description="Fileset ref holding the uploaded NAT project bundle.")
    agent_fileset: str = Field(default="", description="Fileset ref holding the scaffold resolved from the agent.")
    workflow: str = Field(default="", description="Chosen workflow path within the project.")
    launch_mode: str = Field(default="", description="Victim launch mode ('workflow'|'byo').")
    dockerfile: str = Field(default="", description="Project-relative Dockerfile the victim image is built from.")
    binaries: list[str] = Field(default_factory=list, description="In-container glob patterns allowed to egress.")
    manifest_yaml: str = Field(default="", description="The resolved agent-hardener.yaml content.")
    port: int = Field(default=0, description="Victim port the war-game will target.")
    secrets: list[str] = Field(default_factory=list, description="Secret names the victim agent requires.")
    egress: list[str] = Field(default_factory=list, description="Allow-listed egress host[:port] entries.")
    env: StringMap = Field(default_factory=dict, description="Non-secret environment variables for the victim.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal notes from scaffolding.")
    benign_suite: list[BenignSuiteRow] = Field(default_factory=list, description="Cached benign test suite rows.")
    benign_interview: list[BenignSuiteRow] = Field(default_factory=list, description="Interview Q&A rows.")
    defenders: list[str] = Field(default_factory=list, description="Enabled defender keys.")
    attack_intensity: AttackIntensity = Field(default="standard", description="Attacker effort preset.")
    rounds: int = Field(default=1, ge=1, description="Number of iterative hardening rounds.")
    models: WarGameModels = Field(default_factory=WarGameModels, description="Stored default model selection.")
    id: str | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None
    entity_id: str | None = None
    parent: str | None = None
    db_version: int | None = None

    @classmethod
    def from_agent_resolution(
        cls,
        *,
        name: str,
        workspace: str,
        agent_ref: str,
        manifest_yaml: str,
        port: int,
        secrets: list[str],
        warnings: list[str],
        egress: list[str] | None = None,
        env: StringMap | None = None,
        models: WarGameModels | None = None,
        agent_fileset: str = "",
    ) -> AgentHardenerManifest:
        """Build an ``agent``-source manifest entity from a resolved agent scaffold."""
        return cls(
            name=name,
            workspace=workspace,
            agent=agent_ref,
            source_type="agent",
            manifest_yaml=manifest_yaml,
            agent_fileset=agent_fileset,
            port=port,
            secrets=secrets,
            egress=egress or [],
            env=env or {},
            warnings=warnings,
            models=models or WarGameModels(),
        )


class WarGameSpec(BaseModel):
    """Canonical war-game inputs — the shape ``run()`` and ``compile()`` see.

    Supply either a saved ``manifest_id`` (the Studio path — materialized on the host from the stored
    agent ref) or a ready ``config`` manifest path (the CLI path).
    """

    config: str | None = None
    manifest_id: str | None = None
    env_file: str | None = None
    driver: str | None = None
    stop_after_synth: bool = False
    replay_hitlog_fileset: str | None = None
    benign_suite_fileset: str | None = None
    port: int | None = None
    defenders: list[str] | None = None
    attack_intensity: str | None = None
    rounds: int | None = None
    validate_only: bool = False
    defense_guardrails: str | None = None
    defense_policy: str | None = None
    models: WarGameModels | None = None
    source_run: str | None = None


class SynthBenignSpec(BaseModel):
    """Inputs for the benign-suite synthesis phase (the shape ``run()``/``compile()`` see)."""

    manifest_id: str
    driver: str = "native"
    env_file: str | None = None
    interview: str = "interactive"
    run_name: str | None = None
    source_run: str | None = None


class WarGameJobRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    project: str | None = None
    spec: WarGameSpec
    profile: str | None = None
    options: JsonMap | None = None
    ownership: JsonMap | None = None
    custom_fields: JsonMap | None = None
    output_location: str | None = None

    _validate_output_location = field_validator("output_location")(validate_output_location)


class SynthBenignJobRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    project: str | None = None
    spec: SynthBenignSpec
    profile: str | None = None
    options: JsonMap | None = None
    ownership: JsonMap | None = None
    custom_fields: JsonMap | None = None
    output_location: str | None = None

    _validate_output_location = field_validator("output_location")(validate_output_location)


class WarGameJob(BaseModel):
    id: str | None = None
    name: str
    description: str | None = None
    project: str | None = None
    workspace: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    spec: WarGameSpec
    status: PlatformJobStatus | None = None
    status_details: JsonMap | None = None
    error_details: JsonMap | None = None
    ownership: JsonMap | None = None
    custom_fields: JsonMap | None = None


class SynthBenignJob(BaseModel):
    id: str | None = None
    name: str
    description: str | None = None
    project: str | None = None
    workspace: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    spec: SynthBenignSpec
    status: PlatformJobStatus | None = None
    status_details: JsonMap | None = None
    error_details: JsonMap | None = None
    ownership: JsonMap | None = None
    custom_fields: JsonMap | None = None


class ManifestFilter(Filter):
    """Query filter for ``GET /v2/workspaces/{workspace}/manifests``."""

    agent: str | None = Field(default=None, description="Filter to manifests for this agent reference.")


class RunFilter(Filter):
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


class JobsListFilter(Filter):
    created_at: DatetimeFilter | None = Field(default=None, description="Jobs created after/before a datetime.")
    name: StringFilter | str | None = Field(default=None, description="Name of the job.")
    workspace: str | None = Field(default=None, description="Workspace of the job.")
    project: str | None = Field(default=None, description="Project containing the job.")
    status: PlatformJobStatus | None = Field(default=None, description="The current status.")
    updated_at: DatetimeFilter | None = Field(default=None, description="Jobs updated after/before a datetime.")


WarGameJobsListFilter = JobsListFilter
SynthBenignJobsListFilter = JobsListFilter
WarGameJobsSortField = JobsSortField
SynthBenignJobsSortField = JobsSortField


class ManifestInit(BaseModel):
    """Body for ``POST /v2/workspaces/{workspace}/manifests`` — scaffold a named manifest.

    Two sources. ``agent`` is a registered platform agent, which the resolver reads and renders.
    ``project`` is an uploaded project bundle — an image whose author owns the Dockerfile, which a
    Fabric ``agent.yaml`` cannot express. The user never writes ``agent-hardener.yaml`` either way: the
    project source derives it and asks only for the fields a project cannot state about itself.
    """

    name: str = Field(description="User-defined manifest id (unique within the workspace).")
    source_type: ManifestSource = Field(
        default="agent",
        description="Where the victim comes from: a registered platform agent, or an uploaded project bundle.",
    )
    agent: str | None = Field(
        default=None,
        description="Agent reference (``name`` or ``workspace/name``) to war-game. Required when "
        "``source_type`` is 'agent'.",
    )
    project_fileset: str | None = Field(
        default=None,
        description="Fileset ref (``workspace/name``) of the uploaded project bundle. Required when "
        "``source_type`` is 'project'.",
    )
    dockerfile: str | None = Field(
        default=None,
        description="Dockerfile path relative to the project root. Derived when the project holds exactly one.",
    )
    start_command: str | None = Field(
        default=None,
        description="Command that serves the agent. Derived from the Dockerfile's ENTRYPOINT/CMD when it is "
        "an exec form we can resolve.",
    )
    binaries: list[str] | None = Field(
        default=None,
        description="Glob(s) matching the victim's interpreter, for the sandbox's egress policy. A glob that "
        "matches no process grants nothing while looking like it grants something, so this is confirmed "
        "rather than silently guessed.",
    )
    harness: str | None = Field(
        default=None,
        description="Which harness the agent runs, so the run can say up front whether a guardrail can refuse "
        "a tool call. Not knowable from the project.",
    )
    relay_integration_confirmed: bool = Field(
        default=False,
        description="The author confirms NeMo Relay is attached (middleware + plugin.initialize()). Not "
        "knowable from the project; without Relay the victim emits no telemetry and cannot be scored.",
    )
    port: int | None = Field(default=None, description="Victim port (defaults to 8000).")
    secrets: list[str] | None = Field(
        default=None,
        description="Env-var names the victim requires. Derived from the agent's own declarations "
        "(``models.*.api_key_env``, MCP server env) when omitted.",
    )
    egress: list[str] | None = Field(
        default=None,
        description="Allow-listed egress host[:port] entries the victim may reach (external hosts the agent "
        "calls, e.g. inference-api.nvidia.com). The sandbox is default-deny, so a host missing here has its "
        "traffic dropped mid-run.",
    )
    env: StringMap | None = Field(
        default=None,
        description="Non-secret environment variables for the victim (agent-hardener's `agent.env`). Stored in "
        "plaintext on the manifest — credentials belong in `secrets`, which names them and "
        "resolves the values from the Secrets store at run time.",
    )
    backends: list[str] | None = Field(
        default=None,
        description="Route-only host backends the agent's tools call, each 'NAME:PORT[,PORT2]' (e.g. "
        "'finance:8086'). Rewrites the agent's localhost:PORT to host.docker.internal:PORT and opens the "
        "sandbox->host route.",
    )
    models: WarGameModels | None = Field(
        default=None,
        description="Stored default model selection (attack/analysis/agent groups); omit to use agent-hardener's "
        "built-in defaults.",
    )

    @model_validator(mode="after")
    def _source_matches_fields(self) -> "ManifestInit":
        """Reject a body whose source and fields disagree, rather than resolving the wrong one.

        Both fields being free-form strings, a request that names an agent *and* a project bundle has no
        obviously-correct reading — and picking one silently would war-game a target the caller did not ask
        for.
        """
        required, forbidden = (
            ("agent", "project_fileset")
            if self.source_type == "agent"
            else (
                "project_fileset",
                "agent",
            )
        )
        if not getattr(self, required):
            raise ValueError(f"source_type '{self.source_type}' requires '{required}'")
        if getattr(self, forbidden):
            raise ValueError(f"source_type '{self.source_type}' does not accept '{forbidden}'")
        return self


class ManifestUpdate(BaseModel):
    """Body for ``PATCH /v2/workspaces/{workspace}/manifests/{name}`` — edit an existing manifest.

    Only editable fields; omitted fields are left unchanged. The agent source is immutable (delete +
    recreate to retarget).
    """

    benign_suite: list[BenignSuiteRow] | None = Field(
        default=None, description="Replace the cached benign suite (tool,payload,label,rationale,persona rows)."
    )
    port: int | None = Field(default=None, description="Victim port the war-game will target.")
    egress: list[str] | None = Field(
        default=None, description="Allow-listed egress host[:port] entries the victim may reach."
    )
    env: StringMap | None = Field(
        default=None,
        description="Non-secret environment variables for the victim (agent-hardener's `agent.env`). Stored in "
        "plaintext on the manifest — credentials belong in `secrets`, which names them and "
        "resolves the values from the Secrets store at run time.",
    )
    defenders: list[str] | None = Field(
        default=None,
        description="Enabled defender keys ('guardrails','openshell'); empty means agent-hardener defaults.",
    )
    attack_intensity: AttackIntensity | None = Field(default=None, description="Attacker (garak) effort preset.")
    rounds: int | None = Field(
        default=None, ge=1, description="Number of iterative hardening rounds (agent-hardener `run --rounds`)."
    )
    models: WarGameModels | None = Field(
        default=None, description="Replace the stored default model selection (attack/analysis/agent groups)."
    )


CreateManifestRequest = ManifestInit
UpdateManifestRequest = ManifestUpdate


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
    """Body for ``POST /v2/workspaces/{workspace}/manifests/inspect-project`` — read an uploaded project."""

    project_fileset: str = Field(description="Fileset ref of the uploaded project bundle to inspect.")
    dockerfile: str | None = Field(
        default=None,
        description="Which Dockerfile builds the agent, when the bundle holds more than one.",
    )


class InspectProjectResponse(BaseModel):
    """What the project states about itself, plus what it cannot.

    ``unresolved`` is the contract with the caller: everything else on this model is a usable value, and
    these are the only fields a human still has to supply. It is the difference between a form that asks
    for everything and one that asks for what is genuinely unknowable.
    """

    dockerfile: str = Field(default="", description="Dockerfile path relative to the project root.")
    dockerfiles: list[str] = Field(
        default_factory=list, description="Every Dockerfile found, when the choice is ambiguous."
    )
    start_command: str = Field(default="", description="Derived from the Dockerfile's ENTRYPOINT/CMD.")
    binaries: list[str] = Field(default_factory=list, description="Proposed interpreter globs, for confirmation.")
    port: int = Field(default=8000, description="Derived from EXPOSE / ENV PORT.")
    secrets: list[str] = Field(default_factory=list, description="Secret names derived from .env and ENV.")
    egress: list[str] = Field(default_factory=list, description="Hosts the project's own files name.")
    env: StringMap = Field(default_factory=dict, description="Non-secret environment from the Dockerfile.")
    unresolved: list[str] = Field(
        default_factory=list,
        description="Fields the project cannot state about itself; the caller must supply these.",
    )
    warnings: list[str] = Field(default_factory=list, description="Non-fatal notes about the derivation.")


class InspectAgentRequest(BaseModel):
    """Body for ``POST /v2/workspaces/{workspace}/manifests/inspect-agent`` — a deployed agent ref."""

    agent: str = Field(description="Deployed agent reference (``workspace/name`` or ``name``).")


class InspectAgentResponse(BaseModel):
    """Auto-derived defaults for the deployed-agent create form (port + secret names, editable)."""

    agent: str = Field(description="Resolved ``workspace/name`` of the agent.")
    port: int = Field(description="Victim port derived from the running deployment (else the default).")
    secrets: list[str] = Field(default_factory=list, description="Secret names derived from the agent config.")
    egress: list[str] = Field(
        default_factory=list,
        description="Hosts the agent's own config names (model endpoints, network MCP servers). Shown so "
        "the form does not read as 'no egress' for an agent that has some.",
    )
    warnings: list[str] = Field(default_factory=list, description="Non-fatal notes (e.g. no running deployment).")


class ApplyMitigationRequest(BaseModel):
    """Body for ``POST /v2/workspaces/{workspace}/runs/{name}/apply-mitigation`` — adopt the hardened workflow.

    The client passes the hardened workflow YAML from the run's mitigations artifact. The endpoint reverses
    the Inference-Gateway injection and writes it onto the run's target agent config (no redeploy).
    """

    guardrails_toml: str = Field(description="Hardened Relay guardrail set (the mitigations 'after' document).")


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

    # The published OpenAPI schema exposes this as an arbitrary JSON object.
    mitigations: dict[str, Any] = Field(description="The run's mitigations artifact (its 'defenses'/workflow/policy).")
    selected_defense_ids: list[str] = Field(
        default_factory=list, description="Ids of the defenses to keep (guardrail ids and/or 'openshell_policy')."
    )


class ComposeDefenseResponse(BaseModel):
    """The composed workflow + policy for the selected defenses."""

    guardrails_toml: str | None = Field(
        default=None, description="Plugin config with only the selected guardrails, or null."
    )
    policy_yaml: str | None = Field(
        default=None, description="Hardened policy if selected, else the baseline, or null."
    )


class EventIn(BaseModel):
    """Body for ``POST /runs/{name}/events``."""

    event: str
    payload: JsonMap = Field(default_factory=dict)


class EventsResponse(BaseModel):
    """Response for ``GET /runs/{name}/events``."""

    events: list[JsonMap]


ListWarGameJobsQueryParams = TypedDict(
    "ListWarGameJobsQueryParams",
    {
        "page": NotRequired[int],
        "page_size": NotRequired[int],
        "sort": NotRequired[str],
        "filter": NotRequired[str],
    },
    total=False,
)

ListSynthBenignJobsQueryParams = TypedDict(
    "ListSynthBenignJobsQueryParams",
    {
        "page": NotRequired[int],
        "page_size": NotRequired[int],
        "sort": NotRequired[str],
        "filter": NotRequired[str],
    },
    total=False,
)

ListManifestsQueryParams = TypedDict(
    "ListManifestsQueryParams",
    {
        "page": NotRequired[int],
        "page_size": NotRequired[int],
        "sort": NotRequired[str],
        "filter": NotRequired[str],
        "filter[agent]": NotRequired[str],
    },
    total=False,
)

ListRunsQueryParams = TypedDict(
    "ListRunsQueryParams",
    {
        "page": NotRequired[int],
        "page_size": NotRequired[int],
        "sort": NotRequired[str],
        "filter": NotRequired[str],
        "filter[agent]": NotRequired[str],
        "filter[manifest_id]": NotRequired[str],
        "filter[status]": NotRequired[str],
    },
    total=False,
)

GetEventsQueryParams = TypedDict(
    "GetEventsQueryParams",
    {
        "after": NotRequired[int],
    },
    total=False,
)

JobLogsQueryParams = TypedDict(
    "JobLogsQueryParams",
    {
        "limit": NotRequired[int],
        "page_cursor": NotRequired[str],
        "tail": NotRequired[int],
    },
    total=False,
)

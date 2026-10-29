# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for the Agents service HTTP contract."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal, NotRequired, Self, TypeAlias, TypedDict

from nemo_helix_plugin.images import normalize_reference
from nemo_helix_plugin.jobs.schemas import (
    HelixJobListResultResponse,
    HelixJobLog,
    HelixJobResultResponse,
    HelixJobStatus,
    HelixJobStatusResponse,
)
from nemo_helix_plugin.openshell.types import GatewayProviderBinding
from nemo_helix_plugin.refs import ENTITY_REF_PATTERN, FilesetRef
from nemo_helix_plugin.schema import Page
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

JsonMap: TypeAlias = dict[str, Any]
JsonObject: TypeAlias = JsonMap
StringMap: TypeAlias = dict[str, str]

DeploymentStatus: TypeAlias = Literal["pending", "starting", "running", "failed", "deleting"]
DeploymentMode: TypeAlias = Literal["subprocess", "docker", "k8s", "openshell"]
EndpointProtocol: TypeAlias = Literal["http", "https", "grpc", "tcp"]
SessionLifecycleStatus: TypeAlias = Literal["active", "expired", "lost", "closed"]
AgentJobCollection: TypeAlias = Literal[
    "evaluate",
    "execute",
    "optimize",
    "package",
]

NAT_WORKFLOW_CONFIG_FORMAT = "nat-workflow-v1"
NEMO_AGENTS_SPEC_CONFIG_FORMAT = "nemo-agents-spec-v1"

# Label keys follow Kubernetes (an optional ``prefix/`` plus a name) except that
# neither part may contain a dot: list filters address a label as
# ``labels.<key>`` and the entity store splits that path on dots. Values may
# hold a ``workspace/name`` ref, so they also allow ``/``.
_LABEL_SEGMENT_PATTERN = re.compile(r"[A-Za-z0-9]([A-Za-z0-9_-]{0,61}[A-Za-z0-9])?")
_LABEL_VALUE_PATTERN = re.compile(r"([A-Za-z0-9]([A-Za-z0-9_./-]{0,251}[A-Za-z0-9])?)?")


def _validate_labels(labels: StringMap) -> StringMap:
    for key, value in labels.items():
        prefix, separator, name = key.rpartition("/")
        if not _LABEL_SEGMENT_PATTERN.fullmatch(name) or (separator and not _LABEL_SEGMENT_PATTERN.fullmatch(prefix)):
            raise ValueError(
                f"Label key {key!r} must be '<name>' or '<prefix>/<name>', each part 1-63 characters of "
                "letters, digits, '-' and '_', starting and ending with a letter or digit."
            )
        if not _LABEL_VALUE_PATTERN.fullmatch(value):
            raise ValueError(
                f"Label value {value!r} for key {key!r} must be empty or at most 253 characters of letters, "
                "digits, '-', '_', '.' and '/', starting and ending with a letter or digit."
            )
    return labels


Labels: TypeAlias = Annotated[StringMap, AfterValidator(_validate_labels)]

LABELS_DESCRIPTION = (
    "Key/value labels for filtering and display, never for authorization. Keys are '<name>' or "
    "'<prefix>/<name>' with no dots. Keys NeMo Helix plugins define use the prefix 'helix-<plugin>', "
    "e.g. 'helix-agents/managed-by'; that prefix is a convention the API does not enforce."
)


class EntityMetadata(BaseModel):
    """Common entity metadata emitted by NeMo Helix entity-backed routes."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(default="", description="Entity name within the workspace.")
    workspace: str = Field(default="", description="Workspace identifier.")
    project: str | None = Field(default=None, description="Project associated with this entity.")
    id: str | None = Field(default=None, description="Entity UUID.")
    entity_id: str | None = Field(default=None, description="Compatibility alias for the entity UUID.")
    parent: str | None = Field(default=None, description="Parent entity ID for nested entities.")
    created_at: datetime | None = Field(default=None, description="Timestamp of entity creation.")
    created_by: str | None = Field(default=None, description="Principal id for entity creator.")
    updated_at: datetime | None = Field(default=None, description="Timestamp of last entity update.")
    updated_by: str | None = Field(default=None, description="Principal id for last entity update.")
    db_version: int | None = Field(default=None, description="Database version for optimistic locking.")


class AuthContext(BaseModel):
    """Wire/data mirror of the creator auth context captured on delegated resources."""

    principal_id: str = Field(..., description="The principal's unique identifier.")
    principal_account_id: str | None = Field(
        default=None, description="Stable NeMo account identifier for the principal."
    )
    principal_email: str | None = Field(default=None, description="The principal's email address.")
    principal_groups: list[str] = Field(default_factory=list, description="Groups the principal belongs to.")
    principal_authz_aliases: list[str] = Field(
        default_factory=list,
        description="Alternate trusted identifiers for the principal.",
    )
    principal_on_behalf_of: str | None = Field(
        default=None,
        description="If acting on behalf of another principal, their principal ID.",
    )
    principal_on_behalf_of_groups: list[str] | None = Field(
        default=None,
        description="Groups the on-behalf-of principal belongs to.",
    )
    principal_on_behalf_of_email: str | None = Field(
        default=None,
        description="The on-behalf-of principal's email address.",
    )
    principal_on_behalf_of_account_id: str | None = Field(
        default=None,
        description="Stable NeMo account identifier for the on-behalf-of principal.",
    )
    principal_on_behalf_of_authz_aliases: list[str] = Field(
        default_factory=list,
        description="Alternate trusted identifiers for the on-behalf-of principal.",
    )


class Endpoint(BaseModel):
    """A routable network endpoint for a deployment.

    Mirrors ``nemo_deployments_plugin.types.Endpoint`` so container-mode
    deployments can carry the address the deployments-plugin ``Deployment``
    projected without the agents plugin depending on that plugin at the
    entity-schema layer.
    """

    name: str
    url: str
    protocol: EndpointProtocol = "http"


class ComputeResources(BaseModel):
    """Kubernetes-style resource requests/limits.

    Mirrors ``nemo_deployments_plugin.entities.ResourceRequirements`` so the
    agents entity schema does not depend on the deployments plugin. Compiled
    into the execute container's resources for container deployment modes.
    """

    limits: StringMap = Field(
        default_factory=dict,
        description="k8s resource limits (e.g. cpu, memory, nvidia.com/gpu).",
    )
    requests: StringMap = Field(default_factory=dict, description="k8s resource requests.")


class ComputeSpecInline(BaseModel):
    """Inline compute spec - the resources an invocation runs with."""

    description: str = Field(default="", description="Human-readable description.")
    resources: ComputeResources = Field(
        default_factory=ComputeResources,
        description="k8s-style resource requests/limits for the execute container.",
    )


class ModelProviderOverride(BaseModel):
    """Exceptional external model-provider override.

    Null in the normal case: model selection is on the Agent and the provider
    URL is the Inference Gateway (auto-injected). Set ONLY to point the agent at
    a non-IGW external provider endpoint.
    """

    base_url: str = Field(description="External model-provider endpoint.")
    api_key: str | None = Field(
        default=None,
        description="Secrets-service ref for the provider API key (only needed for external providers).",
    )
    provider: str | None = Field(
        default=None,
        description='Provider selector (e.g. "openai", "anthropic").',
    )


class McpFulfillment(BaseModel):
    """EnvironmentSpec-side fulfillment for one MCP server the Agent declares.

    The Agent DECLARES an MCP dependency by name; the EnvironmentSpec PROVIDES
    the url + env + secrets for that same name. Matched by server-name key at
    compile time; ``secrets`` are merged into the server's ``env``.
    """

    url: str = Field(description="Endpoint the environment provides for this MCP server.")
    env: StringMap = Field(default_factory=dict, description="Non-secret env for the MCP server.")
    secrets: StringMap = Field(
        default_factory=dict,
        description="ENV_NAME -> Secrets-service ref, merged into the MCP server env at compile.",
    )


class EnvironmentSpecInline(BaseModel):
    """Inline environment spec - the dependencies and configuration an agent reaches.

    This is the fulfillment half of a request/fulfill split: the Agent declares
    the dependencies it needs; the EnvironmentSpec provides concrete endpoints
    and secret references. It compiles into the agent.yaml / FabricConfig and
    the injected process environment.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(default="", description="Human-readable description.")
    env: StringMap = Field(default_factory=dict, description="Plaintext, non-secret env vars.")
    secrets: StringMap = Field(default_factory=dict, description="Secret alias -> Secrets-service/plugin ref.")
    model_provider_override: ModelProviderOverride | None = Field(
        default=None,
        description="Set only to point at a non-IGW external model provider.",
    )
    provider: str = Field(default="local", description="local | docker | opensandbox | k8s.")
    workspace_path: str | None = Field(default=None, description="Workspace path visible to the harness.")
    artifacts_path: str | None = Field(default=None, description="Provider-specific artifact output location.")
    control_location: str | None = Field(default=None, description="external_control | in_env_control.")
    ownership: str | None = Field(default=None, description="caller_owned | fabric_owned.")
    connection: JsonMap = Field(
        default_factory=dict,
        description="Provider connection metadata (server url, cred ref, namespace).",
    )
    metadata: JsonMap = Field(default_factory=dict, description="Consumer-provided passthrough metadata.")
    settings: JsonMap = Field(default_factory=dict, description="Provider-specific settings.")
    mcp: dict[str, McpFulfillment] = Field(
        default_factory=dict,
        description="server-name -> fulfillment (url/env/secrets) for an Agent-declared MCP dependency.",
    )


class AgentEnvironmentInline(BaseModel):
    """Inline AgentEnvironment - a composition of environment + compute specs.

    Each part is a ``ref | inline | None`` union: a ``"workspace/name"`` string
    references a stored spec entity, an object provides the spec inline, and
    ``None`` omits it. (A ``sandbox_spec`` is out of scope for now and omitted.)
    """

    description: str = Field(default="", description="Human-readable description.")
    environment_spec: str | EnvironmentSpecInline | None = Field(
        default=None,
        description='"workspace/name" ref to an AgentEnvironmentSpec, an inline spec, or None.',
    )
    compute_spec: str | ComputeSpecInline | None = Field(
        default=None,
        description='"workspace/name" ref to an AgentComputeSpec, an inline spec, or None.',
    )


class AgentInline(BaseModel):
    """Inline agent definition without entity identity."""

    description: str = Field(default="", description="Human-readable description of the agent.")
    config: JsonMap = Field(
        default_factory=dict,
        description="Agent config dict interpreted according to config_format.",
    )
    config_format: str = Field(
        default=NAT_WORKFLOW_CONFIG_FORMAT,
        description="Agent config schema tag.",
    )


class ReferenceConfig(BaseModel):
    """Optional source and container references associated with an agent."""

    model_config = ConfigDict(extra="forbid")

    fileset: FilesetRef | None = Field(
        default=None,
        pattern=ENTITY_REF_PATTERN,
        # StrRef's input schema otherwise omits the field's pattern.
        json_schema_extra={"pattern": ENTITY_REF_PATTERN},
        description="Fileset containing agent source and supporting files, as name or workspace/name.",
    )
    container_image: str | None = Field(
        default=None, description="Container image reference; accepts Docker shorthand, tags, and SHA-256 digests."
    )

    @field_validator("container_image")
    @classmethod
    def _validate_container_image(cls, value: str | None) -> str | None:
        if value is not None:
            if value != value.strip():
                raise ValueError("Container image reference must not contain surrounding whitespace.")
            normalize_reference(value)
        return value


class Agent(EntityMetadata, AgentInline):
    """An agent definition stored in the Agents service."""

    reference: ReferenceConfig | None = Field(default=None, description="Agent source and container references.")
    labels: Labels = Field(default_factory=dict, description=LABELS_DESCRIPTION)


class AgentComputeSpec(EntityMetadata, ComputeSpecInline):
    """A reusable compute spec entity."""


class AgentEnvironmentSpec(EntityMetadata, EnvironmentSpecInline):
    """A reusable environment spec entity."""


class AgentEnvironment(EntityMetadata, AgentEnvironmentInline):
    """A reusable composition of environment and compute specs."""


class AgentDeployment(EntityMetadata):
    """A running, pending, failed, or deleting deployment of an Agent."""

    agent: str = Field(default="", description="Name of the Agent entity this deployment is for.")
    config: JsonMap = Field(
        default_factory=dict,
        description="Resolved agent config used by the deployment.",
    )
    environment: str | AgentEnvironmentInline | None = Field(
        default=None,
        description='"workspace/name" ref to an AgentEnvironment, an inline environment, or None.',
    )
    compute: ComputeSpecInline | None = Field(
        default=None,
        description="Resolved compute snapshot from the referenced environment.",
    )
    secrets: StringMap = Field(default_factory=dict, description="Resolved secret env references.")
    providers: dict[str, GatewayProviderBinding] = Field(
        default_factory=dict, description="Resolved gateway credential bindings."
    )
    status: DeploymentStatus = Field(default="pending", description="Deployment lifecycle status.")
    deployment_mode: DeploymentMode = Field(default="subprocess", description="Runtime backend.")
    endpoint: str = Field(default="", description="Subprocess loopback endpoint.")
    endpoints: list[Endpoint] = Field(default_factory=list, description="Routable endpoints for container modes.")
    image: str = Field(default="", description="Container image for docker/k8s/openshell modes.")
    use_image_entrypoint: bool = Field(
        default=False,
        description="Container modes only: preserve the image ENTRYPOINT/CMD.",
    )
    plugin_deployment: str = Field(default="", description="Linked nemo-deployments Deployment entity name.")
    port: int = Field(default=0, description="Port the agent process is listening on.")
    pid: int = Field(default=0, description="OS process ID of the agent subprocess.")
    error: str = Field(default="", description="Error message if status is failed.")
    auth_context: AuthContext | None = Field(default=None, description="Creator auth context for delegated access.")


class AgentSession(EntityMetadata):
    """A durable logical conversation associated with an AgentDeployment."""

    deployment_id: str = Field(
        min_length=1,
        description="Immutable ID of the AgentDeployment this session belongs to.",
    )
    status: SessionLifecycleStatus = Field(default="active", description="Session lifecycle status.")
    first_active_at: datetime | None = Field(default=None, description="UTC timestamp of first accepted invocation.")
    last_active_at: datetime | None = Field(
        default=None,
        description="UTC timestamp of latest accepted or completed invocation.",
    )
    expires_at: datetime | None = Field(default=None, description="UTC rolling idle deadline.")


class RegisterAgentRequest(BaseModel):
    """Request body for ``POST /v2/workspaces/{workspace}/agents``."""

    name: str = Field(description="Unique agent name within the workspace.")
    description: str = Field(default="", description="Human-readable description.")
    reference: ReferenceConfig | None = Field(default=None, description="Agent source and container references.")
    labels: Labels = Field(default_factory=dict, description=LABELS_DESCRIPTION)
    config: JsonMap = Field(
        default_factory=dict, description="Agent config dict interpreted according to config_format."
    )
    config_format: str = Field(
        default_factory=lambda data: (
            data["config"].get("config_format", NAT_WORKFLOW_CONFIG_FORMAT)
            if data.get("config")
            else NEMO_AGENTS_SPEC_CONFIG_FORMAT
        ),
        validate_default=True,
        description=(
            "Config format identifier. Uses config.config_format when present; otherwise defaults to "
            "nemo-agents-spec-v1 for an empty config or nat-workflow-v1 for a nonempty config. "
            "An explicit value must match config.config_format."
        ),
    )

    @model_validator(mode="after")
    def _validate_config_format(self) -> Self:
        if self.config.get("config_format", self.config_format) != self.config_format:
            raise ValueError("config_format must match config.config_format")
        return self


class CreateSampleAgentRequest(BaseModel):
    """Model to use when provisioning the packaged email security sample."""

    model: str = Field(description="Workspace-qualified model entity, for example default/my-model.")


class SampleAgentResponse(BaseModel):
    """Provisioned sample agent and its workspace and deployment state."""

    status: Literal["created", "resumed", "already_exists"]
    workspace: str
    studio_url: str = Field(description="Studio path relative to the platform origin.")
    agent: str
    deployment: str
    deployment_status: DeploymentStatus


SampleAgentComponent: TypeAlias = Literal["workspace", "agent", "deployment", "dataset", "evaluation_config"]
SampleAgentComponentStatus: TypeAlias = Literal["created", "existing", "submitted", "uploaded", "updated"]


class SampleAgentStreamEvent(BaseModel):
    """One NDJSON frame from sample-agent provisioning."""

    kind: Literal["progress", "done", "error"]
    component: SampleAgentComponent | None = None
    status: SampleAgentComponentStatus | None = None
    workspace: str | None = None
    result: SampleAgentResponse | None = None
    message: str | None = None
    status_code: int | None = None
    failed_step: str | None = None
    retryable: bool | None = None


class SampleAgentRetryDetail(BaseModel):
    """Workspace retained when sample provisioning fails and can be retried."""

    workspace: str
    studio_url: str = Field(description="Studio path relative to the platform origin.")
    failed_step: Literal["agent deployment", "sample files"]
    retryable: Literal[True]


class SampleAgentConflictResponse(BaseModel):
    """Error response when the existing deployment locks a different model."""

    detail: str


class SampleAgentRetryResponse(BaseModel):
    """Error response for a retryable sample provisioning failure."""

    detail: SampleAgentRetryDetail


class CreateDeploymentRequest(BaseModel):
    """Request body for ``POST /v2/workspaces/{workspace}/deployments``."""

    agent: str = Field(description="Name of the Agent to deploy.")
    name: str | None = Field(
        default=None,
        description="Optional deployment name.  Auto-generated from agent name + random suffix if omitted.",
    )
    deployment_mode: DeploymentMode = Field(
        default="subprocess",
        description="Runtime backend: subprocess (default), docker, k8s, or openshell.",
    )
    image: str = Field(
        default="",
        description="Container image for docker/k8s/openshell modes. Ignored for subprocess.",
    )
    use_image_entrypoint: bool = Field(
        default=False,
        description=(
            "For docker/k8s modes, leave the container command and args empty so the image's "
            "ENTRYPOINT/CMD starts the agent server."
        ),
    )
    environment: str | AgentEnvironmentInline | None = Field(
        default=None,
        description=(
            'Optional AgentEnvironment: a "workspace/name" ref, an inline environment, or None. '
            "Resolved and snapshotted onto the deployment at create time."
        ),
    )


CreateAgentDeploymentRequest: TypeAlias = CreateDeploymentRequest


class CreateSessionRequest(BaseModel):
    """Request body for ``POST /v2/workspaces/{workspace}/sessions``."""

    deployment_id: str = Field(min_length=1, description="ID of the AgentDeployment to create a session for.")
    name: str | None = Field(
        default=None,
        description="Optional session name. Auto-generated from deployment name + random suffix if omitted.",
    )


class CreateEnvironmentRequest(AgentEnvironmentInline):
    """Request body for ``POST /v2/workspaces/{workspace}/environments``."""

    name: str = Field(description="Unique environment name within the workspace.")


class CreateEnvironmentSpecRequest(EnvironmentSpecInline):
    """Request body for ``POST /v2/workspaces/{workspace}/environment-specs``."""

    name: str = Field(description="Unique environment-spec name within the workspace.")


class CreateComputeSpecRequest(ComputeSpecInline):
    """Request body for ``POST /v2/workspaces/{workspace}/compute-specs``."""

    name: str = Field(description="Unique compute-spec name within the workspace.")


class InvokeAgentRequest(BaseModel):
    """OpenAI chat-completions request body for agent invocation."""

    model_config = ConfigDict(extra="allow")

    messages: list[JsonMap] = Field(default_factory=list)
    stream: bool = False


class LogLine(BaseModel):
    """One deployment log line shaped like ``HelixJobLog`` for UI reuse."""

    timestamp: str = Field(description="ISO-8601 timestamp parsed from the line; empty when absent.")
    job: str = Field(default="", description="Empty - kept for jobs-log shape compatibility.")
    job_step: str = Field(default="", description="Empty - kept for jobs-log shape compatibility.")
    job_task: str = Field(default="", description="Empty - kept for jobs-log shape compatibility.")
    message: str = Field(description="The raw log line minus any parsed timestamp prefix.")


class DeploymentLogsResponse(BaseModel):
    """Response body for ``GET /deployments/{name}/logs``."""

    data: list[LogLine]
    total_lines: int = Field(description="Number of lines actually returned.")
    next_offset: int | str = Field(description="Resume cursor just past the returned tail.")


class AgentJobRequest(BaseModel):
    """Request body for creating one Agents job collection item."""

    name: str | None = None
    description: str | None = None
    project: str | None = None
    spec: JsonMap
    profile: str | None = None
    options: JsonMap | None = None
    ownership: JsonMap | None = None
    custom_fields: JsonMap | None = None
    output_location: str | None = None


class CreateExecuteJobRequest(AgentJobRequest):
    """Compatibility request body for the agents.execute job collection."""

    spec: JsonObject = Field(description="ExecuteAgentJobConfig payload owned by the Agents plugin.")


class PostRunActionInfo(BaseModel):
    """A post-run action installed on the platform, as set on an execute job's ``post_run_action.kind``."""

    kind: str = Field(description="Kind to set on post_run_action.kind; the action's entry-point name.")
    title: str = Field(description="Human-readable name of the action.")
    description: str = Field(description="One-line summary of what the action does after the agent runs.")


class AgentJob(BaseModel):
    """Agents job response with collection-specific spec kept as JSON."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str = ""
    description: str | None = None
    project: str | None = None
    workspace: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    spec: JsonMap = Field(default_factory=dict)
    status: HelixJobStatus | None = None
    status_details: JsonMap | None = None
    error_details: JsonMap | None = None
    ownership: JsonMap | None = None
    custom_fields: JsonMap | None = None


AgentPage = Page[Agent]
DeploymentPage = Page[AgentDeployment]
SessionPage = Page[AgentSession]
EnvironmentPage = Page[AgentEnvironment]
EnvironmentSpecPage = Page[AgentEnvironmentSpec]
ComputeSpecPage = Page[AgentComputeSpec]
AgentJobPage = Page[AgentJob]
AgentJobResultListResponse = HelixJobListResultResponse
AgentJobResult = HelixJobResultResponse
AgentJobStatusResponse = HelixJobStatusResponse
AgentJobLog = HelixJobLog


class ListAgentsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class ListDeploymentsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


ListSessionsQueryParams = TypedDict(
    "ListSessionsQueryParams",
    {
        "page": NotRequired[int],
        "page_size": NotRequired[int],
        "sort": NotRequired[str],
        "filter": NotRequired[str | JsonMap],
        "filter[deployment_id]": NotRequired[str],
    },
    total=False,
)


class ListEnvironmentResourcesQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str | JsonMap]


class DeploymentLogsQueryParams(TypedDict, total=False):
    tail: NotRequired[int]


class ListAgentJobsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str | JsonMap]


class AgentJobLogsQueryParams(TypedDict, total=False):
    limit: NotRequired[int]
    page_cursor: NotRequired[str]
    tail: NotRequired[int]


class ListAgentJobResultsQueryParams(TypedDict, total=False):
    sort: NotRequired[str]

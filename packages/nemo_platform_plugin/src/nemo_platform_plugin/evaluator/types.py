# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for the Evaluator service.

Single source of truth for the HTTP contract. Replaces the Stainless-generated
evaluator types and the hand-written ``nemo_evaluator.sdk`` resource layer's
direct ``NeMoPlatform._client`` usage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, NotRequired, Self, TypeAlias, TypedDict

from nemo_platform_plugin.jobs.schemas import PlatformJobStatus
from nemo_platform_plugin.jobs.types import validate_output_location
from nemo_platform_plugin.schema import Page
from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel, field_validator, model_validator

FlatQueryParams: TypeAlias = dict[str, str | int | bool | None]

_ENTITY_NAME_SEGMENT = r"[a-z](?:[a-z0-9@.+_]|-[a-z0-9@.+_]){1,62}"
_QUALIFIED_MODEL_REF_PATTERN = rf"^{_ENTITY_NAME_SEGMENT}/{_ENTITY_NAME_SEGMENT}$"
_SECRET_REF_PATTERN = r"^[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)?$"

Truncation: TypeAlias = Literal["end", "start"]
RetrieveEvalModelFormat: TypeAlias = Literal["nim", "openai", "llama_stack"]
_RetrieveEvalModelRefRoot: TypeAlias = Annotated[
    str,
    Field(
        pattern=_QUALIFIED_MODEL_REF_PATTERN,
        description="Reference to a model (format: workspace/name).",
        examples=["workspace/model_name"],
    ),
]

# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


class EvaluatorHealth(BaseModel):
    """Evaluator plugin health response."""

    model_config = ConfigDict(extra="allow")

    plugin: str | None = None
    status: str | None = None
    service: str | None = None
    jobs: list[str] = Field(default_factory=list)


class HelloResponse(BaseModel):
    """Evaluator plugin hello response."""

    message: str


class EvaluateJob(BaseModel):
    """Response from an evaluate job route."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    status: str | None = None
    spec: dict[str, Any] | None = None
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AgentEvalJob(BaseModel):
    """Response from an agent-evaluate job route."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    status: str | None = None
    spec: dict[str, Any] | None = None
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


EvaluatorJobResponse: TypeAlias = EvaluateJob


class RetrieveEvalFilesetRef(RootModel[str]):
    """Reference to a persisted platform fileset, optionally with a file fragment."""

    root: str = Field(description="Reference to a Fileset (format: workspace/fileset-name).")


class RetrieveEvalModelRef(RootModel[_RetrieveEvalModelRefRoot]):
    """Reference to a model resolved by the evaluator backend."""


class RetrieveEvalSecretRef(RootModel[str]):
    """Reference to a platform secret or local environment variable."""

    root: str = Field(
        description="Reference to a platform secret or local environment variable. "
        "Format: 'secret_name' (uses request workspace) or 'workspace/secret_name' (explicit workspace).",
        pattern=_SECRET_REF_PATTERN,
        examples=[
            "my-secret",
            "my-workspace/my-secret",
            "NVIDIA_API_KEY",
        ],
    )


class RetrieveEvalModel(BaseModel):
    """Inline model definition accepted by retrieve-eval specs."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="URL of the model.")
    name: str = Field(description="Name of the model.")
    host_url: str | None = Field(
        default=None,
        description="Direct NIM endpoint URL populated when a model reference is resolved.",
    )
    api_key_secret: RetrieveEvalSecretRef | None = Field(
        default=None,
        description="API key secret reference for the model.",
    )
    format: RetrieveEvalModelFormat = Field(
        default="openai",
        description="Inference API format used by the model.",
    )


class RetrieveEvalRetrievalInputSpec(BaseModel):
    """Submitter-facing retrieval target before platform model refs are resolved."""

    model_config = ConfigDict(extra="forbid")

    embeddings: RetrieveEvalModel | RetrieveEvalModelRef
    reranker: RetrieveEvalModel | RetrieveEvalModelRef | None = None
    first_stage_k: int = Field(default=100, ge=1, description="Dense-search cutoff before reranking.")
    truncate_long_documents: Truncation | None = Field(
        default="end",
        description="How to cap passages at 65535 characters: keep the start ('end'), the tail ('start'), or error (null).",
        json_schema_extra={"nullable": True},
    )
    batch_size: int = Field(default=32, ge=1, description="Embedding HTTP batch size.")
    embedding_dimensions: int | None = Field(
        default=None,
        gt=0,
        description="Expected embedding width. Omit to accept the model's native width.",
    )


RetrieveEvalSubmitTarget: TypeAlias = RetrieveEvalRetrievalInputSpec | RetrieveEvalModel | RetrieveEvalModelRef


class RetrieveEvalInputSpec(BaseModel):
    """Submitter-facing BEIR retrieval evaluation spec."""

    model_config = ConfigDict(extra="forbid")

    dataset: RetrieveEvalFilesetRef = Field(description="Fileset containing a BEIR test split.")
    target: RetrieveEvalSubmitTarget = Field(
        description="Retrieval pipeline, embedding NIM, or platform model reference."
    )
    baseline: RetrieveEvalSubmitTarget | None = Field(
        default=None,
        description="Optional baseline retrieval pipeline used for relative nDCG and recall at 10.",
    )
    k: list[int] = Field(default_factory=lambda: [1, 5, 10, 100], min_length=1)

    @model_validator(mode="after")
    def validate_k(self) -> Self:
        """Require unique positive cutoffs."""
        if any(cutoff < 1 for cutoff in self.k):
            raise ValueError("retrieval cutoffs must be positive")
        if len(set(self.k)) != len(self.k):
            raise ValueError("retrieval cutoffs must be unique")
        self.k.sort()
        return self


class RetrieveEvalRetrievalSpec(BaseModel):
    """Canonical retrieval target returned after model refs are resolved."""

    model_config = ConfigDict(extra="forbid")

    embeddings: RetrieveEvalModel = Field(description="Embedding NIM used to encode queries and passages.")
    reranker: RetrieveEvalModel | None = Field(
        default=None, description="Optional ranking NIM applied after dense search."
    )
    first_stage_k: int = Field(default=100, ge=1, description="Dense-search cutoff before reranking.")
    truncate_long_documents: Truncation | None = Field(
        default="end",
        description="How to cap passages at 65535 characters: keep the start ('end'), the tail ('start'), or error (null).",
        json_schema_extra={"nullable": True},
    )
    batch_size: int = Field(default=32, ge=1, description="Embedding HTTP batch size.")
    embedding_dimensions: int | None = Field(
        default=None,
        gt=0,
        description="Expected embedding width. Omit to accept the model's native width.",
    )


class RetrieveEvalSpec(BaseModel):
    """Canonical BEIR retrieval evaluation spec."""

    model_config = ConfigDict(extra="forbid")

    dataset: RetrieveEvalFilesetRef
    target: RetrieveEvalRetrievalSpec
    baseline: RetrieveEvalRetrievalSpec | None = None
    k: list[int] = Field(default_factory=lambda: [1, 5, 10, 100], min_length=1)

    @model_validator(mode="after")
    def validate_k(self) -> Self:
        """Require unique positive cutoffs in canonical task payloads."""
        if any(cutoff < 1 for cutoff in self.k):
            raise ValueError("retrieval cutoffs must be positive")
        if len(set(self.k)) != len(self.k):
            raise ValueError("retrieval cutoffs must be unique")
        self.k.sort()
        return self


class RetrieveEvalJob(BaseModel):
    """Response from a retrieve-eval job route."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str
    description: str | None = None
    project: str | None = None
    workspace: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    spec: RetrieveEvalSpec
    status: PlatformJobStatus | None = None
    status_details: dict[str, JsonValue] | None = None
    error_details: dict[str, JsonValue] | None = None
    ownership: dict[str, JsonValue] | None = None
    custom_fields: dict[str, JsonValue] | None = None


class _WorkspaceResource(BaseModel):
    """Permissive base for evaluator workspace resources."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    workspace: str | None = None
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class AgentEvalResult(_WorkspaceResource):
    """Agent evaluation result record."""


AgentEvalResultPage = Page[AgentEvalResult]


class EvalResult(_WorkspaceResource):
    """Row evaluation result record."""


EvalResultPage = Page[EvalResult]


class Metric(_WorkspaceResource):
    """Stored metric bundle entity."""


MetricBundle: TypeAlias = Metric
MetricBundlePage = Page[Metric]


class Task(_WorkspaceResource):
    """Stored evaluator task entity."""


TaskPage = Page[Task]


class Taskset(_WorkspaceResource):
    """Stored evaluator taskset entity."""


TasksetPage = Page[Taskset]


class Revision(BaseModel):
    """Published task or taskset revision."""

    model_config = ConfigDict(extra="allow")

    content_hash: str | None = None
    revision: int | str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


RevisionPage = Page[Revision]


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------


class SubmitEvaluateJobRequest(BaseModel):
    """Request body for POST /evaluate/jobs."""

    model_config = ConfigDict(extra="forbid")

    spec: dict[str, Any]


class SubmitAgentEvalJobRequest(BaseModel):
    """Request body for POST /agent-evaluate/jobs."""

    model_config = ConfigDict(extra="forbid")

    spec: dict[str, Any]


class SubmitRetrieveEvalJobRequest(BaseModel):
    """Request body for POST /retrieve-eval/jobs."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    project: str | None = None
    spec: RetrieveEvalInputSpec
    profile: str | None = None
    options: dict[str, JsonValue] | None = None
    ownership: dict[str, JsonValue] | None = None
    custom_fields: dict[str, JsonValue] | None = None
    output_location: str | None = None

    _validate_output_location = field_validator("output_location")(validate_output_location)


class MetricMetadata(BaseModel):
    """User-facing metadata captured with a bundled metric."""

    model_config = ConfigDict(extra="allow", revalidate_instances="never")

    description: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class BundledMetricOutputSpec(BaseModel):
    """JSON-safe projection of a runtime metric output spec."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    value_json_schema: dict[str, JsonValue]


class MetricSecretRef(RootModel[str]):
    root: str = Field(
        description="Reference to a platform secret or local environment variable. "
        "Format: 'secret_name' (uses request workspace) or 'workspace/secret_name' (explicit workspace).",
        pattern=_SECRET_REF_PATTERN,
        examples=[
            "my-secret",
            "my-workspace/my-secret",
            "NVIDIA_API_KEY",
        ],
    )


class CloudpickleMetricPayload(BaseModel):
    """Wire schema for a cloudpickle-serialized metric payload."""

    model_config = ConfigDict(extra="forbid", ser_json_bytes="base64", val_json_bytes="base64")

    kind: Literal["cloudpickle"] = Field(description="Payload format discriminator.")
    python_version: str = Field(description="Python version the metric was pickled with.")
    cloudpickle_version: str = Field(description="cloudpickle version used to serialize the metric.")
    pickle_protocol: int = Field(description="Pickle protocol used.")
    blob: bytes = Field(description="Base64-encoded cloudpickled metric object.")
    digest: str | None = Field(
        default=None,
        description="SHA-256 digest of the payload bytes. Informational; recomputed server-side.",
    )


class InlineMetricPayload(BaseModel):
    """Wire schema for an inline metric payload."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["inline"] = Field(description="Payload format discriminator.")
    metric: dict[str, JsonValue] = Field(
        description="JSON-serialized built-in metric configuration, discriminated by its own `type`."
    )
    digest: str | None = Field(
        default=None,
        description="SHA-256 digest of the canonical metric JSON. Informational; recomputed server-side.",
    )

    @field_validator("metric")
    @classmethod
    def _metric_must_declare_type(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        metric_type = value.get("type")
        if not isinstance(metric_type, str) or not metric_type:
            raise ValueError("inline metric payload must include a non-empty 'type'")
        return value


MetricPayload: TypeAlias = Annotated[CloudpickleMetricPayload | InlineMetricPayload, Field(discriminator="kind")]


class CreateMetricRequest(BaseModel):
    """Request body for POST /metrics/{name}."""

    model_config = ConfigDict(extra="forbid")

    bundle_kind: Literal["metric-bundle"] = "metric-bundle"
    bundle_format_version: Literal["v1"] = "v1"
    metric_type: str = Field(min_length=1, description="Runtime metric type name.")
    metadata: MetricMetadata = Field(default_factory=MetricMetadata, description="User-facing metric metadata.")
    outputs: list[BundledMetricOutputSpec] = Field(min_length=1, description="The metric's output contracts.")
    secrets: dict[str, MetricSecretRef] = Field(
        default_factory=dict, description="Secret references required to execute the metric."
    )
    payload: MetricPayload = Field(description="Format-specific serialized metric.")


class CreateTaskRequest(RootModel[dict[str, Any]]):
    """Request body for POST /tasks/{name}."""


class ReplaceTaskRequest(RootModel[dict[str, Any]]):
    """Request body for PUT /tasks/{name}."""


class CreateTasksetRequest(RootModel[dict[str, Any]]):
    """Request body for POST /tasksets/{name}."""


class ReplaceTasksetRequest(RootModel[dict[str, Any]]):
    """Request body for PUT /tasksets/{name}."""


# ---------------------------------------------------------------------------
# Query parameter types
# ---------------------------------------------------------------------------


class ListAgentEvalResultsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class ListEvalResultsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class ListMetricsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]
    include_derived: NotRequired[bool]


class ListTasksQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class ListTasksetsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class EvaluatorJobLogsQueryParams(TypedDict, total=False):
    limit: NotRequired[int]
    page_cursor: NotRequired[str]
    tail: NotRequired[int]


class ListRevisionsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]


class ProjectQueryParams(TypedDict, total=False):
    project: NotRequired[str]


class RevisionQueryParams(TypedDict, total=False):
    revision: NotRequired[str]

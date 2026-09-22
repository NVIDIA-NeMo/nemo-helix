# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/response schemas for the evaluator API — metrics, eval results, and shared filters."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, TypeAlias

from nemo_evaluator.api.fields import (
    LATEST_TAG as LATEST_TAG,
)
from nemo_evaluator.api.fields import (
    REF_FRAGMENT_CHARSET as REF_FRAGMENT_CHARSET,
)
from nemo_evaluator.api.fields import (
    REF_FRAGMENT_SEPARATOR as REF_FRAGMENT_SEPARATOR,
)
from nemo_evaluator.api.fields import (
    CloudpickleMetricPayload as CloudpickleMetricPayload,
)
from nemo_evaluator.api.fields import (
    InlineMetricPayload as InlineMetricPayload,
)
from nemo_evaluator.api.fields import (
    MetadataItem as MetadataItem,
)
from nemo_evaluator.api.fields import (
    MetricInline as MetricInline,
)
from nemo_evaluator.api.fields import (
    MetricPayload as MetricPayload,
)
from nemo_evaluator.api.fields import (
    MetricRef as MetricRef,
)
from nemo_evaluator.api.fields import (
    MetricRefOrInline as MetricRefOrInline,
)
from nemo_evaluator.api.fields import (
    PinnedTaskRefList as PinnedTaskRefList,
)
from nemo_evaluator.api.fields import (
    TaskInputs as TaskInputs,
)
from nemo_evaluator.api.fields import (
    TaskMetadataList as TaskMetadataList,
)
from nemo_evaluator.api.fields import (
    TaskRef as TaskRef,
)
from nemo_evaluator.api.fields import (
    TaskRefList as TaskRefList,
)
from nemo_evaluator.api.fields import (
    TasksetFilesRef as TasksetFilesRef,
)
from nemo_evaluator.api.fields import (
    TasksetRef as TasksetRef,
)
from nemo_evaluator.api.fields import (
    parse_subentity_ref as parse_subentity_ref,
)
from nemo_evaluator.api.task_definitions.evaluator import EvaluatorTaskDefinition as EvaluatorTaskDefinition
from nemo_evaluator.api.task_definitions.harbor import HarborTaskDefinition as HarborTaskDefinition
from nemo_evaluator.shared.metric_bundles.bundles import (
    BundledMetricOutputSpec,
)
from nemo_evaluator_sdk.values.common import SecretRef
from nemo_evaluator_sdk.values.results import AggregatedMetricResult
from nemo_helix_plugin.api.filter import ComparisonOperation, FilterOperation, LogicalOperation
from nemo_helix_plugin.api.parsed_filter import ENTITY_BASE_FIELDS
from nemo_helix_plugin.refs import (
    FILESET_REF_PATTERN as FILESET_REF_PATTERN,
)
from nemo_helix_plugin.schema import DatetimeFilter, Filter
from pydantic import BaseModel, ConfigDict, Field

#: A stored task's content, discriminated by which runner executes it. Widen with more members as
#: runners land — the same way ``AgentRunnerTarget`` does on the target side.
TaskDefinition: TypeAlias = Annotated[EvaluatorTaskDefinition | HarborTaskDefinition, Field(discriminator="kind")]


class DataFilter(Filter):
    """A ``Filter`` whose declared non-base fields are stored under the entity's ``data.*`` column.

    Implements the duck-typed hooks ``make_filter_dep`` looks for, so a custom-field filter (e.g.
    ``metric_type`` or ``job_id``) is rewritten to ``data.<field>`` for the entity store. The plain
    ``Filter`` does no translation, so an un-prefixed custom field reaches the store unresolved and
    500s. (The richer ``nhx.common`` filter does this, but plugins can't depend on it — minimal port.)
    """

    @classmethod
    def _get_entity_field_map(cls) -> dict[str, str]:
        return {name: f"data.{name}" for name in cls.model_fields if name not in ENTITY_BASE_FIELDS}

    @classmethod
    def translate_operation(cls, operation: FilterOperation) -> FilterOperation:
        field_map = cls._get_entity_field_map()

        def _walk(op: FilterOperation) -> FilterOperation:
            if isinstance(op, ComparisonOperation):
                mapped = field_map.get(op.field)
                return op if mapped is None else op.model_copy(update={"field": mapped})
            if isinstance(op, LogicalOperation):
                return op.model_copy(update={"operations": [_walk(child) for child in op.operations]})
            return op

        return _walk(operation)


class Metric(BaseModel):
    """API representation of a stored metric.

    The canonical executable bundle lives in the Files service; the fields here
    are the queryable projection plus the reference and digest needed to load it.
    """

    id: str = Field(description="Unique identifier for the stored metric.")
    name: str = Field(description="Name of the metric, unique within its workspace.")
    workspace: str = Field(description="Workspace the metric belongs to.")
    project: str | None = Field(default=None, description="The project associated with this metric.")
    metric_type: str = Field(description="Runtime metric type name.")
    description: str | None = Field(default=None, description="Description captured from the metric's metadata.")
    labels: dict[str, str] = Field(default_factory=dict, description="Labels captured from the metric's metadata.")
    outputs: list[BundledMetricOutputSpec] = Field(description="The metric's output contracts.")
    secrets: dict[str, SecretRef] = Field(description="Secret references required to execute the metric.")
    payload_kind: str = Field(description="Payload discriminator of the stored bundle.")
    payload_digest: str = Field(description="Digest of the stored payload.")
    bundle_ref: str = Field(description="Files reference to the canonical serialized bundle.")
    derived: bool = Field(
        default=False,
        description="True for a content-addressed metric auto-stored from an inline task metric "
        "(excluded from the default metric listing).",
    )
    created_at: datetime = Field(description="Timestamp the metric was created.")
    updated_at: datetime = Field(description="Timestamp the metric was last updated.")


class MetricSort(StrEnum):
    """Sort fields for metric queries."""

    NAME_ASC = "name"
    NAME_DESC = "-name"
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"


class MetricFilter(DataFilter):
    """Filter for metric queries."""

    workspace: str | None = Field(None, description="Filter by workspace.")
    name: str | None = Field(None, description="Filter by name.")
    metric_type: str | None = Field(None, description="Filter by metric type.")
    description: str | None = Field(None, description="Filter by description.")
    derived: bool | None = Field(None, description="Filter by derived flag.")
    created_at: DatetimeFilter | None = Field(None, description="Filter by creation date.")
    updated_at: DatetimeFilter | None = Field(None, description="Filter by update date.")


# --- Eval result DTOs --------------------------------------------------------
#
# API representation of the persisted result records (the storage entities are
# ``AgentEvalResultEntity`` / ``EvaluateResultEntity``). A separate DTO — like ``Metric`` for
# ``MetricBundleEntity`` — so the wire/SDK contract round-trips cleanly: an ``EntityBase``'s
# ``id`` / ``created_at`` / ``updated_at`` are computed/output-only and don't deserialize from
# the entity's own serialized form, whereas these plain fields do.


class _ResultBase(BaseModel):
    """Fields common to both result DTOs (provenance + aggregated scores + target traits)."""

    id: str = Field(description="Unique identifier for the stored result record.")
    name: str = Field(description="Result record name (equals the producing job's id).")
    workspace: str = Field(description="Workspace the result belongs to.")
    project: str | None = Field(default=None, description="The project associated with this result.")
    job_id: str = Field(description="Identifier of the job run that produced this result.")
    # Nullable traits default to None so they round-trip when the list route serializes with
    # response_model_exclude_none (which drops null values from the payload) — matching ``Metric``.
    target_kind: str | None = Field(
        default=None, description="Target discriminator: 'model', 'agent', or a runner kind."
    )
    target_name: str | None = Field(default=None, description="Model/agent entity name, or the runner's model.")
    target_url: str | None = Field(default=None, description="Endpoint URL, when the target is an HTTP model/agent.")
    scores: AggregatedMetricResult = Field(description="Aggregated metric scores for the run.")
    bundle_ref: str = Field(description="Reference to the full result bundle in the Files service.")
    created_at: datetime = Field(description="Timestamp the result was created.")
    updated_at: datetime = Field(description="Timestamp the result was last updated.")


class AgentEvalResult(_ResultBase):
    """API representation of a persisted agent-evaluation result record."""


class EvaluateResult(_ResultBase):
    """API representation of a persisted (row) evaluation result record."""

    dataset_ref: str | None = Field(
        default=None, description="Reference to the dataset evaluated; None for an inline dataset."
    )
    metric_types: list[str] = Field(description="Runtime metric type names applied in the run.")


class Task(BaseModel):
    """API representation of a stored agent-eval task.

    Maps to the SDK :class:`~nemo_evaluator_sdk.agent_eval.tasks.AgentEvalTask` — the task's stable
    ``id`` is the record ``name`` (unique within its workspace). Metrics are stored in their wire form
    (inline bundles and/or references to stored metrics); references resolve to inline at run time.
    """

    id: str = Field(description="Unique identifier for the stored task record.")
    name: str = Field(description="Task name — the stable task id, unique within its workspace.")
    workspace: str = Field(description="Workspace the task belongs to.")
    project: str | None = Field(default=None, description="The project associated with this task.")
    spec: TaskDefinition = Field(description="The task's content, discriminated by which runner executes it.")
    metadata: TaskMetadataList = Field(default_factory=list, description="Key/value annotations for the task.")
    revision: int = Field(
        description="Ordinal of the published revision this content corresponds to. Every stored task "
        "has at least one revision — creating a task publishes revision 1 — so this is never 0."
    )
    tags: dict[str, int] = Field(
        default_factory=dict,
        description="Tag → revision-ordinal pointers. Reading the record's current content returns "
        "every tag, including 'latest'. Reading a *specific* revision returns only the tags pointing "
        "at that revision, which may be none — so do not assume 'latest' is present.",
    )
    created_at: datetime = Field(description="Timestamp the task was created.")
    updated_at: datetime = Field(description="Timestamp the task was last updated.")


class TaskInput(BaseModel):
    """Create/replace body for a stored task (the name comes from the path).

    The authorable subset of :class:`Task` — the SDK ``AgentEvalTask`` shape minus server-owned
    fields (id, name, workspace, timestamps).
    """

    model_config = ConfigDict(extra="forbid")

    spec: TaskDefinition = Field(description="The task's content, discriminated by which runner executes it.")
    metadata: TaskMetadataList = Field(default_factory=list, description="Key/value annotations for the task.")
    tags: list[str] = Field(
        default_factory=list,
        description="Tags to point at the revision this request publishes. 'latest' is always applied "
        "server-side and need not be listed.",
    )


class Revision(BaseModel):
    """A published revision of a task or taskset.

    Deliberately thin: it identifies a revision and says when it was cut, without repeating the
    content. Listing a record's history is a "what can I pin to?" question, and answering it with
    full content on every entry would make the response large for no benefit — fetch the record at
    a specific revision to get its content.
    """

    revision: int = Field(description="Monotonic 1-based ordinal within the record.")
    content_hash: str = Field(
        description="Full 64-char hex SHA-256 of the revision's content. This is what a pinned "
        "reference carries: 'workspace/name#<content_hash>'."
    )
    tags: list[str] = Field(default_factory=list, description="Tags currently pointing at this revision, if any.")
    created_at: datetime = Field(description="Timestamp the revision was published.")


class TaskSort(StrEnum):
    """Sort fields for task queries."""

    NAME_ASC = "name"
    NAME_DESC = "-name"
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"


class TaskFilter(Filter):
    """Filter for task queries (top-level entity columns only; custom-field filtering is a follow-up)."""

    workspace: str | None = Field(None, description="Filter by workspace.")
    name: str | None = Field(None, description="Filter by name.")
    created_at: DatetimeFilter | None = Field(None, description="Filter by creation date.")
    updated_at: DatetimeFilter | None = Field(None, description="Filter by update date.")


class Taskset(BaseModel):
    """API representation of a stored taskset — a flexible grouping of tasks with metadata.

    Members are referenced by ``workspace/name`` (there are no inline tasks). Membership is a set:
    order is not significant and duplicate references are rejected.
    """

    id: str = Field(description="Unique identifier for the stored taskset record.")
    name: str = Field(description="Taskset name — the stable id, unique within its workspace.")
    workspace: str = Field(description="Workspace the taskset belongs to.")
    project: str | None = Field(default=None, description="The project associated with this taskset.")
    description: str | None = Field(default=None, description="Human-readable description of the grouping.")
    tasks: TaskRefList = Field(
        default_factory=list, description="References to the member tasks (set semantics; duplicates rejected)."
    )
    files_ref: TasksetFilesRef | None = Field(
        default=None,
        description="Files reference to the taskset's own files — shared by its members, owned by none.",
    )
    metadata: TaskMetadataList = Field(default_factory=list, description="Key/value annotations for the taskset.")
    revision: int = Field(
        description="Ordinal of the published revision this content corresponds to. Every stored "
        "taskset has at least one revision, so this is never 0."
    )
    tags: dict[str, int] = Field(
        default_factory=dict,
        description="Tag → revision-ordinal pointers. Reading the record's current content returns "
        "every tag, including 'latest'. Reading a *specific* revision returns only the tags pointing "
        "at that revision, which may be none — so do not assume 'latest' is present.",
    )
    created_at: datetime = Field(description="Timestamp the taskset was created.")
    updated_at: datetime = Field(description="Timestamp the taskset was last updated.")


class TasksetInput(BaseModel):
    """Create/replace body for a stored taskset (the name comes from the path).

    The authorable subset of :class:`Taskset` — minus server-owned fields (id, name, workspace,
    timestamps).
    """

    model_config = ConfigDict(extra="forbid")

    description: str | None = Field(default=None, description="Human-readable description of the grouping.")
    tasks: TaskRefList = Field(
        default_factory=list,
        description="References to the member tasks (set semantics; duplicates rejected). Each may be "
        "bare, tag-pinned ('task-a#latest'), or digest-pinned; all are resolved to an exact digest "
        "when stored, so the grouping cannot change underneath you when a member republishes. "
        "Because membership is a set, the stored order is canonical rather than the submitted order: "
        "reordering the same members is not a content change and publishes no revision.",
    )
    files_ref: TasksetFilesRef | None = Field(
        default=None,
        description="Files reference to the taskset's own files — shared by its members, owned by none. "
        "Upload them to the Files service first and point here ('workspace/fileset#prefix'). The "
        "reference is part of the taskset's content, so repointing it publishes a revision; pin the "
        "content by referencing a location that is not rewritten.",
    )
    metadata: TaskMetadataList = Field(default_factory=list, description="Key/value annotations for the taskset.")
    tags: list[str] = Field(
        default_factory=list,
        description="Tags to point at the revision this request publishes. 'latest' is always applied "
        "server-side and need not be listed.",
    )


class TasksetSort(StrEnum):
    """Sort fields for taskset queries."""

    NAME_ASC = "name"
    NAME_DESC = "-name"
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"


class TasksetFilter(Filter):
    """Filter for taskset queries (top-level entity columns only; custom-field filtering is a follow-up)."""

    workspace: str | None = Field(None, description="Filter by workspace.")
    name: str | None = Field(None, description="Filter by name.")
    created_at: DatetimeFilter | None = Field(None, description="Filter by creation date.")
    updated_at: DatetimeFilter | None = Field(None, description="Filter by update date.")

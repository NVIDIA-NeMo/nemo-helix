# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/response schemas for the evaluator API — metrics, eval results, and shared filters."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Annotated, ClassVar, Self, TypeAlias

from nemo_evals.api.fields import (
    LATEST_TAG as LATEST_TAG,
)
from nemo_evals.api.fields import (
    REF_FRAGMENT_CHARSET as REF_FRAGMENT_CHARSET,
)
from nemo_evals.api.fields import (
    REF_FRAGMENT_SEPARATOR as REF_FRAGMENT_SEPARATOR,
)
from nemo_evals.api.fields import (
    AgentRef as AgentRef,
)
from nemo_evals.api.fields import (
    CloudpickleMetricPayload as CloudpickleMetricPayload,
)
from nemo_evals.api.fields import (
    InlineMetricPayload as InlineMetricPayload,
)
from nemo_evals.api.fields import (
    MetadataItem as MetadataItem,
)
from nemo_evals.api.fields import (
    MetricInline as MetricInline,
)
from nemo_evals.api.fields import (
    MetricPayload as MetricPayload,
)
from nemo_evals.api.fields import (
    MetricRef as MetricRef,
)
from nemo_evals.api.fields import (
    MetricRefOrInline as MetricRefOrInline,
)
from nemo_evals.api.fields import (
    PinnedTaskRefList as PinnedTaskRefList,
)
from nemo_evals.api.fields import (
    TaskInputs as TaskInputs,
)
from nemo_evals.api.fields import (
    TaskMetadataList as TaskMetadataList,
)
from nemo_evals.api.fields import (
    TaskRef as TaskRef,
)
from nemo_evals.api.fields import (
    TaskRefList as TaskRefList,
)
from nemo_evals.api.fields import (
    TasksetFilesRef as TasksetFilesRef,
)
from nemo_evals.api.fields import (
    TasksetRef as TasksetRef,
)
from nemo_evals.api.fields import (
    parse_subentity_ref as parse_subentity_ref,
)
from nemo_evals.api.task_definitions.evaluator import EvaluatorTaskDefinition as EvaluatorTaskDefinition
from nemo_evals.api.task_definitions.harbor import HarborTaskDefinition as HarborTaskDefinition
from nemo_evals.content_hash import DIGEST_PATTERN
from nemo_evals.shared.metric_bundles.bundles import (
    BundledMetricOutputSpec,
)
from nemo_helix_plugin.api.filter import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.api.parsed_filter import ENTITY_BASE_FIELDS
from nemo_helix_plugin.filter_ops import ElemMatchScalar, parse_elem_match_criteria
from nemo_helix_plugin.refs import (
    FILESET_REF_PATTERN as FILESET_REF_PATTERN,
)
from nemo_helix_plugin.schema import DatetimeFilter, Filter
from nhx_evals_sdk.agent_eval.results import AgentEvalMetricOutputCoverage, AgentEvalSummary
from nhx_evals_sdk.values.common import SecretRef
from nhx_evals_sdk.values.results import AggregatedMetricResult
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


_METADATA_FIELD = "metadata"
_METADATA_PREFIX = f"{_METADATA_FIELD}."

#: Stored ref arrays a filter matches by ``workspace/name``; unqualified filter refs take the route's workspace.
_REF_ARRAY_FIELDS = frozenset({"data.spec.metrics", "data.tasks"})


def _one_or_any(op: ComparisonOperation, match: Callable[[object], FilterOperation]) -> FilterOperation:
    if op.operator == FilterOperator.EQ:
        return match(op.value)
    if op.operator == FilterOperator.IN and isinstance(op.value, list) and op.value:
        return LogicalOperation(operator=FilterOperator.OR, operations=[match(value) for value in op.value])
    raise ValueError(f"'{op.field}' supports only $eq and a non-empty $in")


def _string_operand(field: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{field}' filters take a non-empty string")
    return value


def _metadata_match(key: str, operator: FilterOperator, value: object) -> ComparisonOperation:
    if not key:
        raise ValueError("metadata filters need a key, e.g. 'metadata.<key>'")
    if operator in (FilterOperator.IN, FilterOperator.NIN) and isinstance(value, str):
        value = value.split(",")
    criteria = {"key": key, "value": value if operator == FilterOperator.EQ else {operator.value: value}}
    try:
        parse_elem_match_criteria(criteria)
    except ValueError as exc:
        raise ValueError(f"'{_METADATA_PREFIX}{key}': {exc}") from None
    return ComparisonOperation(operator=FilterOperator.ELEM_MATCH, field=f"data.{_METADATA_FIELD}", value=criteria)


def _has_tag(value: object) -> FilterOperation:
    return ComparisonOperation(operator=FilterOperator.HAS_KEY, field="data.tags", value=_string_operand("tags", value))


def _uses_metric(value: object) -> FilterOperation:
    ref = _string_operand("metrics", value)
    return ComparisonOperation(operator=FilterOperator.CONTAINS, field="data.spec.metrics", value=ref)


def _has_member(value: object) -> FilterOperation:
    ref = _string_operand("tasks", value)
    if REF_FRAGMENT_SEPARATOR not in ref:
        return ComparisonOperation(
            operator=FilterOperator.ELEM_MATCH,
            field="data.tasks",
            value={FilterOperator.STARTS_WITH.value: f"{ref}{REF_FRAGMENT_SEPARATOR}"},
        )
    if not re.fullmatch(DIGEST_PATTERN, ref.split(REF_FRAGMENT_SEPARATOR, 1)[1]):
        raise ValueError(
            "'tasks' filters match a member at any revision ('workspace/name') or at one pinned digest "
            "('workspace/name#<digest>'); stored members are pinned by digest, so a tag fragment never matches"
        )
    return ComparisonOperation(operator=FilterOperator.CONTAINS, field="data.tasks", value=ref)


def _operator_operands(value: object) -> list[tuple[str, object]]:
    if isinstance(value, dict) and value and all(str(key).startswith("$") for key in value):
        return [(str(operator), operand) for operator, operand in value.items()]
    return [(FilterOperator.EQ.value, value)]


def _translate_metadata_comparison(op: ComparisonOperation) -> FilterOperation:
    if op.field == _METADATA_FIELD:
        if op.operator != FilterOperator.EQ or not isinstance(op.value, dict) or not op.value:
            raise ValueError("'metadata' filters take key/value pairs, e.g. 'metadata.<key>' or 'metadata[<key>]'")
        pairs: list[FilterOperation] = [
            _translate_metadata_comparison(
                ComparisonOperation(operator=FilterOperator(operator), field=f"{_METADATA_PREFIX}{key}", value=operand)
            )
            for key, value in op.value.items()
            for operator, operand in _operator_operands(value)
        ]
        return pairs[0] if len(pairs) == 1 else LogicalOperation(operator=FilterOperator.AND, operations=pairs)
    return _metadata_match(op.field.removeprefix(_METADATA_PREFIX), op.operator, op.value)


def qualify_ref_filters(operation: FilterOperation | None, workspace: str) -> FilterOperation | None:
    """Qualify bare ``name`` refs in metric and member filters with ``workspace``."""

    def qualify(ref: str) -> str:
        return ref if "/" in ref.split(REF_FRAGMENT_SEPARATOR, 1)[0] else f"{workspace}/{ref}"

    if isinstance(operation, ComparisonOperation):
        if operation.field not in _REF_ARRAY_FIELDS:
            return operation
        value = operation.value
        if operation.operator == FilterOperator.CONTAINS and isinstance(value, str):
            return operation.model_copy(update={"value": qualify(value)})
        prefix = value.get(FilterOperator.STARTS_WITH.value) if isinstance(value, dict) else None
        if operation.operator == FilterOperator.ELEM_MATCH and len(value) == 1 and isinstance(prefix, str):
            return operation.model_copy(update={"value": {FilterOperator.STARTS_WITH.value: qualify(prefix)}})
        return operation
    if isinstance(operation, LogicalOperation):
        return operation.model_copy(
            update={"operations": [qualify_ref_filters(child, workspace) for child in operation.operations]}
        )
    return operation


class RecordFilter(DataFilter):
    """A ``DataFilter`` for revisioned records: metadata pairs, revision tags, and per-model value matchers."""

    metadata: dict[str, ElemMatchScalar] | None = Field(
        None,
        description="Filter by metadata annotations: `metadata.<key>` (or `metadata[<key>]`) matches records "
        "whose metadata has that key with a matching value. Supports `$eq`, `$in`, `$nin`, `$like`, `$startsWith`, "
        "`$endsWith`, `$lt`, `$lte`, `$gt`, and `$gte`.",
    )
    tags: str | None = Field(
        None,
        description="Filter by revision tag: matches records carrying the tag, e.g. `stable`. Supports `$eq` and `$in`.",
    )

    _VALUE_MATCHERS: ClassVar[dict[str, Callable[[object], FilterOperation]]] = {"tags": _has_tag}

    @classmethod
    def _get_entity_namespace_map(cls) -> dict[str, str]:
        return {_METADATA_FIELD: f"data.{_METADATA_FIELD}"}

    @classmethod
    def translate_operation(cls, operation: FilterOperation) -> FilterOperation:
        def _walk(op: FilterOperation) -> FilterOperation:
            if isinstance(op, ComparisonOperation):
                if op.field == _METADATA_FIELD or op.field.startswith(_METADATA_PREFIX):
                    return _translate_metadata_comparison(op)
                if op.field in cls._VALUE_MATCHERS:
                    return _one_or_any(op, cls._VALUE_MATCHERS[op.field])
            if isinstance(op, LogicalOperation):
                return op.model_copy(update={"operations": [_walk(child) for child in op.operations]})
            return op

        return super().translate_operation(_walk(operation))


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


class AgentEvalResultSummary(BaseModel):
    """The queryable part of an agent-eval bundle's ``summary.json``, under the same field names.

    The per-trial tables (``task_metric_values``, ``error_trial_ids``) stay in the bundle because
    they scale with trial count; ``scores`` lives on the result record itself.
    """

    model_config = ConfigDict(extra="forbid")

    task_count: int = Field(ge=0, description="Tasks represented in the run.")
    trial_count: int = Field(ge=0, description="Trial records in the run.")
    score_count: int = Field(ge=0, description="Metric scores computed for the run.")
    error_count: int = Field(ge=0, description="Trials that reported a producer error.")
    metric_coverage: dict[str, dict[str, AgentEvalMetricOutputCoverage]] = Field(
        default_factory=dict, description="Per-metric, per-output coverage counts (total/scored/failed/missing)."
    )

    @classmethod
    def from_summary(cls, summary: AgentEvalSummary) -> Self:
        return cls(
            task_count=summary.task_count,
            trial_count=summary.trial_count,
            score_count=summary.score_count,
            error_count=summary.error_count,
            metric_coverage=summary.metric_coverage,
        )


class AgentEvalResult(_ResultBase):
    """API representation of a persisted agent-evaluation result record."""

    summary: AgentEvalResultSummary | None = Field(
        default=None, description="Run rollup from the bundle summary; None for records stored before it existed."
    )


class EvaluateResult(_ResultBase):
    """API representation of a persisted (row) evaluation result record."""

    dataset_ref: str | None = Field(
        default=None, description="Reference to the dataset evaluated; None for an inline dataset."
    )
    metric_types: list[str] = Field(description="Runtime metric type names applied in the run.")
    row_count: int | None = Field(default=None, description="Dataset rows the run evaluated; None for older records.")
    error_row_count: int | None = Field(
        default=None,
        description="Rows where inference or a metric recorded an error; None for older records.",
    )


class Task(BaseModel):
    """API representation of a stored agent-eval task.

    Maps to the SDK :class:`~nhx_evals_sdk.agent_eval.tasks.AgentEvalTask` — the task's stable
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


class TaskFilter(RecordFilter):
    """Filter for task queries."""

    workspace: str | None = Field(None, description="Filter by workspace.")
    name: str | None = Field(None, description="Filter by name.")
    kind: str | None = Field(None, description="Filter by task kind (the runner that executes it), e.g. `harbor`.")
    intent: str | None = Field(None, description="Filter by an evaluator task's intent, e.g. `$like` for text search.")
    native_task_id: str | None = Field(None, description="Filter by a Harbor task's native task id.")
    metrics: str | None = Field(
        None,
        description="Filter by metric: matches tasks that use the metric `workspace/name`. Supports `$eq` and `$in`.",
    )
    created_at: DatetimeFilter | None = Field(None, description="Filter by creation date.")
    updated_at: DatetimeFilter | None = Field(None, description="Filter by update date.")

    _VALUE_MATCHERS: ClassVar[dict[str, Callable[[object], FilterOperation]]] = {
        **RecordFilter._VALUE_MATCHERS,
        "metrics": _uses_metric,
    }

    @classmethod
    def _get_entity_field_map(cls) -> dict[str, str]:
        return {
            **super()._get_entity_field_map(),
            "kind": "data.spec.kind",
            "intent": "data.spec.intent",
            "native_task_id": "data.spec.native_task_id",
        }


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
        description="Reference to taskset-level files shared across member tasks, such as a common scoring script.",
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

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"oneOf": [{"required": ["tasks"]}, {"required": ["task_ids"]}]},
    )

    description: str | None = Field(default=None, description="Human-readable description of the grouping.")
    tasks: TaskRefList = Field(
        default_factory=list,
        description="References to the member tasks (set semantics; duplicates rejected). Each may be "
        "bare, tag-pinned ('task-a#latest'), or digest-pinned; all are resolved to an exact digest "
        "when stored, so the grouping cannot change underneath you when a member republishes. "
        "Because membership is a set, the stored order is canonical rather than the submitted order: "
        "reordering the same members is not a content change and publishes no revision.",
    )
    task_ids: list[Annotated[str, Field(min_length=1)]] = Field(
        default_factory=list, description="Stored task record IDs; mutually exclusive with tasks. Resolved on write."
    )

    @model_validator(mode="after")
    def exclusive_membership(self) -> TasksetInput:
        """Require exactly one membership representation and reject duplicate task IDs.

        Returns:
            The validated taskset input.

        Raises:
            ValueError: Both or neither membership fields are supplied, or task IDs repeat.
        """
        if len(self.model_fields_set & {"tasks", "task_ids"}) != 1:
            raise ValueError("Supply exactly one of tasks or task_ids")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("Duplicate task IDs")
        return self

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


class TasksetFilter(RecordFilter):
    """Filter for taskset queries."""

    workspace: str | None = Field(None, description="Filter by workspace.")
    name: str | None = Field(None, description="Filter by name.")
    description: str | None = Field(None, description="Filter by description, e.g. `$like` for text search.")
    tasks: str | None = Field(
        None,
        description="Filter by member task: `workspace/name` matches it at any revision, `workspace/name#<digest>` "
        "at that pinned revision. Supports `$eq` and `$in`.",
    )

    _VALUE_MATCHERS: ClassVar[dict[str, Callable[[object], FilterOperation]]] = {
        **RecordFilter._VALUE_MATCHERS,
        "tasks": _has_member,
    }
    created_at: DatetimeFilter | None = Field(None, description="Filter by creation date.")
    updated_at: DatetimeFilter | None = Field(None, description="Filter by update date.")

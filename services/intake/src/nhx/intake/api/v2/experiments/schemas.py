# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request, response, and filter schemas for the Evaluations API.

Response models are standalone: they translate from the stored entity via
``from_entity`` and carry rollup fields hydrated from ClickHouse at read time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self

from nhx.common.entities.values import DatetimeFilter, Filter, NumberFilter, map_entity_field
from nhx.intake.entities.experiments import ColumnLayout, Experiment, ExperimentGroup, ParetoConfig
from nhx.intake.repository.evaluation_session import EvaluationSessionRow
from nhx.intake.spans.domain import (
    INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT,
    IntakeResponseMode,
    SpanStatus,
)
from nhx.intake.spans.storage import text_for_mode
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, computed_field, model_validator

EvaluationSessionMode = IntakeResponseMode


class ExperimentRequest(BaseModel):
    """Request body for creating an Experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Workspace-unique experiment name.")
    description: str | None = Field(default=None, description="Human-readable purpose of the experiment.")
    insight_id: str | None = Field(
        default=None, description="Reference to an external insight that seeded this experiment, if any."
    )
    summary: str | None = Field(
        default=None, description="Human- or agent-authored summary of the experiment's findings."
    )
    metadata: dict[str, str] | None = Field(default=None, description="Free-form producer metadata for the experiment.")
    default_sort: str = Field(
        default="-created_at",
        description=(
            "Default sort for this experiment's evaluations list, as a `sort`-param string: a comma-separated, "
            "ordered list of fields where the first is the primary sort and the rest break ties (leading "
            "'-' on a field = descending), e.g. '-evaluators.reward.mean,cost_usd.mean'. Defaults to "
            "'-created_at'. Accepts any field the evaluations list `sort` param does; clients apply it as "
            "the list `sort` param."
        ),
    )
    pareto: ParetoConfig | None = Field(
        default=None,
        description=(
            "Default X/Y metrics for the experiment's Pareto view. Omit to preserve the existing value on "
            "update; on create, defaults to cost vs. latency."
        ),
    )
    column_layout: ColumnLayout | None = Field(
        default=None,
        description=(
            "Saved column order and column visibility for the experiment's evaluations table. Omit to "
            "leave the saved layout unchanged; on create that means no layout is saved. An empty order "
            "and hidden pair is itself a saved layout — one that shows every column — and is distinct "
            "from having none."
        ),
    )
    is_favorite: bool = Field(
        default=False,
        description=(
            "Whether this Experiment is marked as a favorite. Defaults to false on create; omit on update to "
            "preserve the existing value."
        ),
    )
    show_evaluations_over_time: bool = Field(
        default=False,
        description=(
            "Whether Studio should display this Experiment's Evaluation results over time. Defaults to false "
            "on create; omit on update to preserve the existing value."
        ),
    )


class ExperimentUpdateRequest(ExperimentRequest):
    """Request body for updating an Experiment."""

    baseline_evaluation_name: str | None = Field(
        default=None,
        description=(
            "Name of this Experiment's baseline Evaluation. The Evaluation must already be a live member "
            "of the Experiment. Set null to clear the selected baseline."
        ),
        json_schema_extra={"nullable": True},
    )


class EvaluationRequest(BaseModel):
    """Request body for creating an Evaluation."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Producer-supplied, workspace-unique evaluation id.")
    experiment_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Entity ids of the Experiments this Evaluation belongs to (>=1). Preferred; each experiment "
            "must already exist. When omitted, the deprecated experiment_group_id is used instead."
        ),
    )
    experiment_group_id: str | None = Field(
        default=None,
        deprecated=True,
        description=(
            "Deprecated single-experiment field; provide experiment_ids instead. Coalesced into experiment_ids "
            "when experiment_ids is omitted."
        ),
    )
    dataset_name: str = Field(description="Producer-supplied dataset name.")
    dataset_version: str | None = Field(default=None, description="Producer-supplied dataset version.")
    source_link: AnyUrl | None = Field(default=None, description="Optional URL for the source evaluation.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Free-form producer metadata.")
    description: str | None = Field(default=None, description="Human-readable description.")
    parent_evaluation_id: str | None = Field(
        default=None,
        description="Entity id of the evaluation this one was derived from (e.g. a variant of a baseline), if any.",
    )
    status: str | None = Field(default=None, description="Producer-defined lifecycle status of the evaluation.")
    root_cause: str | None = Field(
        default=None,
        description="Human- or agent-authored explanation of the evaluation's outcome (e.g. why it was killed).",
    )

    @model_validator(mode="after")
    def _resolve_group_membership(self) -> Self:
        """Require >=1 experiment. Coalesce the deprecated ``experiment_group_id`` into ``experiment_ids``
        when the latter is omitted; reject a request that supplies neither.

        Read the deprecated value via ``__dict__`` to avoid tripping its deprecation warning per request.
        """
        if not self.experiment_ids:
            group_id = self.__dict__.get("experiment_group_id")
            if group_id:
                self.experiment_ids = [group_id]
        if not self.experiment_ids:
            raise ValueError(
                "An evaluation must belong to at least one experiment: provide experiment_ids or experiment_group_id."
            )
        # Membership is a set — drop duplicate experiment ids (order-preserving) so counts aren't inflated.
        self.experiment_ids = list(dict.fromkeys(self.experiment_ids))
        return self


class EvaluationPatchRequest(BaseModel):
    """Partial-update body for an Evaluation: only fields present in the request are applied.

    Unset fields are left unchanged (PATCH semantics — same pattern as the models service's PATCH).
    Immutable fields (name, dataset_name, dataset_version) aren't accepted here. ``experiment_ids``,
    when provided, must be non-empty: an evaluation must always belong to at least one experiment.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_ids: list[str] | None = Field(
        default=None,
        description=(
            "Replace the Experiments this Evaluation belongs to. Must be non-empty when provided; "
            "each experiment must already exist. Omit to leave membership unchanged."
        ),
    )
    source_link: AnyUrl | None = Field(default=None, description="Optional URL for the source evaluation.")
    metadata: dict[str, str] | None = Field(default=None, description="Free-form producer metadata.")
    description: str | None = Field(default=None, description="Human-readable description.")
    parent_evaluation_id: str | None = Field(
        default=None,
        description="Entity id of the evaluation this one was derived from (e.g. a variant of a baseline), if any.",
    )
    status: str | None = Field(default=None, description="Producer-defined lifecycle status of the evaluation.")
    root_cause: str | None = Field(
        default=None,
        description="Human- or agent-authored explanation of the evaluation's outcome (e.g. why it was killed).",
    )


class ExperimentResponse(BaseModel):
    """Experiment as served by the API."""

    id: str
    name: str
    workspace: str
    description: str | None = None
    insight_id: str | None = None
    summary: str | None = None
    metadata: dict[str, str] | None = None
    default_sort: str
    pareto: ParetoConfig = Field(default_factory=ParetoConfig)
    column_layout: ColumnLayout | None = Field(
        default=None,
        description=(
            "Saved column order and column visibility for the experiment's evaluations table. Null when "
            "no layout has been saved, which is distinct from a saved layout that hides nothing."
        ),
        json_schema_extra={"nullable": True},
    )
    is_favorite: bool = Field(default=False, description="Whether this Experiment is marked as a favorite.")
    show_evaluations_over_time: bool = Field(
        default=False,
        description="Whether Studio should display this Experiment's Evaluation results over time.",
    )
    baseline_evaluation_name: str | None = Field(
        default=None,
        description="Name of this Experiment's selected baseline Evaluation, if any.",
    )
    created_at: datetime | None = None
    updated_at: datetime | None = None
    evaluation_count: int = Field(
        default=0,
        description="Number of live (non-soft-deleted) evaluations in this experiment.",
    )

    @classmethod
    def from_entity(cls, entity: ExperimentGroup) -> ExperimentResponse:
        return cls(
            id=entity.id,
            name=entity.name,
            workspace=entity.workspace,
            description=entity.description,
            insight_id=entity.insight_id,
            summary=entity.summary,
            metadata=entity.metadata,
            default_sort=entity.default_sort,
            pareto=entity.pareto,
            column_layout=entity.column_layout,
            is_favorite=entity.is_favorite,
            show_evaluations_over_time=entity.show_evaluations_over_time,
            baseline_evaluation_name=entity.baseline_evaluation_name,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )


class EvaluatorAggregate(BaseModel):
    """Aggregate statistics over evaluator scores or session-level metric values."""

    sum: float | None = None
    mean: float | None = None
    median: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    count: int = 0


class EvaluationResponse(BaseModel):
    """Evaluation as served by the API, including ClickHouse-hydrated rollups."""

    id: str
    name: str
    workspace: str
    experiment_ids: list[str] = Field(
        description="Entity ids of the Experiments this Evaluation belongs to (>=1).",
    )
    dataset_name: str
    dataset_version: str | None = None
    source_link: AnyUrl | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    description: str | None = None
    parent_evaluation_id: str | None = None
    status: str | None = None
    root_cause: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    pinned_at: datetime | None = Field(
        default=None,
        description=(
            "Timestamp at which the evaluation was pinned, or null if unpinned. "
            "Managed via POST/DELETE /evaluations/{name}/pin."
        ),
        json_schema_extra={"nullable": True},
    )

    evaluator_names: list[str] = Field(default_factory=list)
    model_names: list[str] = Field(
        default_factory=list,
        description="Distinct model names observed across ingested sessions for this evaluation.",
        json_schema_extra={"uniqueItems": True},
    )
    agent_names: list[str] = Field(
        default_factory=list,
        description="Distinct agent names observed across ingested sessions for this evaluation.",
        json_schema_extra={"uniqueItems": True},
    )
    agent_versions: list[str] = Field(
        default_factory=list,
        description="Distinct agent versions observed across ingested sessions for this evaluation.",
        json_schema_extra={"uniqueItems": True},
    )
    aggregate_scores: dict[str, EvaluatorAggregate] | None = None
    run_count: int = Field(
        default=0,
        description="Number of distinct ingested evaluation sessions; one session is treated as one run.",
    )
    test_case_count: int = Field(
        default=0,
        description=(
            "Number of distinct test cases in the evaluation, i.e. distinct test_case_name values "
            "(sessions with no test_case_name each count as their own). A test case run k times counts once; "
            "the rollup metrics are averaged per test case before pooling across test cases."
        ),
    )
    cost_usd: EvaluatorAggregate | None = None
    latency_ms: EvaluatorAggregate | None = None
    tokens: EvaluatorAggregate | None = Field(
        default=None,
        description="Average total tokens (input + output) per test case, aggregated across the evaluation.",
    )

    @computed_field(  # type: ignore[prop-decorator]
        deprecated=True,
        description="Deprecated single-experiment alias; the first of experiment_ids. Use experiment_ids.",
    )
    @property
    def experiment_group_id(self) -> str:
        return self.experiment_ids[0]

    @classmethod
    def from_entity(cls, entity: Experiment) -> EvaluationResponse:
        return cls(
            id=entity.id,
            name=entity.name,
            workspace=entity.workspace,
            experiment_ids=entity.experiment_ids,
            dataset_name=entity.dataset_name,
            dataset_version=entity.dataset_version,
            source_link=entity.source_link,
            metadata=entity.metadata,
            description=entity.description,
            parent_evaluation_id=entity.parent_experiment_id,
            status=entity.status,
            root_cause=entity.root_cause,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            pinned_at=entity.pinned_at,
        )


class MetricStatFilters(BaseModel):
    """Numeric range filters keyed by rollup aggregate stat.

    Declaring each stat explicitly (rather than an open ``dict[str, NumberFilter]``) makes the valid
    stats visible in the OpenAPI schema, e.g. ``filter[cost_usd.mean][$lte]=0.5``. These stats must
    stay in sync with the runtime sort/filter grammar (``_METRIC_STATS`` in the evaluations
    endpoints); a unit test guards the parity.
    """

    model_config = ConfigDict(extra="forbid")

    sum: NumberFilter | None = None
    mean: NumberFilter | None = None
    median: NumberFilter | None = None
    p90: NumberFilter | None = None
    p95: NumberFilter | None = None
    p99: NumberFilter | None = None
    count: NumberFilter | None = None


class ExperimentFilter(Filter):
    """Filter for listing Experiments."""

    name: str | None = Field(default=None, description="Filter experiments by name.")
    insight_id: str | None = Field(
        default=None,
        description="Filter experiments by the id of the insight that seeded them.",
    )
    is_favorite: bool | None = Field(default=None, description="Filter experiments by favorite status.")
    show_evaluations_over_time: bool | None = Field(
        default=None,
        description="Filter experiments by whether Studio should display Evaluation results over time.",
    )
    baseline_evaluation_name: str | None = Field(
        default=None,
        description="Filter experiments by the name of their selected baseline Evaluation.",
    )
    is_deleted: bool | None = Field(
        default=None,
        description="When true, returns only soft-deleted experiments. Omit (or false) to see only live experiments.",
    )
    metadata: Annotated[dict[str, str] | None, map_entity_field("data.metadata", namespace=True)] = Field(
        default=None,
        description="Filter by a metadata key/value pair, e.g. filter[metadata.model]=claude-opus-4-8.",
    )


class EvaluationFilter(Filter):
    """Filter for listing Evaluations."""

    name: str | None = Field(default=None, description="Filter evaluations by name.")
    experiment_id: str | None = Field(
        default=None,
        description=(
            "Filter evaluations by experiment group membership: matches evaluations whose "
            "`experiment_ids` include this group id (legacy rows still on `experiment_group_id` also "
            "match). This is the canonical membership filter — it mirrors the write-side `experiment_ids`."
        ),
    )
    experiment_group_id: str | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated alias for `experiment_id`; filter evaluations by experiment group membership.",
    )
    dataset_name: str | None = Field(default=None, description="Filter evaluations by dataset name.")
    dataset_version: str | None = Field(default=None, description="Filter evaluations by dataset version.")
    created_by: str | None = Field(default=None, description="Filter evaluations by the principal that created them.")
    created_at: DatetimeFilter | None = Field(
        default=None,
        description="Filter evaluations by creation timestamp; supports `$gte` and `$lte` for ranges.",
    )
    updated_at: DatetimeFilter | None = Field(
        default=None,
        description="Filter evaluations by last-updated timestamp; supports `$gte` and `$lte` for ranges.",
    )
    is_deleted: bool | None = Field(
        default=None,
        description=("When true, returns only soft-deleted evaluations. Omit (or false) to see only live evaluations."),
    )
    is_pinned: bool | None = Field(
        default=None,
        description=(
            "When true, returns only pinned evaluations. When false, returns only unpinned evaluations. "
            "Omit to return both."
        ),
    )
    metadata: Annotated[dict[str, str] | None, map_entity_field("data.metadata", namespace=True)] = Field(
        default=None,
        description="Filter by a metadata key/value pair, e.g. filter[metadata.model]=claude-opus-4-8.",
    )
    # Name-facet filters. Each is a scalar that matches evaluations whose denormalized list facet
    # *contains* the value (the endpoint rewrites the parsed equality into a $contains membership match,
    # like experiment_id). These are entity-store predicates (workspace-scoped, indexed prefix), so they
    # filter the whole workspace without touching ClickHouse. The facets are refreshed after ingest, so a
    # just-ingested name can lag by up to denormalization_interval_seconds.
    agent_name: Annotated[str | None, map_entity_field("data.agent_names")] = Field(
        default=None,
        description="Filter evaluations that observed this agent name in any ingested session.",
    )
    agent_version: Annotated[str | None, map_entity_field("data.agent_versions")] = Field(
        default=None,
        description="Filter evaluations that observed this agent version in any ingested session.",
    )
    model_name: Annotated[str | None, map_entity_field("data.model_names")] = Field(
        default=None,
        description="Filter evaluations that observed this model name in any ingested session.",
    )
    # Rollup-metric filters. These live in ClickHouse, not the entity store, so they're declared as
    # self-mapping namespaces (the path is left untranslated) and applied in the application layer
    # after rollup hydration rather than forwarded to Postgres. Stat sub-paths mirror the sort grammar:
    # filter[cost_usd.mean][gte]=0.8, filter[evaluators.<name>.mean][lte]=0.5, filter[run_count][gte]=5.
    run_count: Annotated[NumberFilter | None, map_entity_field("run_count")] = Field(
        default=None, description="Filter by run count, e.g. filter[run_count][$gte]=5."
    )
    cost_usd: Annotated[MetricStatFilters | None, map_entity_field("cost_usd", namespace=True)] = Field(
        default=None, description="Filter by a cost_usd rollup stat, e.g. filter[cost_usd.mean][$lte]=0.5."
    )
    latency_ms: Annotated[MetricStatFilters | None, map_entity_field("latency_ms", namespace=True)] = Field(
        default=None, description="Filter by a latency_ms rollup stat, e.g. filter[latency_ms.p95][$lte]=1000."
    )
    tokens: Annotated[MetricStatFilters | None, map_entity_field("tokens", namespace=True)] = Field(
        default=None, description="Filter by a tokens rollup stat, e.g. filter[tokens.mean][$lte]=5000."
    )
    evaluators: Annotated[dict[str, MetricStatFilters] | None, map_entity_field("evaluators", namespace=True)] = Field(
        default=None,
        description="Filter by an evaluator rollup stat, e.g. filter[evaluators.<name>.mean][$gte]=0.8.",
    )


class EvaluationSessionFilter(Filter):
    """Filter for listing EvaluationSessions."""

    test_case_name: str | None = Field(default=None, description="Filter by test case name.")
    test_case_id: str | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated alias for test_case_name. Use test_case_name instead.",
    )
    status: str | None = Field(
        default=None, description="Filter by root-span status (success, error, cancelled, unknown)."
    )


class EvaluationSessionResponse(BaseModel):
    """One ingested session of an Evaluation — a single test case execution.

    Hydrated from ClickHouse at read time by reading root/session membership from
    ``trace_index`` and joining page-bounded span/evaluator rollups.
    """

    workspace: str
    evaluation_name: str
    session_id: str
    test_case_name: str | None = Field(
        default=None,
        description="Test case name; null when the producer did not set one.",
    )
    test_case_id: str | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated alias for test_case_name. Use test_case_name instead.",
    )
    trace_id: str
    root_span_id: str

    started_at: datetime
    ended_at: datetime | None = None
    latency_ms: float | None = None

    status: SpanStatus = Field(description="Root-span status: success, error, cancelled, or unknown.")
    input: str | None = Field(
        default=None,
        description=(
            f"Root-span input text. Omitted in summary mode and truncated to {INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT} "
            "characters in preview mode."
        ),
    )
    output: str | None = Field(
        default=None,
        description=(
            f"Root-span output text. Omitted in summary mode and truncated to {INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT} "
            "characters in preview mode."
        ),
    )

    input_tokens: int | None = Field(default=None, description="Sum of input tokens across this session's spans.")
    output_tokens: int | None = Field(default=None, description="Sum of output tokens across this session's spans.")
    cached_tokens: int | None = Field(default=None, description="Sum of cached tokens across this session's spans.")
    cost_total_usd: float | None = Field(default=None, description="Sum of cost across this session's spans.")

    evaluator_scores: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-evaluator session-mean score. Includes NUMERIC and BOOLEAN evaluator results only; "
            "text/categorical results are omitted."
        ),
    )

    @classmethod
    def from_row(
        cls,
        row: EvaluationSessionRow,
        *,
        mode: EvaluationSessionMode = "detailed",
    ) -> EvaluationSessionResponse:
        return cls(
            workspace=row.workspace,
            evaluation_name=row.evaluation_name,
            session_id=row.session_id,
            test_case_name=row.test_case_name,
            test_case_id=row.test_case_name,
            trace_id=row.trace_id,
            root_span_id=row.root_span_id,
            started_at=row.started_at,
            ended_at=row.ended_at,
            latency_ms=row.latency_ms,
            status=row.status,
            input=text_for_mode(row.input, mode=mode),
            output=text_for_mode(row.output, mode=mode),
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            cached_tokens=row.cached_tokens,
            cost_total_usd=row.cost_total_usd,
            evaluator_scores=row.evaluator_scores,
        )

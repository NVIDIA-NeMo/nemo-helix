# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment and ExperimentGroup entity definitions for the Intake service.

These are entity-store rows, distinct from ClickHouse telemetry. They hold the
durable, producer-supplied metadata that organizes telemetry into leaderboard
views. Rollups are derived from ClickHouse at read time.

NOTE: The public API and Studio already call this concept an "Evaluation" — but
the entity here is intentionally still ``Experiment`` (``__entity_type__ =
"experiment"``, ``parent_experiment_id``). Renaming the entity, its
``__entity_type__``, and its stored fields is a breaking storage change that
requires a one-time data migration of existing rows, so it is deferred to a
later pass. Until then the API layer maps Evaluation ⇄ this Experiment entity.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, ClassVar

from nhx.common.entities.client import EntityBase
from pydantic import AnyUrl, BaseModel, Field, field_validator, model_validator


def _stringify_metadata(value: Any) -> Any:
    """Schema-on-read coercion for ``metadata``, which tightened from ``dict[str, Any]`` to
    ``dict[str, str]``. Legacy rows may hold non-string values; stringify them (JSON-encoding
    structured values) so old rows still read. Non-dict values pass through for pydantic to handle."""
    if not isinstance(value, dict):
        return value
    return {key: val if isinstance(val, str) else json.dumps(val) for key, val in value.items()}


class ParetoConfig(BaseModel):
    """Default X/Y metrics for a group's cost-vs-accuracy Pareto view.

    Metric ids use the same vocabulary as the evaluations list sort/filter fields — ``cost_usd``,
    ``latency_ms``, or ``evaluators.<name>``. Defaults to cost (x) vs latency (y): both exist for
    every group, so the chart always has something to render before anyone customizes it.
    """

    x_metric: str = Field(default="cost_usd", description="Metric plotted on the Pareto X axis.")
    y_metric: str = Field(default="latency_ms", description="Metric plotted on the Pareto Y axis.")

    @field_validator("x_metric", "y_metric")
    @classmethod
    def _validate_metric(cls, value: str) -> str:
        """Restrict axes to metrics Studio can actually plot: the two fixed metrics or a dynamic
        ``evaluators.<name>``. Evaluator names are customer-specific, so only the prefix is checked."""
        if value in ("cost_usd", "latency_ms"):
            return value
        if value.startswith("evaluators.") and value != "evaluators.":
            return value
        raise ValueError(
            f"Unsupported Pareto metric {value!r}; expected 'cost_usd', 'latency_ms', or 'evaluators.<name>'."
        )


class ColumnLayout(BaseModel):
    """A saved table layout for a group's evaluations list: column order and which columns are hidden.

    Column ids are Studio's and cannot be enumerated here — the table builds a column per evaluator
    and metadata key found in the rows — so ids are stored and echoed back unvalidated.

    Visibility is stored as the *hidden* ids rather than a map over every column, so a column that
    appears later (a new evaluator, a new metadata key) shows up by default.
    """

    order: list[str] = Field(
        default_factory=list,
        description=(
            "Column ids in display order. Empty means no saved order: the table falls back to its natural column order."
        ),
    )
    hidden: list[str] = Field(
        default_factory=list,
        description="Column ids hidden from the table. Any column not listed is shown.",
    )


class ExperimentGroup(EntityBase):
    """A named container of Experiments pursuing a single optimization goal.

    A group does not constrain dataset or agent identity across its Experiments.
    """

    __entity_type__: ClassVar[str] = "experiment_group"

    description: str | None = Field(default=None, description="Human-readable purpose of the group.")
    insight_id: str | None = Field(
        default=None,
        description="Reference to an external insight that seeded this group, if any.",
    )
    summary: str | None = Field(default=None, description="Human- or agent-authored summary of the group's findings.")
    metadata: dict[str, str] | None = Field(default=None, description="Free-form producer metadata for the group.")

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> Any:
        return _stringify_metadata(value)

    default_sort: str = Field(
        default="-created_at",
        description=(
            "Default sort for this group's experiments list, as a `sort`-param string: a comma-separated, "
            "ordered list of fields where the first is the primary sort and the rest break ties (leading "
            "'-' on a field = descending), e.g. '-evaluators.reward.mean,cost_usd.mean'. Defaults to "
            "'-created_at'. Accepts any field the experiments list `sort` param does. The client applies "
            "it as the list `sort` param; this endpoint does not consult it."
        ),
    )

    @field_validator("default_sort", mode="before")
    @classmethod
    def _default_sort_fallback(cls, value: Any) -> Any:
        """Upgrade legacy rows to the default. Groups persisted before this field was a non-null string
        stored ``default_sort`` as ``null`` (or, earlier, a ``SortCriterion`` list); the entity store is
        schema-on-read, so coerce anything that isn't a usable string to the default on read."""
        return value if isinstance(value, str) else "-created_at"

    pareto: ParetoConfig = Field(
        default_factory=ParetoConfig,
        description=(
            "Default X/Y metrics for this group's cost-vs-accuracy Pareto view. Defaults to cost (x) "
            "vs latency (y) so the chart always renders before it's customized."
        ),
    )

    @field_validator("pareto", mode="before")
    @classmethod
    def _pareto_fallback(cls, value: Any) -> Any:
        """Schema-on-read: groups persisted before this field stored no ``pareto`` (or ``null``);
        coerce anything that isn't a config mapping to the cost-vs-latency default."""
        return value if isinstance(value, dict | ParetoConfig) else ParetoConfig()

    column_layout: ColumnLayout | None = Field(
        default=None,
        description=(
            "Saved column order and column visibility for this group's evaluations table. Null when no "
            "layout has been saved, which is distinct from a saved layout that hides nothing: the "
            "client applies its own default hidden columns only in the null case."
        ),
    )

    is_favorite: bool = Field(
        default=False,
        description="Whether this Experiment is marked as a favorite.",
    )
    show_evaluations_over_time: bool = Field(
        default=False,
        description="Whether this Experiment should display Evaluation results over time.",
    )
    baseline_evaluation_name: str | None = Field(
        default=None,
        description="Name of this Experiment's baseline Evaluation, if one has been selected.",
    )

    is_deleted: bool = Field(
        default=False,
        description=(
            "Soft-delete flag. DELETE flips this to true and cascades to child experiments. "
            "Deleted groups are hidden from list/get unless `filter[is_deleted]=true` is supplied."
        ),
    )


class Experiment(EntityBase):
    """A single agent/config run against a dataset: one row on a leaderboard.

    ``name`` is the producer-supplied, workspace-unique experiment id.

    Exposed as "Evaluation" by the API/Studio; still stored as ``experiment``
    here pending the entity rename + data migration (see module docstring).
    """

    __entity_type__: ClassVar[str] = "experiment"

    experiment_ids: list[str] = Field(
        min_length=1,
        description=(
            "Entity ids of the ExperimentGroups this Experiment belongs to (>=1). An Experiment "
            "always belongs to at least one group. Validated at create/update time. Deleting a Group "
            "removes it from this set; an Experiment whose sole membership was that Group is deleted "
            "with it, while one shared with another Group survives there."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_group_membership(cls, data: Any) -> Any:
        """Schema-on-read: coerce a legacy single ``experiment_group_id`` into ``experiment_ids``.

        Rows written before many-to-many membership stored a scalar ``experiment_group_id`` and no
        ``experiment_ids``. Synthesize the one-element list so old rows read back unchanged, with no
        data migration. When ``experiment_ids`` is already present it wins (a re-saved row drops the
        stale scalar). Non-dict input passes through for pydantic to handle.
        """
        if isinstance(data, dict) and not data.get("experiment_ids") and data.get("experiment_group_id"):
            return {**data, "experiment_ids": [data["experiment_group_id"]]}
        return data

    dataset_name: str = Field(description="Producer-supplied dataset name.")
    dataset_version: str | None = Field(default=None, description="Producer-supplied dataset version.")
    source_link: AnyUrl | None = Field(default=None, description="Optional URL for the source experiment.")

    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Free-form producer metadata (config snapshot, domain-specific attributes, etc.).",
    )

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: Any) -> Any:
        return _stringify_metadata(value)

    description: str | None = Field(default=None, description="Human-readable description of the experiment.")

    parent_experiment_id: str | None = Field(
        default=None,
        description="Entity id of the experiment this one was derived from (e.g. a variant of a baseline), if any.",
    )
    status: str | None = Field(
        default=None,
        description="Producer-defined lifecycle status of the experiment.",
    )
    root_cause: str | None = Field(
        default=None,
        description="Human- or agent-authored explanation of the experiment's outcome (e.g. why it was killed).",
    )
    is_deleted: bool = Field(
        default=False,
        description=(
            "Soft-delete flag. DELETE flips this to true; on delete the entity is also renamed "
            "(`<name>-deleted-<utc-iso>`) so the original name is free for reuse. Deleted experiments "
            "are hidden from list/get and rejected by ATIF ingest unless `filter[is_deleted]=true`."
        ),
    )

    pinned_at: datetime | None = Field(
        default=None,
        description=(
            "Timestamp at which the experiment was pinned to the top of the list, or null if unpinned. "
            "Managed via POST/DELETE /experiments/{name}/pin (not via the create or update body). "
            "Pin state is workspace-shared: every user with workspace access sees the same pinned set."
        ),
    )

    # System-managed denormalized name fields. These mirror the distinct name sets that the ClickHouse
    # rollup already derives from this evaluation's sessions; a background refresher writes them here
    # (see nhx.intake.experiments.denormalizer) so the workspace-wide Evaluations list can filter by
    # agent/model name against the entity store ($contains) instead of scanning the session table. Not
    # accepted on the create/update body (the request schemas omit them); default empty until refreshed.
    # Unlike computed rollups, these are raw observed strings with no formula, so they never need a
    # backfill/recompute when aggregation logic changes — only when a new name is ingested.
    agent_names: list[str] = Field(
        default_factory=list,
        description="System-managed: distinct agent names observed across this evaluation's ingested sessions.",
    )
    agent_versions: list[str] = Field(
        default_factory=list,
        description="System-managed: distinct agent versions observed across this evaluation's ingested sessions.",
    )
    model_names: list[str] = Field(
        default_factory=list,
        description="System-managed: distinct model names observed across this evaluation's ingested sessions.",
    )

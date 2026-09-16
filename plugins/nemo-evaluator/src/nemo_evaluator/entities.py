# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored entities for the evaluator plugin (metrics and agent-eval tasks).

A :class:`MetricBundleEntity` is the persisted, queryable index for a metric.
The full executable :class:`~nemo_evaluator.shared.metric_bundles.bundles.MetricBundle`
(including its potentially multi-MiB serialized payload) lives in the Files
service; the entity stores only the lightweight, searchable projection plus a
reference (``bundle_ref``) and integrity digest (``payload_digest``) that point
back at the canonical copy.

A :class:`TaskEntity` is the persisted form of an agent-eval task — the SDK
:class:`~nemo_evaluator_sdk.agent_eval.tasks.AgentEvalTask`, addressed by
``workspace/name`` (the task's stable id is the record name) and reusable across runs.
"""

from __future__ import annotations

from typing import ClassVar

from nemo_evaluator.api.schemas import (
    PinnedTaskRefList,
    TaskDefinition,
    TaskMetadataList,
    TaskRefList,
    TasksetFilesRef,
)
from nemo_evaluator.content_hash import DIGEST_LENGTH, DIGEST_PATTERN
from nemo_evaluator.shared.metric_bundles.bundles import BundledMetricOutputSpec
from nemo_evaluator_sdk.values.common import SecretRef
from nemo_evaluator_sdk.values.results import AggregatedMetricResult
from nemo_platform_plugin.entities import EntityBase
from pydantic import BaseModel, Field

# Constants are intentionally local: nmp_common's entity constants are not
# re-exported to plugins. Keep these aligned with
# ``nmp.common.entities.constants``.
MAX_NAME_LENGTH = 255
MAX_DESCRIPTION_LENGTH = 1000
NAME_PATTERN = r"^[\w\-\.]+$"

#: Fields on a publishable record that are revision *bookkeeping*, not content. Excluded when
#: digesting a head record so that hashing the head yields the same digest as hashing the
#: corresponding revision. Two things depend on that equality: a publish recognizing "the head
#: already matches the current revision", and a read re-hashing a revision to check it against the
#: digest stored beside it.
REVISION_POINTER_FIELDS = frozenset({"latest_revision", "tags"})

#: The mirror of :data:`REVISION_POINTER_FIELDS` on a revision record. A revision's digest covers
#: neither itself nor the ordinal that was assigned because of it.
REVISION_SELF_FIELDS = frozenset({"content_hash", "revision"})

#: Spec fields excluded from the revision digest.
#:
#: The rule for what belongs *in* the digest: any field that affects the output of a task's
#: execution or the mechanism used to grade it. A field may only be excluded if it is a derived
#: view of content the digest already covers by another route.
#:
#: ``HarborTaskDefinition.config`` qualifies. It is a projection of ``task.toml``, which lives
#: inside the task archive, and Harbor reads the real ``task.toml`` out of the materialized tree at
#: run time — this copy is never an execution input, only a queryable convenience. ``source.files_hash``
#: is authoritative over every file in that directory including ``task.toml``, so a config change
#: that actually alters execution or grading already moves the digest. Hashing the projection too
#: would add no coverage and would make revision history sensitive to Harbor's serialization: a
#: release that reordered keys or emitted a new defaulted field would cut a revision for
#: byte-identical files.
#:
#: That makes ``source.files_hash`` load-bearing. If a Harbor field ever becomes an execution input in
#: its own right — read from the stored record rather than from the tree — it must be digested.
_DERIVED_SPEC_FIELDS = {"config"}

#: What a *head* record excludes when digesting: its revision pointers, plus derived spec fields.
#: Nested form, because the derived fields live inside ``spec``.
REVISION_POINTER_EXCLUDE: dict[str, object] = {
    **dict.fromkeys(REVISION_POINTER_FIELDS, True),
    "spec": set(_DERIVED_SPEC_FIELDS),
}

#: The mirror for a *revision* record. Both must exclude the same derived fields, or the head and
#: its revision would digest differently and publish-time dedup would never fire.
REVISION_SELF_EXCLUDE: dict[str, object] = {
    **dict.fromkeys(REVISION_SELF_FIELDS, True),
    "spec": set(_DERIVED_SPEC_FIELDS),
}


class MetricBundleEntity(EntityBase):
    """Persisted index for a stored metric, addressed by workspace/name.

    The canonical, executable bundle is stored in the Files service and
    referenced by ``bundle_ref``; the fields here are a denormalized projection
    kept for display and filtering without downloading the payload.
    """

    __entity_type__: ClassVar[str] = "metric_bundle"

    metric_type: str = Field(
        description="Runtime metric type name captured from the bundled metric.",
        max_length=MAX_NAME_LENGTH,
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels captured from the bundled metric's metadata.",
    )
    outputs: list[BundledMetricOutputSpec] = Field(
        default_factory=list,
        description="JSON-safe projection of the metric's output contracts.",
    )
    secrets: dict[str, SecretRef] = Field(
        default_factory=dict,
        description="Secret environment-variable references required to execute the metric.",
    )
    payload_kind: str = Field(
        description="Payload discriminator of the stored bundle (e.g. 'cloudpickle').",
        max_length=MAX_NAME_LENGTH,
    )
    payload_digest: str = Field(
        description="Format-specific digest of the stored payload, used to verify integrity on load.",
        max_length=MAX_NAME_LENGTH,
    )
    bundle_ref: str = Field(
        description="Files reference to the canonical serialized MetricBundle (format: workspace/fileset#path).",
    )
    description: str | None = Field(
        default=None,
        description="Description captured from the bundled metric's metadata.",
        max_length=MAX_DESCRIPTION_LENGTH,
    )
    derived: bool = Field(
        default=False,
        description="True for a content-addressed metric auto-stored from an inline task metric. "
        "Derived metrics are excluded from the default metric listing (they're task internals, not "
        "curated metrics), but are addressable by reference.",
    )


# --- Eval result entities ----------------------------------------------------
#
# A result entity is the persisted, *queryable* record of one eval run: the
# aggregated scores plus the traits you'd filter on (target, dataset). The
# detailed per-row / per-trial output that doesn't fit a concise record stays in
# the run's fileset bundle, referenced here by ``bundle_ref``. The entity — not
# Intake — is the evaluator's source of truth; Intake is a denormalized, optional
# downstream copy.
#
# Both result types share the SAME record (``_EvalResultCommon``): provenance, the target it ran
# against, the aggregated ``scores`` rollup, and a ``bundle_ref`` to the full detail. They differ
# only where the domain genuinely differs — row-eval has *referenceable inputs* (its dataset fileset
# + metric refs), which the entity records; agent-eval's tasks are inline, so it has no input ref yet
# (the "Taskset" gap). Run counts / per-metric coverage are derivable rollups that live in the
# bundle's summary, not on the record. This keeps the two entities aligned and matches the lean legacy
# ``BaseJobResult`` → ``MetricJobResult`` / ``BenchmarkJobResult`` shape (refs + scores).


class _EvalResultCommon(BaseModel):
    """Fields shared by every persisted eval-result record (aggregates + filterable traits).

    A mixin (not itself an ``EntityBase``) so the concrete result entities can each declare their own
    ``__entity_type__`` — same split as the legacy ``BaseJobResult`` → ``MetricJobResult`` /
    ``BenchmarkJobResult``.

    Every field is required — a result is only persisted once the run has produced all of it, so the
    caller populates each value (no schema defaults papering over missing data). "What it ran
    against" is denormalized into flat ``target_*`` fields so the list route can filter by them (the
    entity filter matches top-level fields; a nested object wouldn't filter cleanly); they're nullable
    because an offline run (precomputed trials) has no target, but the caller must still pass them.

    (``labels`` and a run ``status`` are intentionally absent: there's no labels source on the spec
    yet, and persistence happens only on success — both would be schema defaults with no real data.
    Add them when there's a source — labels alongside a spec ``labels`` field, status if/when partial
    or failed runs are persisted.)
    """

    job_id: str = Field(description="Identifier of the job run that produced this result (one result per run).")
    target_kind: str | None = Field(
        description="Target discriminator: 'model', 'agent', or a runner kind e.g. 'fabric'."
    )
    target_name: str | None = Field(description="Model/agent entity name, or the runner's model — filterable trait.")
    target_url: str | None = Field(description="Endpoint URL, when the target is an HTTP model/agent.")
    scores: AggregatedMetricResult = Field(
        description="Aggregated metric scores for the run (the concise, queryable rollup)."
    )
    bundle_ref: str = Field(
        description="Reference to the full result bundle in the Files service (rows/trials), e.g. a 'fileset://...' URL.",
    )


class AgentEvalResultEntity(_EvalResultCommon, EntityBase):
    """Persisted, queryable record of an ``AgentEvalJob`` run.

    Carries only the shared record — its tasks are inline, so (unlike row-eval) it has no input ref
    to record yet. Trials, per-metric coverage, and run counts live in the bundle's summary.
    """

    __entity_type__: ClassVar[str] = "agent_eval_result"


class EvaluateResultEntity(_EvalResultCommon, EntityBase):
    """Persisted, queryable record of an ``EvaluateJob`` (row-eval) run.

    Adds the run's *referenceable inputs* — the evaluated dataset and the metrics applied — which the
    shared record can't capture. Row-level detail lives in the bundle.
    """

    __entity_type__: ClassVar[str] = "evaluate_result"

    dataset_ref: str | None = Field(
        description="Reference to the dataset evaluated (e.g. 'workspace/fileset'); None for an inline dataset."
    )
    metric_types: list[str] = Field(
        description="Runtime metric type names applied in the run (e.g. 'exact_match'). Not metric refs: "
        "by run time the submitted refs are resolved to inline bundles, so the originals aren't available."
    )


class _RevisionedCommon(BaseModel):
    """Mutable revision bookkeeping carried by a record that can be published.

    Both fields are pointers into the record's immutable revision children, and both are excluded
    from the content digest — they describe *which* content is current, not what the content is.
    Including them would make the digest change whenever a tag moved.
    """

    latest_revision: int = Field(
        default=0,
        description="Ordinal of the most recently published revision; 0 before the first publish. "
        "The next publish allocates ``latest_revision + 1`` under the record's optimistic lock, so a "
        "concurrent publisher that raced loses with a conflict and retries.",
        ge=0,
    )
    tags: dict[str, int] = Field(
        default_factory=dict,
        description="Mutable tag → revision-ordinal pointers. ``latest`` is reserved and re-applied "
        "on every publish; other tags are user-supplied and may be moved after the fact. Tags are "
        "resolution *inputs* only — anything persisted (e.g. a published taskset's membership) "
        "stores the resolved content digest, never the tag, so a moved tag cannot re-point it. "
        "Ordinals rather than digests because a revision's identity within its parent is its "
        "ordinal, which makes tag resolution a direct child lookup with no query.",
    )


class TaskEntity(_RevisionedCommon, EntityBase):
    """Persisted, queryable task, addressed by workspace/name.

    A task is an evaluation unit; ``spec`` says what it is and which runner executes it. Both kinds
    live in one record type so a user manages every evaluation unit in one place, and so a taskset
    can group them without caring how each one runs — the same way ``AgentRunnerTarget`` already
    treats fabric/gym/harbor as members of one union on the target side.

    Content is nested under ``spec`` rather than flattened with nullable per-kind fields, so each
    variant's required fields stay genuinely required and the revision digest covers the spec as one
    unit. An agent-eval task's ``metrics`` are stored as references (inline metrics submitted on
    create are normalized to derived stored metrics); a Harbor task's files live in a fileset, and
    the spec holds a reference to them.
    """

    __entity_type__: ClassVar[str] = "task"

    spec: TaskDefinition = Field(description="The task's content, discriminated by which runner executes it.")
    metadata: TaskMetadataList = Field(default_factory=list, description="Key/value annotations for the task.")


class TasksetEntity(_RevisionedCommon, EntityBase):
    """Persisted, queryable taskset, addressed by workspace/name.

    A taskset is a flexible grouping of stored tasks: it holds references to its members
    (``workspace/name``) plus free-form annotations. Membership is a set — order is not significant
    and duplicate references are rejected. Referenced tasks are validated to exist at create time.

    ``files_ref`` points at the grouping's own files: shared by its members, owned by none of
    them, the way a Harbor dataset ships a metric script beside its tasks. One reference rather
    than a list of per-file entries — a grouping's files are a directory, and the fileset already
    knows what is in it. It is a first-class field rather than an annotation because it is
    content: the revision digest covers it, so repointing publishes a revision.
    """

    __entity_type__: ClassVar[str] = "taskset"

    description: str | None = Field(
        default=None,
        description="Human-readable description of the grouping.",
        max_length=MAX_DESCRIPTION_LENGTH,
    )
    tasks: TaskRefList = Field(
        default_factory=list,
        description="References to the member tasks (set semantics; duplicates rejected).",
    )
    files_ref: TasksetFilesRef | None = Field(
        default=None,
        description="Files reference to the taskset's own files (workspace/fileset[#prefix]).",
    )
    metadata: TaskMetadataList = Field(default_factory=list, description="Key/value annotations for the taskset.")


# --- Revisions ---------------------------------------------------------------
#
# A revision is an immutable snapshot of a task's or taskset's content, addressed by a digest of
# that content (see ``nemo_evaluator.content_hash``). Revisions exist so a reference can be *pinned*
# — ``workspace/task-name#<digest>`` resolves to exactly the content that was published, and a
# consumer re-derives the digest on read to confirm it.
#
# Persistence shape. A revision is a **child entity** of the record it snapshots, so entity-store
# parent-scoped uniqueness — unique within ``(workspace, entity_type, parent, name)`` — makes an
# ordinal collide rather than silently duplicate. Allocation of the next ordinal rides on the
# parent's ``db_version`` optimistic lock: bump ``latest_revision`` on the parent, and a concurrent
# publisher that raced loses with a conflict and retries.
#
# Naming. Revisions are named ``rev.<n>``, NOT by their digest. Entity names are capped at 63 chars
# and must start with a lowercase letter (``entity_naming.NAME_PATTERN``); a full 64-char hex digest
# violates both. Truncating it to fit — as derived metric names do — would shrink the collision
# bound for no benefit here, since a ref carries the digest in its ``#`` fragment (governed by the
# ref pattern, not the entity-name rules) and the digest lives on the record as an ordinary field
# with no length pressure. Resolving a pinned ref is a filtered lookup on ``content_hash`` scoped to
# the parent. As a bonus, ``rev.7`` is legible in a log line in a way a hex string is not.
#
# Head vs history. The parent record keeps its content fields as the *head* — the current version —
# and revisions accumulate alongside. This keeps the change additive: existing stored records stay
# valid, ``get_task`` and the ``Task`` DTO are unchanged, and nothing needs migrating. The cost is a
# denormalized copy: head and its corresponding revision can drift if a write fails between the two
# (there is no cross-entity transaction). Publishing reconciles — it hashes the head and creates the
# revision only if no revision carries that digest — so a torn write self-heals on the next publish
# rather than persisting a lie.


class _RevisionCommon(BaseModel):
    """Fields shared by every revision record: its digest and its ordinal.

    A mixin rather than an ``EntityBase`` so each concrete revision type declares its own
    ``__entity_type__``, matching the ``_EvalResultCommon`` split above.

    Both fields are excluded from the content digest — a revision's digest cannot cover the ordinal
    that was assigned *because of* that digest, nor cover itself. See
    ``content_hash.content_hash(..., exclude=...)``.
    """

    content_hash: str = Field(
        description="Full 64-char lowercase hex SHA-256 of this revision's content. Never truncated: "
        "a shortened digest collapses the birthday bound, and there is no length pressure here "
        "because this is a field, not an entity name.",
        min_length=DIGEST_LENGTH,
        max_length=DIGEST_LENGTH,
        pattern=DIGEST_PATTERN,
    )
    revision: int = Field(
        description="Monotonic 1-based ordinal within the parent record. Matches the ``rev.<n>`` "
        "entity name; carried as a field too so it can be sorted and filtered on directly.",
        ge=1,
    )


class TaskRevisionEntity(_RevisionCommon, EntityBase):
    """An immutable published snapshot of a :class:`TaskEntity`'s content.

    Child of the task it snapshots. Content fields mirror ``TaskEntity`` exactly — a revision is
    that content frozen, not a different shape.
    """

    __entity_type__: ClassVar[str] = "task_revision"

    spec: TaskDefinition = Field(description="The task's content as of this revision.")
    metadata: TaskMetadataList = Field(default_factory=list, description="Key/value annotations for the task.")


class TasksetRevisionEntity(_RevisionCommon, EntityBase):
    """An immutable published snapshot of a :class:`TasksetEntity`'s content.

    Child of the taskset it snapshots. Its ``tasks`` are **fully pinned** — every member ref carries
    a ``#<digest>`` fragment resolved at publish time. A stored revision must never retain a tag
    fragment (``#latest``, ``#some-tag``): tags move, and a moved tag would silently re-point a
    published taskset's membership, which is the failure this whole design exists to prevent.
    """

    __entity_type__: ClassVar[str] = "taskset_revision"

    description: str | None = Field(
        default=None,
        description="Human-readable description of the grouping.",
        max_length=MAX_DESCRIPTION_LENGTH,
    )
    tasks: PinnedTaskRefList = Field(
        default_factory=list,
        description="Digest-pinned references to the member tasks, resolved at publish time.",
    )
    files_ref: TasksetFilesRef | None = Field(
        default=None,
        description="Files reference as published. Unlike a member ref there is nothing to resolve: "
        "a Files reference already names an exact location.",
    )
    metadata: TaskMetadataList = Field(default_factory=list, description="Key/value annotations for the taskset.")

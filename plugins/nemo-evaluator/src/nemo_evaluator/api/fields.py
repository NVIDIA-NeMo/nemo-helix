# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared field types for the evaluator API: entity references and metric payloads.

Split out of :mod:`nemo_evaluator.api.schemas` so the per-kind task definitions can use these
without importing the module that composes them into DTOs — the definitions are imported *by*
``schemas``, so they cannot import from it.

What counts as a ``workspace/name`` reference is **not** decided here: the shape
(:data:`~nemo_platform_plugin.refs.ENTITY_REF_PATTERN`) and the parser
(:func:`~nemo_platform_plugin.refs.parse_entity_ref`) are the platform's, shared with every other
plugin. This module only adds what is specific to a *revisioned* evaluator entity — the ``#fragment``
that selects a revision.

Everything here is re-exported from ``schemas`` for callers that already import it from there.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, TypeAlias

from nemo_evaluator.content_hash import DIGEST_PATTERN
from nemo_evaluator.shared.metric_bundles.bundles import (
    BundledMetricOutputSpec,
    MetricMetadata,
)
from nemo_evaluator_sdk.values.common import SecretRef
from nemo_platform_plugin.refs import ENTITY_REF_PATTERN, FILESET_REF_PATTERN, parse_entity_ref
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, RootModel, field_validator


class CloudpickleMetricPayload(BaseModel):
    """Wire schema for a cloudpickle-serialized metric payload.

    Mirrors the runtime ``CloudpickleMetricPayload`` so the API contract is
    explicit in the OpenAPI spec. The runtime bundle model serializes payloads
    polymorphically (typed as an abstract base), which renders as an opaque
    object in the spec; this concrete DTO documents the actual fields.
    """

    model_config = ConfigDict(extra="forbid", ser_json_bytes="base64", val_json_bytes="base64")

    kind: Literal["cloudpickle"] = Field(description="Payload format discriminator.")
    python_version: str = Field(description="Python version the metric was pickled with (must match at execution).")
    cloudpickle_version: str = Field(description="cloudpickle version used to serialize the metric.")
    pickle_protocol: int = Field(description="Pickle protocol used.")
    blob: bytes = Field(description="Base64-encoded cloudpickled metric object.")
    digest: str | None = Field(
        default=None,
        description="SHA-256 digest of the payload bytes. Informational; recomputed server-side.",
    )


class InlineMetricPayload(BaseModel):
    """Wire schema for an inline (config-serialized) metric payload.

    Mirrors the runtime ``InlineMetricPayload``. The metric is stored as its own
    JSON configuration and reconstructed from the metric type union at execution,
    so no code is shipped or executed on load. Used for platform-recognized
    built-in metric types.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["inline"] = Field(description="Payload format discriminator.")
    metric: dict[str, Any] = Field(
        description="JSON-serialized built-in metric configuration, discriminated by its own `type`."
    )
    digest: str | None = Field(
        default=None,
        description="SHA-256 digest of the canonical metric JSON. Informational; recomputed server-side.",
    )

    @field_validator("metric")
    @classmethod
    def _metric_must_declare_type(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Reject payloads without a metric ``type`` discriminator at the API boundary.

        The metric body stays an open object (the concrete shape is validated when
        the bundle is hydrated against the metric type union), but a non-empty
        ``type`` is required so malformed payloads fail fast rather than at execution.
        """
        metric_type = value.get("type")
        if not isinstance(metric_type, str) or not metric_type:
            raise ValueError("inline metric payload must include a non-empty 'type'")
        return value


# Discriminated on ``kind`` so additional payload formats can join the union
# without changing the field type.
MetricPayload = Annotated[CloudpickleMetricPayload | InlineMetricPayload, Field(discriminator="kind")]


class MetricInline(BaseModel):
    """An executable metric submitted to the platform.

    Carries the bundled metric — type, metadata, output contracts, secret
    references, and a format-specific payload — used both as the create-request
    body and as an inline metric in an evaluation job.
    """

    model_config = ConfigDict(extra="forbid")

    bundle_kind: Literal["metric-bundle"] = "metric-bundle"
    bundle_format_version: Literal["v1"] = "v1"
    metric_type: str = Field(min_length=1, description="Runtime metric type name.")
    metadata: MetricMetadata = Field(default_factory=MetricMetadata, description="User-facing metric metadata.")
    outputs: list[BundledMetricOutputSpec] = Field(min_length=1, description="The metric's output contracts.")
    secrets: dict[str, SecretRef] = Field(
        default_factory=dict, description="Secret references required to execute the metric."
    )
    payload: MetricPayload = Field(description="Format-specific serialized metric.")


#: The charset a ``#fragment`` may use. Exported because anything that *mints* a fragment — notably
#: revision tag names — has to be constrained by it: a value outside this set can be stored happily
#: and then never appear in a reference, which is a silent dead end rather than an error.
REF_FRAGMENT_CHARSET = r"[\w\-.]+"

# A *sub-entity* reference adds an optional ``#fragment`` to the platform's ``ENTITY_REF_PATTERN``,
# which is the standard way of addressing something contained within an entity (filesets address a
# contained file the same way: ``workspace/fileset#path``). For a revisioned entity the fragment
# selects a revision — either a tag (``#latest``, ``#candidate``) or a full 64-char content digest.
#
# Deliberately a sibling of ``ENTITY_REF_PATTERN`` rather than a widening of it: that constant is
# still shared by ``MetricRef``, which has no revisions, and admitting a fragment there would accept
# input nothing is built to resolve. ``TaskRef`` and ``TasksetRef`` both use this pattern, since both
# name revisioned records; ``MetricRef`` joins them when (if) metrics gain revisions.
#
# The base alternation is spliced in from the shared constant (minus its anchors) so the two shapes
# cannot drift: widening what counts as a ``workspace/name`` widens both at once.
_SUBENTITY_REF_PATTERN = rf"^{ENTITY_REF_PATTERN.removeprefix('^').removesuffix('$')}(#{REF_FRAGMENT_CHARSET})?$"
#: The fragment separator for sub-entity references. Matches the fileset/job ref convention.
REF_FRAGMENT_SEPARATOR = "#"
#: The tag applied to every publish and used when a ref carries no fragment.
LATEST_TAG = "latest"


def parse_subentity_ref(root: str, default_workspace: str) -> tuple[str, str, str]:
    """Split a reference into ``(workspace, name, fragment)``.

    The ``workspace/name`` split is delegated to the platform's :func:`~nemo_platform_plugin.refs.
    parse_entity_ref`; this only adds the revision fragment on top, so evaluator refs and every other
    plugin's refs agree on what a ``workspace/name`` is. Callers that don't care about revisions
    discard the third element — that, rather than a second parser, is how a pinned ref is read
    unpinned.

    An absent fragment resolves to :data:`LATEST_TAG` — a bare ``workspace/name`` means "the current
    revision", never "unpinned". The fragment is returned verbatim: it may be a tag or a content
    digest, and telling them apart is resolution's job, not parsing's.
    """
    base, separator, fragment = root.partition(REF_FRAGMENT_SEPARATOR)
    parsed = parse_entity_ref(base, default_workspace)
    return parsed.workspace, parsed.name, fragment if separator and fragment else LATEST_TAG


class MetricRef(RootModel[str]):
    """Reference to a persisted metric (format: ``workspace/name`` or ``name``)."""

    root: str = Field(
        pattern=ENTITY_REF_PATTERN,
        description="Reference to a stored metric (format: workspace/metric-name, or metric-name in the job workspace).",
    )


#: A wire metric is either an inline bundle DTO or a reference to a stored metric. Lives here (next to
#: ``MetricInline``) rather than in ``metric_refs`` so entity/DTO modules can use it without importing
#: the ref-resolution logic (which depends on ``entities`` and would cycle); ``metric_refs`` re-exports.
MetricRefOrInline: TypeAlias = MetricInline | MetricRef


class TaskRef(RootModel[str]):
    """Reference to a persisted task (format: ``workspace/name``, ``name``, or either with a
    ``#revision`` fragment).

    A taskset points at its member tasks by reference (there are no inline tasks), so a stored
    taskset only ever holds refs. Unlike :class:`MetricRef`, a task ref may address a specific
    revision via the platform's standard ``#`` sub-entity fragment.

    The fragment is optional *on input* and means :data:`LATEST_TAG` when absent — a bare
    ``workspace/name`` is "the current revision", not "unpinned". It may name a tag or a content
    digest. Anything **persisted** as a published snapshot must carry a resolved digest: tags move,
    and a stored tag fragment would silently re-point published membership.
    """

    root: str = Field(
        pattern=_SUBENTITY_REF_PATTERN,
        description="Reference to a stored task (format: workspace/task-name, or task-name in the "
        "taskset workspace), optionally pinned to a revision with '#<tag-or-digest>'.",
    )


class TasksetRef(RootModel[str]):
    """Reference to a persisted taskset (format: ``workspace/name`` or ``name``, optionally ``#rev``).

    Same shape and charset as :class:`TaskRef`. Lets an evaluation reference a stored taskset in place
    of an inline task list; the taskset's member tasks are loaded and expanded during spec resolution.

    An optional ``#`` fragment pins the taskset revision to expand — a tag or a full content digest,
    with an absent fragment meaning ``latest``.

    What each form guarantees, precisely. A taskset revision pins its members by digest, so a member
    task publishing new content never changes what *any* ref expands to. A **bare** ref still tracks
    the taskset's own revisions, and republishing the taskset re-resolves its members on write — so a
    ``replace`` can change both which members are named and the content they resolve to, even if the
    submitted member names were identical. A **pinned** ref is fixed against that too, and is what an
    evaluation needs to stay comparable across a ``replace``.
    """

    root: str = Field(
        pattern=_SUBENTITY_REF_PATTERN,
        description="Reference to a stored taskset (format: workspace/taskset-name, or taskset-name in the "
        "job workspace), optionally pinned to a revision with '#<tag-or-digest>'.",
    )


class TaskInputs(BaseModel):
    """Inputs supplied to a task.

    ``instruction`` is the agent's prompt; the runtime falls back to the task ``intent`` when it is unset.
    Additional task- or runtime-specific inputs, such as `gym_row` for Gym tasks, are also accepted fields.
    """

    model_config = ConfigDict(extra="allow")

    instruction: str | None = Field(
        default=None, description="The agent's instruction (its prompt). Falls back to the task `intent` when unset."
    )


class MetadataItem(BaseModel):
    """A single key/value annotation on a task."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="Annotation key.")
    value: JsonValue = Field(description="JSON-serializable annotation value.")


def _reject_duplicate_metadata_keys(items: list[MetadataItem]) -> list[MetadataItem]:
    """Metadata is a key→value map expressed as a list; duplicate keys would silently collapse (e.g.
    when folded into a mapping for the runtime), so reject them at validation rather than lose data."""
    seen: set[str] = set()
    for item in items:
        if item.key in seen:
            raise ValueError(f"duplicate metadata key: {item.key!r}")
        seen.add(item.key)
    return items


#: A task's metadata: key/value annotations with unique keys (duplicates rejected at validation).
TaskMetadataList: TypeAlias = Annotated[list[MetadataItem], AfterValidator(_reject_duplicate_metadata_keys)]

#: Where a taskset's own files live: one Files reference, ``workspace/fileset#prefix``. A
#: grouping's files are a directory, not a set of independently addressed blobs, so one reference
#: says everything a list of per-file entries would — while making it impossible for the two to
#: disagree about what the taskset ships.
#:
#: Same shape as ``bundle_ref`` and Harbor tree references, so the fragment is required; here it names the
#: prefix the files sit under rather than a single file. Use a prefix such as ``#files`` to mean
#: "the whole of this fileset's file area".
#:
#: Pinning is arranged by pointing at a location that is not rewritten — a fileset written once, or
#: a content-addressed prefix inside a shared one. A reference into a location that *is* later
#: rewritten resolves to whatever it holds when read, which is the contract the Files service gives
#: every other consumer.
TasksetFilesRef: TypeAlias = Annotated[str, Field(pattern=FILESET_REF_PATTERN)]


def _reject_duplicate_task_refs(refs: list[TaskRef]) -> list[TaskRef]:
    """A taskset's members are an unordered set expressed as a list; a repeated ref is ambiguous
    (it can't mean anything more than membership), so reject duplicates at validation."""
    seen: set[str] = set()
    for ref in refs:
        if ref.root in seen:
            raise ValueError(f"duplicate task reference: {ref.root!r}")
        seen.add(ref.root)
    return refs


#: A list of task references with set semantics (order not significant, duplicates rejected).
TaskRefList: TypeAlias = Annotated[list[TaskRef], AfterValidator(_reject_duplicate_task_refs)]
#: Shape of a content digest in a ref fragment: full-length lowercase hex, never truncated.
_DIGEST_FRAGMENT_PATTERN = re.compile(DIGEST_PATTERN)


def _require_pinned_task_refs(refs: list[TaskRef]) -> list[TaskRef]:
    """Every member of a *published* taskset revision must name an exact content digest.

    Enforced on the field rather than in the publish path so it cannot be bypassed by any other
    writer. A ref that is bare (``workspace/name``) or tag-pinned (``#latest``, ``#candidate``)
    resolves through a mutable pointer: the moment that tag moves, the published revision's
    membership silently changes under it, and a "reproducible" dataset stops being reproducible.
    Tags are resolution *inputs*, resolved to digests at publish time; only digests persist.
    """
    for ref in refs:
        _, _, fragment = parse_subentity_ref(ref.root, "")
        if not _DIGEST_FRAGMENT_PATTERN.match(fragment):
            raise ValueError(
                f"task reference {ref.root!r} is not pinned to a content digest: a published taskset "
                f"revision must reference an exact revision (got fragment {fragment!r}). Tags move; "
                "resolve them to a digest before persisting."
            )
    return refs


#: Member refs of a published taskset revision: set semantics *and* every ref digest-pinned.
PinnedTaskRefList: TypeAlias = Annotated[
    list[TaskRef], AfterValidator(_reject_duplicate_task_refs), AfterValidator(_require_pinned_task_refs)
]

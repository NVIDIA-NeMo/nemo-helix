# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical content-hash tests.

The digest is what a consumer recomputes when reading a pinned ref, so these tests pin the
properties that make that comparison meaningful: determinism, insensitivity to identity and to
server-owned fields, and — the part most likely to break quietly — that near-miss content
variations hash *differently* rather than collapsing onto one digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pytest
from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    HarborTaskDefinition,
    MetadataItem,
    MetricRef,
    TaskInputs,
    TaskRef,
)
from nemo_evaluator.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash
from nemo_evaluator.content_hash import DIGEST_PATTERN, canonical_payload, content_hash
from nemo_evaluator.entities import TaskEntity, TasksetEntity
from nemo_evaluator.revisions import head_digest
from nemo_evaluator_sdk.agent_eval.tasks import SemanticReducer, SemanticView, ViewSignal
from pydantic import ValidationError

_DEFAULT_VIEWS = {
    "correctness": SemanticView(
        reducer=SemanticReducer.SINGLE,
        signals=[ViewSignal(metric="exact-match", output="score")],
    )
}
_DEFAULT_METADATA = [MetadataItem(key="suite", value="smoke")]


def _task(
    *,
    name: str = "task-1",
    workspace: str = "default",
    project: str | None = None,
    intent: str = "Answer the question.",
    inputs: TaskInputs | None = None,
    reference: dict[str, Any] | None = None,
    metrics: list[MetricRef] | None = None,
    views: dict[str, SemanticView] | None = None,
    metadata: list[MetadataItem] | None = None,
) -> TaskEntity:
    return TaskEntity(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent=intent,
            inputs=inputs if inputs is not None else TaskInputs(instruction="What is 2+2?"),
            reference=reference if reference is not None else {},
            metrics=metrics if metrics is not None else [MetricRef("default/stored-metric")],
            views=views if views is not None else _DEFAULT_VIEWS,
        ),
        name=name,
        workspace=workspace,
        project=project,
        metadata=metadata if metadata is not None else _DEFAULT_METADATA,
    )


def _harbor_task(*, config: dict[str, Any] | None = None, files_hash: str = "a" * 64) -> TaskEntity:
    return TaskEntity(
        spec=HarborTaskDefinition(
            kind="harbor",
            harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
            source=HarborArchiveSource(
                fileset_ref="default/harbor#packages/o-n/abc/files",
                files_hash=files_hash,
            ),
            config=config if config is not None else {},
        ),
        name="harbor-1",
        workspace="default",
    )


# --- Shape -------------------------------------------------------------------


def test_digest_is_full_length_lowercase_hex() -> None:
    """Never truncated: a shortened prefix would collapse the birthday bound from 2**128."""
    assert re.match(DIGEST_PATTERN, content_hash(_task()))


def test_digest_matches_sha256_of_canonical_payload() -> None:
    """The payload is the compatibility contract; the digest is just its SHA-256."""
    entity = _task()
    expected = hashlib.sha256(canonical_payload(entity).encode("utf-8")).hexdigest()
    assert content_hash(entity) == expected


def test_canonical_payload_is_compact_and_key_sorted() -> None:
    payload = canonical_payload(_task())
    assert ", " not in payload and '": ' not in payload
    keys = list(json.loads(payload).keys())
    assert keys == sorted(keys)


# --- Determinism -------------------------------------------------------------


def test_identical_content_hashes_identically() -> None:
    """The property publish-time dedup depends on: republishing the same content is a no-op."""
    assert content_hash(_task()) == content_hash(_task())


def test_mapping_insertion_order_does_not_affect_digest() -> None:
    """Canonicalization sorts mapping keys, so two equal mappings built in different orders are one
    piece of content — otherwise republishing an unchanged task would cut a spurious revision.

    ``views`` is the mapping to test this with. ``metadata`` is a *list*, where order is significant
    and genuinely changes the digest (see the ordering test below).
    """
    first = SemanticView(reducer=SemanticReducer.SINGLE, signals=[ViewSignal(metric="exact-match", output="score")])
    second = SemanticView(reducer=SemanticReducer.SINGLE, signals=[ViewSignal(metric="contains", output="score")])

    a = _task(views={"correctness": first, "coverage": second})
    b = _task(views={"coverage": second, "correctness": first})

    assert content_hash(a) == content_hash(b)


# --- Identity is not an input ------------------------------------------------


def test_name_and_workspace_do_not_affect_digest() -> None:
    """Content-only. Salting with identity would break dedup and defeat verify-on-read: the
    recomputed digest would match whenever identity matched, regardless of the content beneath."""
    assert content_hash(_task()) == content_hash(_task(name="task-2", workspace="other"))


def test_project_does_not_affect_digest() -> None:
    """``project`` is a server-owned base field, not content."""
    assert content_hash(_task()) == content_hash(_task(project="proj-a"))


def test_extra_exclude_is_honoured() -> None:
    """Revisioned entities exclude their own revision/tag bookkeeping — a revision's digest must
    not cover the index that was assigned because of that digest."""
    entity = _task()
    assert content_hash(entity, exclude={"metadata"}) != content_hash(entity)


# --- Near misses: these must NOT collide -------------------------------------


def test_differing_intent_changes_digest() -> None:
    assert content_hash(_task()) != content_hash(_task(intent="Do something else."))


def test_populated_and_empty_metrics_differ() -> None:
    """Dropping a task's metrics is a content change, so it must cut a new revision.

    Note this is *not* "absent differs from empty": ``metrics`` defaults to ``[]`` and
    ``model_dump`` materializes defaults, so an unset list and an explicitly empty one hash
    identically — see :func:`test_absent_field_collapses_onto_its_default`.
    """
    assert content_hash(_task(metrics=[])) != content_hash(_task())


def test_metric_ref_order_changes_digest() -> None:
    """Sequence order is significant: this function cannot tell a set from an ordered list, so a
    set-semantics field must be normalized by its own model before hashing."""
    a = _task(metrics=[MetricRef("default/m-a"), MetricRef("default/m-b")])
    b = _task(metrics=[MetricRef("default/m-b"), MetricRef("default/m-a")])
    assert content_hash(a) != content_hash(b)


def test_grader_only_reference_changes_digest() -> None:
    """``reference`` decides what a metric grades *against*, so it is task content.

    Two revisions that score the same output differently must not share a digest — otherwise
    publish-time dedup would collapse them and a pin would no longer fix the grading. This is the
    general rule for the digest: it covers anything affecting a task's execution output or the
    mechanism used to grade it.
    """
    assert content_hash(_task(reference={"expected": "Paris"})) != content_hash(_task())
    assert content_hash(_task(reference={"expected": "Paris"})) != content_hash(_task(reference={"expected": "Lyon"}))


def test_nested_view_change_changes_digest() -> None:
    """Nested sub-models participate; a change buried in a view must not be invisible."""
    changed = _task(
        views={
            "correctness": SemanticView(
                reducer=SemanticReducer.SINGLE,
                signals=[ViewSignal(metric="exact-match", output="other")],
            )
        }
    )
    assert content_hash(_task()) != content_hash(changed)


def test_empty_and_populated_metadata_value_differ() -> None:
    a = _task(metadata=[MetadataItem(key="suite", value="smoke")])
    b = _task(metadata=[MetadataItem(key="suite", value="")])
    assert content_hash(a) != content_hash(b)


def test_metadata_rejects_non_json_value() -> None:
    with pytest.raises(ValidationError):
        MetadataItem.model_validate({"key": "invalid", "value": object()})


def test_absent_field_collapses_onto_its_default() -> None:
    """Documented behavior, asserted so a future change to it is deliberate: ``model_dump``
    materializes defaults, so "unset" and "set to the default" are indistinguishable. A model
    needing that distinction must express it (e.g. an optional defaulting to ``None``)."""
    assert content_hash(_task(inputs=TaskInputs())) == content_hash(_task(inputs=TaskInputs(instruction=None)))


def test_int_and_float_render_distinctly() -> None:
    """``1`` and ``1.0`` are distinguishable values, and JSON renders them distinctly."""
    assert content_hash(_task(inputs=TaskInputs.model_validate({"value": 1}))) != content_hash(
        _task(inputs=TaskInputs.model_validate({"value": 1.0}))
    )


# --- Harbor: the one deliberate exclusion ------------------------------------


def test_harbor_config_does_not_change_digest() -> None:
    """``config`` is a *projection* of ``task.toml``, never an execution input.

    Harbor reads the real ``task.toml`` out of the materialized tree at run time, so this copy
    affects neither execution nor grading. Hashing it would buy no coverage and would make revision
    history sensitive to Harbor's serialization — a release that reordered keys or emitted a new
    defaulted field would cut a revision for byte-identical files.

    Exercised through ``head_digest`` rather than ``content_hash``: the exclusion lives in
    ``REVISION_POINTER_EXCLUDE``, not in the hashing primitive.
    """
    plain = _harbor_task()
    configured = _harbor_task(config={"verifier": {"type": "pytest"}, "agent": {"timeout": 600}})
    assert head_digest(plain) == head_digest(configured)


def test_harbor_files_hash_changes_digest() -> None:
    """The invariant that makes excluding ``config`` safe.

    ``files_hash`` is authoritative over every file in the task directory, ``task.toml``
    included — so a config change that genuinely alters execution or grading moves *this* field and
    is covered. If this ever stopped holding, excluding ``config`` would become a real gap.
    """
    assert head_digest(_harbor_task()) != head_digest(_harbor_task(files_hash="b" * 64))


# --- Tasksets ----------------------------------------------------------------


def _taskset(
    *,
    name: str = "set-1",
    workspace: str = "default",
    description: str | None = "A grouping.",
    tasks: list[TaskRef] | None = None,
) -> TasksetEntity:
    return TasksetEntity(
        name=name,
        workspace=workspace,
        description=description,
        tasks=tasks if tasks is not None else [TaskRef("default/task-a"), TaskRef("default/task-b")],
        metadata=[],
    )


def test_taskset_digest_is_content_only() -> None:
    assert content_hash(_taskset()) == content_hash(_taskset(name="set-2", workspace="other"))


def test_taskset_membership_change_changes_digest() -> None:
    """A dataset's identity is its membership: changing a member must change the digest."""
    changed = _taskset(tasks=[TaskRef("default/task-a"), TaskRef("default/task-c")])
    assert content_hash(_taskset()) != content_hash(changed)


def test_taskset_description_change_changes_digest() -> None:
    assert content_hash(_taskset()) != content_hash(_taskset(description="Different."))


@pytest.mark.parametrize("field", ["fileset_ref", "files_hash"])
def test_harbor_source_fields_all_affect_revision(field):
    original = _harbor_task()
    changed = original.model_copy(deep=True)
    assert isinstance(changed.spec, HarborTaskDefinition)
    tree = changed.spec.source
    setattr(tree, field, "d" * 64 if field == "files_hash" else f"default/other#{field}")
    assert head_digest(original) != head_digest(changed)

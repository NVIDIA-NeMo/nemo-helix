# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Revision entity shape and the head/revision digest invariant.

The load-bearing property here: hashing a *head* record (excluding its revision pointers) must
yield the same digest as hashing the corresponding *revision* (excluding its own digest and
ordinal). Publish depends on it to recognize "this content is already published" — if the two
digests can't be compared, every publish allocates a new revision and dedup never fires.
"""

from __future__ import annotations

import re

import pytest
from nemo_evaluator.api.schemas import EvaluatorTaskDefinition, MetadataItem, MetricRef, TaskInputs, TaskRef
from nemo_evaluator.content_hash import content_hash
from nemo_evaluator.entities import (
    REVISION_POINTER_EXCLUDE,
    REVISION_SELF_EXCLUDE,
    TaskEntity,
    TaskRevisionEntity,
    TasksetEntity,
    TasksetRevisionEntity,
)
from nemo_helix_plugin.entity_naming import NAME_MAX_LENGTH, NAME_PATTERN
from pydantic import ValidationError

_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64

_INTENT = "Answer the question."
_INPUTS = TaskInputs(instruction="What is 2+2?")
_METRICS = [MetricRef("default/stored-metric")]
_ANNOTATIONS = [MetadataItem(key="suite", value="smoke")]

_DESCRIPTION = "A grouping."
_MEMBERS = [TaskRef(f"default/task-a#{_DIGEST}")]


def _task_head(*, intent: str = _INTENT, latest_revision: int = 0, tags: dict[str, int] | None = None) -> TaskEntity:
    return TaskEntity(
        spec=EvaluatorTaskDefinition(kind="evaluator", intent=intent, inputs=_INPUTS, metrics=_METRICS),
        name="task-1",
        workspace="default",
        metadata=_ANNOTATIONS,
        latest_revision=latest_revision,
        tags=tags or {},
    )


def _task_revision(*, intent: str = _INTENT, revision: int = 1, digest: str = _DIGEST) -> TaskRevisionEntity:
    return TaskRevisionEntity(
        spec=EvaluatorTaskDefinition(kind="evaluator", intent=intent, inputs=_INPUTS, metrics=_METRICS),
        name=f"rev.{revision}",
        workspace="default",
        content_hash=digest,
        revision=revision,
        metadata=_ANNOTATIONS,
    )


def _taskset_head(*, members: list[TaskRef] | None = None) -> TasksetEntity:
    return TasksetEntity(
        name="set-1",
        workspace="default",
        description=_DESCRIPTION,
        tasks=members if members is not None else _MEMBERS,
    )


def _taskset_revision(*, members: list[TaskRef] | None = None) -> TasksetRevisionEntity:
    return TasksetRevisionEntity(
        name="rev.1",
        workspace="default",
        content_hash=_DIGEST,
        revision=1,
        description=_DESCRIPTION,
        tasks=members if members is not None else _MEMBERS,
    )


# --- The head/revision invariant ---------------------------------------------


def test_task_head_and_revision_digests_agree() -> None:
    assert content_hash(_task_head(), exclude=REVISION_POINTER_EXCLUDE) == content_hash(
        _task_revision(), exclude=REVISION_SELF_EXCLUDE
    )


def test_taskset_head_and_revision_digests_agree() -> None:
    assert content_hash(_taskset_head(), exclude=REVISION_POINTER_EXCLUDE) == content_hash(
        _taskset_revision(), exclude=REVISION_SELF_EXCLUDE
    )


def test_moving_a_tag_does_not_change_the_head_digest() -> None:
    """Tags are pointers, not content. If they were digested, every retag would fork history."""
    tagged = _task_head(latest_revision=7, tags={"latest": 7, "candidate": 3})
    assert content_hash(_task_head(), exclude=REVISION_POINTER_EXCLUDE) == content_hash(
        tagged, exclude=REVISION_POINTER_EXCLUDE
    )


def test_ordinal_does_not_change_the_revision_digest() -> None:
    """Two revisions of identical content digest identically regardless of when they were cut."""
    assert content_hash(_task_revision(revision=1), exclude=REVISION_SELF_EXCLUDE) == content_hash(
        _task_revision(revision=9), exclude=REVISION_SELF_EXCLUDE
    )


def test_content_change_changes_the_revision_digest() -> None:
    assert content_hash(_task_revision(), exclude=REVISION_SELF_EXCLUDE) != content_hash(
        _task_revision(intent="Do something else."), exclude=REVISION_SELF_EXCLUDE
    )


def test_membership_change_changes_the_taskset_revision_digest() -> None:
    """A published dataset's identity is its membership — including which revision of each member."""
    repinned = _taskset_revision(members=[TaskRef(f"default/task-a#{_OTHER_DIGEST}")])
    assert content_hash(_taskset_revision(), exclude=REVISION_SELF_EXCLUDE) != content_hash(
        repinned, exclude=REVISION_SELF_EXCLUDE
    )


# --- Published membership must be digest-pinned ------------------------------


@pytest.mark.parametrize(
    "unpinned",
    [
        "default/task-a",  # bare — resolves through `latest`
        "task-a",  # bare, workspace-relative
        "default/task-a#latest",  # the reserved moving tag
        "default/task-a#candidate",  # a user tag, equally mutable
        f"default/task-a#{'a' * 63}",  # truncated digest is not a digest
        f"default/task-a#{'A' * 64}",  # uppercase hex
    ],
)
def test_published_taskset_rejects_unpinned_members(unpinned: str) -> None:
    """Enforced on the field, not in the publish path, so no writer can bypass it. A tag-pinned
    member would silently re-point the published revision the moment that tag moved."""
    with pytest.raises(ValidationError):
        _taskset_revision(members=[TaskRef(unpinned)])


def test_published_taskset_accepts_digest_pinned_members() -> None:
    assert _taskset_revision(members=[TaskRef(f"other/task-a#{_OTHER_DIGEST}")]).tasks[0].root.endswith(_OTHER_DIGEST)


def test_head_taskset_still_accepts_unpinned_members() -> None:
    """Only *published* membership must be pinned. The head is a working record; a bare ref there
    means `latest`, which is exactly what publish resolves."""
    assert _taskset_head(members=[TaskRef("default/task-a")]).tasks[0].root == "default/task-a"


# --- Digest field validation -------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "a" * 63,  # too short
        "a" * 65,  # too long
        "A" * 64,  # uppercase
        "g" * 64,  # non-hex
        f"sha256:{'a' * 57}",  # algorithm-prefixed
    ],
)
def test_revision_rejects_malformed_digest(bad: str) -> None:
    """The digest is what a consumer compares against on read; a malformed one must not persist."""
    with pytest.raises(ValidationError):
        _task_revision(digest=bad)


def test_revision_ordinal_is_one_based() -> None:
    with pytest.raises(ValidationError):
        _task_revision(revision=0)


# --- Naming ------------------------------------------------------------------


def test_ordinal_names_are_legal_entity_names() -> None:
    """``rev.<n>`` exists because the entity-name rules reject the alternatives."""
    for ordinal in (1, 9, 10, 12345):
        assert re.match(NAME_PATTERN, f"rev.{ordinal}")


def test_a_full_digest_is_not_a_legal_entity_name() -> None:
    """Why revisions aren't named by digest: 64 chars exceeds the cap, and a hex digest usually
    starts with a digit while names must start with a lowercase letter."""
    assert len(_DIGEST) > NAME_MAX_LENGTH
    assert not re.match(NAME_PATTERN, "0" + "a" * 63)


def test_bare_ordinal_is_not_a_legal_entity_name() -> None:
    """Why the ``rev.`` prefix exists rather than naming a revision ``1``."""
    assert not re.match(NAME_PATTERN, "1")

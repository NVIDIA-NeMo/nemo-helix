# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for task/taskset revisions through the SDK against a real platform.

Everything else about revisions is covered by unit tests against an in-memory fake. This suite
exists because that fake has repeatedly been *wrong* in ways that hid real behavior — it initially
ignored filters, ignored ``sort``, and aliased stored and in-memory objects so optimistic-lock
conflicts could never fire. Each of those made correct-looking tests pass over broken assumptions.

So these tests target exactly the things only a real entity store can confirm:

- child records are addressable under a parent, and ordinals collide per-parent rather than globally;
- the ``(parent, data.content_hash)`` query really resolves a digest to its revision;
- server-side ``-created_at`` ordering actually returns revisions newest-first;
- deleting a task cascades to its revisions;
- a published revision is immutable in practice: reading a pinned digest returns the old content
  after the task has moved on.

Pure CRUD (no online target or IGW), so it only needs the host subprocess backend. Shares the evaluator-plugin
integration opt-in (``RUN_AGENT_EVAL_INTEGRATION``) and the session-scoped ``subprocess_platform``.
"""

from __future__ import annotations

import os
import uuid

import pytest
from nemo_evaluator.api.schemas import (
    EvaluatorTaskDefinition,
    HarborTaskDefinition,
    TaskInput,
    TaskInputs,
    TaskRef,
    TasksetInput,
)
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.sdk import NeMoPlatform
from nemo_platform_plugin.workspaces.client import WorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceRequest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RUN_AGENT_EVAL_INTEGRATION"),
        reason="opt-in; set RUN_AGENT_EVAL_INTEGRATION=1 to run (spins real nemo services platforms)",
    ),
]

WORKSPACE = "default"


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _task_input(intent: str = "Answer the question.", *, tags: list[str] | None = None) -> TaskInput:
    return TaskInput(
        spec=EvaluatorTaskDefinition(
            kind="evaluator",
            intent=intent,
            inputs=TaskInputs(instruction="What is 2+2?"),
        ),
        tags=tags or [],
    )


def _client(base_url: str) -> NeMoPlatform:
    client = NeMoPlatform(base_url=base_url, max_retries=2)
    client_from_platform(client, WorkspacesClient).create_workspace(
        exist_ok=True, body=CreateWorkspaceRequest(name=WORKSPACE)
    ).data()
    return client


@pytest.mark.timeout(300)
def test_publish_and_read_a_pinned_revision(subprocess_platform: str) -> None:
    """The core promise: a digest-pinned read returns what was published, not what is current."""
    client = _client(subprocess_platform)
    name = _unique("task")
    try:
        created = client.evaluator.tasks.create(name, task=_task_input("First."), workspace=WORKSPACE)
        assert created.revision == 1
        assert created.tags["latest"] == 1

        first_digest = client.evaluator.tasks.list_revisions(name, workspace=WORKSPACE).data[0].content_hash

        replaced = client.evaluator.tasks.replace(name, task=_task_input("Second."), workspace=WORKSPACE)
        assert replaced.revision == 2

        pinned = client.evaluator.tasks.retrieve(name, revision=first_digest, workspace=WORKSPACE)
        assert isinstance(pinned.spec, EvaluatorTaskDefinition)
        assert pinned.spec.intent == "First."
        assert pinned.revision == 1
        current = client.evaluator.tasks.retrieve(name, workspace=WORKSPACE)
        assert isinstance(current.spec, EvaluatorTaskDefinition)
        assert current.spec.intent == "Second."
    finally:
        client.evaluator.tasks.delete(name, workspace=WORKSPACE)


@pytest.mark.timeout(300)
def test_republishing_identical_content_cuts_no_revision(subprocess_platform: str) -> None:
    """Publish-time dedup against a real store: the ``(parent, content_hash)`` query must find the
    existing revision, or every republish would allocate a new ordinal."""
    client = _client(subprocess_platform)
    name = _unique("task")
    try:
        client.evaluator.tasks.create(name, task=_task_input("Same."), workspace=WORKSPACE)
        again = client.evaluator.tasks.replace(name, task=_task_input("Same."), workspace=WORKSPACE)

        assert again.revision == 1
        assert client.evaluator.tasks.list_revisions(name, workspace=WORKSPACE).pagination.total_results == 1
    finally:
        client.evaluator.tasks.delete(name, workspace=WORKSPACE)


@pytest.mark.timeout(300)
def test_revisions_come_back_newest_first(subprocess_platform: str) -> None:
    """Ordering is server-side (``-created_at``); the fake ignored ``sort`` entirely at first, so
    this is the only place the real ordering is confirmed."""
    client = _client(subprocess_platform)
    name = _unique("task")
    try:
        client.evaluator.tasks.create(name, task=_task_input("One."), workspace=WORKSPACE)
        client.evaluator.tasks.replace(name, task=_task_input("Two."), workspace=WORKSPACE)
        client.evaluator.tasks.replace(name, task=_task_input("Three."), workspace=WORKSPACE)

        page = client.evaluator.tasks.list_revisions(name, workspace=WORKSPACE)
        assert [r.revision for r in page.data] == [3, 2, 1]
    finally:
        client.evaluator.tasks.delete(name, workspace=WORKSPACE)


@pytest.mark.timeout(300)
def test_ordinals_are_scoped_per_task(subprocess_platform: str) -> None:
    """Two tasks each own a ``rev.1``. Parent-scoped uniqueness is what allows that; without it the
    second task's first publish would collide on the name."""
    client = _client(subprocess_platform)
    first, second = _unique("task-a"), _unique("task-b")
    try:
        a = client.evaluator.tasks.create(first, task=_task_input("A."), workspace=WORKSPACE)
        b = client.evaluator.tasks.create(second, task=_task_input("B."), workspace=WORKSPACE)
        assert a.revision == b.revision == 1

        a_digest = client.evaluator.tasks.list_revisions(first, workspace=WORKSPACE).data[0].content_hash
        b_digest = client.evaluator.tasks.list_revisions(second, workspace=WORKSPACE).data[0].content_hash
        assert a_digest != b_digest
    finally:
        client.evaluator.tasks.delete(first, workspace=WORKSPACE)
        client.evaluator.tasks.delete(second, workspace=WORKSPACE)


@pytest.mark.timeout(300)
def test_identical_content_under_two_tasks_does_not_cross_resolve(subprocess_platform: str) -> None:
    """Same content under two tasks yields one digest, so the digest alone cannot identify a
    revision — resolution must be parent-scoped. Only a real store proves the query is."""
    client = _client(subprocess_platform)
    first, second = _unique("task-a"), _unique("task-b")
    try:
        client.evaluator.tasks.create(first, task=_task_input("Shared."), workspace=WORKSPACE)
        client.evaluator.tasks.create(second, task=_task_input("Shared."), workspace=WORKSPACE)

        a_digest = client.evaluator.tasks.list_revisions(first, workspace=WORKSPACE).data[0].content_hash
        b_digest = client.evaluator.tasks.list_revisions(second, workspace=WORKSPACE).data[0].content_hash
        assert a_digest == b_digest, "identical content must digest identically"

        # Each resolves under its own parent, and neither leaks the other's record.
        assert client.evaluator.tasks.retrieve(first, revision=a_digest, workspace=WORKSPACE).name == first
        assert client.evaluator.tasks.retrieve(second, revision=b_digest, workspace=WORKSPACE).name == second
    finally:
        client.evaluator.tasks.delete(first, workspace=WORKSPACE)
        client.evaluator.tasks.delete(second, workspace=WORKSPACE)


@pytest.mark.timeout(300)
def test_tagging_an_older_revision_leaves_latest_alone(subprocess_platform: str) -> None:
    client = _client(subprocess_platform)
    name = _unique("task")
    try:
        client.evaluator.tasks.create(name, task=_task_input("First."), workspace=WORKSPACE)
        first_digest = client.evaluator.tasks.list_revisions(name, workspace=WORKSPACE).data[0].content_hash
        client.evaluator.tasks.replace(name, task=_task_input("Second."), workspace=WORKSPACE)

        tagged = client.evaluator.tasks.tag(name, tag="blessed", revision=first_digest, workspace=WORKSPACE)

        assert tagged.tags["blessed"] == 1
        assert tagged.tags["latest"] == 2, "latest is machine-managed and must not follow a manual tag"
        blessed = client.evaluator.tasks.retrieve(name, tag="blessed", workspace=WORKSPACE)
        assert isinstance(blessed.spec, EvaluatorTaskDefinition)
        assert blessed.spec.intent == "First."
    finally:
        client.evaluator.tasks.delete(name, workspace=WORKSPACE)


@pytest.mark.timeout(300)
def test_deleting_a_task_removes_its_revisions(subprocess_platform: str) -> None:
    """Cascade is a DB-level FK behavior, so it can only be confirmed against real persistence."""
    client = _client(subprocess_platform)
    name = _unique("task")
    client.evaluator.tasks.create(name, task=_task_input("One."), workspace=WORKSPACE)
    client.evaluator.tasks.replace(name, task=_task_input("Two."), workspace=WORKSPACE)

    client.evaluator.tasks.delete(name, workspace=WORKSPACE)

    # Recreating under the same name starts from revision 1 — the old children are gone, so the
    # ordinal is free. A surviving `rev.1` would make this publish conflict.
    recreated = client.evaluator.tasks.create(name, task=_task_input("Fresh."), workspace=WORKSPACE)
    try:
        assert recreated.revision == 1
        assert client.evaluator.tasks.list_revisions(name, workspace=WORKSPACE).pagination.total_results == 1
    finally:
        client.evaluator.tasks.delete(name, workspace=WORKSPACE)


@pytest.mark.timeout(300)
def test_taskset_membership_is_pinned_and_stays_pinned(subprocess_platform: str) -> None:
    """The reproducibility guarantee, end to end: a published taskset keeps naming the member
    revision it was created with, even after that member publishes new content."""
    client = _client(subprocess_platform)
    task_name, set_name = _unique("task"), _unique("ts")
    try:
        client.evaluator.tasks.create(task_name, task=_task_input("Original."), workspace=WORKSPACE)

        created = client.evaluator.tasksets.create(
            set_name, taskset=TasksetInput(tasks=[TaskRef(task_name)]), workspace=WORKSPACE
        )
        member = created.tasks[0].root
        assert "#" in member, "membership must be stored digest-pinned"
        pinned_digest = member.split("#", 1)[1]
        # A fragment alone only proves *some* sub-entity was named. Pin it to the member's actual
        # content digest, which is what makes the reference content-addressed rather than a label.
        task_revisions = client.evaluator.tasks.list_revisions(task_name, workspace=WORKSPACE)
        assert pinned_digest == task_revisions.data[0].content_hash

        # The member moves on; the published taskset must not.
        client.evaluator.tasks.replace(task_name, task=_task_input("Updated."), workspace=WORKSPACE)

        assert client.evaluator.tasksets.retrieve(set_name, workspace=WORKSPACE).tasks[0].root == member
        pinned_task = client.evaluator.tasks.retrieve(task_name, revision=pinned_digest, workspace=WORKSPACE)
        assert isinstance(pinned_task.spec, EvaluatorTaskDefinition)
        assert pinned_task.spec.intent == "Original."
    finally:
        client.evaluator.tasksets.delete(set_name, workspace=WORKSPACE)
        client.evaluator.tasks.delete(task_name, workspace=WORKSPACE)


@pytest.mark.timeout(300)
def test_republishing_a_taskset_after_a_member_moves_cuts_a_revision(subprocess_platform: str) -> None:
    """Members re-resolve on every write, so identical member *names* can still be a real change.
    This is the semantic that is hardest to change later, so it is worth pinning against a real
    store rather than only against a fake."""
    client = _client(subprocess_platform)
    task_name, set_name = _unique("task"), _unique("ts")
    try:
        client.evaluator.tasks.create(task_name, task=_task_input("v1."), workspace=WORKSPACE)
        created = client.evaluator.tasksets.create(
            set_name, taskset=TasksetInput(tasks=[TaskRef(task_name)]), workspace=WORKSPACE
        )
        assert created.revision == 1

        client.evaluator.tasks.replace(task_name, task=_task_input("v2."), workspace=WORKSPACE)
        republished = client.evaluator.tasksets.replace(
            set_name, taskset=TasksetInput(tasks=[TaskRef(task_name)]), workspace=WORKSPACE
        )

        assert republished.revision == 2, "the grouping names different content, so it is a new revision"
        assert republished.tasks[0].root != created.tasks[0].root
    finally:
        client.evaluator.tasksets.delete(set_name, workspace=WORKSPACE)
        client.evaluator.tasks.delete(task_name, workspace=WORKSPACE)


# --- Harbor-kind tasks --------------------------------------------------------


def _harbor_input(digest: str = "a" * 64, *, config: dict | None = None) -> TaskInput:
    return TaskInput(
        spec=HarborTaskDefinition(
            kind="harbor",
            archive_ref="default/harbor-tasks#packages/org-name/abc/dist.tar.gz",
            archive_digest=digest,
            instruction="Fix the failing test.",
            config=config if config is not None else {"verifier": {"type": "pytest"}},
        )
    )


@pytest.mark.timeout(300)
def test_harbor_and_evaluator_tasks_coexist(subprocess_platform: str) -> None:
    """Both kinds are one record type, so they list together and a taskset can group them — the
    point of managing every evaluation unit in one place."""
    client = _client(subprocess_platform)
    harbor_name, evaluator_name = _unique("harbor"), _unique("evaluator")
    try:
        harbor = client.evaluator.tasks.create(harbor_name, task=_harbor_input(), workspace=WORKSPACE)
        evaluator = client.evaluator.tasks.create(evaluator_name, task=_task_input(), workspace=WORKSPACE)

        assert harbor.spec.kind == "harbor"
        assert evaluator.spec.kind == "evaluator"

        listed = {t.name: t.spec.kind for t in client.evaluator.tasks.list(workspace=WORKSPACE, page_size=1000).data}
        assert listed[harbor_name] == "harbor"
        assert listed[evaluator_name] == "evaluator"
    finally:
        client.evaluator.tasks.delete(harbor_name, workspace=WORKSPACE)
        client.evaluator.tasks.delete(evaluator_name, workspace=WORKSPACE)


@pytest.mark.timeout(300)
def test_harbor_task_round_trips_through_the_store(subprocess_platform: str) -> None:
    """The discriminated union has to survive the entity store's JSON column, which is the one
    thing a unit test against an in-memory fake cannot confirm."""
    client = _client(subprocess_platform)
    name = _unique("harbor")
    try:
        client.evaluator.tasks.create(name, task=_harbor_input(), workspace=WORKSPACE)

        fetched = client.evaluator.tasks.retrieve(name, workspace=WORKSPACE)
        assert isinstance(fetched.spec, HarborTaskDefinition)
        assert fetched.spec.kind == "harbor"
        assert fetched.spec.archive_digest == "a" * 64
        assert fetched.spec.config == {"verifier": {"type": "pytest"}}
        assert fetched.spec.instruction == "Fix the failing test."
    finally:
        client.evaluator.tasks.delete(name, workspace=WORKSPACE)


@pytest.mark.timeout(300)
def test_harbor_config_changes_do_not_cut_a_revision(subprocess_platform: str) -> None:
    """`config` is excluded from the digest because it is a projection of task.toml inside the
    archive. Confirmed end-to-end, since the exclusion is applied where the digest is computed."""
    client = _client(subprocess_platform)
    name = _unique("harbor")
    try:
        client.evaluator.tasks.create(name, task=_harbor_input(), workspace=WORKSPACE)

        same = client.evaluator.tasks.replace(
            name, task=_harbor_input(config={"verifier": {"type": "pytest"}, "new_field": 1}), workspace=WORKSPACE
        )
        assert same.revision == 1, "a config-only change must not publish"
        assert isinstance(same.spec, HarborTaskDefinition)
        assert same.spec.config["new_field"] == 1

        moved = client.evaluator.tasks.replace(name, task=_harbor_input(digest="b" * 64), workspace=WORKSPACE)
        assert moved.revision == 2, "an archive change must publish"
    finally:
        client.evaluator.tasks.delete(name, workspace=WORKSPACE)
